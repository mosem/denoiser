import itertools

import torch
import torch.nn.functional as F
from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.demucs_hifi_generator import DemucsHifi
from denoiser.models.hifi_gan_loss_functions import discriminator_loss, \
    feature_loss, generator_loss
from denoiser.models.modules import HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator
from torchaudio.transforms import MelSpectrogram


class DemucsHifiBS(BatchSolver):
    GEN = 'gen'
    MPD = 'mpd'
    MSD = 'msd'
    G_OPT = "gen_opt"
    D_OPT = "disc_opt"

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._models = self._construct_models()
        self._optimizers = self._construct_optimizers()
        self._losses_names = ['L1', 'Gen_loss', 'Disc_loss']
        self._mel = MelSpectrogram(sample_rate=args.experiment.sample_rate,
                                   n_fft=args.experiment.demucs_hifi_bs.n_fft,
                                   win_length=args.experiment.demucs_hifi_bs.win_size,
                                   hop_length=args.experiment.demucs_hifi_bs.hop_size,
                                   n_mels=args.experiment.demucs_hifi_bs.n_mels).to(args.device)
        self.ft_loss = self.args.experiment.demucs_hifi.include_ft
        if self.ft_loss: # room for support - not yet implemented
            self.ft_model = None
            self.ft_factor = 0
        else:
            self.ft_model = None
            self.ft_factor = 0

    def estimate_output_length(self, input_length):
        return self._get_generator_model().estimate_valid_length(input_length)

    def get_generator_for_evaluation(self, best_states):
        generator = self._get_generator_model()
        generator.load_state_dict(self._get_generator_state(best_states))
        return generator

    def _get_generator_model(self):
        return self._models[self.GEN]

    def _get_generator_state(self, best_states):
        return best_states[self.GEN]

    def _construct_models(self):
        gen_args = dict(**self.args.experiment.demucs_hifi, **{"output_length": self.args.experiment.sample_rate *
                                        self.args.experiment.segment})
        gen = DemucsHifi(**gen_args).to(self.args.device)
        mpd = HifiMultiPeriodDiscriminator().to(self.args.device)
        msd = HifiMultiScaleDiscriminator().to(self.args.device)
        return {self.GEN: gen, self.MPD: mpd, self.MSD: msd}

    def _construct_optimizers(self):
        gen_opt = torch.optim.AdamW(self._models[self.GEN].parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        disc_opt = torch.optim.AdamW(itertools.chain(self._models[self.MPD].parameters(),
                                                     self._models[self.MSD].parameters()),
                                     lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        return {self.G_OPT: gen_opt, self.D_OPT: disc_opt}

    @staticmethod
    def _disc_step(mpd, msd, y, y_g_hat, cross_valid, optim_d):
        """
        takes a discriminator step during training
        """

        if not cross_valid:
            optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        if not cross_valid:
            loss_disc_all.backward()
            optim_d.step()

        del y_ds_hat_r, y_df_hat_r

        return loss_disc_all

    def _gen_step(self, cross_valid, optim_g, mpd, msd, y, y_g_hat, x_ft=0):
        """
        takes a generator step during training
        """
        # Generator
        if not cross_valid:
            optim_g.zero_grad()

        # Loss calc
        loss_gen_all = 0
        _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_gen_all += feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_all += feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_all += generator_loss(y_df_hat_g)[0]
        loss_gen_all += generator_loss(y_ds_hat_g)[0]

        if self.ft_loss:
            with torch.no_grad:
                y_ft = self.ft_model(y)
                asr_ft_loss = F.l1_loss(y_ft, x_ft) * self.ft_factor
        else:
            asr_ft_loss = 0

        if self.args.experiment.demucs_hifi_bs.with_spec:
            y_g_hat = self._mel(y_g_hat.squeeze(1))
            y = self._mel(y.squeeze(1))
        loss_audio = F.l1_loss(y, y_g_hat) * self.args.experiment.demucs_hifi_bs.l1_factor

        loss_gen_all = self.args.experiment.demucs_hifi_bs.gen_factor * loss_gen_all + loss_audio + asr_ft_loss

        if not cross_valid:
            loss_gen_all.backward()
            optim_g.step()

        del y, y_ds_hat_g, y_df_hat_g, fmap_s_r, fmap_s_g, fmap_f_r, fmap_f_g

        return loss_gen_all, loss_audio

    def run(self, data, cross_valid=False):
        x, y = data
        generator, mpd, msd= self._models[self.GEN], self._models[self.MPD], self._models[self.MSD]

        optim_g, optim_d = self._optimizers[self.G_OPT], self._optimizers[self.D_OPT]

        if self.ft_loss:
            y_g_hat, x_ft = generator(x)
        else:
            y_g_hat = generator(x)
            x_ft = 0

        # take discriminator step
        loss_disc_all = self._disc_step(mpd, msd, y, y_g_hat, cross_valid, optim_d)

        # take generator step
        loss_gen_all, loss_audio = self._gen_step(cross_valid, optim_g, mpd, msd, y, y_g_hat, x_ft)

        return {self._losses_names[0]: loss_audio.item(),
                self._losses_names[1]: loss_gen_all.item() - loss_audio.item(),
                self._losses_names[2]: loss_disc_all.item()}

    def get_evaluation_loss(self, losses_dict):
        return losses_dict[self._losses_names[1]] * self.args.experiment.demucs_hifi_bs.gen_factor + losses_dict[self._losses_names[0]]
