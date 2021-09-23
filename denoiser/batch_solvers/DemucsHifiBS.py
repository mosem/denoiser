import itertools
import torch
import torch.nn.functional as F
from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.demucs_hifi_gen import DemucsHifi
from denoiser.models.hifi_gan_models import HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator, discriminator_loss, \
    feature_loss, generator_loss
from torchaudio.transforms import MelSpectrogram


class DemucsHifiBS(BatchSolver):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._models_dict = self._construct_models()
        self._opt_dict = self._construct_optimizers()
        self._mel = MelSpectrogram(sample_rate=args.sample_rate,
                                   n_fft=args.hifi.n_fft,
                                   win_length=args.hifi.win_size,
                                   hop_length=args.hifi.hop_size,
                                   n_mels=args.hifi.n_mels)

    def get_models(self):
        return self._models_dict

    def get_optimizers(self):
        return self._opt_dict

    def _construct_models(self):
        sample_rate = self.args.sample_rate
        d2e_args = self.args.demucs2embedded
        d2e_args.sample_rate = sample_rate
        gen = DemucsHifi(self.args.demucs, d2e_args, self.args.hifi)
        mpd = HifiMultiPeriodDiscriminator()
        msd = HifiMultiScaleDiscriminator()
        return {'gen': gen, 'mpd': mpd, 'msd': msd}

    def _construct_optimizers(self):
        gen_opt = torch.optim.AdamW(self._models_dict['gen'].parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        disc_opt = torch.optim.AdamW(itertools.chain(self._models_dict['mpd'].parameters(),
                                                     self._models_dict['msd'].parameters()),
                                     lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        return {'gen_opt': gen_opt, "disc_opt": disc_opt}

    def run(self, data, cross_valid=False):
        x, y = data
        generator = self._models_dict['gen']
        mpd = self._models_dict['mpd']
        msd = self._models_dict['msd']

        optim_g, optim_d = self._opt_dict['gen_opt'], self._opt_dict['disc_opt']

        y_g_hat = generator(x)

        if y.shape[2] < y_g_hat.shape[2]:
            y_g_hat = y_g_hat[:, :, :y.shape[2]]
        elif y.shape[2] > y_g_hat.shape[2]:
            y = y[:, :, :y_g_hat.shape[2]]

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

        if self.args.hifi.with_spec:
            y_g_hat = self._mel(y_g_hat.squeeze(1))
            y = self._mel(y.squeeze(1))
        loss_audio = F.l1_loss(y, y_g_hat) * self.args.hifi.l1_factor

        loss_gen_all = self.args.hifi.gen_factor * loss_gen_all + loss_audio

        if not cross_valid:
            loss_gen_all.backward()
            optim_g.step()

        del y, y_ds_hat_g, y_df_hat_g, fmap_s_r, fmap_s_g, fmap_f_r, fmap_f_g, \
            y_ds_hat_r, y_ds_hat_g, y_df_hat_r, y_df_hat_g

        return {'L1': loss_audio, 'Gen_loss': loss_gen_all - loss_audio, 'Disc_loss': loss_disc_all}

    def get_evaluation_loss(self, losses_dict):
        return losses_dict['Gen_loss'] * self.args.hifi.gen_factor + losses_dict['L1']

