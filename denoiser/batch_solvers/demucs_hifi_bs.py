import itertools

import torch
import logging

import torch.nn.functional as F
import torchaudio

from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.demucs_hifi import DemucsHifi
from denoiser.models.modules import HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator

from denoiser.stft_loss import MultiResolutionSTFTLoss
from denoiser.utils import load_lexical_model

logger = logging.getLogger(__name__)
GEN = "generator"
G_OPT = "generator_optimizer"
MPD = "mpd"
MSD = "msd"
D_OPT = "discriminator optimizer"


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class DemucsHifiBS(BatchSolver):

    def __init__(self, args):
        super(DemucsHifiBS, self).__init__(args)
        self.device = args.device
        self.include_disc = args.experiment.demucs_hifi_bs.include_disc
        self.include_ft = args.experiment.demucs_hifi.include_ft

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        self._models = self._generate_models(args)
        self._optimizers = self._generate_optimizers()
        self._losses_names = ['L1', 'Gen_loss', 'Disc_loss'] if self.include_disc else ['L1']
        self.l1_factor = args.experiment.demucs_hifi_bs.l1_factor
        self.gen_factor = args.experiment.demucs_hifi_bs.gen_factor
        self.disc_factor = args.experiment.demucs_hifi_bs.disc_factor

        if self.include_ft:
            self.ft_model = load_lexical_model(args.experiment.features_model.feature_model,
                                               args.experiment.features_model.state_dict_path,
                                               args.device)
            self.ft_factor = args.experiment.features_model.features_factor

    def _generate_optimizers(self):
        gen_opt = torch.optim.Adam(self._models[GEN].parameters(), lr=self.args.lr,
                                    betas=(self.args.beta1, self.args.beta2))
        if self.include_disc:
            disc_opt = torch.optim.Adam(itertools.chain(self._models[MPD].parameters(),
                                                         self._models[MSD].parameters()),
                                         lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
            return {G_OPT: gen_opt, D_OPT: disc_opt}
        else:
            return {G_OPT: gen_opt}

    def _generate_models(self, args):
        gen = DemucsHifi(**args.experiment.demucs_hifi).to(self.args.device)
        if self.include_disc:
            mpd = HifiMultiPeriodDiscriminator().to(self.args.device)
            msd = HifiMultiScaleDiscriminator().to(self.args.device)
            return {GEN: gen, MPD: mpd, MSD: msd}
        else:
            return {GEN: gen}

    def get_generator_for_evaluation(self, best_states):
        generator = self._models[GEN]
        generator.load_state_dict(best_states[GEN])
        return generator

    def estimate_valid_length(self, input_length):
        return self._models[GEN].estimate_valid_length(input_length)

    def run(self, data, cross_valid=False):
        noisy, clean = data
        estimate = self._models[GEN](noisy)
        losses = self._get_loss(clean, estimate)
        if not cross_valid:
            self._optimize(losses)
        return {k: v.item() for k, v in losses.items()}

    def get_evaluation_loss(self, losses_dict):
        return losses_dict[self._losses_names[0]]

    def _gen_loss(self, clean, predicted):

        if self.include_ft:
            estimate, x_ft = predicted
        else:
            estimate = predicted
            x_ft = 0

        audio_loss = F.l1_loss(clean, estimate) * self.l1_factor

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._models[MPD](clean, estimate.detach())
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._models[MSD](clean, estimate.detach())
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * self.gen_factor
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * self.gen_factor
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        if self.include_ft:
            with torch.no_grad():
                y_ft = self.ft_model.extract_feats(clean)
                x_ft = torchaudio.transforms.Resample(x_ft.shape[-1], y_ft.shape[-1]).to(self.args.device)(x_ft)
                if x_ft.shape[-2] != y_ft.shape[-2]:
                    x_ft = torchaudio.transforms.Resample(x_ft.shape[-2], y_ft.shape[-2]).to(self.args.device)(
                        x_ft.permute(0, 2, 1)).permute(0, 2, 1)
                asr_ft_loss = F.l1_loss(y_ft, x_ft) * self.ft_factor
        else:
            asr_ft_loss = 0

        return audio_loss, loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + audio_loss + asr_ft_loss

    def _disc_loss(self, clean, predicted):

        if self.include_ft:
            estimate, _ = predicted
        else:
            estimate = predicted

        mpd, msd = self._models[MPD], self._models[MSD]

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(clean, estimate.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(clean, estimate.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        return loss_disc_s + loss_disc_f

    def _optimize(self, losses):
        with torch.autograd.set_detect_anomaly(True):
            if self.include_disc:
                self._optimizers[G_OPT].zero_grad()
                losses[self._losses_names[1]].backward()
                self._optimizers[G_OPT].step()
                self._optimizers[D_OPT].zero_grad()
                disc_tot_loss = losses[self._losses_names[2]] * self.disc_factor
                disc_tot_loss.backward()
                self._optimizers[D_OPT].step()
            else:
                self._optimizers[G_OPT].zero_grad()
                losses[self._losses_names[0]].backward()
                self._optimizers[G_OPT].step()

    def _get_loss(self, clean, estimate):
        with torch.autograd.set_detect_anomaly(True):

            if self.include_disc:
                disc_loss = self._disc_loss(clean, estimate)
                audio_loss, gen_loss = self._gen_loss(clean, estimate)
                return {self._losses_names[0]: audio_loss, self._losses_names[1]: gen_loss, self._losses_names[2]: disc_loss}
            else:
                audio_loss = F.l1_loss(clean, estimate)
                # MultiResolution STFT loss
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                    audio_loss += sc_loss + mag_loss

            return {self._losses_names[0]: audio_loss}