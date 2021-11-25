import itertools

import torch.nn as nn
import torch
import torchaudio
from torch.nn import Conv1d, functional as F, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm

from denoiser.models.dataclasses import FeaturesConfig
from denoiser.utils import get_padding


class HifiDiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(HifiDiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 128, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(128, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = nn.SiLU()(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class HifiMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HifiMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiDiscriminatorP(2),
            HifiDiscriminatorP(3),
            HifiDiscriminatorP(5),
            HifiDiscriminatorP(7),
            HifiDiscriminatorP(11),
        ])
        self._init_args_kwargs = (None, None)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HifiDiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(HifiDiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 32, 15, 1, padding=7)),
            norm_f(Conv1d(32, 32, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(32, 64, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(64, 64, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(64, 128, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(128, 128, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(128, 128, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(128, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = nn.SiLU()(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class HifiMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HifiMultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            HifiDiscriminatorS(use_spectral_norm=True),
            HifiDiscriminatorS(),
            HifiDiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])
        self._init_args_kwargs = (None, None)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HifiJointDiscriminator(nn.Module):
    def __init__(self, device='cuda', l1_factor=45, gen_factor=2, ft_conf:FeaturesConfig=None):
        super().__init__()
        self.mpd = HifiMultiPeriodDiscriminator().to(device)
        self.msd = HifiMultiScaleDiscriminator().to(device)
        self.include_ft = ft_conf.include_ft if ft_conf is not None else False
        self.ft_conf = ft_conf
        self.l1_factor = l1_factor
        self.ft_factor = gen_factor

    def forward(self, clean, predicted):
        if self.include_ft:
            estimate, _ = predicted
        else:
            estimate = predicted
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(clean, estimate.detach())
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(clean, estimate.detach())
        return y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for disc_real, disc_generated in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - disc_real) ** 2)
            g_loss = torch.mean(disc_generated ** 2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def get_disc_loss(self, clean, estimate):
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(clean, estimate.detach())
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(clean, estimate.detach())
        loss_disc_f, _, _ = self.discriminator_loss(y_df_hat_r, y_df_hat_g)
        loss_disc_s, _, _ = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        return loss_disc_s + loss_disc_f

    def get_params_for_optimizer(self):
        return itertools.chain(self.mpd.parameters(), self.msd.parameters())

    def get_gen_loss(self, clean, predicted):

        if self.include_ft:
            estimate, x_ft = predicted
            with torch.no_grad():
                y_ft = self.ft_model.extract_feats(clean)
                x_ft = torchaudio.transforms.Resample(x_ft.shape[-1], y_ft.shape[-1]).to(self.args.device)(x_ft)
                if x_ft.shape[-2] != y_ft.shape[-2]:
                    x_ft = torchaudio.transforms.Resample(x_ft.shape[-2], y_ft.shape[-2]).to(self.args.device)(
                        x_ft.permute(0, 2, 1)).permute(0, 2, 1)
                asr_ft_loss = F.l1_loss(y_ft, x_ft) * self.ft_factor
        else:
            estimate = predicted
            asr_ft_loss = 0

        audio_loss = F.l1_loss(clean, estimate) * self.l1_factor

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(clean, estimate.detach())
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(clean, estimate.detach())
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g) * self.gen_factor
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g) * self.gen_factor
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)

        return audio_loss, loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + audio_loss + asr_ft_loss



