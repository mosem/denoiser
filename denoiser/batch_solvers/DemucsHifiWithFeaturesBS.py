import itertools
import torch
import torch.nn.functional as F
from denoiser.batch_solvers.DemucsHifiBS import DemucsHifiBS
from denoiser.models.demucs_hifi_gen import load_features_model, DemucsEn, DemucsToEmbeddedDim
from denoiser.models.hifi_gan_models import HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator, discriminator_loss, \
    feature_loss, generator_loss, HifiGenerator


class DemucsHifiWithFeaturesBS(DemucsHifiBS):

    def __init__(self, args):
        super().__init__(args)

    def _construct_models(self):
        sample_rate = self.args.sample_rate
        d2e_args = self.args.demucs2embedded
        d2e_args.sample_rate = sample_rate
        encoder = DemucsEn(self.args.demucs)
        d2e = DemucsToEmbeddedDim(d2e_args)
        decoder = HifiGenerator(**self.args.hifi)
        ft = load_features_model(self.args.features_model.feature_model, self.args.features_model.state_dict_path)
        mpd = HifiMultiPeriodDiscriminator()
        msd = HifiMultiScaleDiscriminator()
        return {'enc': encoder, 'd2e': d2e, 'ft': ft, 'dec': decoder, 'mpd': mpd, 'msd': msd}

    def _construct_optimizers(self):
        gen_opt = torch.optim.AdamW(itertools.chain(self._models_dict['enc'].parameters(),
                                                    self._models_dict['ft'].parameters(),
                                                    self._models_dict['dec'].parameters()),
                                    lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        disc_opt = torch.optim.AdamW(itertools.chain(self._models_dict['mpd'].parameters(),
                                                     self._models_dict['msd'].parameters()),
                                     lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        return {'gen_opt': gen_opt, "disc_opt": disc_opt}

    def run(self, data, cross_valid=False):
        x, y = data
        encoder = self._models_dict['enc']
        d2e = self._models_dict['d2e']
        decoder = self._models_dict['dec']
        ft = self._models_dict['ft']
        mpd = self._models_dict['mpd']
        msd = self._models_dict['msd']

        optim_g, optim_d = self._opt_dict['gen_opt'], self._opt_dict['disc_opt']

        x = d2e(encoder(x))

        # collect embedded dim loss
        with torch.no_grad():
            y_ft = ft.extract_feats(y)
        emb_loss = F.l1_loss(x, y_ft)

        y_g_hat = decoder(x)

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

        loss_gen_all = self.args.hifi.gen_factor * loss_gen_all + loss_audio \
                       + self.args.features_model.features_factor * emb_loss

        if not cross_valid:
            loss_gen_all.backward()
            optim_g.step()

        del y, y_ds_hat_g, y_df_hat_g, fmap_s_r, fmap_s_g, fmap_f_r, fmap_f_g, \
            y_ds_hat_r, y_ds_hat_g, y_df_hat_r, y_df_hat_g

        return {'L1': loss_audio, 'Gen_loss': loss_gen_all - loss_audio,
                'Disc_loss': loss_disc_all, 'Embedded_L1_loss': emb_loss}

    def get_eval_loss(self, losses_dict):
        return losses_dict['Gen_loss'] * self.args.hifi.gen_factor + losses_dict['L1'] \
               + self.args.features_model.features_factor * losses_dict['Embedded_L1_loss']

