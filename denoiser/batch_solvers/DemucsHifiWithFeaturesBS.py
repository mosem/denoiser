import itertools
import torch
import torch.nn.functional as F
from denoiser.batch_solvers.DemucsHifiBS import DemucsHifiBS
from denoiser.models.demucs_hifi_gen import load_features_model, DemucsEn, DemucsToEmbeddedDim
from denoiser.models.hifi_gan_models import HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator, discriminator_loss, \
    feature_loss, generator_loss, HifiGenerator


class DemucsHifiWithFeaturesBS(DemucsHifiBS):
    ENC = "enc"
    D2E = "d2e"
    FT = "ft"
    DEC = "dec"

    def __init__(self, args):
        super().__init__(args)
        self.LOSS_NAMES = ['L1', 'Gen_loss', 'Disc_loss', 'Embedded_L1_loss']

    def _construct_models(self):
        sample_rate = self.args.sample_rate
        d2e_args = self.args.demucs2embedded
        d2e_args.sample_rate = sample_rate
        device = 'cuda' if torch.cuda.is_available() and self.args.device != 'cpu' else 'cpu'
        encoder = DemucsEn(self.args.demucs).to(device)
        d2e = DemucsToEmbeddedDim(d2e_args).to(device)
        decoder = HifiGenerator(**self.args.hifi).to(device)
        ft = load_features_model(self.args.features_model.feature_model, self.args.features_model.state_dict_path).to(device)
        mpd = HifiMultiPeriodDiscriminator().to(device)
        msd = HifiMultiScaleDiscriminator().to(device)
        return {self.ENC: encoder, self.D2E: d2e, self.FT: ft, self.DEC: decoder, self.MPD: mpd, self.MSD: msd}

    def _construct_optimizers(self):
        gen_opt = torch.optim.AdamW(itertools.chain(self._models_dict[self.ENC].parameters(),
                                                    self._models_dict[self.D2E].parameters(),
                                                    self._models_dict[self.DEC].parameters()),
                                    lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        disc_opt = torch.optim.AdamW(itertools.chain(self._models_dict[self.MPD].parameters(),
                                                     self._models_dict[self.MSD].parameters()),
                                     lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        return {self.G_OPT: gen_opt, self.D_OPT: disc_opt}

    def get_generator_for_evaluation(self, best_states):
        encoder = self._models_dict[self.ENC]
        d2e = self._models_dict[self.D2E]
        decoder = self._models_dict[self.DEC]
        encoder.load_state_dict(best_states[self.ENC])
        d2e.load_state_dict(best_states[self.D2E])
        decoder.load_state_dict(best_states[self.DEC])
        return torch.nn.Sequential(encoder, d2e, decoder)

    def run(self, data, cross_valid=False):
        x, y = data
        encoder = self._models_dict[self.ENC]
        d2e = self._models_dict[self.D2E]
        decoder = self._models_dict[self.DEC]
        ft = self._models_dict[self.FT]
        mpd = self._models_dict[self.MPD]
        msd = self._models_dict[self.MSD]

        optim_g, optim_d = self._opt_dict[self.G_OPT], self._opt_dict[self.D_OPT]

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

        return {self.LOSS_NAMES[0]: loss_audio, self.LOSS_NAMES[1]: loss_gen_all - loss_audio,
                self.LOSS_NAMES[2]: loss_disc_all, self.LOSS_NAMES[3]: emb_loss}

    def get_evaluation_loss(self, losses_dict):
        return losses_dict['Gen_loss'] * self.args.hifi.gen_factor + losses_dict['L1'] \
               + self.args.features_model.features_factor * losses_dict['Embedded_L1_loss']

