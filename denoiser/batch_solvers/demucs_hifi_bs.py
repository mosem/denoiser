import itertools

import torch
import logging

import torch.nn.functional as F
import torchaudio

from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.autoencoder_composer import Autoencoder
from denoiser.models.dataclasses import FeaturesConfig

from denoiser.stft_loss import MultiResolutionSTFTLoss
from denoiser.utils import load_lexical_model

GEN = "generator"
G_OPT = "generator_optimizer"
DISC = "discriminator"
D_OPT = "discriminator optimizer"


class DemucsHifiBS(BatchSolver):

    def __init__(self, args, generator, discriminator, features_config: FeaturesConfig=None):
        super(DemucsHifiBS, self).__init__(args)
        self.device = args.device
        self.include_disc = args.experiment.demucs_hifi_bs.include_disc
        self.include_ft = features_config.include_ft if features_config is not None else False

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        self._models = {GEN: generator, DISC: discriminator} if discriminator is not None else {GEN: generator}
        self._optimizers = self._generate_optimizers()
        self._losses = ['L1', 'Gen_loss', 'Disc_loss'] if self.include_disc else ['L1']
        self._losses_names = self._losses
        self.first_disc_epoch = args.experiment.demucs_hifi_bs.disc_first_epoch if args.experiment.pass_epochs else 0
        self.epoch = 0

        if self.include_ft:
            self.ft_model = load_lexical_model(features_config.feature_model,
                                               features_config.state_dict_path,
                                               args.device)
            self.ft_factor = features_config.features_factor

    def _generate_optimizers(self):
        gen_opt = torch.optim.Adam(self._models[GEN].parameters(), lr=self.args.lr,
                                   betas=(self.args.beta1, self.args.beta2))
        if DISC in self._models.keys():
            disc_opt = torch.optim.Adam(self._models[DISC].get_params_for_optimizer(),
                                        lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
            return {G_OPT: gen_opt, D_OPT: disc_opt}
        else:
            return {G_OPT: gen_opt}

    def get_generator_for_evaluation(self, best_states):
        generator = self._models[GEN]
        generator.load_state_dict(best_states[GEN])
        return generator

    def estimate_valid_length(self, input_length):
        return self._models[GEN].estimate_valid_length(input_length)

    def run(self, data, cross_valid=False, epoch=0):
        self._losses_names = self._losses if epoch >= self.first_disc_epoch else [self._losses[0]]
        noisy, clean = data
        estimate = self._models[GEN](noisy)
        losses = self._get_loss(clean, estimate, epoch)
        if not cross_valid:
            self._optimize(losses, epoch)
        return {k: v.item() for k, v in losses.items()}

    def get_evaluation_loss(self, losses_dict):
        return losses_dict[self._losses_names[0]]

    def _optimize(self, losses, epoch):
        with torch.autograd.set_detect_anomaly(True):
            if self.include_disc and epoch >= self.first_disc_epoch:
                self._optimizers[G_OPT].zero_grad()
                losses[self._losses_names[1]].backward()
                self._optimizers[G_OPT].step()
                self._optimizers[D_OPT].zero_grad()
                disc_tot_loss = losses[self._losses_names[2]]
                disc_tot_loss.backward()
                self._optimizers[D_OPT].step()
            else:
                self._optimizers[G_OPT].zero_grad()
                losses[self._losses_names[0]].backward()
                self._optimizers[G_OPT].step()

    def _get_loss(self, clean, estimate, epoch):
        with torch.autograd.set_detect_anomaly(True):

            if self.include_disc and epoch >= self.first_disc_epoch:
                disc_loss = self._models[DISC].get_disc_loss(clean, estimate)
                audio_loss, gen_loss = self._models[DISC].get_gen_loss(clean, estimate)
                return {self._losses_names[0]: audio_loss, self._losses_names[1]: gen_loss, self._losses_names[2]: disc_loss}
            else:
                audio_loss = F.l1_loss(clean, estimate)

                # MultiResolution STFT loss
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                    audio_loss += sc_loss + mag_loss

            return {self._losses_names[0]: audio_loss}