import torch
import logging
import os

import torch.nn.functional as F

from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.demucs import Demucs

from denoiser.stft_loss import MultiResolutionSTFTLoss

logger = logging.getLogger(__name__)

class DemucsBS(BatchSolver):


    def __init__(self, args):
        super(DemucsBS, self).__init__(args)
        self.device = args.device

        generator = Demucs(**args.demucs, scale_factor=args.scale_factor)
        if torch.cuda.is_available():
            generator.cuda()
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, args.beta2))

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        self.models = {'generator': generator}
        self.optimizers = {'generator_optimizer': generator_optimizer}
        self.losses_names = ['generator']


    def calculate_valid_length(self, length):
        return self.models['generator'].calculate_valid_length(length)

    def set_target_training_length(self, target_length):
        self.models['generator'].target_training_length = target_length

    def run(self, data, cross_valid=False):
        noisy, clean = data
        target_length = clean.size(-1) if cross_valid else None
        estimate = self.models['generator'](noisy, target_length)
        loss = self._get_loss(clean, estimate)
        if not cross_valid:
            self._optimize(loss)
        losses = {self.losses_names[0]: loss.item()}
        return losses


    def get_evaluation_loss(self, losses_dict):
        return losses_dict[self.losses_names[0]]


    def get_generator_model(self):
        return self.models['generator']


    def get_generator_state(self, best_states):
        return best_states['generator']


    def _optimize(self, loss):
        self.optimizers['generator_optimizer'].zero_grad()
        loss.backward()
        self.optimizers['generator_optimizer'].step()


    def _get_loss(self, clean, estimate):
        with torch.autograd.set_detect_anomaly(True):
            if self.args.loss == 'l1':
                loss = F.l1_loss(clean, estimate)
            elif self.args.loss == 'l2':
                loss = F.mse_loss(clean, estimate)
            elif self.args.loss == 'huber':
                loss = F.smooth_l1_loss(clean, estimate)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            # MultiResolution STFT loss
            if self.args.stft_loss:
                sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                loss += sc_loss + mag_loss

            return loss