import torch
import logging
import os

import torch.nn.functional as F

from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.models.demucs import Demucs

logger = logging.getLogger(__name__)

class DemucsBS(BatchSolver):


    def __init__(self, args):
        super().__init__(args)
        self.args = args

        generator = Demucs(**args.demucs, scale_factor=args.scale_factor)
        if args.optim == "adam":
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, args.beta2))
        else:
            logger.fatal('Invalid optimizer %s', args.optim)
            os._exit(1)

        self.models = {'generator': generator}
        self.optimizers = {'generator_optimizer': generator_optimizer}

    def train(self):
        for model in self.models.values():
            model.train()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def get_losses_names(self):
        return ['generator']

    def get_models(self):
        return self.models

    def get_optimizers(self):
        return self.optimizers

    def run(self, data, cross_valid=False):
        noisy, clean = data
        estimate = self.models['generator'](noisy)
        losses = {'generator': self._get_loss(clean, estimate, cross_valid)}
        return losses

    def _get_loss(self, clean, estimate, cross_valid):
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

            # optimize model in training mode
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            return loss.item()

    def get_eval_loss(self, losses_dict):
        pass