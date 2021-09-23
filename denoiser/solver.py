# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
import math
from pathlib import Path
import os
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from . import augment, distrib, pretrained
from .augment import Augment
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from .resample import downsample2, upsample2
from .preprocess import  TorchSignalToFrames, TorchOLA

logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, batch_solver, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.batch_solver = batch_solver

        self.augment = Augment(args)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_states = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)
        self._reset()

    def _serialize(self):
        package = {}
        package['models'], package['optimizers'] = self.batch_solver.serialize()
        package['history'] = self.history
        package['best_states'] = self.best_states
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        models = package['models']
        for name, best_state in package['best_states']:
            models[name]['state'] = best_state
            model_name = name + '_' + self.best_file.name
            tmp_path = os.path.join(self.best_file.parent, model_name) + ".tmp"
            torch.save(models[name], tmp_path)
            model_path = Path(self.best_file.parent / model_name)
            os.rename(tmp_path, model_path)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            self.batch_solver.load(package, load_best)
            if keep_history:
                self.history = package['history']
            self.best_states = package['best_states']

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.batch_solver.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            losses = self._run_one_epoch(epoch)
            logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' + ' | '.join([f'{k} Loss {v:.5f}' for k,v in losses.items()])
            logger.info(bold(logger_msg))

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.batch_solver.eval()
                with torch.no_grad():
                    losses = self._run_one_epoch(epoch, cross_valid=True)
                logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                             + ' | '.join([f'{k} Loss {v:.5f}' for k, v in losses.items()])
                logger.info(bold(logger_msg))
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {**losses, 'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_states = self.batch_solver.copy_models_states()

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                generator = self.batch_solver.models['generator']
                # We switch to the best known model for testing
                with swap_state(generator, self.best_states['generator']):
                    pesq, stoi = evaluate(self.args, generator, self.tt_loader)

                metrics.update({'pesq': pesq, 'stoi': stoi})

                # enhance some samples
                logger.info('Enhance and save samples...')
                enhance(self.args, generator, self.samples_dir)

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())


    def _run_one_epoch(self, epoch, cross_valid=False):
        total_losses = {k:0 for k in self.batch_solver.get_keys()}
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)

        for i, (noisy, clean) in enumerate(logprog):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            if not cross_valid:
                noisy, clean = self.augment.augment_data(noisy, clean)

            losses = self.batch_solver.run((noisy, clean))
            for k in self.batch_solver.get_keys():
                total_losses[k] += losses[k]
            losses_info = {k: format(v/(i+1), ".5f") for k,v in total_losses.items()}
            logprog.update(losses_info)
            del losses

        for k,v in total_losses.items():
            total_losses[k] = v/(i+1)

        return total_losses

    def get_loss(self, clean, estimate, cross_valid):
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

            return loss

    def get_adversarial_losses(self, clean, estimate, cross_valid):
        disc_fake_detached = self.disc(estimate.detach())
        disc_real = self.disc(clean)

        loss_D = 0
        for scale in disc_fake_detached:
            loss_D += F.relu(1+ scale[-1]).mean() # TODO: check if this is a mean over time domain or batch domain

        for scale in disc_real:
            loss_D += F.relu(1-scale[-1]).mean()  # TODO: check if this is a mean over time domain or batch domain

        if not cross_valid:
            # self.dDisc.zero_grad() # should I do this?
            self.disc_opt.zero_grad()
            loss_D.backward()
            self.disc_opt.step()

        disc_fake = self.disc(estimate)

        loss_G = 0
        for scale in disc_fake:
            loss_G += F.relu(1-scale[-1]).mean()  # TODO: check if this is a mean over time domain or batch domain

        loss_feat = 0
        feat_weights = 4.0 / (self.args.discriminator.n_layers + 1)
        D_weights = 1.0 / self.args.discriminator.num_D
        wt = D_weights * feat_weights

        for i in range(self.args.discriminator.num_D):
            for j in range(len(disc_fake[i]) -1):
                loss_feat += wt * F.l1_loss(disc_fake[i][j], disc_real[i][j].detach())

        total_loss_G = (loss_G + self.args.lambda_feat * loss_feat)
        if not cross_valid:
            # self.dModel.zero_grad() # should I do this?
            self.optimizer.zero_grad()
            total_loss_G.backward()
            self.optimizer.step()

        return total_loss_G, loss_D

