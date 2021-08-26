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
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from .resample import downsample2
from .preprocess import  TorchSignalToFrames, TorchOLA

logger = logging.getLogger(__name__)


class MultipleInputsSequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Solver(object):
    def __init__(self, data, model, optimizer, args, disc=None, disc_opt=None):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer

        self.disc = disc
        self.disc_opt = disc_opt
        if self.disc is not None:
            self.dDisc = distrib.wrap(disc)

        if args.model == "caunet":
            self.signalPreProcessor = TorchSignalToFrames(frame_size=args.frame_size,
                                                          frame_shift=args.frame_shift)
        if args.model == "seanet":
            self.signalPreProcessor = TorchSignalToFrames(frame_size=args.frame_size,
                                                          frame_shift=args.frame_shift)
            self.framesToSignal = TorchOLA()

        # data augment
        augments = []
        sources_sample_rate = math.ceil(args.sample_rate / args.scale_factor)
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, source_sample_rate=sources_sample_rate,
                                             target_sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same, args.scale_factor))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho, target_sample_rate=args.sample_rate, scale_factor=args.scale_factor))
        self.augment = MultipleInputsSequential(*augments)

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
            self.best_disc_file = Path('discriminator_' + args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.best_state_discriminator = None
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
        package['model'] = serialize_model(self.model)
        if self.args.adversarial_mode:
            package['disc'] = serialize_model(self.disc)
            package['disc_opt'] = self.disc_opt.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['best_state_discriminator'] = self.best_state_discriminator
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

        if self.args.adversarial_mode:
            disc = package['disc']
            disc['state'] = self.best_state_discriminator
            tmp_disc_path = str(self.best_disc_file) + '.tmp'
            torch.save(disc,tmp_disc_path)
            os.rename(tmp_disc_path, self.best_disc_file)

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
            if load_best:
                self.model.load_state_dict(package['best_state'])
                self.disc.load_state_dict(package['best_state_discriminator'])
            else:
                self.model.load_state_dict(package['model']['state'])
                if self.disc:
                    self.disc.load_state_dict(package['disc']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if 'dics_opt' in package and not load_best:
                self.disc_opt.load_state_dict(package['dict_opt'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']
            self.best_state_discriminator = package['best_state_discriminator']
        continue_pretrained = self.args.continue_pretrained
        if continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            model = getattr(pretrained, self.args.continue_pretrained)()
            self.model.load_state_dict(model.state_dict())

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            if self.args.adversarial_mode:
                train_loss, discriminator_loss = self._run_one_epoch(epoch)
                logger.info(
                    bold(f'Train Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f} | Discrim. Loss {discriminator_loss:.5f}'))
            else:
                train_loss, _ = self._run_one_epoch(epoch)
                logger.info(
                    bold(f'Train Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                if self.args.adversarial_mode:
                    with torch.no_grad():
                        valid_loss, discriminator_loss = self._run_one_epoch(epoch, cross_valid=True)
                    logger.info(
                        bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f} | Discrim. Loss {discriminator_loss:.5f}'))
                else:
                    with torch.no_grad():
                        valid_loss, _ = self._run_one_epoch(epoch, cross_valid=True)
                    logger.info(
                        bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            if self.args.adversarial_mode:
                metrics = {'train': train_loss, 'discriminator': discriminator_loss, 'valid': valid_loss, 'best': best_loss}
            else:
                metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())
                if self.args.adversarial_mode:
                    self.best_state_discriminator = copy_state(self.disc.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                with swap_state(self.model, self.best_state):
                    pesq, stoi = evaluate(self.args, self.model, self.tt_loader)


                metrics.update({'pesq': pesq, 'stoi': stoi})

                # enhance some samples
                logger.info('Enhance and save samples...')
                enhance(self.args, self.model, self.samples_dir)

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


    def augment_data(self, noisy, clean):
        if self.args.scale_factor == 1:
            sources = torch.stack([noisy - clean, clean])
        elif self.args.scale_factor == 2:
            noisy_downsampled = downsample2(noisy)
            clean_downsampled=  downsample2(clean)
            noise = noisy_downsampled - clean_downsampled
            sources = torch.stack([noise, clean_downsampled])
        elif self.args.scale_factor == 4:
            noisy_downsampled = downsample2(noisy)
            noisy_downsampled = downsample2(noisy_downsampled)
            clean_downsampled = downsample2(clean)
            clean_downsampled = downsample2(clean_downsampled)
            noise = noisy_downsampled - clean_downsampled
            sources = torch.stack([noise, clean_downsampled])
        sources, target = self.augment(sources, clean)
        source_noise, source_clean = sources
        source_noisy = source_noise + source_clean
        return source_noisy, target

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        total_G_loss = 0
        total_D_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, (noisy, clean) in enumerate(logprog):
            if self.args.model == "caunet":
                noisy = self.signalPreProcessor(noisy)
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            if not cross_valid:
                # logger.info(f"noisy shape:{noisy.shape}")
                # logger.info(f"clean shape:{clean.shape}")
                # logger.info(f"clean downsampled shape:{clean_downsampled.shape}")
                noisy, clean = self.augment_data(noisy, clean)

            # with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
            estimate = self.dmodel(noisy)
            # logger.info(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

            # logger.info(f"estimate shape:{estimate.shape}")
            # logger.info(f"noisy shape:{noisy.shape}")
            # logger.info(f"clean shape:{clean.shape}")
            # logger.info(f"clean_downsampled shape:{clean_downsampled.shape}")
            # apply a loss function after each layer
            if self.args.adversarial_mode:
                loss_G, loss_D = self.get_adversarial_losses(clean, estimate, cross_valid)
                total_G_loss += loss_G.item()
                total_D_loss += loss_D.item()
                logprog.update(loss_G=format(total_G_loss / (i + 1), ".5f"), loss_D=format(total_D_loss / (i + 1), ".5f"))
                # Just in case, clear some memory
                del loss_G, loss_D, estimate
            else:
                loss = self.get_loss(clean, estimate, cross_valid)

                total_loss += loss.item()
                logprog.update(loss=format(total_loss / (i + 1), ".5f"))
                # Just in case, clear some memory
                del loss, estimate
        if self.args.adversarial_mode:
            return distrib.average([total_G_loss / (i + 1)], i + 1)[0], distrib.average([total_D_loss / (i + 1)], i + 1)[0]
        else:
            return distrib.average([total_loss / (i + 1)], i + 1)[0], None


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
        disc_fake_detached = self.dDisc(estimate.detach())
        disc_real = self.dDisc(clean)

        loss_D = 0
        for scale in disc_fake_detached:
            loss_D += F.relu(1+ scale[-1]).mean() # TODO: check is this is a mean over time domain or batch domain

        for scale in disc_real:
            loss_D += F.relu(1-scale[-1]).mean()  # TODO: check is this is a mean over time domain or batch domain


        if not cross_valid:
            # self.dDisc.zero_grad() # should I do this?
            self.disc_opt.zero_grad()
            loss_D.backward()
            self.disc_opt.step()


        disc_fake = self.dDisc(estimate)

        loss_G = 0
        for scale in disc_fake:
            loss_G += F.relu(1-scale[-1]).mean()  # TODO: check is this is a mean over time domain or batch domain

        loss_feat = 0
        feat_weights = 4.0 / (self.args.n_layers_D + 1)
        D_weights = 1.0 / self.args.num_D
        wt = D_weights * feat_weights

        for i in range(self.args.num_D):
            for j in range(len(disc_fake[i]) -1):
                loss_feat += wt * F.l1_loss(disc_fake[i][j], disc_real[i][j].detach())

        total_loss_G = (loss_G + self.args.lambda_feat * loss_feat)
        if not cross_valid:
            # self.dModel.zero_grad() # should I do this?
            self.optimizer.zero_grad()
            total_loss_G.backward()
            self.optimizer.step()

        return total_loss_G, loss_D



