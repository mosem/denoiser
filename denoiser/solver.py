# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

import torch
import torch.nn.functional as F

from gan_models import generator_loss, feature_loss, discriminator_loss
from . import augment, distrib, pretrained
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress, mel_spectrogram

logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, models, optimizers, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.models = models
        self.dmodel = distrib.wrap(models[0])
        self.optimizers = optimizers

        # data augment
        augments = []
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho))
        self.augment = torch.nn.Sequential(*augments)

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

    # def _serialize(self):
        # package = {}
        # package['model'] = serialize_model(self.model)
        # package['optimizer'] = self.optimizer.state_dict()
        # package['history'] = self.history
        # package['best_state'] = self.best_state
        # package['args'] = self.args
        # tmp_path = str(self.checkpoint_file) + ".tmp"
        # torch.save(package, tmp_path)
        # # renaming is sort of atomic on UNIX (not really true on NFS)
        # # but still less chances of leaving a half written checkpoint behind.
        # os.rename(tmp_path, self.checkpoint_file)
        #
        # # Saving only the latest best model.
        # model = package['model']
        # model['state'] = self.best_state
        # tmp_path = str(self.best_file) + ".tmp"
        # torch.save(model, tmp_path)
        # os.rename(tmp_path, self.best_file)

    def _serialize(self):
        package = {}
        if self.args.model in {'hifi', 'demucs_hifi'}:
            package['model'] = [{"class": model.__class__, "state": copy_state(model.state_dict())} for model in
                                self.models]
        else:
            package['model'] = [serialize_model(model) for model in self.models]
        package['optimizer'] = [optimizer.state_dict() for optimizer in self.optimizers]
        package['history'] = self.history
        package['best_state'] = self.best_states
        package['args'] = self.args
        # tmp_path = str(self.checkpoint_file)
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        models = package['model']
        for i, model in enumerate(models):
            model['state'] = self.best_states[i]
            # tmp_path = str(self.best_file) + f"_model_{i}"
            tmp_path = str(self.best_file) + f"_model_{i}.tmp"
            torch.save(model, tmp_path)
            os.rename(tmp_path, str(self.best_file) + f"_model_{i}")

    def _reset(self):
        # """_reset."""
        # load_from = None
        # load_best = False
        # keep_history = True
        # # Reset
        # if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
        #     load_from = self.checkpoint_file
        # elif self.continue_from:
        #     load_from = self.continue_from
        #     load_best = self.args.continue_best
        #     keep_history = False
        #
        # if load_from:
        #     logger.info(f'Loading checkpoint model: {load_from}')
        #     package = torch.load(load_from, 'cpu')
        #     if load_best:
        #         self.model.load_state_dict(package['best_state'])
        #     else:
        #         self.model.load_state_dict(package['model']['state'])
        #     if 'optimizer' in package and not load_best:
        #         self.optimizer.load_state_dict(package['optimizer'])
        #     if keep_history:
        #         self.history = package['history']
        #     self.best_state = package['best_state']
        # continue_pretrained = self.args.continue_pretrained
        # if continue_pretrained:
        #     logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
        #     model = getattr(pretrained, self.args.continue_pretrained)()
        #     self.model.load_state_dict(model.state_dict())
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
                for i, state in enumerate(package['best_state']):
                    self.models[i].load_state_dict(state)
            else:
                for i, model in enumerate(self.models):
                    model.load_state_dict(package['model'][i]['state'])
            if 'optimizer' in package and not load_best:
                for i, optimizer in enumerate(self.optimizers):
                    optimizer.load_state_dict(package['optimizer'][i])
            if keep_history:
                self.history = package['history']
            self.best_states = package['best_state']
        continue_pretrained = self.args.continue_pretrained
        if self.args.model == 'demucs' and continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            self.models[0] = getattr(pretrained, self.args.continue_pretrained)()
            self.models[0].load_state_dict(self.models[0].state_dict())

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            for model in self.models:
                model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                for model in self.models:
                    model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                for i, model in enumerate(self.models):
                    if self.best_states is None:
                        self.best_states = [[copy_state(model.state_dict())]]
                    elif len(self.best_states) < i + 1:
                        self.best_states.append([copy_state(model.state_dict())])
                    self.best_states[i] = copy_state(model.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                with swap_state(self.models[0], self.best_states[0]):
                    pesq, stoi = evaluate(self.args, self.models[0], self.tt_loader)

                metrics.update({'pesq': pesq, 'stoi': stoi})

                # enhance some samples
                logger.info('Enhance and save samples...')
                enhance(self.args, self.models[0], self.samples_dir)

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
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch
        step_func = self.get_step_func()

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, batch in enumerate(logprog):
            losses, estimate = step_func(batch, cross_valid)

            total_loss += sum([v.item() for v in losses.values()])
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del losses, estimate
        return distrib.average([total_loss / (i + 1)], i + 1)[0]

    def get_step_func(self):
        funcs = {
            'demucs': self._step_func_demucs,
            'hifi': self._step_func_hifi,
            'demucs_hifi': self._step_func_hifi
        }
        return funcs[self.args.model]

    def _step_func_hifi(self, batch, cross_valid=False, *args):

        x, y = [x.to(self.device) for x in batch]
        x = mel_spectrogram(x.squeeze(1), self.args.n_fft, self.args.n_mels,
                            self.args.sample_rate, self.args.hop_size,
                            self.args.win_size)
        generator = self.models[0]
        mpd = self.models[1]
        msd = self.models[2]

        optim_g, optim_d = self.optimizers

        y_g_hat = generator(x)

        if y.shape[2] < y_g_hat.shape[2]:
            y_g_hat = y_g_hat[:, :, :y.shape[2]]
        elif y.shape[2] > y_g_hat.shape[2]:
            y = y[:, :, :y_g_hat.shape[2]]

        # # add masking
        # y_g_hat = y_g_hat * mask

        if not cross_valid:
            optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        del y_df_hat_r, y_df_hat_g

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        # loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        del y_ds_hat_r, y_ds_hat_g

        loss_disc_all = loss_disc_s + loss_disc_f

        if not cross_valid:
            loss_disc_all.backward()
            optim_d.step()

        # Generator
        if not cross_valid:
            optim_g.zero_grad()

        # Loss calc
        # loss_audio = F.l1_loss(y, y_g_hat) * 45
        loss_gen_all = 0
        _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        # y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_gen_all += feature_loss(fmap_f_r, fmap_f_g)
        del fmap_f_r, fmap_f_g
        loss_gen_all += feature_loss(fmap_s_r, fmap_s_g)
        del fmap_s_r, fmap_s_g
        # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        # loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_all += generator_loss(y_df_hat_g)[0]
        del y_df_hat_g
        # loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_all += generator_loss(y_ds_hat_g)[0]
        del y_ds_hat_g
        # loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        if self.args.with_spec:
            y_g_hat = mel_spectrogram(y_g_hat.squeeze(1), self.args.n_fft, self.args.n_mels,
                                          self.args.sample_rate, self.args.hop_size,
                                          self.args.win_size)
            y = mel_spectrogram(y.squeeze(1), self.args.n_fft, self.args.n_mels,
                                    self.args.sample_rate, self.args.hop_size,
                                    self.args.win_size)
        loss_audio = F.l1_loss(y, y_g_hat) * 45
        del y
        loss_gen_all = 2 * loss_gen_all + loss_audio
        # loss_gen_all = 2 * (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f) + loss_audio

        if not cross_valid:
            loss_gen_all.backward()
            optim_g.step()

        return {'L1': loss_audio, 'Gen_loss': loss_gen_all - loss_audio, 'Disc_loss': loss_disc_all}, y_g_hat

    def _step_func_demucs(self, batch, cross_valid, *args):
        noisy, clean = [x.to(self.device) for x in batch]
        if not cross_valid:
            sources = torch.stack([noisy - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            noisy = noise + clean
        estimate = self.dmodel(noisy)
        # apply a loss function after each layer
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
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()

        return {self.args.loss: loss}, estimate