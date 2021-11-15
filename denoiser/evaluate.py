# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss
import math
import os
from concurrent.futures import ProcessPoolExecutor
import logging

import torchaudio
import torchvision.utils
from torch.nn import functional as F
from torchaudio.transforms import Spectrogram
from scipy.signal import spectrogram
import wandb

from pesq import pesq
from pystoi import stoi
import torch

from .enhance import get_estimate
from . import distrib
from .resample import upsample2
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)


def evaluate(args, model, data_loader, epoch):
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    model.eval()

    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                (noisy, noisy_path), (clean, clean_path) = data
                filename = os.path.basename(clean_path[0]).rstrip('_clean.wav')
                noisy = noisy.to(args.device)
                clean = clean.to(args.device)
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(estimate_and_run_metrics, clean, model, noisy, filename, args))
                else:
                    estimate = get_estimate(model, noisy)
                    clean, noisy_upsampled, estimate = _process_signals_lengths(args, clean, noisy, estimate)
                    clean = clean.cpu()
                    noisy_upsampled = noisy_upsampled.cpu()
                    estimate = estimate.cpu()
                    estimate_pesq, estimate_stoi = get_metrics(clean, estimate, args)
                    estimate_snr = _snr(estimate, estimate - clean).item()
                    log_file_to_wandb(estimate, (estimate_pesq, estimate_stoi, estimate_snr), filename, args, epoch)
                    pendings.append(
                        pool.submit(run_metrics, (clean, noisy_upsampled, estimate), filename, args))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i

    metrics = [total_pesq, total_stoi]
    pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}.'))
    return pesq, stoi


def init_wandb_table():
    columns = ['filename', 'clean audio', 'clean spectogram', 'noisy audio', 'noisy spectogram', 'enhanced audio',
               'enhanced spectogram', 'noisy snr', 'enhanced snr', 'pesq', 'stoi']
    table = wandb.Table(columns=columns)
    return table


def log_file_to_wandb(enhanced, enhanced_metrics, filename, args, epoch):
    logger.info(f'logging {filename} to wandb.')
    spectrogram_transform = Spectrogram()

    enhanced_pesq, enhanced_stoi, enhanced_snr = enhanced_metrics

    enhaced_spectrogram = wandb.Image(spectrogram_transform(enhanced).log2()[0, :, :].numpy(), caption=filename)
    ehnahced_wandb_audio = wandb.Audio(enhanced.squeeze().numpy(), sample_rate=args.experiment.sample_rate,
                                    caption=filename)
    wandb.log({f'test samples/{filename}/pesq': enhanced_pesq,
               f'test samples/{filename}/stoi': enhanced_stoi,
               f'test samples/{filename}/snr': enhanced_snr,
               f'test samples/{filename}/spectrogram': enhaced_spectrogram,
               f'test samples/{filename}/audio': ehnahced_wandb_audio},
              step=epoch)


def add_data_to_wandb_table(signals, metrics, filename, args, wandb_table):
    logger.info(f'adding {filename} to wandb table.')
    clean, noisy, enhanced = signals

    spectogram_transform = Spectrogram()

    epsilon = 1e-13
    clean_spec = wandb.Image(spectogram_transform(clean).log2()[0, :, :].numpy())
    noisy_spec = wandb.Image((epsilon + spectogram_transform(noisy)).log2()[0, :, :].numpy())
    enhaced_spec = wandb.Image(spectogram_transform(enhanced).log2()[0, :, :].numpy())
    pesq, stoi = metrics
    noisy_snr = _snr(noisy, noisy - clean).item()
    enhanced_snr = _snr(enhanced, enhanced - clean).item()

    clean_sr = args.experiment.sample_rate

    clean_wandb_audio = wandb.Audio(clean.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_clean')
    noisy_wandb_audio = wandb.Audio(noisy.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_noisy')
    enhanced_wandb_audio = wandb.Audio(enhanced.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_enhanced')

    wandb_table.add_data(filename, clean_wandb_audio, clean_spec, noisy_wandb_audio, noisy_spec, enhanced_wandb_audio,
                         enhaced_spec, noisy_snr, enhanced_snr, pesq, stoi)

def estimate_and_run_metrics(clean, model, noisy, filename, args, wandb_table=None):
    estimate = get_estimate(model, noisy)
    signals = (clean, noisy, estimate)
    return run_metrics(signals, filename, args, wandb_table)


def run_metrics(signals, filename, args, wandb_table=None):
    clean, noisy, estimate = signals
    estimate_numpy = estimate.numpy()[:, 0]
    clean_numpy = clean.numpy()[:, 0]

    if args.pesq:
        pesq_i = get_pesq(clean_numpy, estimate_numpy, sr=args.experiment.sample_rate)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean_numpy, estimate_numpy, sr=args.experiment.sample_rate)
    # log_to_wandb_table(signals, (pesq_i, stoi_i), filename, args, wandb_table)
    return pesq_i, stoi_i


def get_metrics(clean, estimate, args):
    estimate_numpy = estimate.numpy()[:, 0]
    clean_numpy = clean.numpy()[:, 0]
    pesq_i = get_pesq(clean_numpy, estimate_numpy, sr=args.experiment.sample_rate)
    stoi_i = get_stoi(clean_numpy, estimate_numpy, sr=args.experiment.sample_rate)
    return pesq_i, stoi_i


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        tmp = pesq(sr, ref_sig[i], out_sig[i], 'wb')  # from pesq
        pesq_val += tmp
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def _snr(signal, noise):
    return (signal**2).mean()/(noise**2).mean()


def _process_signals_lengths(args, clean, noisy, enhanced):
    if args.experiment.scale_factor == 2:
        noisy = upsample2(noisy)
    elif args.experiment.scale_factor == 4:
        noisy = upsample2(noisy)
        noisy = upsample2(noisy)

    if clean.shape[-1] < noisy.shape[-1]:
        logger.info(f'padding clean with {noisy.shape[-1] - clean.shape[-1]} samples.')
        clean = F.pad(clean, (0, noisy.shape[-1] - clean.shape[-1]))
    if enhanced.shape[-1] < noisy.shape[-1]:
        logger.info(f'padding enhanced with {noisy.shape[-1] - enhanced.shape[-1]} samples.')
        enhanced = F.pad(enhanced, (0, noisy.shape[-1] - enhanced.shape[-1]))

    return clean, noisy, enhanced