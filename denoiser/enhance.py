# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import math
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os

import torch
import torchaudio
from torch.nn import functional as F

from .audio import Audioset, find_audio_files
from . import distrib
from .resample import downsample2

from .utils import LogProgress

logger = logging.getLogger(__name__)


def get_estimate_and_trim(model, noisy, target_length):
    torch.set_num_threads(1)
    with torch.no_grad():
        estimate = model(noisy)
    if estimate.shape[-1] > target_length:
        # logger.info(f'trimming {estimate.shape[-1] - target_length}')
        estimate = estimate[..., :target_length]
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, noisy_sr=16_000, enhanced_sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(noisy, filename + "_noisy.wav", sr=noisy_sr)
        write(estimate, filename + "_enhanced.wav", sr=enhanced_sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(args):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    if paths.noisy_json:
        with open(paths.noisy_json) as f:
            files = json.load(f)
    elif paths.noisy_dir:
        files = find_audio_files(paths.noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by either noisy_dir or noisy_json. "
            "Skipping enhancement.")
        return None
    return Audioset(files, with_path=True, sample_rate=args.experiment.sample_rate)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, sample_rate, target_length):
    estimate = get_estimate_and_trim(model, noisy_signals, target_length)
    save_wavs(estimate, noisy_signals, filenames, out_dir, sr=sample_rate)


def _pad_signal_to_valid_length(signal, calc_valid_length_func, scale_factor):
    valid_length = calc_valid_length_func(math.ceil(signal.shape[-1] / scale_factor))

    if valid_length > signal.shape[-1]:
        # logger.info(f'padding signal: {valid_length - signal.shape[-1]}')
        signal = F.pad(signal, (0, valid_length - signal.shape[-1]))

    return signal


def enhance(args, model, out_dir, calc_valid_length_func):

    model.eval()

    dset = get_dataset(args)
    if dset is None:
        return
    loader = distrib.loader(dset, batch_size=1)

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signal, filenames = data
            noisy_signal = noisy_signal.to(args.device)

            target_length = noisy_signal.shape[-1]

            noisy_signal = _pad_signal_to_valid_length(noisy_signal, calc_valid_length_func,
                                                       args.experiment.scale_factor)

            if args.experiment.scale_factor == 2:
                noisy_signal = downsample2(noisy_signal)
            elif args.experiment.scale_factor == 4:
                noisy_signal = downsample2(noisy_signal)
                noisy_signal = downsample2(noisy_signal)

            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, noisy_signal, filenames, out_dir, args.experiment.sample_rate, target_length))
            else:
                # Forward
                estimate = get_estimate_and_trim(model, noisy_signal, target_length)
                noisy_sr = math.ceil(args.experiment.sample_rate / args.experiment.scale_factor)
                save_wavs(estimate, noisy_signal, filenames, out_dir, noisy_sr=noisy_sr, enhanced_sr=args.experiment.sample_rate)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()