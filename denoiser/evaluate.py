# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

from concurrent.futures import ProcessPoolExecutor
import logging

from pesq import pesq
from pystoi import stoi
import torch

from .enhance import get_estimate
from . import distrib
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)


def evaluate(args, model, data_loader):
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
                noisy = noisy.to(args.device)
                clean = clean.to(args.device)
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(run_metrics, clean, estimate, args))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i

    metrics = [total_pesq, total_stoi]
    pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}.'))
    return pesq, stoi


def estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy)
    return run_metrics(clean, estimate, args)


def run_metrics(clean, estimate, args):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]

    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=args.experiment.sample_rate)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=args.experiment.sample_rate)
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
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')  # from pesq
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
