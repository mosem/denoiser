import json
import os
import argparse
import sys

import pandas as pd
import torchaudio

from denoiser.data import NoisyCleanSet

from denoiser import distrib
from denoiser.utils import LogProgress
from .resample import upsample2
from denoiser.evaluate import _run_metrics

from .audio import find_audio_files

import logging
logger = logging.getLogger(__name__)


def _snr(signal, noise):
    return (signal**2).mean()/(noise**2).mean()


def _create_results_df(args):
    logger.info('creating results dataframe...')
    df = pd.DataFrame(columns=['filename', 'snr', 'estimated snr', 'pesq', 'stoi'])
    files = find_audio_files(args.samples_dir, progress=False)
    clean_paths = [str(data[0]) for data in files if '_clean' in str(data[0])]
    enhanced_paths = [str(data[0]) for data in files if '_enhanced' in str(data[0])]
    noisy_paths = [str(data[0]) for data in files if '_noisy' in str(data[0])]
    for i, (clean_path, enhanced_path, noisy_path) in enumerate(zip(clean_paths, enhanced_paths, noisy_paths)):
        clean, clean_sr = torchaudio.load(clean_path)
        noisy, noisy_sr = torchaudio.load(noisy_path)
        enhanced, enhanced_sr = torchaudio.load(enhanced_path)

        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)

        if args.experiment.scale_factor == 2:
            noisy = upsample2(noisy)
        elif args.experiment.scale_factor == 4:
            noisy = upsample2(noisy)
            noisy = upsample2(noisy)

        noisy_snr = _snr(noisy, noisy - clean).item()
        estimated_snr = _snr(enhanced, enhanced - clean).item()
        pesq, stoi = _run_metrics(clean, enhanced, args)

        filename = os.path.basename(clean_path).rstrip('_clean.wav')
        df.loc[i] = [filename, noisy_snr, estimated_snr, pesq, stoi]
    return df


def _create_results_histogram_df(results_df, n_bins):
    logger.info('creating histogram dataframe...')
    results_histogram_df = pd.DataFrame(columns=['range', 'n_samples','avg pesq', 'avg stoi'])
    bin_indices, bins = pd.cut(results_df['snr'], n_bins, labels=False, retbins=True, right=False)
    for i in range(n_bins):
        bin_range = (float("{:.2f}".format(bins[i])), float("{:.2f}".format(bins[i + 1])))
        n_samples_per_bin = len(bin_indices[bin_indices == i])
        bin_avg_pesq = results_df.pesq[bin_indices == i].mean()
        bin_avg_stoi = results_df.stoi[bin_indices == i].mean()
        results_histogram_df.loc[i] = [bin_range, n_samples_per_bin, bin_avg_pesq, bin_avg_stoi]
    return results_histogram_df


def log_results(args):
    results_out_path = 'results.csv'
    if os.path.isfile(results_out_path):
        results_df = pd.read_csv(results_out_path)
    else:

        results_df = _create_results_df(args)
        results_df.to_csv(results_out_path)

    n_bins = args.n_bins
    histogram_out_path = 'results_histogram_' + str(n_bins) + '.csv'
    if not os.path.isfile(histogram_out_path):
        results_histogram_df = _create_results_histogram_df(results_df, n_bins)
        results_histogram_df.to_csv(histogram_out_path)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--sample_rate', default=16_000, type=int, help='target sample rate')
    parser.add_argument('--source_sample_rate', default=8_000, type=int, help='source sample rate')
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                        default=logging.INFO, help="more loggging")
    parser.add_argument('--pesq', action='store_false')
    parser.add_argument('--stoi', action='store_false')
    parser.add_argument('--json_dir', help='dir containing json files') #
    parser.add_argument('--results_dir', help='dir containing estimated audio files') #
    parser.add_argument('--n_bins', default=10, help='number of bins for histogram') #


def main():
    parser = argparse.ArgumentParser(
        'denoiser.log_results',
        description="Log PESQ/STOI data on enhanced files")
    add_flags(parser)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.info(args)

    log_results(args)




if __name__ == "__main__":
    main()
    logger.info('done logging results.')