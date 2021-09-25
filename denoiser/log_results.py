import json
import os
import argparse
import sys

import pandas as pd
import torchaudio

from denoiser.audio import Audioset, find_audio_files
from denoiser.data import NoisyCleanSet


from denoiser import distrib, pretrained
from denoiser.utils import LogProgress
from denoiser.evaluate import _run_metrics

import logging
logger = logging.getLogger(__name__)


def get_clean_dataset(args):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    clean_json_path = os.path.join(paths.test, 'clean.json')
    with open(clean_json_path) as f:
        files = json.load(f)
    return Audioset(files, with_path=True, sample_rate=args.sample_rate)


def snr(signal, noise):
    return (signal**2).mean()/(noise**2).mean()


def log_results(args):
    out_path = os.path.join(args.results_dir, 'results.csv')
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        return df

    logger.info('logging results...')
    df = pd.DataFrame(columns=['filename', 'snr', 'estimated snr', 'pesq', 'stoi'])
    dataset = NoisyCleanSet(args, args.json_dir, with_path=True)
    loader = distrib.loader(dataset, batch_size=1)
    iterator = LogProgress(logger, loader, name="Iterate over noisy/clean dataset")
    for i, ((noisy, noisy_path), (clean, clean_path)) in enumerate(iterator):
        noisy_path = str(noisy_path)
        clean_path = str(clean_path)
        filename = os.path.basename(clean_path).rsplit(".", 1)[0]
        estimated_path = os.path.join(args.results_dir, 'samples' , filename) + "_enhanced.wav"
        estimated, sr = torchaudio.load(estimated_path)
        estimated = estimated.unsqueeze(0) # add batch size of 1

        if clean.shape != estimated.shape:
            min_len = min(estimated.shape[-1], clean.shape[-1])
            estimated = estimated[..., :min_len]
            clean = clean[..., :min_len]
            noisy = noisy[..., :min_len]

        noisy_snr = snr(noisy, noisy-clean).item()
        estimated_snr = snr(estimated, estimated-clean).item()
        pesq, stoi = _run_metrics(clean, estimated, args)

        df.loc[i] = [filename, noisy_snr, estimated_snr, pesq, stoi]
    df.to_csv(out_path)
    return df


def calc_results_histogram(results_df, args):
    n_bins = args.n_bins
    out_path = os.path.join(args.results_dir, 'results_histogram_' + str(n_bins) + '.csv')
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
        return df

    logger.info('calculating histogram...')
    results_histogram_df = pd.DataFrame(columns=['range', 'avg pesq', 'avg stoi'])
    bin_indices, bins = pd.cut(results_df['snr'], n_bins, labels=False, retbins=True, right=False)
    for i in range(n_bins):
        bin_range = (float("{:.2f}".format(bins[i])), float("{:.2f}".format(bins[i+1])))
        bin_avg_pesq = results_df.pesq[bin_indices == i].mean()
        bin_avg_stoi = results_df.stoi[bin_indices == i].mean()
        results_histogram_df.loc[i] = [bin_range, bin_avg_pesq, bin_avg_stoi]
    results_histogram_df.to_csv(out_path)
    return results_histogram_df





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
    parser.add_argument('--json_dir', help='dir contatining json files')
    parser.add_argument('--results_dir', help='dir containing estimated audio files')
    parser.add_argument('--n_bins', default=10, help='number of bins for histogram')



def main():
    parser = argparse.ArgumentParser(
        'denoiser.log_results',
        description="Log PESQ/STOI data on enhanced files")
    add_flags(parser)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.info(args)

    results_csv = log_results(args)
    results_hist = calc_results_histogram(results_csv, args)




if __name__ == "__main__":
    main()
    logger.info('done.')