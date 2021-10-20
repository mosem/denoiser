import os
import pandas as pd
import torchaudio
from torch.nn import functional as F

from .resample import upsample2
from .evaluate import run_metrics
from .audio import find_audio_files

import logging
logger = logging.getLogger(__name__)


def _snr(signal, noise):
    return (signal**2).mean()/(noise**2).mean()


def _process_signals_lengths(args, clean, noisy, enhanced):
    if args.experiment.scale_factor == 2:
        noisy = upsample2(noisy)
    elif args.experiment.scale_factor == 4:
        noisy = upsample2(noisy)
        noisy = upsample2(noisy)

    if clean.shape[-1] < noisy.shape[-1]:
        clean = F.pad(clean, (0, noisy.shape[-1] - clean.shape[-1]))
    if enhanced.shape[-1] < noisy.shape[-1]:
        enhanced = F.pad(enhanced, (0, noisy.shape[-1] - enhanced.shape[-1]))

    return clean, noisy, enhanced

def create_results_df(args):
    df = pd.DataFrame(columns=['filename', 'snr', 'enhanced snr', 'pesq', 'stoi'])
    files = find_audio_files(args.samples_dir, progress=False)
    clean_paths = [str(data[0]) for data in files if '_clean' in str(data[0])]
    noisy_paths = [str(data[0]) for data in files if '_noisy' in str(data[0])]
    enhanced_paths = [str(data[0]) for data in files if '_enhanced' in str(data[0])]
    for i, (clean_path, noisy_path, enhanced_path) in enumerate(zip(clean_paths, noisy_paths, enhanced_paths)):
        clean, clean_sr = torchaudio.load(clean_path)
        noisy, noisy_sr = torchaudio.load(noisy_path)
        enhanced, enhanced_sr = torchaudio.load(enhanced_path)

        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)

        clean, noisy, enhanced = _process_signals_lengths(args, clean, noisy, enhanced)

        noisy_snr = _snr(noisy, noisy - clean).item()
        enhanced_snr = _snr(enhanced, enhanced - clean).item()
        pesq, stoi = run_metrics(clean, enhanced, args)

        filename = os.path.basename(clean_path).rstrip('_clean.wav')
        df.loc[i] = [filename, noisy_snr, enhanced_snr, pesq, stoi]
    return df


def create_results_histogram_df(results_df, n_bins):
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
    logger.info('logging results...')
    results_out_path = 'results.csv'
    if os.path.isfile(results_out_path):
        results_df = pd.read_csv(results_out_path)
    else:
        results_df = create_results_df(args)
        results_df.to_csv(results_out_path)

    n_bins = args.n_bins
    histogram_out_path = 'results_histogram_' + str(n_bins) + '.csv'
    if not os.path.isfile(histogram_out_path):
        results_histogram_df = create_results_histogram_df(results_df, n_bins)
        results_histogram_df.to_csv(histogram_out_path)