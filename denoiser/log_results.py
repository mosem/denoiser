import os
import pandas as pd
import torchaudio
import wandb

from .evaluate import get_metrics, _snr, _process_signals_lengths, init_wandb_table, add_data_to_wandb_table
from .audio import find_audio_files

import logging
logger = logging.getLogger(__name__)


def create_results_df(args):
    wandb_table = init_wandb_table()
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

        clean, noisy_upsampled, enhanced = _process_signals_lengths(args, clean, noisy, enhanced)

        noisy_snr = _snr(noisy_upsampled, noisy_upsampled - clean).item()
        enhanced_snr = _snr(enhanced, enhanced - clean).item()
        pesq, stoi = get_metrics(clean, enhanced, args)

        filename = os.path.basename(clean_path).rstrip('_clean.wav')
        df.loc[i] = [filename, noisy_snr, enhanced_snr, pesq, stoi]
        add_data_to_wandb_table((clean, noisy_upsampled, enhanced), (pesq, stoi), filename, args, wandb_table)

    wandb.log({"Results": wandb_table})
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