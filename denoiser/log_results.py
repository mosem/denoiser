import os
import pandas as pd
import torchaudio
import wandb
from torchaudio.transforms import Spectrogram

from .evaluate import get_metrics, get_snr, upsample_noisy, pad_signals_to_noisy_length, logger
from .audio import find_audio_files

import logging

from .utils import convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

RESULTS_DF_FILENAME = 'filename'
RESULTS_DF_NOISY_SNR = 'noisy snr'
RESULTS_DF_ENHANCED_SNR = 'enhanced snr'
RESULTS_DF_PESQ = 'pesq'
RESULTS_DF_STOI = 'stoi'

HISTOGRAM_DF_RANGE = 'range'
HISTOGRAM_DF_N_SAMPLES = 'n_samples'
HISTOGRAM_DF_AVG_PESQ = 'avg pesq'
HISTOGRAM_DF_AVG_STOI = 'avg stoi'


def create_results_df(args):
    wandb_table = init_wandb_table()
    df = pd.DataFrame(columns=[RESULTS_DF_FILENAME, RESULTS_DF_NOISY_SNR, RESULTS_DF_ENHANCED_SNR, RESULTS_DF_PESQ,
                               RESULTS_DF_STOI])
    files = find_audio_files(args.samples_dir, progress=False)
    clean_paths = [str(data[0]) for data in files if '_clean' in str(data[0])]
    noisy_paths = [str(data[0]) for data in files if '_noisy' in str(data[0])]
    enhanced_paths = [str(data[0]) for data in files if '_enhanced' in str(data[0])]
    for i, (clean_path, noisy_path, enhanced_path) in enumerate(zip(clean_paths, noisy_paths, enhanced_paths)):
        clean, clean_sr = torchaudio.load(clean_path)
        noisy, noisy_sr = torchaudio.load(noisy_path)
        enhanced, enhanced_sr = torchaudio.load(enhanced_path)

        noisy = upsample_noisy(args, noisy)

        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)

        clean, enhanced = pad_signals_to_noisy_length(clean, noisy.shape[-1], enhanced)

        noisy_snr = get_snr(noisy, noisy - clean).item()
        pesq, stoi, enhanced_snr = get_metrics(clean, enhanced, args.experiment.sample_rate)

        filename = os.path.basename(clean_path).rstrip('_clean.wav')
        df.loc[i] = [filename, noisy_snr, enhanced_snr, pesq, stoi]
        add_data_to_wandb_table((clean, noisy, enhanced), (pesq, stoi), filename, args, wandb_table)

    wandb.log({"Results": wandb_table})
    return df


def create_results_histogram_df(results_df, n_bins):
    results_histogram_df = pd.DataFrame(columns=[HISTOGRAM_DF_RANGE, HISTOGRAM_DF_N_SAMPLES, HISTOGRAM_DF_AVG_PESQ,
                                                 HISTOGRAM_DF_AVG_STOI])
    bin_indices, bins = pd.cut(results_df[RESULTS_DF_NOISY_SNR], n_bins, labels=False, retbins=True, right=False)
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
        logger.info('results.csv file already exists.')
        results_df = pd.read_csv(results_out_path, index_col=False)
    else:
        results_df = create_results_df(args)
        results_df.to_csv(results_out_path)

    n_bins = args.n_bins
    histogram_out_path = 'results_histogram_' + str(n_bins) + '.csv'
    if not os.path.isfile(histogram_out_path):
        results_histogram_df = create_results_histogram_df(results_df, n_bins)
        results_histogram_df.to_csv(histogram_out_path)
    else:
        logger.info('histogram file already exists.')


def init_wandb_table():
    columns = ['filename', 'clean audio', 'clean spectogram', 'noisy audio', 'noisy spectogram', 'enhanced audio',
               'enhanced spectogram', 'noisy snr', 'enhanced snr', 'pesq', 'stoi']
    table = wandb.Table(columns=columns)
    return table


def add_data_to_wandb_table(signals, metrics, filename, args, wandb_table):
    clean, noisy, enhanced = signals

    spectrogram_transform = Spectrogram()

    epsilon = 1e-13
    clean_spectrogram = spectrogram_transform(clean).log2()[0, :, :].numpy()
    noisy_spectrogram = (epsilon + spectrogram_transform(noisy)).log2()[0, :, :].numpy()
    enhanced_spectrogram = spectrogram_transform(enhanced).log2()[0, :, :].numpy()
    clean_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(clean_spectrogram))
    noisy_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(noisy_spectrogram))
    enhaced_wandb_spec = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram))
    pesq, stoi = metrics
    noisy_snr = get_snr(noisy, noisy - clean).item()
    enhanced_snr = get_snr(enhanced, enhanced - clean).item()

    clean_sr = args.experiment.sample_rate

    clean_wandb_audio = wandb.Audio(clean.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_clean')
    noisy_wandb_audio = wandb.Audio(noisy.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_noisy')
    enhanced_wandb_audio = wandb.Audio(enhanced.squeeze().numpy(), sample_rate=clean_sr, caption=filename + '_enhanced')

    wandb_table.add_data(filename, clean_wandb_audio, clean_wandb_spec, noisy_wandb_audio, noisy_wandb_spec,
                         enhanced_wandb_audio, enhaced_wandb_spec, noisy_snr, enhanced_snr, pesq, stoi)
