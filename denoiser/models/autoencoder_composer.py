import math

import torch
from torch import nn

from torch.nn.functional import interpolate

from denoiser.utils import capture_init
from denoiser.resample import ResampleTransform

import logging
logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):

    @capture_init
    def __init__(self, encoder, attention_module, decoder, skips, input_sr: int, target_sr: int, normalize: bool, floor=1e-3):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.skips = skips
        self.floor = floor
        self.normalize = normalize
        self.input_sr = input_sr
        self.target_sr = target_sr

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.resample_transform = ResampleTransform(input_sr, target_sr)

    def calculate_valid_output_length(self, out_len):
        latent_len = self.decoder.calculate_input_length(out_len)
        min_valid_output_len = self.decoder.calculate_output_length(math.floor(latent_len))
        max_valid_output_len = self.decoder.calculate_output_length(math.ceil(latent_len))
        if out_len == min_valid_output_len:
            return min_valid_output_len
        else:
            return max_valid_output_len

    def _calculate_valid_input_len(self, out_len):
        logger.info(f'_calculate_valid_input_len, out_len: {out_len}')
        latent_len = self.decoder.calculate_input_length(out_len)
        logger.info(f'_calculate_valid_input_len, latent_len: {latent_len}')
        in_len_min, in_len_max = self.encoder.calculate_input_range(math.floor(latent_len))
        logger.info(f'_calculate_valid_input_len, in_len_min: {in_len_min}, in_len_max: {in_len_max}')
        return in_len_min

    def _resample_input_signal(self, signal):
        signal_len = signal.shape[-1]
        scale_factor = self.target_sr / self.input_sr
        logger.info(f'resample_input_signal - signal_len: {signal_len}')
        target_len = signal_len * scale_factor
        logger.info(f'resample_input_signal - target_len: {target_len}')
        input_len = self._calculate_valid_input_len(target_len)
        logger.info(f'resample_input_signal - input_len: {input_len}')
        return interpolate(signal, size=input_len, mode='linear')

    def forward(self, signal):
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1

        logger.info(f'forward - signal shape: {signal.shape}')
        signal = self._resample_input_signal(signal)
        logger.info(f'forward - signal shape: {signal.shape}')

        if self.skips:
            latent, skips_signals = self.encoder(signal)
            latent = self.attention_module(latent)
            out = self.decoder(latent, skips_signals)
        else:
            latent = self.encoder(signal)
            logger.info(f'encoder output shape: {latent.shape}')
            latent = self.attention_module(latent)
            logger.info(f'attention output shape: {latent.shape}')
            out = self.decoder(latent)

        logger.info(f'out shape: {out.shape}')

        return std * out
