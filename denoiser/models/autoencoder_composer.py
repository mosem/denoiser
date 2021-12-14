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

    def estimate_valid_length(self, input_length):
        logger.info(f'estimate_valid_length - input_length: {input_length}')
        scale_factor = self.target_sr / self.input_sr
        target_length = scale_factor*input_length
        logger.info(f'target_length: {target_length}')
        latent_len = self.decoder.calculate_input_length(target_length)
        logger.info(f'estimate_valid_length - latent_len: {latent_len}')
        in_len_min, in_len_max = self.encoder.calculate_input_range(math.floor(latent_len))
        logger.info(f'estimate_valid_length - in_len_min: {in_len_min}, in_len_max: {in_len_max}')
        return in_len_min if self.training else in_len_max+1

        # logger.info(f'input_length: {input_length}')
        # resampled_in_len = self.get_resampled_in_len(input_length)
        # logger.info(f'resampled length: {resampled_in_len}')
        # encoder_output_length = self.encoder.estimate_output_length(resampled_in_len)
        # logger.info(f'encoder_output_length: {encoder_output_length}')
        # attention_output_length = self.attention_module.estimate_output_length(encoder_output_length)
        # logger.info(f'attention_output_length: {attention_output_length}')
        # decoder_output_length = self.decoder.estimate_output_length(attention_output_length)
        # logger.info(f'decoder_output_length: {decoder_output_length}')
        # return decoder_output_length

    def calculate_valid_output_length(self, out_len):
        latent_len = self.decoder.calculate_input_length(out_len)
        min_valid_output_len = self.decoder.estimate_output_length(math.floor(latent_len))
        max_valid_output_len = self.decoder.estimate_output_length(math.ceil(latent_len))
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

    # def calculate_input_length(self, out_len):
    #     logger.info(f'calculate_input_length - out_len: {out_len}')
    #     latent_len = self.decoder.calculate_input_length(out_len)
    #     logger.info(f'calculate_input_length - latent_len: {latent_len}')
    #     in_len_min, in_len_max = self.encoder.calculate_input_range(math.floor(latent_len))
    #     logger.info(f'calculate_input_length - in_len_min: {in_len_min}, in_len_max: {in_len_max}')
    #     return in_len_min if self.training else in_len_max + 1

    # def get_resampled_in_len(self, in_len):
    #     scale_factor = self.target_sr / self.input_sr
    #     tmp_in_len = in_len * scale_factor
    #     return tmp_in_len

    def resample_input_signal(self, signal):
        signal_len = signal.shape[-1]
        scale_factor = self.target_sr / self.input_sr
        logger.info(f'resample_input_signal - signal_len: {signal_len}')
        target_len = signal_len * scale_factor
        logger.info(f'resample_input_signal - target_len: {target_len}')
        input_len = self._calculate_valid_input_len(target_len)
        logger.info(f'resample_input_signal - input_len: {input_len}')
        return interpolate(signal, size=input_len, mode='linear')
        #
        # tmp_sr = tmp_in_len/signal_len*self.input_sr
        # logger.info(f'input_sr: {self.input_sr}, tmp_sr: {tmp_sr}')
        # self.resample_transform = ResampleTransform(self.input_sr, tmp_sr).to(self.device)
        # return self.resample_transform(signal)

    # def get_model_ratio(self, input_length):
    #     encoder_output_length = self.encoder.estimate_output_length(input_length)
    #     attention_output_length = self.attention_module.estimate_output_length(encoder_output_length)
    #     decoder_output_length = self.decoder.estimate_output_length(attention_output_length)
    #
    #     encoder_ratio = encoder_output_length / input_length
    #     attention_ratio = attention_output_length / encoder_output_length
    #     decoder_ratio = decoder_output_length / attention_output_length
    #
    #     logger.info(f'input_length: {input_length}, encoder_output_length: {encoder_output_length}, '
    #                 f'attention_output_length: {attention_output_length}, decoder_output_length: {decoder_output_length}')
    #     logger.info(f'encoder ratio: {encoder_ratio}')
    #     logger.info(f'attention ratio: {attention_ratio}')
    #     logger.info(f'decoder ratio: {decoder_ratio}')
    #
    #     return encoder_ratio*attention_ratio*decoder_ratio

    def forward(self, signal):
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1

        logger.info(f'forward - signal shape: {signal.shape}')
        signal = self.resample_input_signal(signal)
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
