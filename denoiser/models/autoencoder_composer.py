import math

from torch import nn

from denoiser.utils import capture_init

import logging
logger = logging.getLogger(__name__)

class Autoencoder(nn.Module):

    @capture_init
    def __init__(self, encoder, attention_module, decoder, skips, normalize, floor=1e-3):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.skips = skips
        self.floor = floor
        self.normalize = normalize

    def estimate_valid_length(self, input_length):
        # logger.info(f'input_length: {input_length}')
        encoder_output_length = self.encoder.estimate_output_length(input_length)
        # logger.info(f'encoder_output_length: {encoder_output_length}')
        attention_output_length = self.attention_module.estimate_output_length(encoder_output_length)
        # logger.info(f'attention_output_length: {attention_output_length}')
        decoder_output_length = self.decoder.estimate_output_length(attention_output_length)
        # logger.info(f'decoder_output_length: {decoder_output_length}')
        encoder_ratio = input_length / encoder_output_length

        return decoder_output_length

    def forward(self, signal):
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1

        # logger.info(f'signal shape: {signal.shape}')

        if self.skips:
            latent, skips_signals = self.encoder(signal)
            latent = self.attention_module(latent)
            out = self.decoder(latent, skips_signals)
        else:
            latent = self.encoder(signal)
            # logger.info(f'encoder output shape: {latent.shape}')
            latent = self.attention_module(latent)
            # logger.info(f'attention output shape: {latent.shape}')
            out = self.decoder(latent)

        # logger.info(f'out shape: {out.shape}')

        return std * out
