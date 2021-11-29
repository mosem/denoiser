from torch import nn

from denoiser.models.dataclasses import FeaturesConfig
from denoiser.utils import capture_init


class Autoencoder(nn.Module):

    @capture_init
    def __init__(self, encoder, attention_module, decoder, skips, normalize, floor=1e-3, include_ft_in_output=False):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.skips = skips
        self.floor = floor
        self.normalize = normalize
        self.include_features_in_output = include_ft_in_output

    def estimate_valid_length(self, input_length):
        encoder_output_length = self.encoder.estimate_output_length(input_length)
        attention_output_length = self.attention_module.estimate_output_length(encoder_output_length)
        decoder_output_length = self.decoder.estimate_output_length(attention_output_length)
        return decoder_output_length

    def forward(self, signal):
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1

        if self.skips:
            latent, skips_signals = self.encoder(signal)
            latent = self.attention_module(latent)
            out = self.decoder(latent, skips_signals)
        else:
            latent = self.encoder(signal)
            latent = self.attention_module(latent)
            out = self.decoder(latent)

        if self.include_features_in_output:
            return std*out, latent
        return std * out
