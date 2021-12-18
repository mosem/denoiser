import torch
from torch import nn
from denoiser.models.ft_conditioner import FtConditioner
from denoiser.utils import capture_init


class Autoencoder(nn.Module):

    @capture_init
    def __init__(self, encoder, attention_module, decoder, skips, normalize, floor=1e-3,
                 features_module: FtConditioner = None):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.skips = skips
        self.floor = floor
        self.normalize = normalize
        self.include_features_in_output = features_module is not None and \
                                          features_module.include_ft and \
                                          not features_module.use_as_conditioning
        self.post_attn = features_module is not None and features_module.get_ft_after_lstm
        self.ft_module = features_module

    def estimate_output_length(self, input_length):
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

        input_signal = signal
        if self.ft_module is not None and self.ft_module.use_as_conditioning:
            with torch.no_grad():
                features = self.ft_module.extract_feats(input_signal.detach())
        else:
            features = None

        if self.skips:
            pre_attn, skips_signals = self.encoder(signal)
            pre_attn = self.ft_module(pre_attn, features) if self.ft_module is not None and not self.post_attn else pre_attn
            post_attn = self.attention_module(pre_attn)
            post_attn = self.ft_module(post_attn, features) if self.ft_module is not None and self.post_attn else post_attn
            out = self.decoder(post_attn, skips_signals)
        else:
            pre_attn = self.encoder(signal)
            pre_attn = self.ft_module(pre_attn, features) if self.ft_module is not None and not self.post_attn else pre_attn
            post_attn = self.attention_module(pre_attn)
            post_attn = self.ft_module(post_attn, features) if self.ft_module is not None and self.post_attn else post_attn
            out = self.decoder(post_attn)

        if self.include_features_in_output:
            return std * out, post_attn if self.post_attn else pre_attn
        return std * out
