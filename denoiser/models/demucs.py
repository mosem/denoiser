# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math

import torch
from torch import nn

from denoiser.models.dataclasses import FeaturesConfig, DemucsConfig
from denoiser.models.ft_conditioner import FtConditioner
from denoiser.models.modules import BLSTM
from denoiser.resample import downsample2, upsample2
from denoiser.utils import capture_init


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.

    """
    @capture_init
    def __init__(self, demucs_config: DemucsConfig,
                 features_module: FtConditioner = None):

        super().__init__()
        if demucs_config.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = demucs_config.chin
        self.chout = demucs_config.chout
        self.hidden = demucs_config.hidden
        self.depth = demucs_config.depth
        self.kernel_size = demucs_config.kernel_size
        self.stride = demucs_config.stride
        self.causal = demucs_config.causal
        self.floor = demucs_config.floor
        self.resample = demucs_config.resample
        self.normalize = demucs_config.normalize
        self.scale_factor = demucs_config.scale_factor
        self.shift = demucs_config.shift
        self.include_features_in_output = features_module is not None and \
                                          features_module.include_ft and \
                                          not features_module.use_as_conditioning
        self.get_ft_after_lstm = features_module is not None and features_module.get_ft_after_lstm
        self.ft_module = features_module

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if demucs_config.glu else nn.ReLU()
        ch_scale = 2 if demucs_config.glu else 1
        chin, hidden, chout = demucs_config.chin, demucs_config.hidden, demucs_config.chout

        for index in range(demucs_config.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, demucs_config.kernel_size, demucs_config.stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, demucs_config.kernel_size, demucs_config.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(demucs_config.growth * hidden), demucs_config.max_hidden)

        self.lstm = BLSTM(chin, bi=not demucs_config.causal)
        if demucs_config.rescale:
            rescale_module(self, reference=demucs_config.rescale)

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length -= self.shift
        length = math.ceil(length * self.scale_factor)
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample)) + self.scale_factor * self.shift
        return int(length)

    def forward(self, signal, expected_size: int=None):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)

        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1
        signal = signal
        input_signal = signal
        if self.ft_module is not None and self.ft_module.use_as_conditioning:
            with torch.no_grad():
                features = self.ft_module.extract_feats(input_signal.detach())
        else:
            features = None

        if self.scale_factor == 2:
            signal = upsample2(signal)
        elif self.scale_factor == 4:
            signal = upsample2(signal)
            signal = upsample2(signal)

        if self.resample == 2:
            signal = upsample2(signal)
        elif self.resample == 4:
            signal = upsample2(signal)
            signal = upsample2(signal)
        skips = []
        for encode in self.encoder:
            signal = encode(signal)
            skips.append(signal)
        pre_lstm = signal
        pre_lstm = self.ft_module(pre_lstm, features) if self.ft_module is not None and not self.get_ft_after_lstm else pre_lstm
        post_lstm = self.lstm(pre_lstm)
        post_lstm = self.ft_module(post_lstm, features) if self.ft_module is not None and self.get_ft_after_lstm else post_lstm
        signal = post_lstm
        for decode in self.decoder:
            skip = skips.pop(-1)
            signal = signal + skip[..., :signal.shape[-1]]
            signal = decode(signal)
        if self.resample == 2:
            signal = downsample2(signal)
        elif self.resample == 4:
            signal = downsample2(signal)
            signal = downsample2(signal)
        else:
            pass

        if expected_size is not None:  # force trimming in case of augmentations
            signal = signal[..., :expected_size]

        if self.include_features_in_output:
            return std * signal, post_lstm if self.get_ft_after_lstm else pre_lstm
        return std * signal
