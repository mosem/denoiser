import torch.nn as nn
import math
from denoiser.models.modules import ResnetBlock, WNConv1d, WNConvTranspose1d, weights_init
from denoiser.resample import downsample2, upsample2
from denoiser.utils import capture_init


class Seanet(nn.Module):

    @capture_init
    def __init__(self,
                 latent_space_size=128,
                 ngf=32, n_residual_layers=3,
                 resample=1,
                 normalize=True,
                 floor=1e-3,
                 ratios=[8, 8, 2, 2],
                 scale_factor=1):
        super().__init__()

        self.resample = resample
        self.normalize = normalize
        self.floor = floor
        self.scale_factor = scale_factor

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.ratios = ratios
        mult = int(2 ** len(ratios))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(latent_space_size, mult * ngf, kernel_size=7, padding=0),
        ]

        encoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(mult * ngf, latent_space_size, kernel_size=7, padding=0)
        ]

        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        for i, r in enumerate(ratios):
            encoder_block = [
                nn.LeakyReLU(0.2),
                WNConv1d(mult * ngf // 2,
                         mult * ngf,
                         kernel_size=r * 2,
                         stride=r,
                         padding=r // 2 + r % 2,
                         ),
            ]

            decoder_block = [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers - 1, -1, -1):
                encoder_block = [ResnetBlock(mult * ngf // 2, dilation=3 ** j)] + encoder_block

            for j in range(n_residual_layers):
                decoder_block += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

            self.encoder.insert(0, nn.Sequential(*encoder_block))
            self.decoder.append(nn.Sequential(*decoder_block))

        encoder_wrapper_conv_layer = [
            nn.ReflectionPad1d(3),
            WNConv1d(1, ngf, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        self.apply(weights_init)

    def estimate_valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.scale_factor)
        length = math.ceil(length * self.resample)
        depth = len(self.ratios)
        for idx in range(depth - 1, -1, -1):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            length = math.ceil((length - kernel_size + 2 * padding) / stride) + 1
            length = max(length, 1)
        for idx in range(depth):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            output_padding = stride % 2
            length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, signal):

        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1
        x = signal

        if self.scale_factor == 2:
            x = upsample2(x)
        elif self.scale_factor == 4:
            x = upsample2(x)
            x = upsample2(x)

        if self.resample == 2:
            x = upsample2(x)
        skips = []
        for i, encode in enumerate(self.encoder):
            skips.append(x)
            x = encode(x)
        for j, decode in enumerate(self.decoder):
            x = decode(x)
            skip = skips.pop(-1)
            x = x + skip
        if self.resample == 2:
            x = downsample2(x)

        return std * x
