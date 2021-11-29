import torch.nn as nn
import math
from denoiser.models.modules import ResnetBlock, WNConv1d, WNConvTranspose1d, weights_init
from denoiser.resample import downsample2, upsample2
from denoiser.utils import capture_init


class SeanetDecoder(nn.Module):

    @capture_init
    def __init__(self,
                 latent_space_size,
                 ngf=32, n_residual_layers=3,
                 resample=1,
                 ratios=[8, 8, 2, 2],
                 scale_factor=1):
        super().__init__()

        self.resample = resample
        self.scale_factor = scale_factor

        self.decoder = nn.ModuleList()

        self.ratios = ratios
        mult = int(2 ** len(ratios))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(latent_space_size, mult * ngf, kernel_size=7, padding=0),
        ]

        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        for i, r in enumerate(ratios):

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

            for j in range(n_residual_layers):
                decoder_block += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

            self.decoder.append(nn.Sequential(*decoder_block))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        self.apply(weights_init)

    def estimate_output_length(self, input_length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = input_length
        depth = len(self.ratios)
        for idx in range(depth):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            output_padding = stride % 2
            length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, signal, skips=None):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1) #TODO: is this necessary?

        x = signal

        for j, decode in enumerate(self.decoder):
            if skips is not None:
                skip = skips.pop(-1)
                x = x + skip
            x = decode(x) #TODO: should the decoding be done before adding the skip (like in seanet)?

        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        else:
            pass

        return x
