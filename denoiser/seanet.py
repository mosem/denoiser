import torch.nn as nn
import torch.nn.functional as F
import torch
# from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
from torch.nn import ConstantPad1d
import numpy as np
import math

from .resample import downsample2, upsample2

from .utils import capture_init


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Seanet(nn.Module):

    @capture_init
    def __init__(self,
                 latent_space_size=128,
                 ngf=32, n_residual_layers=3,
                 resample=1,
                 normalize=True,
                 floor=1e-3,
                 ratios =[8, 8, 2, 2],
                 scale_factor=1):
        super().__init__()

        self.resample = resample
        self.normalize = normalize
        self.floor = floor
        self.scale_factor = scale_factor

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.ratios = ratios
        self.hop_length = np.prod(ratios)
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
                       kernel_size = r * 2,
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

    def valid_length(self, length):
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
            kernel_size = 2*stride
            padding = stride // 2 + stride % 2
            length = math.ceil((length - kernel_size + 2*padding) / stride) + 1
            length = max(length, 1)
        for idx in range(depth):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            output_padding = stride % 2
            length = (length - 1) * stride + kernel_size - 2*padding + output_padding
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, mix):

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1

        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))

        if self.scale_factor == 2:
            x = upsample2(x)
        elif self.scale_factor == 4:
            x = upsample2(x)
            x = upsample2(x)

        if self.resample == 2:
            x = upsample2(x)
        skips = []
        for i,encode in enumerate(self.encoder):
            skips.append(x)
            x = encode(x)
        for j,decode in enumerate(self.decoder):
            x = decode(x)
            skip = skips.pop(-1)[..., :x.shape[-1]]
            x = x + skip
        if self.resample == 2:
            x = downsample2(x)
        trim_length = length * self.scale_factor
        if x.size(-1) < trim_length:
            pad = ConstantPad1d((0, trim_length - x.size(-1)), 0)
            x = pad(x)
        elif x.size(-1) > trim_length:
            x = x[..., :trim_length]
        return std * x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    @capture_init
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results


