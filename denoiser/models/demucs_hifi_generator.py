import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.nn import Conv1d

from denoiser.models.modules import BLSTM, HifiResBlock1, HifiResBlock2
from denoiser.resample import upsample2
from denoiser.utils import capture_init, init_weights


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

LRELU_SLOPE = 0.1

class DemucsHifi(nn.Module):
    @capture_init
    def __init__(self,
                 # demucs args
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 scale_factor=1,
                 # embedded dim args
                 include_ft=False,
                 include_skip_connections=True,
                 # hifi args
                 resblock=2,
                 resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 output_length=6_000):
        super().__init__()

        # demucs related
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.include_skip = include_skip_connections
        self.ft_loss = include_ft
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.target_training_length = output_length
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        # hifi generator
        self.num_kernels = len(resblock_kernel_sizes)
        resblock = HifiResBlock1 if str(resblock) == '1' else HifiResBlock2
        channels = []

        self.resblocks = nn.ModuleList()
        self.conv_post = weight_norm(Conv1d(hidden // ch_scale, 1, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            # hifi-gan resblocks
            for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.insert(0, resblock(hidden, k, d))

            # decoding
            decode = []
            decode += [
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

            channels.append(chout // ch_scale)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)


    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        x = mix

        if self.scale_factor == 2:
            x = upsample2(x)
        elif self.scale_factor == 4:
            x = upsample2(x)
            x = upsample2(x)

        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            if self.include_skip:
                skips.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        # embedded dim creation
        if self.ft_loss:
            ft = self.resampler(x)

        # decode to original dims
        for i, decode in enumerate(self.decoder):
            if self.include_skip:
                skip = skips.pop(-1)
                x = x + skip[..., :x.shape[-1]]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = decode(x)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        if self.ft_loss:
            return x * std, ft
        return x * std

    def estimate_valid_length(self, input_length):
        length = math.ceil(input_length * self.scale_factor)
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil(
                (length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)
