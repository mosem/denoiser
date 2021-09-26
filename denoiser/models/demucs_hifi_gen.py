import math
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from denoiser.models.hifi_gan_models import HifiGenerator
from denoiser.models.modules import BLSTM
from denoiser.resample import upsample2
from denoiser.utils import capture_init

# asr models
from lese.models.hubert import huBERT
from lese.models.cpc import CPC
from lese.models.asr import AsrFeatExtractor


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


class DemucsToEmbeddedDim(nn.Module):

    def __init__(self, sample_rate=16000, slice_rate=20e-3, ch_in=128, len_in=256, emb_dim=768):
        """

        """
        super().__init__()
        self.l1 = nn.Linear(len_in, int(sample_rate * slice_rate)) # first match the num of expected feature vectors dim
        # self.l2 = nn.Linear(ch_in, emb_dim) # then match embedded dim, requires transpose

    def forward(self, x):
         x = nn.ReLU()(self.l1(x))
         # x = torch.transpose(x, 1, 2)
         return x
         # return nn.ReLU()(self.l2(x))


class DemucsEn(nn.Module):
    """
    Demucs speech enhancement model (only encoder - No skip connections).
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
    def __init__(self,
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
                 scale_factor=1):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.scale_factor = scale_factor

        self.encoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

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
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        # length = mix.shape[-1]
        x = mix
        # x = F.pad(x, (0, self.valid_length(length) - length))

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
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        return std * x


class DemucsHifi(nn.Module):

    def __init__(self, demucs_args, demucs2embedded_args, hifi_args):
        super().__init__()
        self.d = DemucsEn(**demucs_args)
        self.d2e = DemucsToEmbeddedDim(**demucs2embedded_args)
        self.h = HifiGenerator(**hifi_args)
        self.upscale = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        kw = {"d2e": demucs2embedded_args, "hifi": hifi_args}
        self._init_args_kwargs = (demucs_args, kw)

    def forward(self, x):
        x_len = x.shape[-1]
        x_upscaled = self.upscale(x)
        x = self.d(x)
        x = self.d2e(x)
        x = self.h(x)
        return x + x_upscaled


def load_features_model(feature_model, state_dict_path):
    if feature_model == 'hubert':
        return huBERT(state_dict_path, 6)
    elif feature_model == 'cpc':
        return CPC()
    elif feature_model == 'asr':
        return AsrFeatExtractor()
    else:
        raise ValueError("Unknown model.")
