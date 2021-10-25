import math
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, ConstantPad1d
from denoiser.models.hifi_gan_models import ResBlock1, ResBlock2

from denoiser.models.hifi_gan_models import HifiGenerator
from denoiser.models.modules import BLSTM
from denoiser.resample import upsample2
from denoiser.utils import capture_init, init_weights

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

LRELU_SLOPE = 0.1

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
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        return std * x


def load_features_model(feature_model, state_dict_path, device):
    if feature_model == 'hubert':
        return huBERT(state_dict_path, 6, device=device)
    elif feature_model == 'cpc':
        return CPC(device=device)
    elif feature_model == 'asr':
        return AsrFeatExtractor(device=device)
    else:
        raise ValueError("Unknown model.")


class DemucsHifi(nn.Module):
    def __init__(self, args, output_length):
        super().__init__()

        # demucs related
        demucs_args = args.experiment.demucs
        demucs2embedded_args = args.experiment.demucs2embedded
        hifi_args = args.experiment.hifi
        self.chin = demucs_args.chin
        self.chout = demucs_args.chout
        self.hidden = demucs_args.hidden
        self.max_hidden = demucs_args.max_hidden
        self.depth = demucs_args.depth
        self.kernel_size = demucs_args.kernel_size
        self.stride = demucs_args.stride
        self.growth = demucs_args.growth
        self.dialation = demucs2embedded_args.dialation
        self.causal = demucs_args.causal
        self.floor = demucs_args.floor
        self.resample = demucs_args.resample
        self.normalize = demucs_args.normalize
        self.scale_factor = demucs_args.scale_factor
        self.include_skip = demucs2embedded_args.include_skip_connections
        self.ft_loss = demucs2embedded_args.include_ft
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.target_training_length = output_length
        activation = nn.GLU(1) if demucs_args.glu else nn.ReLU()
        # ch_scale = 1
        ch_scale = 2 if demucs_args.glu else 1

        kw = {"d2e": demucs2embedded_args, "hifi": hifi_args}
        self._init_args_kwargs = (demucs_args, kw)

        # hifi generator
        self.num_kernels = len(hifi_args.resblock_kernel_sizes)
        self.num_upsamples = len(hifi_args.upsample_rates)
        resblock = ResBlock1 if str(hifi_args.resblock) == '1' else ResBlock2
        args = {
            "resblock": resblock,
            "upsample_rates": hifi_args.upsample_rates,
            "upsample_kernel_sizes": hifi_args.upsample_kernel_sizes,
            "upsample_dialation_sizes": hifi_args.upsample_dialation_sizes,
            "upsample_initial_channel": hifi_args.upsample_initial_channel,
            "input_initial_channel": hifi_args.input_initial_channel,
            "resblock_kernel_sizes": hifi_args.resblock_kernel_sizes,
            "resblock_dilation_sizes": hifi_args.resblock_dilation_sizes,
        }
        self._init_args_kwargs = (args, None)

        channels = []

        self.resblocks = nn.ModuleList()
        self.conv_post = weight_norm(Conv1d(self.hidden // ch_scale, 1, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

        for index in range(demucs_args.depth):
            encode = []
            encode += [
                nn.Conv1d(self.chin, self.hidden, self.kernel_size, self.stride),
                nn.ReLU(),
                nn.Conv1d(self.hidden, self.hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            # hifi-gan resblocks
            for (k, d) in zip(hifi_args.resblock_kernel_sizes, hifi_args.resblock_dilation_sizes):
                self.resblocks.insert(0, resblock(self.hidden, k, d))

            # decoding
            decode = []
            decode += [
                nn.ConvTranspose1d(self.hidden, self.chout, self.kernel_size, self.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            self.chout = self.hidden
            self.chin = self.hidden
            self.hidden = min(int(self.growth * self.hidden), self.max_hidden)

            channels.append(self.chout // ch_scale)

        self.lstm = BLSTM(self.chin, bi=not demucs_args.causal)
        if demucs_args.rescale:
            rescale_module(self, reference=demucs_args.rescale)

        # linear embedded dim
        if self.ft_loss:
            self.resampler = torchaudio.transforms.Resample(output_length // (2**demucs_args.depth),
                                                            int(args.experiment.source_sample_rate *
                                                                args.experiment.segment *
                                                                demucs2embedded_args.slice_rate))

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

        # embedded dim createion
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

