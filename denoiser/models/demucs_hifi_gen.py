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
        kw = {"d2e": demucs2embedded_args, "hifi": hifi_args}
        self._init_args_kwargs = (demucs_args, kw)

    def forward(self, x):
        x = self.d(x)
        x = self.d2e(x)
        x = self.h(x)
        return x


def load_features_model(feature_model, state_dict_path):
    if feature_model == 'hubert':
        return huBERT(state_dict_path, 6)
    elif feature_model == 'cpc':
        return CPC()
    elif feature_model == 'asr':
        return AsrFeatExtractor()
    else:
        raise ValueError("Unknown model.")


class DemucsHifiWithSkipConnections(nn.Module):
    def __init__(self, demucs_args, demucs2embedded_args, hifi_args, output_length):
        super().__init__()

        # demucs related
        self.chin = demucs_args.chin
        self.chout = demucs_args.chout
        self.hidden = demucs_args.hidden
        self.depth = demucs_args.depth
        self.kernel_size = demucs_args.kernel_size
        self.stride = demucs_args.stride
        self.causal = demucs_args.causal
        self.floor = demucs_args.floor
        self.resample = demucs_args.resample
        self.normalize = demucs_args.normalize
        self.scale_factor = demucs_args.scale_factor
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.target_training_length = output_length
        activation = nn.GLU(1) if demucs_args.glu else nn.ReLU()
        # ch_scale = 1
        ch_scale = 2 if demucs_args.glu else 1

        kw = {"d2e": demucs2embedded_args, "hifi": hifi_args}
        self._init_args_kwargs = (demucs_args, kw)

        channels = []

        for index in range(demucs_args.depth):
            encode = []
            encode += [
                nn.Conv1d(self.chin, self.hidden, demucs_args.kernel_size, demucs_args.stride),
                nn.ReLU(),
                nn.Conv1d(self.hidden, self.hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d( self.hidden, ch_scale * self.hidden, 1), activation,
                nn.ConvTranspose1d(self.hidden, self.chout, self.kernel_size, self.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            self.chout = self.hidden
            self.chin = self.hidden
            self.hidden = min(int(demucs_args.growth * self.hidden), demucs_args.max_hidden)

            channels.append(self.chout // ch_scale)

        self.lstm = BLSTM(self.chin, bi=not demucs_args.causal)
        if demucs_args.rescale:
            rescale_module(self, reference=demucs_args.rescale)

        # linear embedded dim
        # self.l1 = nn.Linear(demucs2embedded_args.len_in,
        #                     int(demucs2embedded_args.sample_rate * demucs2embedded_args.slice_rate)) # TODO: this supports a fixed length of input, change this?
        #
        # self.l2 = nn.Linear(int(demucs2embedded_args.sample_rate * demucs2embedded_args.slice_rate), demucs2embedded_args.len_in)

        # hifi generator
        self.num_kernels = len(hifi_args.resblock_kernel_sizes)
        self.num_upsamples = len(hifi_args.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(hifi_args.input_initial_channel, hifi_args.upsample_initial_channel, 7, 1, padding=3))
        # self.conv_pre = weight_norm(Conv1d(1, h.upsample_initial_channel, 7, 1, padding=3))
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

        # self.ups = nn.ModuleList()
        # for i, (u, d, k) in enumerate(zip(hifi_args.upsample_rates, hifi_args.upsample_dialation_sizes, hifi_args.upsample_kernel_sizes)):
        #     self.ups.append(weight_norm(
        #         ConvTranspose1d(in_channels=max(hifi_args.upsample_initial_channel // (2 ** i), 1),
        #                         out_channels=max(hifi_args.upsample_initial_channel // (2 ** (i + 1)), 1),
        #                         kernel_size=k,
        #                         stride=u,
        #                         padding=(k - u) // 2,
        #                         dilation=d)))

        self.resblocks = nn.ModuleList()
        ch = 24
        for i in range(len(self.decoder)):
            # ch = upsample_initial_channel
            # ch = max(hifi_args.upsample_initial_channel // (2 ** (i + 1)), 1)
            ch = channels.pop()
            for j, (k, d) in enumerate(zip(hifi_args.resblock_kernel_sizes, hifi_args.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.last_upscaling_conv_transpose = ConvTranspose1d(in_channels=ch,
                                                             out_channels=ch,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             dilation=5)
        self.last_resblocks = nn.ModuleList()
        for (k, d) in zip(hifi_args.resblock_kernel_sizes, hifi_args.resblock_dilation_sizes):
            self.last_resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        # self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

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

        # embedded dim createion
        # x = F.leaky_relu(self.l1(x), LRELU_SLOPE)
        # x = F.leaky_relu(self.l2(x), LRELU_SLOPE)

        # decode to original dims
        for i, decode in enumerate(self.decoder):
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            # if i == 0:
            #     x = F.leaky_relu(self.conv_pre(x), LRELU_SLOPE)
            x = decode(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # upscale to proper length
        x = F.leaky_relu(self.last_upscaling_conv_transpose(x), LRELU_SLOPE)
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.last_resblocks[j](x)
            else:
                xs += self.last_resblocks[j](x)
        x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        target_length = self.target_training_length

        if x.size(-1) < target_length:
            pad = ConstantPad1d((0, target_length-x.size(-1)), 0)
            x = pad(x)
        elif x.size(-1) > target_length:
            x = x[..., :target_length]
        return x * std

