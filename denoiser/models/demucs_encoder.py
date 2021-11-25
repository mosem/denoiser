import math
from torch import nn

from denoiser.models.dataclasses import DemucsConfig
from denoiser.models.demucs import  rescale_module
from denoiser.resample import upsample2
from denoiser.utils import capture_init


class DemucsEncoder(nn.Module):
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
    def __init__(self,demucs_conf: DemucsConfig):

        super().__init__()
        if demucs_conf.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = demucs_conf.chin
        self.hidden = demucs_conf.hidden
        self.depth = demucs_conf.depth
        self.kernel_size = demucs_conf.kernel_size
        self.stride = demucs_conf.stride
        self.resample = demucs_conf.resample
        self.scale_factor = demucs_conf.scale_factor
        self.skips = demucs_conf.skips

        self.encoder = nn.ModuleList()
        activation = nn.GLU(1) if demucs_conf.glu else nn.ReLU()
        ch_scale = 2 if demucs_conf.glu else 1
        chin, hidden, chout = self.chin, self.hidden, self.chout

        for index in range(demucs_conf.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, demucs_conf.kernel_size, demucs_conf.stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            chin = hidden
            hidden = min(int(demucs_conf.growth * hidden), demucs_conf.max_hidden)

        self.n_chout = chin

        if demucs_conf.rescale:
            rescale_module(self, reference=demucs_conf.rescale)

    def get_n_chout(self):
        return self.n_chout

    def estimate_output_length(self, input_length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(input_length * self.scale_factor)
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        return int(length)

    def forward(self, signal):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)

        x = signal

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
        if self.skips:
            skips_signals = []
            for encode in self.encoder:
                x = encode(x)
                skips_signals.append(x)

            return x, skips_signals

        else:
            for encode in self.encoder:
                x = encode(x)
            return x