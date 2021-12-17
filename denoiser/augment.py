# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
import math
import random
import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from . import dsp
from .resample import downsample2


class Remix(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def forward(self, sources, target):
        noise, clean_source = sources
        bs, *other = noise.shape
        device = noise.device
        perm = th.argsort(th.rand(bs, device=device), dim=0)
        out = th.stack([noise[perm], clean_source]), target
        return out



class RevEcho(nn.Module):
    """
    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).

    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.

    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, target_sample_rate=16000, scale_factor=1):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.source_sample_rate = math.ceil(target_sample_rate / scale_factor)
        self.target_sample_rate = target_sample_rate
        self.scale_factor = scale_factor

    def _reverb(self, source, initial, first_delay, rt60, sample_rate):
        """
        Return the reverb for a single source.
        """
        length = source.shape[-1]
        reverb = th.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(
                    1 + int(jitter * first_delay * sample_rate),
                    length)
                # Delay the echo in time by padding with zero on the left
                echo = F.pad(echo[:, :, :-delay], (delay, 0))
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def forward(self, sources, target):
        if random.random() >= self.proba:
            return sources, target
        noise, clean_source = sources
        clean = target
        
        # Sample characteristics for the reverb
        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        reverb_noise = self._reverb(noise, initial, first_delay, rt60, self.source_sample_rate)
        # Reverb for the noise is always added back to the noise
        noise += reverb_noise

        reverb_clean = self._reverb(clean, initial, first_delay, rt60, self.target_sample_rate)
        if self.scale_factor == 2:
            reverb_clean_downsampled = downsample2(reverb_clean)
        elif self.scale_factor == 4:
            reverb_clean_downsampled = downsample2(reverb_clean)
            reverb_clean_downsampled = downsample2(reverb_clean_downsampled)
        # Split clean reverb among the clean speech and noise
        clean += self.keep_clean * reverb_clean
        clean_source += self.keep_clean * reverb_clean_downsampled
        noise += (1 - self.keep_clean) * reverb_clean_downsampled

        out = th.stack([noise, clean_source]), clean
        return out


class BandMask(nn.Module):
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, scale_factor=1, target_sample_rate=16_000):
        """__init__.

        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param source_sample_rate: signal sample rate
        """
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.source_sample_rate = math.ceil(target_sample_rate / scale_factor)
        self.target_sample_rate = target_sample_rate


    def forward(self, sources, target):
        bands = self.bands
        bandwidth = int(abs(self.maxwidth) * bands)
        mels = dsp.mel_frequencies(bands, 40, self.source_sample_rate / 2) / self.source_sample_rate
        low = random.randrange(bands)
        high = random.randrange(low, min(bands, low + bandwidth))
        filters = dsp.LowPassFilters([mels[low], mels[high]]).to(sources.device)
        sources_low, sources_midlow = filters(sources)
        
        # band pass filtering
        sources_out = sources - sources_midlow + sources_low
        target_mels = dsp.mel_frequencies(bands, 40, self.target_sample_rate / 2) / self.target_sample_rate
        target_filters = dsp.LowPassFilters([target_mels[low], target_mels[high]]).to(sources.device)
        targets_low, targets_midlow = target_filters(target)
        target_out = target - targets_midlow + targets_low
        out = sources_out, target_out
        return out


class Shift(nn.Module):
    """Shift."""

    def __init__(self, shift=8192, same=False, target_scale_factor=1):
        """__init__.

        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same = same
        self.target_scale_factor = target_scale_factor

    def forward(self, sources, target):
        n_sources, batch, channels, length = sources.shape
        _ , _, target_length = target.shape
        target_scale = target_length / length
        shift_length = np.random.randint(self.shift // 2, self.shift+1)
        output_sources = th.roll(sources, shifts=shift_length, dims=-1)
        output_sources[..., :shift_length] *= 0
        output_targets = th.roll(sources, shifts=shift_length*target_scale, dims=-1)
        output_targets[..., :shift_length*target_scale] *= 0
        return output_sources, output_targets


class Augment(object):

    def __init__(self, args):
        np.random.seed(args.seed)
        self.args = args
        augments = []
        self.r = Remix() if args.remix else None
        self.b = BandMask(args.bandmask, scale_factor=args.experiment.scale_factor,
                                             target_sample_rate=args.experiment.sample_rate) if args.bandmask else None
        self.s = Shift(args.shift, args.shift_same, args.experiment.scale_factor) if args.shift else None
        self.re = RevEcho(args.revecho, target_sample_rate=args.experiment.sample_rate, scale_factor=args.experiment.scale_factor) if args.revecho else None
        self.augment = self.r is not None or self.s is not None or self.b is not None or self.re is not None

    def augment_data(self, noisy, clean):
        if not self.augment:
            return noisy, clean

        if self.args.experiment.scale_factor == 1:
            clean_downsampled = clean
        elif self.args.experiment.scale_factor == 2:
            clean_downsampled = downsample2(clean)
        elif self.args.experiment.scale_factor == 4:
            clean_downsampled = downsample2(clean)
            clean_downsampled = downsample2(clean_downsampled)
        noise = noisy - clean_downsampled
        sources = th.stack([noise, clean_downsampled])
        if self.r is not None:
            sources, target = self.r(sources, clean)
        if self.b is not None:
            sources, target = self.b(sources, target)
        if self.s is not None:
            sources, target = self.s(sources, target)
        if self.re is not None:
            sources, target = self.re(sources, target)
        source_noise, source_clean = sources
        source_noisy = source_noise + source_clean
        return source_noisy, target
