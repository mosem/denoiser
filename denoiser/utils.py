# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import functools
import logging
from contextlib import contextmanager
import inspect
import time

import torch
from librosa.filters import mel as librosa_mel_fn
logger = logging.getLogger(__name__)


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def deserialize_model(package, strict=False):
    """deserialize_model.

    """
    klass = package['class']
    if strict:
        model = klass(*package['args'], **package['kwargs'])
    else:
        sig = inspect.signature(klass)
        kw = package['kwargs']
        for key in list(kw):
            if key not in sig.parameters:
                logger.warning("Dropping inexistant parameter %s", key)
                del kw[key]
        model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin=0, fmax=None, center=False):
    fmax = sampling_rate // 2 if fmax is None else fmax
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
