# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import math
import os
import re

from .audio import Audioset
from .resample import downsample2
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    def __init__(self, json_dir, calc_valid_length_func, matching="sort", clean_length=None, stride=None,
                 pad=True, sample_rate=None, scale_factor=1, with_path=False, is_training=False):
        """__init__.
        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param clean_length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        self.scale_factor = scale_factor
        self.with_path = with_path
        self.clean_length = clean_length
        self.calc_valid_length_func = calc_valid_length_func
        self.is_training = is_training

        if self.is_training:
            input_training_length = math.ceil(self.clean_length / self.scale_factor)
            self.valid_length = self.calc_valid_length_func(input_training_length)
        else:
            self.valid_length = None

        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': self.valid_length, 'stride': stride, 'pad': pad, 'with_path': with_path}
        self.clean_set = Audioset(clean, sample_rate=sample_rate, **kw)
        self.noisy_set = Audioset(noisy, sample_rate=sample_rate, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        noisy, clean = self.noisy_set[index], self.clean_set[index]

        if not self.is_training:
            self.valid_length = self.calc_valid_length_func(math.ceil(clean.shape[-1]/self.scale_factor))

        logger.info(f'index: {index}')
        logger.info(f'valid_length: {self.valid_length}')
        logger.info(f'clean shape: {clean.shape}')
        logger.info(f'noisy shape: {noisy.shape}')

        if self.valid_length < clean.shape[-1]:
            logger.info(f'trimming clean: {clean.shape[-1]-self.valid_length}')
            clean = clean[..., :self.valid_length]
        elif self.valid_length > clean.shape[-1]:
            logger.info(f'padding clean: {self.valid_length-clean.shape[-1]}')
            clean = F.pad(clean, (0, self.valid_length - clean.shape[-1]))

        if self.valid_length < noisy.shape[-1]:
            logger.info(f'trimming noisy: {noisy.shape[-1] - self.valid_length}')
            noisy = noisy[..., :self.valid_length]
        elif self.valid_length > noisy.shape[-1]:
            logger.info(f'padding noisy: {self.valid_length - noisy.shape[-1]}')
            noisy = F.pad(noisy, (0, self.valid_length - noisy.shape[-1]))

        if self.scale_factor == 2:
            noisy = downsample2(noisy)
        elif self.scale_factor == 4:
            noisy = downsample2(noisy)
            noisy = downsample2(noisy)
        elif self.scale_factor != 1:
            raise RuntimeError(f"Scale factor should be 1, 2, or 4")

        # estimated_length = noisy.shape[-1] * self.scale_factor
        # if estimated_length < clean.shape[-1]:
        #     logger.info('trimming!')
        #     logger.info(f'estimated shape: {estimated_length}')
        #     logger.info(f'clean shape: {clean.shape}')
        #     clean = clean[...,:estimated_length]
        # elif estimated_length > clean.shape[-1]:
        #     logger.info('padding!')
        #     logger.info(f'estimated shape: {estimated_length}')
        #     logger.info(f'clean shape: {clean.shape}')
        #     clean = F.pad(clean, (0, estimated_length - clean.shape[-1]))


        return noisy, clean

    def __len__(self):
        return len(self.noisy_set)