# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os
import re
import numpy as np
import torch

from .audio import Audioset

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


def match_files(fnames: dict, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        noisy, clean = fnames[0], fnames[1]
        match_dns(noisy, clean)
    elif matching == "sort":
        for x in fnames.values():
            x.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, include_features=False, nb_sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')

        # add support for features
        features_json = f"{json_dir}/features.json"

        fnames = dict()
        with open(noisy_json, 'r') as f:
            fnames['noisy'] = json.load(f)
            # noisy = json.load(f)
        with open(clean_json, 'r') as f:
            fnames['clean'] = json.load(f)
            # clean = json.load(f)

        # TODO add support for features and high-res clean
        if include_features:
            if not os.path.exists(features_json):
                raise ValueError("missing features.json file")
            with open(features_json, 'r') as f:
                fnames['features'] = json.load(f)

        match_files(fnames, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        # self.clean_set = Audioset(clean, **kw)
        # self.noisy_set = Audioset(noisy, **kw)
        self.clean_set = Audioset(fnames['clean'], **kw)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'nb_sample_rate': nb_sample_rate}
        self.noisy_set = Audioset(fnames['noisy'], **kw)
        if include_features:
            self.features = fnames['features']
        self.include_features = include_features

        if nb_sample_rate is None or nb_sample_rate == sample_rate:
            assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        if self.include_features:
            return self.noisy_set[index], self.clean_set[index], torch.load(self.features[index])
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)
