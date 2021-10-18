#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

tr_path=egs/debug/tr
cv_path=egs/debug/cv
tt_path=egs/debug/tt
if [[ ! -e $tr_path ]]; then
    mkdir -p $tr_path
fi
if [[ ! -e $cv_path ]]; then
    mkdir -p $cv_path
fi
if [[ ! -e $tt_path ]]; then
    mkdir -p $tt_path
fi

python3 -m denoiser.audio dataset/debug/noisy > $tr_path/noisy.json
python3 -m denoiser.audio dataset/debug/clean > $tr_path/clean.json

python3 -m denoiser.audio dataset/debug/noisy > $cv_path/noisy.json
python3 -m denoiser.audio dataset/debug/clean > $cv_path/clean.json

python3 -m denoiser.audio dataset/debug/noisy > $tt_path/noisy.json
python3 -m denoiser.audio dataset/debug/clean > $tt_path/clean.json
