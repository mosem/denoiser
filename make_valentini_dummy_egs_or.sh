#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

tr_path=egs/valentini_dummy/16k-dummy/tr
cv_path=egs/valentini_dummy/16k-dummy/cv
tt_path=egs/valentini_dummy/16k-dummy/tt



if [[ ! -e $tr_path ]]; then
    mkdir -p $tr_path
fi
if [[ ! -e $cv_path ]]; then
    mkdir -p $cv_path
fi
if [[ ! -e $tt_path ]]; then
    mkdir -p $tt_path
fi

python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini_noisy/test/generated16000 > $tr_path/noisy.json
python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini/test/generated16000 > $tr_path/clean.json

python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini_noisy/test/generated16000 > $cv_path/noisy.json
python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini/test/generated16000 > $cv_path/clean.json

python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini_noisy/test/generated16000 > $tt_path/noisy.json
python3 -m denoiser.audio /cs/labs/adiyoss/ortal1602/generated_valentini/test/generated16000 > $tt_path/clean.json
