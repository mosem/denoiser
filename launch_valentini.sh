#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  model=demucs \
  stft_loss=True \
  segment=2 \
  stride=2 \
  ddp=0 \
  batch_size=16 \
  scale_factor=2 \
  dummy='demucs_bs' \

