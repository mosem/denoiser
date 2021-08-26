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
  bandmask=0 \
  remix=0 \
  shift=0 \
  shift_same=True \
  stft_loss=True \
  segment=2 \
  stride=2 \
  ddp=0 \
  batch_size=16 \
  epochs=1 \
  sample_rate=16000 \
  scale_factor=2 \
  adversarial_mode=False \
  eval_every=1 \
  dummy='refactoring-debug' \

