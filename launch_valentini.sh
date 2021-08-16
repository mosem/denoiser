#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  demucs.causal=1 \
  demucs.hidden=48 \
  seanet.ngf=32 \
  bandmask=0 \
  demucs.resample=2 \
  remix=1 \
  shift=0 \
  shift_same=True \
  stft_loss=True \
  segment=4 \
  stride=2 \
  ddp=0 \
  batch_size=16 \
  restart=False \
  epochs=100 \
  sample_rate=16000 \
  dummy='seanet-debugging' \

