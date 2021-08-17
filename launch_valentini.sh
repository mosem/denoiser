#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  seanet.ngf=32 \
  bandmask=0 \
  remix=1 \
  shift=0 \
  shift_same=True \
  stft_loss=True \
  segment=2 \
  stride=2 \
  ddp=0 \
  batch_size=16 \
  restart=False \
  epochs=100 \
  sample_rate=16000 \
  adversarial_mode=True \
  dummy='seanet-denoising-only-16k-discriminators' \

