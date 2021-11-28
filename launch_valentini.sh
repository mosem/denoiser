#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini_dummy \
  experiment=demucs_seanet_skipless_adversarial_1 \
  experiment.scale_factor=2 \

