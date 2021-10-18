#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

import logging
import os

import hydra

from dora import hydra_main, get_xp

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from . import distrib
    from .data import NoisyCleanSet
    from .models.demucs import Demucs
    from .solver import Solver
    distrib.init()

    model = Demucs(**args.experiment.demucs)

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        no_params = sum(p.numel() for p in model.parameters())
        logger.info('Size: %.1f MB', mb)
        logger.info(f'No params: {no_params}')
        if hasattr(model, 'valid_length'):
            field = model.calculate_valid_length(1)
            logger.info('Field: %.1f ms', field / args.experiment.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.experiment.segment * args.experiment.sample_rate)
    stride = int(args.experiment.stride * args.experiment.sample_rate)
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.calculate_valid_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.experiment.sample_rate}
    # Building datasets and loaders
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.experiment.pad, **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(args.dset.valid, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(args.dset.test, **kwargs)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, model, optimizer, args)
    solver.train()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra_main(config_path="../conf", config_name="config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
