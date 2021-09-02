#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez
import itertools
import logging
import os

import hydra

from denoiser.executor import start_ddp_workers
from gan_models import MelGenerator, HifiMultiScaleDiscriminator, HifiMultiPeriodDiscriminator

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from denoiser import distrib
    from denoiser.data import NoisyCleanSet
    from denoiser.demucs import Demucs
    from denoiser.solver import Solver
    distrib.init(args)
    options = {'demucs': [Demucs], 'hifi': [MelGenerator, HifiMultiPeriodDiscriminator, HifiMultiScaleDiscriminator]}
    models = list()
    k = args.model
    for i, cls in enumerate(options[k]):
        if i == 0:
            attributes = getattr(args, k)
            if args.model == 'hifi':
                models.append(cls(attributes))
            else:
                models.append(cls(**attributes))
        else:
            models.append(cls())


    if args.show:
        for model in models:
            logger.info(model)
            mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
            logger.info('Size: %.1f MB', mb)
            if hasattr(model, 'valid_length'):
                field = model.valid_length(1)
                logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
            return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(models[0], 'valid_length'):
        length = models[0].valid_length(length)
    # kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate,
              "include_features": args.include_features, "nb_sample_rate": args.nb_sample_rate}
    # Building datasets and loaders
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, **kwargs)
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
        for model in models:
            model.cuda()

    # optimizer
    if args.optim == "adam":
        if args.model == 'demucs':
            optimizers = optimizer = torch.optim.Adam(models[0].parameters(), lr=args.lr, betas=(0.9, args.beta2))
        elif args.model == 'hifi':
            optimizers = [torch.optim.AdamW(models[0].parameters(), lr=args.lr, betas=(args.beta1, args.beta2)),
                          torch.optim.AdamW(models[1].parameters(),
                          # torch.optim.AdamW(itertools.chain(models[1].parameters(), models[2].parameters()),
                          lr=args.lr, betas=(args.beta1, args.beta2))]
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, models, optimizers, args)
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
    if args.ddp and args.rank is None:
        start_ddp_workers()
    else:
        run(args)


@hydra.main(config_path="conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
