#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez
import logging
import math
import os
import shutil
from datetime import datetime
import hydra

from denoiser.executor import start_ddp_workers
from denoiser.batch_solvers.BatchSolverFactory import BatchSolverFactory

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from denoiser.batch_solvers import demucs_bs
    from denoiser import distrib
    from denoiser.data import NoisyCleanSet
    from denoiser.solver import Solver
    distrib.init(args)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    # (or) added batch solver factory
    batch_solver = BatchSolverFactory.get_bs(args)

    if args.show:
            logger.info(batch_solver)
            mb = sum(p.numel() for model in batch_solver.models for p in model.parameters()) * 4 / 2**20
            logger.info('Size: %.1f MB', mb)
            if hasattr(batch_solver, 'valid_length'):
                batch_solver.calculate_valid_length(1)
                field = batch_solver.get_valid_length()
                logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
            return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Define a specific number of samples to avoid 0 padding during training
    length = batch_solver.calculate_valid_length(math.ceil(length / args.scale_factor))
    batch_solver.set_target_training_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}
    # Building datasets and loaders
    tr_dataset = NoisyCleanSet(
            args, args.dset.train, length=length, stride=stride, pad=args.pad, scale_factor=args.scale_factor, **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(
            args, args.dset.valid, length=length, stride=stride, pad=args.pad, scale_factor=args.scale_factor, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(
            args, args.dset.test, length=length, stride=stride, pad=args.pad, scale_factor=args.scale_factor, **kwargs)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    # Construct Solver
    solver = Solver(data, batch_solver, args)
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

@hydra.main(config_path="conf", config_name="config_demucs_hifi") #  for latest version of hydra=1.0
# @hydra.main(config_path="conf", config_name="config") #  for latest version of hydra=1.0, general config | TODO change to this one for demucs
def main(args):
    try:
        if "hifi" in args.model:
            os.makedirs(f"../hifi_L-{args.hifi.l1_factor}_G-{args.hifi.gen_factor}", exist_ok=True)
            _main(args)
            now = datetime.now()
            try:
                shutil.move(".", f"../hifi_L-{args.hifi.l1_factor}_G-{args.hifi.gen_factor}/{now.strftime('%d-%m-%Y_%H-%M')}")
            except Exception:
                pass

        else:
            _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
