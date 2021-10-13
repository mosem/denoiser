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
import hydra

from denoiser.executor import start_ddp_workers
from denoiser.batch_solvers.batch_solver_factory import BatchSolverFactory

logger = logging.getLogger(__name__)

def run(args):
    import torch

    from denoiser import distrib
    from denoiser.data import NoisyCleanSet
    from denoiser.solver import Solver
    distrib.init(args)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    batch_solver = BatchSolverFactory.get_bs(args)

    if args.show:
            logger.info(batch_solver)
            mb = sum(p.numel() for model in batch_solver.models for p in model.parameters()) * 4 / 2**20
            logger.info('Size: %.1f MB', mb)
            if hasattr(batch_solver, 'valid_length'):
                batch_solver.calculate_valid_length(1)
                field = batch_solver.get_valid_length()
                logger.info('Field: %.1f ms', field / args.experiment.sample_rate * 1000)
            return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    target_training_length = int(args.experiment.segment * args.experiment.sample_rate)
    training_stride = int(args.experiment.stride * args.experiment.sample_rate)
    # Define a specific number of samples to avoid 0 padding during training
    input_training_length = math.ceil(target_training_length / args.experiment.scale_factor)
    valid_target_training_length = batch_solver.calculate_valid_length(input_training_length)
    batch_solver.set_target_training_length(valid_target_training_length) # TODO: fix this, add this inside.
    kwargs = {"matching": args.dset.matching, "sample_rate": args.experiment.sample_rate}
    # Building datasets and loaders

    tr_dataset = NoisyCleanSet(args.dset.train, batch_solver.calculate_valid_length, clean_length=target_training_length,
                stride=training_stride, pad=args.experiment.pad, scale_factor=args.experiment.scale_factor, is_training=True,
                **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(args.dset.valid, batch_solver.calculate_valid_length,
            scale_factor=args.experiment.scale_factor, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(args.dset.test, batch_solver.calculate_valid_length,
            scale_factor=args.experiment.scale_factor, **kwargs)
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

@hydra.main(config_path="conf", config_name="config") #  for latest version of hydra=1.0
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
