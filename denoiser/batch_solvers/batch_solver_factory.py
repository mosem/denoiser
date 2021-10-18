from denoiser.batch_solvers.demucs_bs import DemucsBS
from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "demucs":
            return DemucsBS(args)
        elif args.experiment.model == "demucs_hifi":
            return DemucsHifiBS(args)
        else:
            raise ValueError(f"Given model name is not supported: {args.experiment.model}")
