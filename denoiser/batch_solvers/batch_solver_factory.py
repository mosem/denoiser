from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS
from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.models.demucs import Demucs


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "demucs":
            generator = Demucs(**args.experiment.demucs)
            return GeneratorBS(args, generator)
        elif args.experiment.model == "demucs_hifi":
            return DemucsHifiBS(args)
        else:
            raise ValueError("Given model name is not supported")
