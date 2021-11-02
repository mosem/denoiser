from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS
from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.models.demucs import Demucs
from denoiser.models.skipless_demucs import SkiplessDemucs
from denoiser.models.caunet import Caunet


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "demucs":
            generator = Demucs(**args.experiment.demucs)
            return GeneratorBS(args, generator)
        elif args.experiment.model == "skipless_demucs":
            generator = SkiplessDemucs(**args.experiment.skipless_demucs)
            return GeneratorBS(args, generator)
        if args.experiment.model == "caunet":
            generator = Caunet(**args.experiment.caunet)
            return GeneratorBS(args, generator)
        elif args.experiment.model == "demucs_hifi":
            return DemucsHifiBS(args)
        else:
            raise ValueError("Given model name is not supported")
