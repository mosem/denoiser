from denoiser.batch_solvers.demucs_bs import DemucsBS
from denoiser.batch_solvers.demucs_adversarial_bs import DemucsAdversarialBS
from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS
from denoiser.batch_solvers.demucs_hifi_with_features_bs import  DemucsHifiWithFeaturesBS


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        bsolvers = {"demucs": DemucsBS,
                    "demucs_adversarial": DemucsAdversarialBS,
                    "demucs_hifi": DemucsHifiBS # this also includes feature support
                    }
        if args.experiment.model not in bsolvers.keys():
            raise ValueError("Given model name is not supported")

        return bsolvers[args.experiment.model](args)
