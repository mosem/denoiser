from denoiser.batch_solvers.demucs_bs import DemucsBS
from denoiser.batch_solvers.demucs_adversarial_bs import DemucsAdversarialBS
from denoiser.batch_solvers.DemucsHifiBS import DemucsHifiBS
from denoiser.batch_solvers.DemucsHifiWithFeaturesBS import  DemucsHifiWithFeaturesBS


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        bsolvers = {"demucs": DemucsBS,
                    "demucs_adversarial": DemucsAdversarialBS,
                    "demucs_hifi": DemucsHifiBS,
                    "demucs_hifi_features": DemucsHifiWithFeaturesBS}
        if args.model not in bsolvers.keys():
            raise ValueError("Given model name is not supported")

        return bsolvers[args.model](args)
