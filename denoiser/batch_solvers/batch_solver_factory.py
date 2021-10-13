from denoiser.batch_solvers.demucs_bs import DemucsBS


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if args.experiment.model == "demucs":
            return DemucsBS(args)
        else:
            raise ValueError("Given model name is not supported")
