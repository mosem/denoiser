from denoiser.batch_solvers.autoencoder_bs import AutoencoderBS
from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS
from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.batch_solvers.adversarial_bs import AdversarialBS
from denoiser.models.demucs_decoder import DemucsDecoder
from denoiser.models.demucs_encoder import DemucsEncoder
from denoiser.models.modules import Discriminator, LaplacianDiscriminator, BLSTM
from denoiser.models.demucs import Demucs
from denoiser.models.caunet import Caunet
from denoiser.models.seanet import Seanet


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if 'adversarial' in args.experiment and args.experiment.adversarial:
            if args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
                if args.experiment.discriminator_model == "laplacian":
                    discriminator = LaplacianDiscriminator(**args.experiment.discriminator)
                else:
                    discriminator = Discriminator(**args.experiment.discriminator)
                return AdversarialBS(args, generator, discriminator)
            else:
                raise ValueError("Given model name is not supported")
        else:
            if args.experiment.model == "demucs":
                encoder = DemucsEncoder(**args.experiment.demucs_encoder)
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(**args.experiment.demucs_decoder)
                return AutoencoderBS(args, encoder, attention, decoder, args.experiment.skips)
                # generator = Demucs(**args.experiment.demucs)
                # return GeneratorBS(args, generator)
            elif args.experiment.model == "caunet":
                generator = Caunet(**args.experiment.caunet)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_hifi":
                return DemucsHifiBS(args)
            elif args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
                return GeneratorBS(args, generator)
            else:
                raise ValueError("Given model name is not supported")
