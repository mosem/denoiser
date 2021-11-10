from denoiser.batch_solvers.demucs_hifi_bs import DemucsHifiBS
from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.batch_solvers.adversarial_bs import AdversarialBS
from denoiser.models.demucs_decoder import DemucsDecoder
from denoiser.models.demucs_encoder import DemucsEncoder
from denoiser.models.modules import Discriminator, LaplacianDiscriminator, BLSTM, OneDimDualTransformer
from denoiser.models.demucs import Demucs
from denoiser.models.caunet import Caunet
from denoiser.models.seanet import Seanet
from denoiser.models.autoencoder_composer import Autoencoder


class BatchSolverFactory:

    @staticmethod
    def get_bs(args):
        if 'adversarial' in args.experiment and args.experiment.adversarial:
            if args.experiment.model == "demucs":
                generator = Demucs(**args.experiment.demucs)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(**args.experiment.demucs_encoder)
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(**args.experiment.demucs_decoder)
                generator = Autoencoder(encoder, attention, decoder, args.experiment.skips)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(**args.experiment.demucs_encoder)
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(**args.experiment.demucs_decoder)
                generator = Autoencoder(encoder, attention, decoder, args.experiment.skips)
            elif args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
            elif args.experiment.model == "caunet":
                generator = Caunet(**args.experiment.caunet)
            else:
                raise ValueError("Given model name is not supported")

            if args.experiment.discriminator_model == "laplacian":
                discriminator = LaplacianDiscriminator(**args.experiment.discriminator)
            else:
                discriminator = Discriminator(**args.experiment.discriminator)

            return AdversarialBS(args, generator, discriminator)
        else:
            if args.experiment.model == "demucs":
                generator = Demucs(**args.experiment.demucs)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(**args.experiment.demucs_encoder)
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(**args.experiment.demucs_decoder)
                generator = Autoencoder(encoder, attention, decoder, args.experiment.skips)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(**args.experiment.demucs_encoder)
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(**args.experiment.demucs_decoder)
                generator = Autoencoder(encoder, attention, decoder, args.experiment.skips)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "caunet":
                generator = Caunet(**args.experiment.caunet)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_hifi":
                return DemucsHifiBS(args)
            else:
                raise ValueError("Given model name is not supported")
