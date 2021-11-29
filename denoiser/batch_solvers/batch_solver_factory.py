from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.batch_solvers.adversarial_bs import AdversarialBS
from denoiser.models.dataclasses import FeaturesConfig, DemucsConfig
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
        ft_config = FeaturesConfig(**args.experiment.features_model) if hasattr(args.experiment, "features_model") else None

        if 'adversarial' in args.experiment and args.experiment.adversarial:
            if args.experiment.model == "demucs":
                generator = Demucs(DemucsConfig(**args.experiment.demucs), include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(DemucsConfig(**args.experiment.demucs_encoder))
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(DemucsConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(DemucsConfig(**args.experiment.demucs_encoder))
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(DemucsConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
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

            return AdversarialBS(args, generator, discriminator, ft_config)
        else:
            if args.experiment.model == "demucs":
                generator = Demucs(DemucsConfig(**args.experiment.demucs), include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
                return GeneratorBS(args, generator, ft_config)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(DemucsConfig(**args.experiment.demucs_encoder))
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(DemucsConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(DemucsConfig(**args.experiment.demucs_encoder))
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(DemucsConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=ft_config.include_ft if ft_config is not None else False)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "caunet":
                generator = Caunet(**args.experiment.caunet)
                return GeneratorBS(args, generator)
            else:
                raise ValueError("Given model name is not supported")
