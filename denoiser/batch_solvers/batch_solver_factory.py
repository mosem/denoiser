from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.batch_solvers.adversarial_bs import AdversarialBS
from denoiser.models.dataclasses import FeaturesConfig, DemucsConfig, DemucsEncoderConfig, DemucsDecoderConfig
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
        include_ft = ft_config.include_ft if ft_config is not None else False
        get_ft_after_lstm = ft_config.get_ft_after_lstm if ft_config is not None else False
        if 'adversarial' in args.experiment and args.experiment.adversarial:
            if args.experiment.model == "demucs":
                generator = Demucs(DemucsConfig(**args.experiment.demucs), include_ft_in_output=include_ft)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(DemucsEncoderConfig(**args.experiment.demucs_encoder))
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(DemucsDecoderConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=include_ft)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(DemucsEncoderConfig(**args.experiment.demucs_encoder))
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(DemucsDecoderConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=include_ft)
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
                generator = Demucs(DemucsConfig(**args.experiment.demucs), include_ft_in_output=include_ft,
                                   get_ft_after_lstm=get_ft_after_lstm)
                return GeneratorBS(args, generator, ft_config)
            elif args.experiment.model == "demucs_skipless":
                encoder = DemucsEncoder(DemucsEncoderConfig(**args.experiment.demucs_encoder))
                attention = BLSTM(dim=encoder.get_n_chout(), **args.experiment.blstm)
                decoder = DemucsDecoder(DemucsDecoderConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=include_ft)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "demucs_with_transformer":
                encoder = DemucsEncoder(DemucsEncoderConfig(**args.experiment.demucs_encoder))
                attention = OneDimDualTransformer(dim=encoder.get_n_chout(), **args.experiment.transformer)
                decoder = DemucsDecoder(DemucsDecoderConfig(**args.experiment.demucs_decoder))
                generator = Autoencoder(encoder, attention, decoder, **args.experiment.autoencoder, include_ft_in_output=include_ft)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "seanet":
                generator = Seanet(**args.experiment.seanet)
                return GeneratorBS(args, generator)
            elif args.experiment.model == "caunet":
                generator = Caunet(**args.experiment.caunet)
                return GeneratorBS(args, generator)
            else:
                raise ValueError("Given model name is not supported")
