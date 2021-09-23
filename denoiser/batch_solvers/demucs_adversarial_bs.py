import torch

from denoiser.batch_solvers.demucs_bs import DemucsBS
from denoiser.models.modules import Discriminator, LaplacianDiscriminator


class DemucsAdversarialBS(DemucsBS):

    def __init__(self, args):
        super().__init__(args)

        discriminator = LaplacianDiscriminator(**args.discriminator) if args.laplacian \
            else Discriminator(**args.discriminator)

        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, args.beta2))
        self.models.update({'discriminator': discriminator})
        self.optimizers.update(({'discriminator': disc_optimizer}))