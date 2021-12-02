import torch
import torch.nn.functional as F

from denoiser.batch_solvers.generator_bs import GeneratorBS
from denoiser.models.dataclasses import FeaturesConfig

GENERATOR_KEY = 'generator'
DISCRIMINATOR_KEY = 'discriminator'
GENERATOR_OPTIMIZER_KEY = 'generator_optimizer'
DISCRIMINATOR_OPTIMIZER_KEY = 'discriminator_optimizer'

class AdversarialBS(GeneratorBS):

    def __init__(self, args, generator, discriminator, features_config: FeaturesConfig=None):
        super().__init__(args, generator, features_config)
        if torch.cuda.is_available():
            discriminator.cuda()
        self._models.update({DISCRIMINATOR_KEY: discriminator})

        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(0.9, self.args.beta2))
        self._optimizers.update(({DISCRIMINATOR_OPTIMIZER_KEY: disc_optimizer}))
        self._losses_names += [DISCRIMINATOR_KEY]


    def run(self, data, cross_valid=False):
        noisy, clean = data

        generator = self._models[GENERATOR_KEY]
        discriminator = self._models[DISCRIMINATOR_KEY]

        prediction = generator(noisy)

        # get features regularization loss if specified
        estimate, estimated_embedded_dim = prediction if self.include_ft else prediction, None
        features_loss = self.get_features_loss(estimated_embedded_dim , clean)

        discriminator_fake_detached = discriminator(estimate.detach())
        discriminator_real = discriminator(clean)
        discriminator_fake = discriminator(estimate)

        loss_discriminator = self._get_discriminator_loss(discriminator_fake_detached, discriminator_real)

        total_loss_generator = features_loss + self._get_total_generator_loss(discriminator_fake, discriminator_real)

        losses = {self._losses_names[0]: total_loss_generator.item(), self._losses_names[1]: loss_discriminator.item()}

        if not cross_valid:
            self._optimize(total_loss_generator, loss_discriminator)

        return losses

    def _optimize(self, total_loss_G, loss_D):
        generator_optimizer = self._optimizers[GENERATOR_OPTIMIZER_KEY]
        discriminator_optimizer = self._optimizers[DISCRIMINATOR_OPTIMIZER_KEY]

        generator_optimizer.zero_grad()
        total_loss_G.backward()
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        loss_D.backward()
        discriminator_optimizer.step()

    def _get_discriminator_loss(self, discriminator_fake, discriminator_real):
        discriminator_loss = 0
        for scale in discriminator_fake:
            discriminator_loss += F.relu(1 + scale[-1]).mean()

        for scale in discriminator_real:
            discriminator_loss += F.relu(1 - scale[-1]).mean()
        return discriminator_loss

    def _get_total_generator_loss(self, discriminator_fake, discriminator_real):
        generator_loss = 0
        for scale in discriminator_fake:
            generator_loss += F.relu(1 - scale[-1]).mean()

        features_loss = 0
        features_weights = 4.0 / (self.args.experiment.discriminator.n_layers + 1)
        discriminator_weights = 1.0 / self.args.experiment.discriminator.num_D
        weights = discriminator_weights * features_weights

        for i in range(self.args.experiment.discriminator.num_D):
            for j in range(len(discriminator_fake[i]) - 1):
                features_loss += weights * F.l1_loss(discriminator_fake[i][j], discriminator_real[i][j].detach())

        return generator_loss + self.args.experiment.features_loss_lambda * features_loss
