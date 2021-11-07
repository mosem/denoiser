import torch
from torch import nn
import torch.nn.functional as F

from denoiser.utils import capture_init
from denoiser.batch_solvers.batch_solver import BatchSolver
from denoiser.stft_loss import MultiResolutionSTFTLoss

class Generator(nn.Module):

    @capture_init
    def __init__(self, encoder, attention_module, decoder, skips):
        super().__init__()
        self.encoder = encoder
        self.attention_module = attention_module
        self.decoder = decoder
        self.skips = skips

    def estimate_valid_length(self, input_length):
        encoder_output_length = self.encoder.estimate_output_length(input_length)
        attention_output_length = self.attention_module.estimate_output_length(encoder_output_length)
        decoder_output_length = self.decoder.estimate_output_length(attention_output_length)
        return decoder_output_length

    def forward(self, signal):
        if self.skips:
            latent, skips_signals = self.encoder(signal)
            latent = self.attention_module(latent)
            out = self.decoder(latent, skips_signals)
        else:
            latent = self.encoder
            latent = self.attention_module(latent)
            out = self.decoder(latent)
        return out


class AutoencoderBS(BatchSolver):

    def __init__(self, args, encoder, attention_module, decoder, skips=False):
        super(AutoencoderBS, self).__init__(args)
        self.device = args.device

        generator = Generator(encoder, attention_module, decoder, skips)

        if torch.cuda.is_available() and args.device == 'cuda':
            generator.cuda()
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, args.beta2))

        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)

        self._models.update({'generator': generator})
        self._optimizers.update({'generator_optimizer': generator_optimizer})
        self._losses_names += ['generator']

    def get_generator_for_evaluation(self, best_states):
        generator = self._models['generator']
        generator.load_state_dict(best_states['generator'])
        return generator

    def estimate_valid_length(self, input_length):
        return self._models['generator'].estimate_valid_length(input_length)

    def run(self, data, cross_valid=False):
        noisy, clean = data
        estimate = self._models['generator'](noisy)
        loss = self._get_loss(clean, estimate)
        if not cross_valid:
            self._optimize(loss)
        losses = {self._losses_names[0]: loss.item()}
        return losses

    def get_evaluation_loss(self, losses_dict):
        return losses_dict[self._losses_names[0]]

    def _optimize(self, loss):
        self._optimizers['generator_optimizer'].zero_grad()
        loss.backward()
        self._optimizers['generator_optimizer'].step()

    def _get_loss(self, clean, estimate):
        with torch.autograd.set_detect_anomaly(True):
            if self.args.loss == 'l1':
                loss = F.l1_loss(clean, estimate)
            elif self.args.loss == 'l2':
                loss = F.mse_loss(clean, estimate)
            elif self.args.loss == 'huber':
                loss = F.smooth_l1_loss(clean, estimate)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            # MultiResolution STFT loss
            if self.args.stft_loss:
                sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                loss += sc_loss + mag_loss

            return loss