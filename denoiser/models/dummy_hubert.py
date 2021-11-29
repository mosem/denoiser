import torch

HUBERT_CH = 768
HUBERT_TIME = 100


class DummyHubert(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        return torch.rand((x.shape[0], HUBERT_CH, HUBERT_TIME)).to(self.device)