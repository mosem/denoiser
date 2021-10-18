from ._base_lexical import BaseLexical

import torch

class CPC(BaseLexical):
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.model = torch.hub.load("facebookresearch/CPC_audio", "CPC_audio").to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        feats = self.model(x, None)[0]
        return feats.detach()
