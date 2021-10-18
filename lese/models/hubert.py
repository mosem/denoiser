from ._base_lexical import BaseLexical

import torch
import fairseq

class huBERT(BaseLexical):
    def __init__(self, model_path, layer, device='cuda'):
        super().__init__()
        self.path = model_path
        self.layer = layer

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.path])
        self.model = models[0] 
        self.model = self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        feats, _ = self.model.extract_features(source=x.squeeze(1), padding_mask=None, mask=False, output_layer=self.layer)
        return feats.detach()
