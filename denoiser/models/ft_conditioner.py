import torch
import torch.nn as nn
from torch.nn import functional as F
from denoiser.models.dataclasses import FeaturesConfig
from denoiser.utils import load_lexical_model


class FtConditioner(nn.Module):

    def __init__(self, device, ft_config: FeaturesConfig=None):
        super().__init__()
        if ft_config is not None:
            self.condition = ft_config.use_as_conditioning
            self.include_ft = ft_config.include_ft
            self.proj = nn.Linear(ft_config.features_dim_for_conditioning + ft_config.features_dim,
                                  ft_config.features_dim_for_conditioning).to(device)
            self.features_factor = ft_config.features_factor
            if self.include_ft or self.condition:
                self.ft_model = load_lexical_model(ft_config.feature_model,
                                                   ft_config.state_dict_path,
                                                   device)
            self.merge_method = ft_config.merge_method
        else:
            self.include_ft = False

    def extract_feats(self, reference_signal):
        return self.ft_model.extract_feats(reference_signal)

    def forward(self, x):
        if self.condition:
            features = self.extract_feats(x)
            if self.merge_method == 'inter':
                x_res = F.interpolate(features.permute(0, 2, 1), x.shape[0]).permute(2, 0, 1)
            elif self.merge_method == 'att':
                x_res = x.permute(1, 0, 2)
                alpha = F.softmax((features.unsqueeze(1) * x_res.unsqueeze(2)).sum(dim=-1), dim=2)
                x_res = torch.bmm(alpha, features).permute(1, 0, 2)
            x = torch.cat([x, x_res], dim=-1)
            x = self.proj(x)
        return x
