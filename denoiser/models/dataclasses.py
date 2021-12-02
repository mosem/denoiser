from dataclasses import dataclass

@dataclass
class FeaturesConfig:
    include_ft: bool = False
    feature_model: str = 'hubert'
    state_dict_path: str = '/cs/labs/adiyoss/shared/pretrained_weights/hubert/hubert_base_ls960.pt'
    features_factor: float = 0.01
