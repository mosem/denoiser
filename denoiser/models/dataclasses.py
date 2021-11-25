from dataclasses import dataclass

# The following dataclasses come to replace the redundancy of passing multiple arguments,
# and offers a simpler manner of doing so.
@dataclass
class DemucsConfig:
    chin: int = 1
    chout: int = 1
    hidden: int = 48
    max_hidden: int = 10000
    causal: bool = True
    floor: float = 1e-3
    glu: bool = True
    depth: int = 5
    kernel_size: int = 8
    stride: int = 2
    normalize: bool = True
    resample: int = 1
    growth: int = 2
    rescale: float = 0.1
    scale_factor: int = 1
    skips: bool = False


@dataclass
class FeaturesConfig:
    include_ft: bool = False
    feature_model: str = 'hubert'
    state_dict_path: str = '/cs/labs/adiyoss/shared/pretrained_weights/hubert/hubert_base_ls960.pt'
    features_factor: float = 0.01


@dataclass
class MRFConfig:
    num_mrfs: int = 0
    resblock: int = 1
    resblock_kernel_sizes: list = (3, 7, 11)
    resblock_dilation_sizes: list = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
