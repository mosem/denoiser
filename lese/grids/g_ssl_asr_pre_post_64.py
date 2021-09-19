from ._explorers import SimpleExplorer
from ._utils import get_dummy_version

import itertools

@SimpleExplorer
def explorer(launcher):
    launcher.slurm_(gpus=2, mem_per_gpu=50, partition="devlab")
    valentini_64 = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 64, 'bandmask': 0.2, 'demucs.resample': 4,
                    'demucs.stride': 4, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}

    methods = ['cpc', 'hubert', 'asr']
    injection = ['pre', 'post']
    
    for lex, inj in itertools.product(methods, injection):
        launcher(valentini_64, {"lexical": lex, "lexical.inject": inj})

        
