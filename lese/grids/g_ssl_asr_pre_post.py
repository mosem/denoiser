from ._explorers import SimpleExplorer
from ._utils import get_dummy_version

import itertools

@SimpleExplorer
def explorer(launcher):
    launcher.slurm_(gpus=2, mem_per_gpu=50, partition="devlab")
    valentini_48 = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 48, 'bandmask': 0.2, 'demucs.resample': 4,
                    'demucs.stride': 4, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}
    #  launcher.bind_({
    #      'dummy': get_dummy_version(1),
    #  })
    

    methods = ['cpc', 'hubert', 'asr']
    injection = ['pre', 'post']
    
    for lex, inj in itertools.product(methods, injection):
        launcher(valentini_48, {"lexical": lex, "lexical.inject": inj})

        
