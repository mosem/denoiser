from ._explorers import SimpleExplorer
from ._utils import get_dummy_version

import itertools

@SimpleExplorer
def explorer(launcher):
    launcher.slurm_(gpus=2, mem_per_gpu=50, partition="learnlab")
    valentini_48 = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 48, 'bandmask': 0.2, 'demucs.resample': 4,
                    'demucs.stride': 4, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}
    #  launcher.bind_({
    #      'dummy': get_dummy_version(1),
    #  })
    

    cpc = {'lexical': 'cpc', 'lexical_loss': True, 'lexical_loss_coef': 1.0, 'lexical.inject': 'sup'}
    hubert = {'lexical': 'hubert', 'lexical_loss': True, 'lexical_loss_coef': 0.5, 'lexical.inject': 'sup'}
    asr = {'lexical': 'asr', 'lexical_loss': True, 'lexical_loss_coef': 0.1, 'lexical.inject': 'sup'}
    
    launcher(valentini_48, cpc)
    launcher(valentini_48, hubert)
    launcher(valentini_48, asr)

        
