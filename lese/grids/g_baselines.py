from ._explorers import SimpleExplorer
from ._utils import get_dummy_version

import itertools

@SimpleExplorer
def explorer(launcher):
    launcher.bind_({
        'dummy': get_dummy_version(1),
    })
    launcher.slurm_(gpus=2, mem_per_gpu=50, partition="devlab")

    valentini_64 = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 64, 'bandmask': 0.2, 'demucs.resample': 4,
                    'demucs.stride': 4, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}
    valentini_64_large = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 64, 'bandmask': 0.2, 'demucs.resample': 2,
                    'demucs.stride': 2, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}
    valentini_48 = {'dset': 'valentini', 'demucs.causal': 0, 'demucs.hidden': 48, 'bandmask': 0.2, 'demucs.resample': 4,
                    'demucs.stride': 4, 'remix': 1, 'shift': 8000, 'shift_same': True, 'stft_loss': True, 'segment': 4.5, 'stride': 0.5, 'batch_size': 32}
    
    # reproduce the results
    launcher(valentini_48)
    launcher(valentini_64)

    launcher.slurm_(gpus=4, mem_per_gpu=50, partition="devlab")
    launcher(valentini_64_large)

        
