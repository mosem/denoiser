from ._explorers import SimpleExplorer
from ._utils import get_dummy_version

@SimpleExplorer
def explorer(launcher):
    launcher.slurm_(gpus=1, mem_per_gpu=50, partition="devlab")
    launcher.bind_({
        'dummy': get_dummy_version(4),
    })

    for bs in [4, 8]:
        launcher(batch_size=bs)

