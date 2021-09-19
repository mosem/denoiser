import inspect
from pathlib import Path


def get_dummy_version(version_num):
    """
    Returns a unique string composed of the calling grid file name and a version to be set as
    a dummy flag for an experiment grid. Assuming the caller is a grid file.
    """
    frame = inspect.stack()[1]
    filename = frame[0].f_code.co_filename
    return Path(filename).stem + f"-v{version_num}"
