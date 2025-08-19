from diffsky.param_utils import diffsky_param_wrapper as dpw
from mpi4py import MPI

from .diffsky_model import (compute_targets_and_weights,
                            downsample_upweight_lc_data, generate_lc_data)


def _not_trange(*args, desc=None, leave=None, **kwargs):
    del desc, leave
    return range(*args, **kwargs)


try:
    if MPI.COMM_WORLD.rank != 0:
        raise ImportError("Progress bar only runs on rank 0")
    from tqdm.auto import trange
except ImportError:
    trange = _not_trange


__all__ = [
    "dpw",
    "compute_targets_and_weights",
    "downsample_upweight_lc_data",
    "generate_lc_data",
    "trange",
]
