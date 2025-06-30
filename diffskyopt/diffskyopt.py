from mpi4py import MPI

from diffsky.param_utils import diffsky_param_wrapper as dpw  # noqa: F401

from .diffsky_model import (  # noqa: F401
    generate_lc_data,
    downsample_upweight_lc_data,
    compute_targets_and_weights,
)


def _not_trange(*args, desc=None, leave=None, **kwargs):
    del desc, leave
    return range(*args, **kwargs)


try:
    if MPI.COMM_WORLD.rank != 0:
        raise ImportError("Progress bar only runs on rank 0")
    from tqdm.auto import trange
except ImportError:
    trange = _not_trange
