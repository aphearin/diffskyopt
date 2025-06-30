import argparse
from mpi4py import MPI

import numpy as np
import jax

from ..lossfuncs.cosmos_fit import CosmosFit


rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def adam_fit(cosmos_fit, out=None, seed=1, nsteps=1000, u_param_init=None,
             learning_rate=0.1, progress=True, modelsamp=True):
    # Run Adam optimization from random initializations and plot loss
    calc = cosmos_fit.get_multi_grad_calc(modelsamp=modelsamp)
    key = jax.random.key(seed)
    if u_param_init is None:
        u_param_init = cosmos_fit.default_u_param_arr

    params, losses = calc.run_adam(
        u_param_init, nsteps=nsteps, learning_rate=learning_rate,
        randkey=key, progress=progress)

    if out:
        np.savez(
            out,
            params=np.asarray(params),
            losses=np.asarray(losses)
        )


parser = argparse.ArgumentParser(
    description="Debug parallel fits to COSMOS data")
parser.add_argument(
    "-i", "--input", type=str, default=None,
    help="Previous results of which to start from the final params")
parser.add_argument(
    "-s", "--seed", type=int, default=1,
    help="Random seed")
parser.add_argument(
    "-k", "--num-kernels", type=int, default=4,
    help="Number of kernels for kdescent")
parser.add_argument(
    "-f", "--num-fourier-positions", type=int, default=2,
    help="Number of Fourier evaluation positions for kdescent")
parser.add_argument(
    "--log-loss", action="store_true",
    help="Perform logarithm on KDE counts for loss computation")
parser.add_argument(
    "--not-modelsamp", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    cosmos_fit = CosmosFit(
        num_kernels=args.num_kernels,
        num_fourier_positions=args.num_fourier_positions,
        hmf_calibration=None, log_loss=args.log_loss)

    if args.input is not None:
        u_params = np.load(args.input)["params"][-1]
    else:
        u_params = cosmos_fit.default_u_param_arr

    calc = cosmos_fit.get_multi_grad_calc(modelsamp=not args.not_modelsamp)

    key = jax.random.key(args.seed)

    params, losses = calc.run_adam(u_params, nsteps=5, randkey=key)
    print(f"({rank=}/{size=}) {params.mean(axis=1)=}, {losses=}",
          flush=True)
