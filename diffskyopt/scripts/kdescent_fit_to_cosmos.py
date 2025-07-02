import argparse
from mpi4py import MPI

import numpy as np
import jax

from ..lossfuncs.cosmos_fit import CosmosFit


def adam_fit(cosmos_fit, out, seed=1, nsteps=1000, u_param_init=None,
             learning_rate=0.1, progress=True, modelsamp=True):
    # Run Adam optimization from random initializations and plot loss
    calc = cosmos_fit.get_multi_grad_calc(modelsamp=modelsamp)
    key = jax.random.key(seed)
    if u_param_init is None:
        u_param_init = cosmos_fit.default_u_param_arr

    params, losses = calc.run_adam(
        u_param_init, nsteps=nsteps, learning_rate=learning_rate,
        randkey=key, progress=progress)

    if not MPI.COMM_WORLD.rank:
        np.savez(
            out,
            params=np.asarray(params),
            losses=np.asarray(losses)
        )


parser = argparse.ArgumentParser(
    description="Test real fits to COSMOS data")
parser.add_argument(
    "-o", "--output", type=str, default="cosmos_fit_results.npz",
    help="Save plots instead of opening interactive window")
parser.add_argument(
    "-i", "--input", type=str, default=None,
    help="Previous results of which to start from the final params")
parser.add_argument(
    "-n", "--nsteps", type=int, default=1000,
    help="Number of adam fitting steps")
parser.add_argument(
    "-l", "--learning-rate", type=float, default=0.1,
    help="Learning rate for adam gradient descent")
parser.add_argument(
    "-s", "--seed", type=int, default=1,
    help="Random seed")
parser.add_argument(
    "--iband-max", type=float, default=25.0,
    help="Only fit to data with apparent mag_i < IBAND_MAX")
parser.add_argument(
    "-m", "--lgmp-min", type=float, default=10.5,
    help="Minimum lgmp value for mc lightcone")
parser.add_argument(
    "-a", "--sky-area-degsq", type=float, default=0.1,
    help="Sky area in square degrees for the mc lightcone")
parser.add_argument(
    "-k", "--num-kernels", type=int, default=40,
    help="Number of kernels for kdescent")
parser.add_argument(
    "-f", "--num-fourier-positions", type=int, default=20,
    help="Number of Fourier evaluation positions for kdescent")
parser.add_argument(
    "--hmf-calibration", type=str, default=None,
    help="Specify diffmahpop params ('smdpl_hmf', 'smdpl_shmf', 'hacc_shmf')")
parser.add_argument(
    "--log-loss", action="store_true",
    help="Perform logarithm on KDE counts for loss computation")
parser.add_argument(
    "--max-n-halos-per-bin", type=int, default=200,
    help="Number of halos to select from each halo_upweight bin")
parser.add_argument(
    "--n-halo-weight-bins", type=int, default=50,
    help="Number of Fourier evaluation positions for kdescent")
parser.add_argument(
    "--num-mag-z-kernels", type=int, default=20,
    help="Number of kernels for 2D mag-z kdescent term")

if __name__ == "__main__":
    args = parser.parse_args()
    cosmos_fit = CosmosFit(
        i_thresh=args.iband_max,
        lgmp_min=args.lgmp_min,
        sky_area_degsq=args.sky_area_degsq,
        num_kernels=args.num_kernels,
        num_fourier_positions=args.num_fourier_positions,
        hmf_calibration=args.hmf_calibration,
        log_loss=args.log_loss,
        max_n_halos_per_bin=args.max_n_halos_per_bin,
        n_halo_weight_bins=args.n_halo_weight_bins,
        num_mag_z_kernels=args.num_mag_z_kernels)

    u_param_init = None
    if args.input is not None:
        u_param_init = np.load(args.input)["params"][-1]

    adam_fit(cosmos_fit, args.output, nsteps=args.nsteps,
             learning_rate=args.learning_rate, u_param_init=u_param_init)
