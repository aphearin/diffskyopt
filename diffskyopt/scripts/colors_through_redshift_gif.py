import argparse
from mpi4py import MPI

if not MPI.COMM_WORLD.rank:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

import numpy as np
import jax
# import jax.numpy as jnp

from ..lossfuncs.cosmos_fit import CosmosFit
from .cosmos_fit_test import TARGET_LABELS
from .. import trange

ran_key = jax.random.key(1)


def colors_gif(
    cosmos_fit, model_params, prefix="",
    num_z_bins=10, bins=30, duration=0.5
):

    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr

    # Gather data targets from all ranks
    data_targets = np.array(cosmos_fit.data_targets)
    data_weights = np.array(cosmos_fit.data_weights)
    assert np.allclose(
        np.array(MPI.COMM_WORLD.allgather(cosmos_fit.data_targets)),
        cosmos_fit.data_targets[None])
    assert np.allclose(
        np.array(MPI.COMM_WORLD.allgather(cosmos_fit.data_weights)),
        cosmos_fit.data_weights[None])

    # Gather model targets from all ranks
    model_targets, model_weights = cosmos_fit.targets_and_weights_from_params(
        model_params, ran_key, modelsamp=False)
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))

    if MPI.COMM_WORLD.rank:
        return

    # Redshift is always the second column (index 1)
    z_data = data_targets[:, 1]
    z_model = model_targets[:, 1]
    data_targets = np.delete(data_targets, 1, axis=1)
    model_targets = np.delete(model_targets, 1, axis=1)
    z_bins = np.linspace(cosmos_fit.zmin, cosmos_fit.zmax, num_z_bins + 1)
    labels = [*TARGET_LABELS[:1], *TARGET_LABELS[2:]]

    fig, axes = plt.subplots(1, 7, figsize=(28, 4), sharey=False)

    def plot_slice(i):
        zlo, zhi = z_bins[i], z_bins[i+1]
        data_mask = (z_data >= zlo) & (z_data < zhi)
        model_mask = (z_model >= zlo) & (z_model < zhi)
        for j, ax in enumerate(axes):
            xidx = j
            yidx = j + 1
            ax.clear()
            # 2D histogram for data
            H_data, xedges, yedges = np.histogram2d(
                data_targets[data_mask, xidx], data_targets[data_mask, yidx],
                bins=bins, weights=data_weights[data_mask])
            level_max = H_data.max() / 5.0
            levels = [level_max, level_max / 10.0, level_max / 100.0]
            X, Y = np.meshgrid(
                0.5 * (xedges[:-1] + xedges[1:]),
                0.5 * (yedges[:-1] + yedges[1:]))
            if np.any(H_data > 0):
                ax.contour(
                    X, Y, H_data.T, levels=levels, colors="C0",
                    linewidths=1.2, alpha=0.7, label="COSMOS")
            # 2D histogram for model
            H_model, _, _ = np.histogram2d(
                model_targets[model_mask,
                              xidx], model_targets[model_mask, yidx],
                bins=[xedges, yedges], weights=model_weights[model_mask])
            if np.any(H_model > 0):
                ax.contour(
                    X, Y, H_model.T, levels=levels, colors="C1",
                    linewidths=1.2, alpha=0.7, label="Model")
            ax.set_xlabel(labels[xidx])
            ax.set_ylabel(labels[yidx])
            # Dummy handles for legend
            ax.plot([], [], color="C0", label="COSMOS")
            ax.plot([], [], color="C1", label="Model")
            ax.legend(frameon=False)
        fig.suptitle(f"$\\rm {zlo:.2f} < z < {zhi:.2f}$")

    pbar = trange(num_z_bins + 1)

    def update(frame):
        plot_slice(frame)
        pbar.update()
        return axes

    ani = animation.FuncAnimation(
        fig, update, frames=num_z_bins, blit=False, repeat=True)

    ani.save(f"{prefix}colors-through-redshift.gif",
             writer='pillow', fps=int(1/duration))
    plt.close(fig)


parser = argparse.ArgumentParser(
    description="Validate model vs COSMOS data with n(z) and n(<m_i)")
parser.add_argument(
    "--prefix", type=str, default="",
    help="Prefix for saved plot filenames, ignored if --save is not set")
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
    "--param-results", type=str, default=None,
    help="Results .npz file to load final parameter value from")
parser.add_argument(
    "--hmf-calibration", type=str, default=None,
    help="Specify diffmahpop params ('smdpl_hmf', 'smdpl_shmf', 'hacc_shmf')")

if __name__ == "__main__":
    args = parser.parse_args()
    cosmos_fit = CosmosFit(
        i_thresh=args.iband_max,
        lgmp_min=args.lgmp_min,
        sky_area_degsq=args.sky_area_degsq,
        hmf_calibration=args.hmf_calibration)

    model_params = None
    if args.param_results:
        results = np.load(args.param_results)
        if "best_params" in results.keys():
            model_params = results["best_params"]
        else:
            model_params = results["params"][-1]

    colors_gif(cosmos_fit, prefix=args.prefix, model_params=model_params)
