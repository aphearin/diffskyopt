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
    num_z_bins=15, bins=30, duration=0.5, dz=0.2, ithresh=None,
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
        model_params, ran_key)
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))

    # Filter data and model targets by apparent magnitude threshold
    if ithresh is not None:
        data_mask = data_targets[:, 0] < ithresh
        model_mask = model_targets[:, 0] < ithresh
        data_targets = data_targets[data_mask]
        data_weights = data_weights[data_mask]
        model_targets = model_targets[model_mask]
        model_weights = model_weights[model_mask]

    if MPI.COMM_WORLD.rank:
        return

    # Redshift is always the second column (index 1)
    z_data = data_targets[:, 1]
    z_model = model_targets[:, 1]
    data_targets = np.delete(data_targets, 1, axis=1)
    model_targets = np.delete(model_targets, 1, axis=1)
    z_lows = np.linspace(cosmos_fit.zmin, cosmos_fit.zmax - dz, num_z_bins)
    z_highs = np.linspace(cosmos_fit.zmin + dz, cosmos_fit.zmax, num_z_bins)
    labels = [*TARGET_LABELS[:1], *TARGET_LABELS[2:]]

    if data_targets.shape[1] < 8:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8),
                                 gridspec_kw={"wspace": 0.3})
        title_prefix = "\n"
    else:
        fig, axes = plt.subplots(3, 3, figsize=(13, 12),
                                 gridspec_kw={"wspace": 0.3})
        title_prefix = "\n\n"
    axes = axes.ravel()

    ranges: list[tuple[float, float]] = []
    both_targets = np.concatenate(
        [data_targets, model_targets])
    both_weights = np.concatenate(
        [data_weights, model_weights])
    for i in range(data_targets.shape[1]):
        quants = np.quantile(
            both_targets[:, i], [0.005, 0.995], weights=both_weights,
            method="inverted_cdf")
        dx = 0.1 * (quants[1] - quants[0])
        range_i = (quants[0] - dx, quants[1] + dx)
        ranges.append(range_i)

    def make_levels(hist2d):
        level_max = hist2d.max() / 3.0
        return level_max / np.array([7.0, 1.0])

    def plot_slice(i):
        zlo, zhi = z_lows[i], z_highs[i]
        data_mask = (z_data >= zlo) & (z_data < zhi)
        model_mask = (z_model >= zlo) & (z_model < zhi)

        for j, ax in enumerate(axes):
            if j > data_targets.shape[1] - 2:
                ax.set_visible(False)
                continue

            xidx = j
            yidx = j + 1
            ax.clear()
            # Scatter plots for data and model
            dw = data_weights[data_mask]
            mw = model_weights[model_mask]
            data_alpha = 3000 * dw / dw.sum()
            data_alpha[data_alpha > 1] = 1
            model_alpha = 3000 * mw / mw.sum()
            model_alpha[model_alpha > 1] = 1
            if data_mask.any():
                ax.scatter(
                    data_targets[data_mask, xidx],
                    data_targets[data_mask, yidx],
                    color="C0", s=0.5, alpha=data_alpha)
            if model_mask.any():
                ax.scatter(
                    model_targets[model_mask, xidx],
                    model_targets[model_mask, yidx],
                    color="C1", s=0.5,
                    alpha=model_alpha)

            # 2D histogram for data and model
            xedges = np.linspace(*ranges[xidx], bins)
            yedges = np.linspace(*ranges[yidx], bins)
            h_data, _, _ = np.histogram2d(
                data_targets[data_mask, xidx], data_targets[data_mask, yidx],
                bins=[xedges, yedges], weights=data_weights[data_mask])
            h_model, _, _ = np.histogram2d(
                model_targets[model_mask, xidx],
                model_targets[model_mask, yidx],
                bins=[xedges, yedges], weights=model_weights[model_mask])
            X, Y = np.meshgrid(
                0.5 * (xedges[:-1] + xedges[1:]),
                0.5 * (yedges[:-1] + yedges[1:]))

            levels = make_levels(h_data)
            if np.any(h_data > 0):
                ax.contour(
                    X, Y, h_data.T, levels=levels, colors="C0",
                    linewidths=2, alpha=0.9, label="COSMOS")
            else:
                levels = make_levels(h_model)
            if np.any(h_model > 0):
                ax.contour(
                    X, Y, h_model.T, levels=levels, colors="C1",
                    linewidths=2, alpha=0.9, label="Model")
            ax.set_xlabel(labels[xidx], fontsize=16)
            ax.set_ylabel(labels[yidx], fontsize=16)
            ax.set_xlim(ranges[xidx])
            ax.set_ylim(ranges[yidx])
        fig.suptitle(
            f"{title_prefix}$\\rm z = {(zlo+zhi)/2:.1f} \\pm {dz/2:.1f}$"
            f", $\\rm m_i < {ithresh:.3g}$", fontsize=24)
        handles = [
            plt.Line2D([], [], color="C0", lw=4, label="COSMOS"),
            plt.Line2D([], [], color="C1", lw=4, label="Diffsky"),
        ]
        fig.legend(
            # handles=handles, loc=(0.5, 0.25), frameon=False, fontsize=20)
            handles=handles, loc=(0.13, 0.89), frameon=False, fontsize=16)

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
    "--num-halos", type=int, default=5000,
    help="Number of halos for the mc lightcone")
parser.add_argument(
    "--param-results", type=str, default=None,
    help="Results .npz file to load final parameter value from")
parser.add_argument(
    "--hmf-calibration", type=str, default=None,
    help="Specify diffmahpop params ('smdpl_hmf', 'smdpl_shmf', 'hacc_shmf')")

if __name__ == "__main__":
    args = parser.parse_args()
    cosmos_fit = CosmosFit(
        num_halos=args.num_halos,
        i_thresh=args.iband_max,
        lgmp_min=args.lgmp_min,
        hmf_calibration=args.hmf_calibration)

    model_params = None
    if args.param_results:
        results = np.load(args.param_results)
        if "best_params" in results.keys():
            model_params = results["best_params"]
        else:
            model_params = results["params"][-1]
    for ithresh in [21, 23, 25]:
        if not MPI.COMM_WORLD.rank:
            print(
                f"Generating colors gif for i-band threshold {ithresh}",
                flush=True)
        colors_gif(cosmos_fit, model_params=model_params,
                   prefix=args.prefix + "m" + str(ithresh) + "_",
                   ithresh=ithresh)
