import argparse
from mpi4py import MPI

if not MPI.COMM_WORLD.rank:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

import numpy as np
import jax
# import jax.numpy as jnp
from astropy.cosmology import Planck15
from astropy import units

from diffsky.experimental.diagnostics import check_smhm
from diffsky.param_utils import diffsky_param_wrapper as dpw

from ..lossfuncs.cosmos_fit import COSMOS_SKY_AREA, CosmosFit
# from .. import trange

ran_key = jax.random.key(1)


def cosmos_volume(z_min, z_max):
    """Volume in spherical shell between z_min < z < z_max in (Mpc/h)^3"""
    full_sky_area = (4 * np.pi) * (180 / np.pi) ** 2  # square degrees
    fsky = COSMOS_SKY_AREA / full_sky_area
    vol_com_sphere = Planck15.comoving_volume(  # type: ignore
        z_max) - Planck15.comoving_volume(z_min)  # type: ignore
    return fsky*vol_com_sphere / (units.Mpc / Planck15.h)**3  # type: ignore


def n_of_z_plot(cosmos_fit, save=False, zbins=30,
                ax=None, show=True, make_legend=True,
                model_params=None, prefix="",
                data_label="COSMOS"):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    # i_threshes = [22.0, 23.0, 24.0, 25.0]
    i_threshes = [20.0, 21.5, 23.0, 25.0]
    # Plot a color-magnitude diagram: g-r vs i-band
    data_targets, data_weights = (cosmos_fit.data_targets,
                                  cosmos_fit.data_weights)

    # Compute model targets + weights with default u_params
    # and overplot with a seaborn.kdeplot
    model_targets, model_weights = \
        cosmos_fit.targets_and_weights_from_params(
            model_params, ran_key)
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))
    if not MPI.COMM_WORLD.rank:
        data_i = data_targets[:, 0]
        model_i = model_targets[:, 0]
        data_z = data_targets[:, 1]
        model_z = model_targets[:, 1]
        z_edges = np.linspace(cosmos_fit.zmin, cosmos_fit.zmax, zbins)
        z_cens = 0.5 * (z_edges[:-1] + z_edges[1:])
        dz = np.diff(z_edges)
        for i, ithresh in enumerate(i_threshes):
            data_n_of_z = np.histogram(
                data_z[data_i < ithresh], z_edges,
                weights=data_weights[data_i < ithresh])[0]
            model_n_of_z = np.histogram(
                model_z[model_i < ithresh], z_edges,
                weights=model_weights[model_i < ithresh])[0]

            # cosmos_volume_per_zbin = np.array([
            #     cosmos_volume(*z_edges[i:i+2]) for i in range(len(z_cens))])
            # data_n_of_z /= cosmos_volume_per_zbin
            # model_n_of_z /= cosmos_volume_per_zbin
            data_n_of_z /= dz
            model_n_of_z /= dz
            model_n_of_z /= (data_n_of_z * dz).sum()
            data_n_of_z /= (data_n_of_z * dz).sum()

            if ax is None:
                ax = plt.subplots()[1]
            ax.step(z_edges, [data_n_of_z[0], *data_n_of_z],
                    color=f"C{i}", ls="--")
            ax.plot(z_cens, model_n_of_z, label=f"$\\rm m_i < {ithresh:.1f}$")
        ax.step([], [], color="k", ls="--", label=data_label)
        ax.semilogy()
        ax.set_xlim(cosmos_fit.zmin, cosmos_fit.zmax)
        ax.set_xlabel("$z$")
        ax.set_ylabel("$\\rm n(z)$")

        if make_legend:
            ax.legend(frameon=False)
        if save:
            plt.savefig(f"{prefix}n_of_z.png", bbox_inches="tight")
            plt.clf()
        elif show:
            plt.show()
            plt.clf()


def n_of_ithresh_plot(cosmos_fit, save=False, ibins=30, ithresh=25.0,
                      ax=None, show=True, make_legend=True,
                      model_params=None, prefix="",
                      data_label="COSMOS", model_label="Diffsky Model"):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    data_targets, data_weights = (cosmos_fit.data_targets,
                                  cosmos_fit.data_weights)

    model_targets, model_weights = \
        cosmos_fit.targets_and_weights_from_params(
            model_params, ran_key)
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))
    if not MPI.COMM_WORLD.rank:
        data_i = data_targets[:, 0]
        model_i = model_targets[:, 0]
        all_i = np.concatenate([data_i, model_i])
        i_vals = np.linspace(all_i.min(), ithresh, ibins)
        data_hist = np.cumsum(
            np.histogram(data_i, i_vals, weights=data_weights)[0])
        model_hist = np.cumsum(
            np.histogram(model_i, i_vals, weights=model_weights)[0])

        ess = np.sum(data_weights)**2 / np.sum(data_weights**2)
        f_ess = ess / np.sum(data_weights)
        data_n_of_ithresh = data_hist / COSMOS_SKY_AREA
        model_n_of_ithresh = model_hist / COSMOS_SKY_AREA
        frac_err = 1 / np.sqrt(f_ess * data_hist) + 0.1
        data_err = data_n_of_ithresh * frac_err

        # Fractional residuals
        frac_resid = (model_n_of_ithresh - data_n_of_ithresh
                      ) / data_n_of_ithresh

        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0)
            ax = fig.add_subplot(gs[0])
            ax_resid = fig.add_subplot(gs[1], sharex=ax)
        else:
            fig = ax.figure
            gs = None
            ax_resid = None  # Not supported if ax is provided

        # Upper panel: cumulative counts
        ax.plot(i_vals[1:], data_n_of_ithresh, label=data_label, color="C0")
        ax.fill_between(i_vals[1:], data_n_of_ithresh + data_err,
                        data_n_of_ithresh - data_err, color="C0", alpha=0.5)
        ax.plot(i_vals[1:], model_n_of_ithresh, label=model_label, color="C1")
        ax.semilogy()
        ax.set_ylabel("$\\rm n(<mag) \\; [deg^{-2}]$")
        if make_legend:
            ax.legend(frameon=False)
        ax.tick_params(labelbottom=False)  # Hide x labels on upper panel

        # Lower panel: fractional residuals
        if ax_resid is not None:
            ax_resid.axhline(0, color="C0")
            ax_resid.fill_between(i_vals[1:], -frac_err, frac_err,
                                  color="C0", alpha=0.5)
            ax_resid.plot(i_vals[1:], frac_resid, color="C1")
            ax_resid.set_ylabel("$\\rm \\Delta n / n$")
            ax_resid.set_xlabel("$\\rm m_i$")
            ax_resid.set_ylim(-1, 1)

            # Remove the top y-tick label to avoid overlap
            yticks = ax_resid.get_yticks()
            yticklabels = [f"{x:g}" for x in yticks]
            if len(yticklabels) > 1:
                yticklabels[-1] = ""
            ax_resid.set_yticklabels(yticklabels)
            # Remove space between panels
            plt.subplots_adjust(hspace=0)
            plt.setp(ax.get_xticklabels(), visible=False)

        if save:
            plt.savefig(f"{prefix}n_of_ithresh.png", bbox_inches="tight")
            plt.clf()
        elif show:
            plt.show()
            plt.clf()


def smhm_drift_plot(save=False, prefix="", model_params=None):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    u_param_collection = dpw.get_u_param_collection_from_u_param_array(
        model_params)
    diffstarpop_params = dpw.get_param_collection_from_u_param_collection(
        *u_param_collection)[0]

    if save:
        check_smhm.plot_diffstarpop_insitu_smhm(
            diffstarpop_params2=diffstarpop_params,
            fname=f"{prefix}smhm_drift.png",
        )


parser = argparse.ArgumentParser(
    description="Validate model vs COSMOS data with n(z) and n(<m_i)")
parser.add_argument(
    "-s", "--save", action="store_true",
    help="Save plots instead of opening interactive window")
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

    n_of_z_plot(cosmos_fit, save=args.save, prefix=args.prefix,
                model_params=model_params)
    n_of_ithresh_plot(cosmos_fit, save=args.save, prefix=args.prefix,
                      ithresh=args.iband_max, model_params=model_params)
    smhm_drift_plot(save=args.save, prefix=args.prefix,
                    model_params=model_params)
