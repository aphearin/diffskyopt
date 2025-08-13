import os
import warnings
import argparse
import glob
from mpi4py import MPI

if not MPI.COMM_WORLD.rank:
    import matplotlib.pyplot as plt
    import corner
    import seaborn

import numpy as np
import jax
import jax.numpy as jnp

from ..lossfuncs.cosmos_fit import CosmosFit
from .. import trange

ran_keys = jax.random.split(jax.random.key(1), 4)
keys = jax.random.split(ran_keys[3], 100)

TARGET_LABELS = [
    "$\\rm m_i$", "$\\rm redshift$", "$\\rm g - r$", "$\\rm r - i$",
    "$\\rm i - z$", "$\\rm z - y$", "$\\rm Y - J$", "$\\rm J - H$",
    "$\\rm H - K_s$",
]

if os.environ.get("DIFFSKYOPT_REMOVE_G_FILTER"):
    TARGET_LABELS.remove("$\\rm g - r$")


def color_magnitude_test(cosmos_fit, save=False, x_index=0, y_index=1,
                         ax=None, show=True, make_legend=True,
                         model_params=None,
                         data_label="COSMOS", model_label="Diffsky Model",
                         model_targets_and_weights=None):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    # Plot a color-magnitude diagram: g-r vs i-band
    targets, weights = cosmos_fit.data_targets, cosmos_fit.data_weights
    if not MPI.COMM_WORLD.rank:
        if ax is None:
            ax = plt.subplots()[1]
        ax.scatter(targets[:, x_index], targets[:, y_index],
                   alpha=weights/weights.max(), color="C0", s=1)
        ax.set_xlabel(TARGET_LABELS[x_index])
        ax.set_ylabel(TARGET_LABELS[y_index])
    # Compute model targets + weights with default u_params
    # and overplot with a seaborn.kdeplot
    if model_targets_and_weights is None:
        model_targets, model_weights = \
            cosmos_fit.targets_and_weights_from_params(
                model_params, keys[0])
        model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
        model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))
    else:
        model_targets, model_weights = model_targets_and_weights
    if not MPI.COMM_WORLD.rank:
        seaborn.kdeplot(
            x=model_targets[:, x_index], y=model_targets[:, y_index],
            weights=model_weights, color="C1", fill=True,
            thresh=0.05, levels=4, alpha=0.3, zorder=10, ax=ax)
        ax.scatter([], [], color="C0", label=data_label)
        ax.plot([], [], color="C1", label=model_label)

        if make_legend:
            ax.legend(frameon=False)
        if save:
            plt.savefig("color_magnitude_test.png", bbox_inches="tight")
            plt.clf()
        elif show:
            plt.show()
            plt.clf()


def targets_triangle_test(cosmos_fit, save=False, data_label="COSMOS",
                          model_label="Diffsky Model", model_params=None):
    # Plot every combination of targets in a triangle plot,
    # using color_magnitude_test() to perform each subplot
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr

    n = len(TARGET_LABELS)
    axes = None
    if not MPI.COMM_WORLD.rank:
        axes = plt.subplots(
            n, n, figsize=(12, 12), sharex="col", sharey="row")[1]

    model_targets, model_weights = cosmos_fit.targets_and_weights_from_params(
        model_params, keys[0])
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))
    tnw = model_targets, model_weights

    if not MPI.COMM_WORLD.rank:
        progress = 0
        print(f"Triangle plot progress (out of {int(n*(n-1)/2)}): 0",
              end=" ", flush=True)
    for i in range(n):
        for j in range(n):
            ax = None if axes is None else axes[i, j]
            if j >= i:
                if ax is not None:
                    ax.set_visible(False)
                continue
            color_magnitude_test(
                cosmos_fit, x_index=j, y_index=i, ax=ax, show=False,
                make_legend=False, model_targets_and_weights=tnw,
                model_label="", data_label="")
            if not MPI.COMM_WORLD.rank:
                progress += 1
                print(progress, end=" ", flush=True)
    if axes is not None:
        print(flush=True)
        ax = axes[1, 0]
        ax.scatter([], [], color="C0", label=data_label)
        ax.plot([], [], color="C1", label=model_label)
        ax.legend(frameon=False, loc=(1.1, 0.1))

    if not MPI.COMM_WORLD.rank:
        if save:
            plt.savefig("targets_triangle_test.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


def targets_corner_test(cosmos_fit, save=False, data_label="COSMOS",
                        model_label="Diffsky Model", model_params=None,
                        prefix="", loghist=True, ithreshes=None):
    """
    Plot a corner plot (using corner.corner) comparing COSMOS data targets to
    model targets generated with the given model_params.
    """
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    if ithreshes is None:
        ithreshes = [21, 23, 25]

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
        model_params, keys[0])
    model_targets = np.concatenate(MPI.COMM_WORLD.allgather(model_targets))
    model_weights = np.concatenate(MPI.COMM_WORLD.allgather(model_weights))

    if not MPI.COMM_WORLD.rank:
        for j in trange(len(ithreshes)):
            ithresh = ithreshes[j]
            data = data_targets[data_targets[:, 0] < ithresh]
            model = model_targets[model_targets[:, 0] < ithresh]
            data_w = data_weights[data_targets[:, 0] < ithresh]
            model_w = model_weights[model_targets[:, 0] < ithresh]

            labels = TARGET_LABELS
            density = False
            ranges = []
            for i in range(data.shape[1]):
                range_i = np.percentile(
                    data[:, i], [1, 99], weights=data_w,
                    method="inverted_cdf")
                dx = 0.2 * (range_i[1] - range_i[0])
                range_i = (range_i[0] - dx, range_i[1] + dx)
                ranges.append(range_i)
            # Plot data as filled contours, model as lines
            histkw = {"density": density,
                      "color": "C0", "histtype": "stepfilled"}
            fig = corner.corner(
                data, weights=data_w, color="C0", bins=18,
                labels=labels, label_kwargs={"fontsize": 12}, alpha=1.0,
                plot_density=True, plot_contours=True, fill_contours=True,
                contour_kwargs={"colors": ["C0"]}, plot_datapoints=False,
                show_titles=False, title_kwargs={"fontsize": 10},
                hist_kwargs=histkw, range=ranges,
            )
            # Save ylims after first call
            axes = np.array(fig.axes).reshape(
                (data.shape[1], data.shape[1]))
            data_ylims = [axes[i, i].get_ylim()
                          for i in range(data.shape[1])]
            if loghist:
                for i in range(data.shape[1]):
                    axes[i, i].semilogy()

            histkw = {"density": density, "color": "C1", "lw": 2}
            with warnings.catch_warnings(action="ignore"):
                corner.corner(
                    model, weights=model_w, color="C1", bins=18,
                    labels=labels, fig=fig, alpha=0.8,
                    plot_density=False, plot_contours=True, fill_contours=True,
                    contour_kwargs={"colors": ["C1"]}, plot_datapoints=False,
                    hist_kwargs=histkw, range=ranges,
                )
            # Save ylims after second call
            model_ylims = [axes[i, i].get_ylim()
                           for i in range(model.shape[1])]

            # Set ylims to encompass both
            for i in range(data.shape[1]):
                ymin = min(data_ylims[i][0], model_ylims[i][0])
                ymax = max(data_ylims[i][1], model_ylims[i][1])
                if loghist:
                    axes[i, i].semilogy()
                    ymin = max(ymin, ymax / 1e4, axes[i, i].get_ylim()[0])
                axes[i, i].set_ylim(ymin, ymax)

            handles = [
                plt.Line2D([], [], color="none", lw=0,
                           label=f"$\\rm m_i < {ithresh:.3g}$"),
                plt.Line2D([], [], color="C0", lw=4, label=data_label),
                plt.Line2D([], [], color="C1", lw=4, label=model_label),
            ]
            fig.legend(
                handles=handles, loc=(0.5, 0.65), frameon=False, fontsize=20)
            if save:
                plt.savefig(
                    f"{prefix}m{ithresh:g}_targets_corner_test.png",
                    bbox_inches="tight")
            else:
                plt.show()
            plt.clf()


def plot_losscurve(losscalc, label=None, fixed_key=False, color=None,
                   **kwargs):
    p_vals = []
    loss_vals = []
    for i in trange(len(keys), desc=label):
        key = keys[0] if fixed_key else keys[i]
        p = losscalc.aux_data["fit_instance"].default_u_param_arr * (
            (i - 50) / 50.0 + 1.0)
        p_vals.append(p)
        loss_vals.append(losscalc.calc_loss_from_params(p, key))
    p_vals = jnp.array(p_vals)
    loss_vals = jnp.array(loss_vals)
    if not MPI.COMM_WORLD.rank:
        plt.plot(
            (p_vals[:, 0]
             / losscalc.aux_data["fit_instance"].default_u_param_arr[0]) - 1,
            loss_vals, label=label, color=color, **kwargs)


def losscurve_test(cosmos_fit, save=False, model_params=None, prefix=""):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    calc = cosmos_fit.get_multi_grad_calc()

    # Characteristic noise: random key varies
    plot_losscurve(calc, label="Characteristic noise",
                   fixed_key=False, color="grey", lw=1)

    # HMF-sampled model: fixed key
    plot_losscurve(calc, label="HMF-sampled model",
                   fixed_key=True, color="C1", lw=2)

    if not MPI.COMM_WORLD.rank:
        plt.semilogy()
        plt.axvline(0, color="k", ls="--")
        plt.xlabel("Fractional change in u_params")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        if save:
            plt.savefig(f"{prefix}losscurve_test.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


def adam_steps_test(cosmos_fit, seed=1, num_learns=15, num_inits=15,
                    save=False, model_params=None, param_scatter_factor=0.0,
                    prefix=""):
    if model_params is None:
        model_params = cosmos_fit.default_u_param_arr
    # Run Adam optimization from random initializations and plot loss
    calc = cosmos_fit.get_multi_grad_calc()
    keys = jax.random.split(jax.random.key(seed), num_inits * 2)
    if not param_scatter_factor:
        u_param_inits = jnp.array([model_params for i in range(num_inits)])
    else:
        cov = jnp.diag(
            (0.2 * param_scatter_factor *
             cosmos_fit.default_u_param_arr) ** 2 + 1e-3)
        u_param_inits = jnp.array([jax.random.multivariate_normal(
            keys[i], model_params, cov)
            for i in range(num_inits)])
    learn_rate_vals = jnp.logspace(-4, 1, num_learns)
    u_param_inits_gather = jnp.array(MPI.COMM_WORLD.allgather(u_param_inits))
    assert jnp.all(u_param_inits[None, :] == u_param_inits_gather)

    loss_improvement = []
    for i in trange(len(u_param_inits), desc="Inits"):
        u_param_init = u_param_inits[i]
        loss_improvement.append([])
        for j in trange(len(learn_rate_vals), leave=False):
            learn_rate = learn_rate_vals[j]
            params, losses = calc.run_adam(
                u_param_init, nsteps=10, learning_rate=learn_rate,
                randkey=keys[num_inits + i], progress=False)
            loss_improvement[-1].append(losses[0] - losses[-1])
    loss_improvement = jnp.array(loss_improvement)

    if not MPI.COMM_WORLD.rank:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.axhline(0, color="grey", ls="--")
        for i in range(num_inits):
            ax.plot(learn_rate_vals, loss_improvement[i],
                    label=f"Init {i+1}", marker="o", zorder=1)
        mean = jnp.mean(loss_improvement, axis=0)
        std = jnp.std(loss_improvement, axis=0, ddof=1)
        ax.fill_between(
            learn_rate_vals, mean + std, mean - std,
            color="C0", alpha=0.3, zorder=2)
        top = abs(max(mean + std))
        ax.set_ylim(bottom=-3 * top, top=1.2 * top)

        # Mark learn_rate value that consistently performs the best
        argmax = jnp.argmax(mean - std)
        best_learn_rate = float(learn_rate_vals[argmax])
        plt.axvline(best_learn_rate, color="k", ls="--", lw=0.5, zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("symlog")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss improvement")
        plt.tight_layout()
        if save:
            plt.savefig(f"{prefix}adam_steps_test.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


def loss_results_plot(cosmos_fit, result_files, save=False,
                      recompute_loss=False, thin=1, seed=1, prefix=""):
    if not result_files:
        raise ValueError(f"No results files given: {result_files=}")
    randkey = jax.random.key(seed)
    results = [np.load(fn) for fn in result_files]
    lens = [len(x["losses"]) for x in results]
    edges = np.cumsum([0, *lens])
    for i in trange(len(lens), desc="Result files"):
        start, end = edges[i:i+2]
        res = results[i]
        if recompute_loss:
            calc = cosmos_fit.get_multi_grad_calc()
            loss = []
            params = res["params"][::thin]
            iterator = trange(
                len(params), desc=f"Recomputing loss for {result_files[i]}")
            for j in iterator:
                loss_i = calc.calc_loss_from_params(params[j], randkey)
                loss.append(loss_i)
            loss = np.array(loss)
        else:
            loss = res["losses"][::thin]
        steps = np.linspace(start, end, len(loss), endpoint=False)
        if not MPI.COMM_WORLD.rank:
            plt.semilogy(steps, loss, color="C0")
            if i > 0:
                plt.axvline(start - thin/2, color="grey", ls="--", lw=0.5)

    if not MPI.COMM_WORLD.rank:
        plt.xlabel("step")
        plt.ylabel("loss")
        if save:
            plt.savefig(f"{prefix}loss_vs_step.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


parser = argparse.ArgumentParser(
    description="Test real fits to COSMOS data")
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
    "-a", "--sky-area-degsq", type=float, default=0.1,
    help="Sky area in square degrees for the mc lightcone")
parser.add_argument(
    "--num-halos", type=int, default=5000,
    help="Number of halos for the mc lightcone")
parser.add_argument(
    "-k", "--num-kernels", type=int, default=40,
    help="Number of kernels for kdescent")
parser.add_argument(
    "-f", "--num-fourier-positions", type=int, default=20,
    help="Number of Fourier evaluation positions for kdescent")
parser.add_argument(
    "--num-mag-z-kernels", type=int, default=40,
    help="Number of kernels for 2D mag-z kdescent term")
parser.add_argument(
    "--skip-corner", action="store_true",
    help="Skip targets_corner_test")
parser.add_argument(
    "--skip-loss", action="store_true",
    help="Skip losscurve_test")
parser.add_argument(
    "--skip-adam", action="store_true",
    help="Skip adam_steps_test")
parser.add_argument(
    "--param-results", type=str, default=None,
    help="Results .npz file to load final parameter value from")
parser.add_argument(
    "--param-scatter-factor", type=float, default=0.0,
    help="In adam_steps_test, scatter init points more or less")
parser.add_argument(
    "--loss-results", type=str, default=None,
    help="Specify globbed result(s) files to plot the loss curve")
parser.add_argument(
    "--recompute-loss-results", action="store_true",
    help="Recompute losses w fixed seed, ignored without --loss-results given")
parser.add_argument(
    "--recompute-loss-thin", type=int, default=1,
    help="Only recompute one loss every THIN steps")
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
    "--kidw", type=float, default=0.0,
    help="Inverse density weighting for kdescent")

if __name__ == "__main__":
    args = parser.parse_args()
    cosmos_fit = CosmosFit(
        num_halos=args.num_halos,
        i_thresh=args.iband_max,
        lgmp_min=args.lgmp_min,
        sky_area_degsq=args.sky_area_degsq,
        num_kernels=args.num_kernels,
        num_fourier_positions=args.num_fourier_positions,
        hmf_calibration=args.hmf_calibration,
        log_loss=args.log_loss,
        max_n_halos_per_bin=args.max_n_halos_per_bin,
        n_halo_weight_bins=args.n_halo_weight_bins,
        num_mag_z_kernels=args.num_mag_z_kernels,
        kde_idw_power=args.kidw)

    model_params = None
    if args.param_results:
        model_params = np.load(args.param_results)["params"][-1]

    if args.loss_results is not None:
        result_files = glob.glob(args.loss_results)
        loss_results_plot(
            cosmos_fit, result_files, save=args.save,
            recompute_loss=args.recompute_loss_results,
            thin=args.recompute_loss_thin, prefix=args.prefix)
    if not args.skip_corner:
        targets_corner_test(
            cosmos_fit, save=args.save, model_params=model_params,
            prefix=args.prefix)
    if not args.skip_loss:
        losscurve_test(
            cosmos_fit, save=args.save, model_params=model_params,
            prefix=args.prefix)
    if not args.skip_adam:
        adam_steps_test(
            cosmos_fit, save=args.save, model_params=model_params,
            param_scatter_factor=args.param_scatter_factor, prefix=args.prefix)
