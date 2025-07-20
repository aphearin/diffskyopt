import argparse
from mpi4py import MPI

if not MPI.COMM_WORLD.rank:
    import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from ..lossfuncs.self_fit import SelfFit
from .. import trange

from .cosmos_fit_test import targets_corner_test

ran_keys = jax.random.split(jax.random.key(1), 4)
keys = jax.random.split(ran_keys[3], 100)


# Helper function for losscurve_test()
def plot_losscurve(losscalc, label=None, fixed_key=False, **kwargs):
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
            loss_vals, label=label, **kwargs)


def losscurve_test(self_fit, save=False, prefix=""):
    calc_sampdata_sampmodel = self_fit.get_multi_grad_calc()
    calc_fulldata_fullmodel = self_fit.get_multi_grad_calc()
    calc_fulldata_sampmodel = self_fit.get_multi_grad_calc()
    calc_sampdata_fullmodel = self_fit.get_multi_grad_calc()

    # One non-fixed key curve to demonstrate characteristic noise
    plot_losscurve(calc_sampdata_sampmodel, label="Characteristic noise",
                   fixed_key=False, color="grey", lw=1)

    # The rest will be run with a fixed key to remove (well, freeze) the noise
    plot_losscurve(calc_fulldata_fullmodel,
                   label="Full data vs full model",
                   fixed_key=True, color="C0", lw=2)
    plot_losscurve(calc_fulldata_sampmodel,
                   label="Full data vs HMF-sampled model",
                   fixed_key=True, color="C1", lw=2)
    plot_losscurve(calc_sampdata_fullmodel,
                   label="HMF-sampled data vs full model",
                   fixed_key=True, color="C2", lw=2)
    plot_losscurve(calc_sampdata_sampmodel,
                   label="HMF-sampled data vs HMF-sampled model",
                   fixed_key=True, color="C3", lw=2)

    if not MPI.COMM_WORLD.rank:
        plt.semilogy()
        plt.axvline(0, color="k", ls="--")
        plt.legend(frameon=False)
        plt.xlabel("Fractional change in u_params")
        plt.ylabel("Loss")
        if save:
            plt.savefig(f"{prefix}losscurve_test.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


def adam_steps_test(self_fit, seed=1, num_learns=15, num_inits=15,
                    save=False, prefix=""):
    calc_sampdata_sampmodel = self_fit.get_multi_grad_calc()
    keys = jax.random.split(jax.random.key(seed), num_inits * 2)

    u_param_inits = [jax.random.multivariate_normal(
        keys[i], self_fit.default_u_param_arr,
        jnp.diag((0.2 * self_fit.default_u_param_arr) ** 2 + 1e-3))
        for i in range(num_inits)]
    learn_rate_vals = jnp.logspace(-4, 1, num_learns)

    offset_improvement = []
    loss_improvement = []
    for i in trange(len(u_param_inits), desc="Inits"):
        u_param_init = u_param_inits[i]
        offset_improvement.append([])
        loss_improvement.append([])
        for j in trange(len(learn_rate_vals), leave=False):
            learn_rate = learn_rate_vals[j]
            params, losses = calc_sampdata_sampmodel.run_adam(
                u_param_init, nsteps=10, learning_rate=learn_rate,
                randkey=keys[num_inits + i], progress=False)
            start_offset = jnp.linalg.norm(
                u_param_init - self_fit.default_u_param_arr)
            final_offset = jnp.linalg.norm(
                params[-1] - self_fit.default_u_param_arr)
            offimp = (start_offset - final_offset) / start_offset
            offset_improvement[-1].append((offimp))
            loss_improvement[-1].append(losses[0] - losses[-1])
    offset_improvement = jnp.array(offset_improvement)
    loss_improvement = jnp.array(loss_improvement)

    if not MPI.COMM_WORLD.rank:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Shade grey region in target range of sct = 0.2 to 0.8 and l imp > 0
        axes[0].axhspan(0, 1.0, color="grey", alpha=0.3)
        axes[1].axhline(0, color="grey", ls="--")

        # On left panel, plot step*cos(theta) vs learning rate for each init
        # On right panel, plot loss improvement vs learning rate for each init
        for i in range(num_inits):
            axes[0].plot(learn_rate_vals, offset_improvement[i],
                         label=f"Init {i+1}", marker="o", zorder=1)
            axes[1].plot(learn_rate_vals, loss_improvement[i],
                         label=f"Init {i+1}", marker="o", zorder=1)

        # Draw blue shaded region with mean+/-std of realizations vs learn rate
        for i in range(2):
            improvement = [offset_improvement, loss_improvement][i]
            mean = jnp.mean(improvement, axis=0)
            std = jnp.std(improvement, axis=0, ddof=1)
            axes[i].fill_between(
                learn_rate_vals, mean + std, mean - std,
                color="C0", alpha=0.3, zorder=2)
            top = abs(max(mean + std))
            axes[i].set_ylim(bottom=-3 * top, top=1.2 * top)
            if i == 1:
                argmax = jnp.argmax(mean - std)
                best_learn_rate = float(learn_rate_vals[argmax])

        # Notate learn_rate that is the best speed vs stability compromise
        axes[0].axvline(best_learn_rate, color="k", ls="--", lw=0.5, zorder=3)
        axes[1].axvline(best_learn_rate, color="k", ls="--", lw=0.5, zorder=3)

        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[1].set_yscale("symlog")
        axes[0].set_xlabel("Learning rate")
        axes[1].set_xlabel("Learning rate")
        axes[0].set_ylabel("Fraction towards target u_params")
        axes[1].set_ylabel("Loss improvement")
        plt.tight_layout()
        if save:
            plt.savefig(f"{prefix}adam_steps_test.png", bbox_inches="tight")
        else:
            plt.show()
        plt.clf()


parser = argparse.ArgumentParser(
    description="Test that self fitting works before running real fits")
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
    "-a", "--sky-area-degsq", type=float, default=0.01,
    help="Sky area in square degrees for mc lightcone")
parser.add_argument(
    "--num-z-grid", type=int, default=100,
    help="Number of redshift grid points for the mc lightcone")
parser.add_argument(
    "--num-m-grid", type=int, default=100,
    help="Number of lgmp grid points for the mc lightcone")
parser.add_argument(
    "-k", "--num-kernels", type=int, default=40,
    help="Number of kernels for kdescent")
parser.add_argument(
    "-f", "--num-fourier-positions", type=int, default=20,
    help="Number of Fourier evaluation positions for kdescent")
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
    "--hmf-calibration", type=str, default=None,
    help="Specify diffmahpop params ('smdpl_hmf', 'smdpl_shmf', 'hacc_shmf')")

if __name__ == "__main__":
    args = parser.parse_args()
    self_fit = SelfFit(
        i_thresh=args.iband_max,
        num_z_grid=args.num_z_grid,
        num_m_grid=args.num_m_grid,
        lgmp_min=args.lgmp_min,
        sky_area_degsq=args.sky_area_degsq,
        num_kernels=args.num_kernels,
        num_fourier_positions=args.num_fourier_positions,
        hmf_calibration=args.hmf_calibration)

    if not args.skip_corner:
        targets_corner_test(self_fit, save=args.save, prefix=args.prefix,
                            data_label="Diffsky Init")
    if not args.skip_loss:
        losscurve_test(self_fit, save=args.save, prefix=args.prefix)
    if not args.skip_adam:
        adam_steps_test(self_fit, save=args.save, prefix=args.prefix)
