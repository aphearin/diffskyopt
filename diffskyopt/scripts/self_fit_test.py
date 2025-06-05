import argparse
import matplotlib.pyplot as plt
import tqdm

import jax
import jax.numpy as jnp

from diffskyopt.lossfuncs.self_fit import (
    default_u_param_arr,
    calc_fulldata_fullmodel,
    calc_fulldata_sampmodel,
    calc_sampdata_fullmodel,
    calc_sampdata_sampmodel,
    targets_and_weights_from_params,
)

ran_keys = jax.random.split(jax.random.key(1), 4)
keys = jax.random.split(ran_keys[3], 100)


# Helper function for losscurve_test()
def plot_losscurve(losscalc, label=None, fixed_key=False, **kwargs):
    p_vals = []
    loss_vals = []
    for i in tqdm.trange(len(keys), desc=label):
        key = keys[0] if fixed_key else keys[i]
        p = default_u_param_arr * ((i - 50) / 50.0 + 1.0)
        p_vals.append(p)
        loss_vals.append(losscalc.calc_loss_from_params(p, key))
    p_vals = jnp.array(p_vals)
    loss_vals = jnp.array(loss_vals)
    plt.plot((p_vals[:, 0] / default_u_param_arr[0]) - 1, loss_vals,
             label=label, **kwargs)


def color_magnitude_test(save=False):
    # Simply plot a color-magnitude diagram
    targets, weights = targets_and_weights_from_params(
        default_u_param_arr, keys[0], modelsamp=True)
    plt.scatter(targets[:, 0], targets[:, 1], c=weights, s=1)
    plt.xlabel("$\\rm u - g$")
    plt.ylabel("$\\rm m_i$")
    if save:
        plt.savefig("color_magnitude_test.png", bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


def losscurve_test(save=False):
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

    plt.semilogy()
    plt.axvline(0, color="k", ls="--")
    plt.legend(frameon=False)
    plt.xlabel("Fractional change in u_params")
    plt.ylabel("Loss")
    if save:
        plt.savefig("losscurve_test.png", bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


def adam_steps_test(seed=1, num_learns=20, num_inits=25, save=False):
    keys = jax.random.split(jax.random.key(seed), num_inits * 2)

    u_param_inits = [jax.random.multivariate_normal(
        keys[i], default_u_param_arr,
        jnp.diag((0.2 * default_u_param_arr) ** 2 + 1e-3))
        for i in range(num_inits)]
    learn_rate_vals = jnp.logspace(-4, 1, num_learns)

    offset_improvement = []
    loss_improvement = []
    for i in tqdm.trange(len(u_param_inits), desc="Inits"):
        u_param_init = u_param_inits[i]
        offset_improvement.append([])
        loss_improvement.append([])
        for learn_rate in tqdm.tqdm(learn_rate_vals, leave=False):
            params, losses = calc_sampdata_sampmodel.run_adam(
                u_param_init, nsteps=10, learning_rate=learn_rate,
                randkey=keys[num_inits + i], progress=False)
            start_offset = jnp.linalg.norm(u_param_init - default_u_param_arr)
            final_offset = jnp.linalg.norm(params[-1] - default_u_param_arr)
            offimp = (start_offset - final_offset) / start_offset
            offset_improvement[-1].append((offimp))
            loss_improvement[-1].append(losses[0] - losses[-1])
    offset_improvement = jnp.array(offset_improvement)
    loss_improvement = jnp.array(loss_improvement)

    # print("Offset improvement:", offset_improvement)
    # print("Loss improvement:", loss_improvement)
    # import pdb
    # pdb.set_trace()

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

    # Notate learn_rate = 0.02, which is a good speed + stability compromise
    axes[0].axvline(2e-2, color="k", ls="--", lw=0.5, zorder=3)
    axes[1].axvline(2e-2, color="k", ls="--", lw=0.5, zorder=3)

    # if axes[0].get_ylim()[0] < -3:
    #     axes[0].set_ylim(bottom=-3)
    axes[0].set_ylim(-0.2, 0.05)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("symlog")
    axes[0].set_xlabel("Learning rate")
    axes[1].set_xlabel("Learning rate")
    axes[0].set_ylabel("Fraction towards target u_params")
    axes[1].set_ylabel("Loss improvement")
    plt.tight_layout()
    if save:
        plt.savefig("adam_steps_test.png", bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


parser = argparse.ArgumentParser(
    description="Test that self fitting works before running real fits")
parser.add_argument(
    "-s", "--save", action="store_true",
    help="Save plots instead of opening interactive window")

if __name__ == "__main__":
    args = parser.parse_args()

    color_magnitude_test(save=args.save)
    losscurve_test(save=args.save)
    adam_steps_test(save=args.save)
