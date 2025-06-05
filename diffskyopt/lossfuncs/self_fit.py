from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from diffopt import kdescent
from diffopt import multigrad

from diffsky.param_utils import diffsky_param_wrapper as dpw

from ..diffsky_model import (
    generate_lc_data,
    downsample_upweight_lc_data,
    compute_targets_and_weights,
)


u_param_collection = dpw.get_u_param_collection_from_param_collection(
    *dpw.DEFAULT_PARAM_COLLECTION)
default_u_param_arr = dpw.unroll_u_param_collection_into_flat_array(
    *u_param_collection)

LGMP_MIN = 11.0
ran_keys = jax.random.split(jax.random.key(0), 4)

lc_data = generate_lc_data(
    0.4, 0.6, lgmp_min=LGMP_MIN, sky_area_degsq=0.01, ran_key=ran_keys[0])
downsampled_lc_data, halo_upweights = downsample_upweight_lc_data(
    lc_data, lgmp_min=LGMP_MIN, ran_key=ran_keys[1],
    max_n_halos_per_bin=1000, n_halo_weight_bins=10)

default_targets, default_weights = compute_targets_and_weights(
    default_u_param_arr, downsampled_lc_data, ran_key=ran_keys[2],
    halo_upweights=halo_upweights)
default_targets_no_w, default_weights_no_w = compute_targets_and_weights(
    default_u_param_arr, lc_data, ran_key=ran_keys[2])

# Create KCalc instances for full and HMF-sampled data
# ====================================================
num_kernels = 40
num_fourier_kernels = 20
covariant_kernels = False
bandwidth_factor = 2.0
kcalc = kdescent.KCalc(
    default_targets_no_w, default_weights_no_w, num_kernels=num_kernels,
    num_fourier_kernels=num_fourier_kernels,
    covariant_kernels=covariant_kernels, bandwidth_factor=bandwidth_factor)
kcalc_hmfsamp = kdescent.KCalc(
    default_targets, default_weights, num_kernels=num_kernels,
    num_fourier_kernels=num_fourier_kernels,
    covariant_kernels=covariant_kernels, bandwidth_factor=bandwidth_factor)


@partial(jax.jit, static_argnums=[2])
def targets_and_weights_from_params(params, randkey, modelsamp=False):
    data = downsampled_lc_data if modelsamp else lc_data
    hmf_sample_weight = halo_upweights if modelsamp else None

    targets, weights = compute_targets_and_weights(
        params, data, ran_key=randkey,
        halo_upweights=hmf_sample_weight)
    return targets, weights

# Define multigrad-compatible loss function
# =========================================


@partial(jax.jit, static_argnums=[2, 3])
def sumstats_from_params(params, randkey, datasamp=False, modelsamp=False):
    kc = kcalc_hmfsamp if datasamp else kcalc
    keys = jax.random.split(randkey, 3)
    targets, weights = targets_and_weights_from_params(
        params, keys[0], modelsamp=modelsamp)
    counts_model, counts_true = kc.compare_kde_counts(
        keys[1], targets, weights)
    fcounts_model, fcounts_true = kc.compare_fourier_counts(
        keys[2], targets, weights)

    # Summed over ranks
    sumstats = jnp.concatenate([
        counts_model, fcounts_model])

    # Not summed over ranks
    sumstats_aux = jnp.concatenate([
        counts_true, fcounts_true, kc.training_weights.sum(keepdims=True)])

    return sumstats, sumstats_aux


@jax.jit
def loss_from_sumstats(sumstats, sumstats_aux):
    # Unpack the sumstats
    counts_model, fcounts_model = jnp.split(sumstats, [num_kernels])
    counts_true, fcounts_true, training_weights_sum = jnp.split(
        sumstats_aux, [num_kernels, num_kernels + num_fourier_kernels])

    # Compute ~MSE loss
    loss = jnp.mean(jnp.abs(jnp.concatenate([
        counts_model - counts_true,
        fcounts_model - fcounts_true
    ]) / training_weights_sum)**2)

    return loss


@dataclass
class MultiGradModel(multigrad.OnePointModel):
    sumstats_func_has_aux: bool = True  # override param default set by parent

    def calc_partial_sumstats_from_params(self, params, randkey):
        # NOTE: sumstats will automatically be summed over all MPI ranks
        # before getting passed to calc_loss_from_sumstats. However,
        # sumstats_aux will be passed directly without MPI communication
        sumstats, sumstats_aux = sumstats_from_params(
            params, randkey, datasamp=self.aux_data["datasamp"],
            modelsamp=self.aux_data["modelsamp"])
        return sumstats, sumstats_aux

    def calc_loss_from_sumstats(self, sumstats, sumstats_aux, randkey=None):
        # NOTE: randkey kwarg must be accepted by BOTH functions or NEITHER
        # However, we have no need for it in the loss function
        del randkey

        loss = loss_from_sumstats(sumstats, sumstats_aux)
        return loss


# Create multigrad model instances for each case
# ==============================================
calc_fulldata_fullmodel = MultiGradModel(
    aux_data=dict(datasamp=False, modelsamp=False))
calc_fulldata_sampmodel = MultiGradModel(
    aux_data=dict(datasamp=False, modelsamp=True))
calc_sampdata_fullmodel = MultiGradModel(
    aux_data=dict(datasamp=True, modelsamp=False))
calc_sampdata_sampmodel = MultiGradModel(
    aux_data=dict(datasamp=True, modelsamp=True))
