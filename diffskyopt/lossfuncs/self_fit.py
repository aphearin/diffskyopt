from dataclasses import dataclass
from functools import partial
from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

from diffopt import kdescent
from diffopt import multigrad

from diffsky.param_utils import diffsky_param_wrapper as dpw

from ..diffsky_model import (
    generate_weighted_grid_lc_data,
    compute_targets_and_weights,
    lc_data_slice,
)
from .cosmos_fit import COSMOS_SKY_AREA

u_param_collection = dpw.get_u_param_collection_from_param_collection(
    *dpw.DEFAULT_PARAM_COLLECTION)

SIZE, RANK = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank


class SelfFit:
    default_u_param_arr = dpw.unroll_u_param_collection_into_flat_array(
        *u_param_collection)

    def __init__(self, zmin=0.4, zmax=2.0, num_z_grid=100,
                 lgmp_min=10.5, lgmp_max=15.0, num_m_grid=100,
                 sky_area_degsq=COSMOS_SKY_AREA,
                 num_kernels=40, num_fourier_positions=20, i_thresh=25.0,
                 hmf_calibration=None, seed=0):
        self.zmin = zmin
        self.zmax = zmax
        self.num_z_grid = num_z_grid
        self.lgmp_min = lgmp_min
        self.lgmp_max = lgmp_max
        self.num_m_grid = num_m_grid
        self.sky_area_degsq = sky_area_degsq
        self.i_thresh = i_thresh
        self.hmf_calibration = hmf_calibration
        ran_keys = jax.random.split(jax.random.key(seed), 4)

        self.lc_data, self.halo_upweights = generate_weighted_grid_lc_data(
            self.zmin, self.zmax, self.num_z_grid,
            self.lgmp_min, self.lgmp_max, self.num_m_grid,
            sky_area_degsq=self.sky_area_degsq, ran_key=ran_keys[0],
            hmf_calibration=self.hmf_calibration)

        _res = compute_targets_and_weights(
            self.default_u_param_arr, self.lc_data, ran_key=ran_keys[2],
            i_band_thresh=self.i_thresh)
        self.data_targets, self.data_weights = _res

        # Create KCalc instances for full and HMF-sampled data
        self.num_kernels = num_kernels
        self.num_fourier_positions = num_fourier_positions
        covariant_kernels = False
        bandwidth_factor = 2.0
        self.kcalc = kdescent.KCalc(
            self.data_targets, self.data_weights,
            num_kernels=self.num_kernels,
            num_fourier_positions=self.num_fourier_positions,
            covariant_kernels=covariant_kernels,
            bandwidth_factor=bandwidth_factor)

        # Distribute lc_data and downsampled_lc_data across MPI ranks
        cut = jnp.array_split(
            jnp.arange(self.lc_data.z_obs.size), SIZE)[RANK]
        self.lc_data = lc_data_slice(self.lc_data, cut)
        self.halo_upweights = self.halo_upweights[cut]

    def get_multi_grad_calc(self):
        return self.MultiGradModel(
            aux_data=dict(fit_instance=self))

    @partial(jax.jit, static_argnums=[0])
    def targets_and_weights_from_params(self, params, randkey):
        # Each rank must have a UNIQUE key to compute photometry
        targets, weights = compute_targets_and_weights(
            params, self.lc_data,
            ran_key=jax.random.split(randkey, SIZE)[RANK],
            weights=self.halo_upweights, i_band_thresh=self.i_thresh)
        return targets, weights

    @partial(jax.jit, static_argnums=[0])
    def sumstats_from_params(self, params, randkey):
        keys = jax.random.split(randkey, 3)
        _res = self.targets_and_weights_from_params(
            params, keys[0])
        model_targets, model_weights = _res

        # Each rank must have the SAME randkey in kdescent
        model_k, data_k, err_k = self.kcalc.compare_kde_counts(
            keys[1], model_targets, model_weights, return_err=True)
        model_f, data_f, err_f = self.kcalc.compare_fourier_counts(
            keys[2], model_targets, model_weights, return_err=True)

        # Summed over ranks
        sumstats = jnp.concatenate([model_k, model_f])

        # Not summed over ranks
        sumstats_aux = jnp.concatenate([data_k, err_k, data_f, err_f])

        return sumstats, sumstats_aux

    @partial(jax.jit, static_argnums=[0])
    def loss_from_sumstats(self, sumstats, sumstats_aux):
        nk = self.num_kernels
        nf = self.num_fourier_positions
        model_k, model_f = jnp.split(sumstats, [nk])
        data_k, err_k, data_f, err_f = jnp.split(
            sumstats_aux, np.cumsum([nk, nk, nf]))

        # Volume difference already accounted for in halo_upweights
        normalized_residuals = jnp.concatenate([
            (model_k.real - data_k.real) / err_k.real,
            (model_f.real - data_f.real) / err_f.real,
            (model_f.imag - data_f.imag) / err_f.imag
        ])
        reduced_chisq = jnp.mean(normalized_residuals**2)
        return reduced_chisq

    @dataclass
    class MultiGradModel(multigrad.OnePointModel):
        aux_data: dict
        sumstats_func_has_aux: bool = True  # override param default

        def calc_partial_sumstats_from_params(self, params, randkey):
            fit_instance = self.aux_data["fit_instance"]
            sumstats, sumstats_aux = fit_instance.sumstats_from_params(
                params, randkey)
            return sumstats, sumstats_aux

        def calc_loss_from_sumstats(self, sumstats, sumstats_aux,
                                    randkey=None):
            del randkey
            fit_instance = self.aux_data["fit_instance"]
            loss = fit_instance.loss_from_sumstats(sumstats, sumstats_aux)
            return loss
