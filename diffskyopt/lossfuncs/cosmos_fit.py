import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from cosmos20_colors import load_cosmos20
from diffopt import kdescent, multigrad
from diffsky.param_utils import diffsky_param_wrapper as dpw
from mpi4py import MPI

from ..diffsky_model import (
    FILTER_NAMES,
    I_BAND_IND,
    compute_targets_and_weights,
    cosmos_mags_to_colors,
    generate_weighted_sobol_lc_data,
)

u_param_collection = dpw.get_u_param_collection_from_param_collection(
    *dpw.DEFAULT_PARAM_COLLECTION)
COSMOS_SKY_AREA = 1.21
COSMOS_DIR = "/lcrc/project/halotools/COSMOS/"

SIZE, RANK = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank


def load_target_data_and_cat(cat, z_min, z_max, i_band_thresh,
                             thresh_softening=0.1, min_weight=1e-3,
                             filter_names=FILTER_NAMES):
    """
    Perform data-cleaning on the COSMOS catalog, removing NaNs, values
    outside of the redshift/i_mag range, and color outliers. Then, return
    the columns used to construct our kdescent targets, along with their
    corresponding weights.

    """
    # Mask out NaNs
    nan_msk_keys = ("photoz", *filter_names)
    msk_no_nan = np.ones(len(cat), dtype=bool)
    for key in nan_msk_keys:
        msk_no_nan &= ~np.isnan(np.array(cat[key]))
    cosmos = cat[msk_no_nan]

    photoz_arr = np.array(cosmos["photoz"])
    msk_redshift = (photoz_arr > z_min) & (photoz_arr < z_max)
    cosmos = cosmos[msk_redshift]

    mags = jnp.stack(jnp.array(
        [cosmos[name] for name in filter_names]), axis=1)
    colors = cosmos_mags_to_colors(mags, filter_names=filter_names)
    i_mag = np.array(mags[:, I_BAND_IND])
    redshift = np.array(cosmos["photoz"])

    weights = jax.nn.sigmoid(
        # sigmoid weights instead of sharp i-band threshold
        (i_band_thresh - i_mag) / thresh_softening)

    msk_good_colors = np.ones(len(cosmos)).astype(bool)
    for color in colors.T:
        color_lo, color_hi = np.percentile(color, (0.5, 99.5))
        msk_good_colors &= (color > color_lo) & (color < color_hi)

    good_weights = weights > min_weight
    full_mask = np.array(good_weights & msk_good_colors)

    cosmos = cosmos[full_mask]
    i_mag = i_mag[full_mask]
    redshift = redshift[full_mask]
    colors = colors[full_mask]

    weights = jnp.array(weights[msk_good_colors & good_weights])
    return cosmos, i_mag, redshift, colors, weights


class CosmosFit:
    default_u_param_arr = dpw.unroll_u_param_collection_into_flat_array(
        *u_param_collection)

    def __init__(self, num_halos=5000, zmin=0.4, zmax=2.0,
                 lgmp_min=10.5, lgmp_max=15.0,
                 sky_area_degsq=COSMOS_SKY_AREA,
                 num_kernels=40, num_fourier_positions=20, i_thresh=25.0,
                 hmf_calibration=None, log_loss=False, num_mag_z_kernels=20,
                 max_n_halos_per_bin=1000, n_halo_weight_bins=10,
                 kde_idw_power=0.0, seed=0, drn=COSMOS_DIR):
        self.num_halos = num_halos
        self.zmin = zmin
        self.zmax = zmax
        self.lgmp_min = lgmp_min
        self.lgmp_max = lgmp_max
        self.sky_area_degsq = sky_area_degsq
        self.num_kernels = num_kernels
        self.num_fourier_positions = num_fourier_positions
        self.i_thresh = i_thresh
        self.hmf_calibration = hmf_calibration
        self.log_loss = log_loss
        self.max_n_halos_per_bin = max_n_halos_per_bin
        self.n_halo_weight_bins = n_halo_weight_bins
        self.num_mag_z_kernels = num_mag_z_kernels
        self.kde_idw_power = kde_idw_power

        # Load COSMOS data and compile targets and weights arrays
        cat = load_cosmos20(drn=drn)
        _, i_mag, redshift, colors, weights = load_target_data_and_cat(
            cat, self.zmin, self.zmax, self.i_thresh)
        self.data_weights = weights
        self.data_targets = np.stack([i_mag, redshift, *colors.T], axis=1)

        ran_keys = jax.random.split(jax.random.key(seed), 3)
        self.lc_data = generate_weighted_sobol_lc_data(
            self.num_halos, self.zmin, self.zmax, self.lgmp_min, self.lgmp_max,
            sky_area_degsq=self.sky_area_degsq, ran_key=ran_keys[0],
            hmf_calibration=self.hmf_calibration, comm=MPI.COMM_WORLD)
        self.halo_upweights = self.lc_data.nhalos

        covariant_kernels = False
        bandwidth_factor = 2.0
        ktrain_fn = (f"ktrain_nk{self.num_kernels}_"
                     f"nf{self.num_fourier_positions}.npz")
        mag_z_ktrain_fn = (f"mag_z_ktrain_nk{self.num_mag_z_kernels}_"
                           f"idwp{self.kde_idw_power:.2g}.npz")
        if os.path.exists(ktrain_fn):
            ktrain = kdescent.KPretrainer.load(ktrain_fn)
        else:
            ktrain = kdescent.KPretrainer.from_training_data(
                self.data_targets, self.data_weights,
                num_eval_kernels=self.num_kernels,
                num_eval_fourier_positions=self.num_fourier_positions,
                bandwidth_factor=bandwidth_factor,
                covariant_kernels=covariant_kernels,
                comm=MPI.COMM_WORLD,
            )
            if not RANK:
                ktrain.save(ktrain_fn)
        if os.path.exists(mag_z_ktrain_fn):
            mag_z_ktrain = kdescent.KPretrainer.load(mag_z_ktrain_fn)
        else:
            mag_z_ktrain = kdescent.KPretrainer.from_training_data(
                self.data_targets[:, :2], self.data_weights,
                num_eval_kernels=self.num_mag_z_kernels,
                num_eval_fourier_positions=0,
                bandwidth_factor=bandwidth_factor,
                covariant_kernels=covariant_kernels,
                inverse_density_weight_power=self.kde_idw_power,
                comm=MPI.COMM_WORLD,
            )
            if not RANK:
                mag_z_ktrain.save(mag_z_ktrain_fn)
        self.kcalc = kdescent.KCalc(ktrain)
        self.mag_z_kcalc = kdescent.KCalc(mag_z_ktrain)

        # Account for volume difference between COSMOS and diffsky lightcone
        self.volume_factor_weight = COSMOS_SKY_AREA / self.sky_area_degsq
        self.halo_upweights *= self.volume_factor_weight

    def get_multi_grad_calc(self):
        return self.MultiGradModel(aux_data=dict(
            fit_instance=self))

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
        keys = jax.random.split(randkey, 4)
        _res = self.targets_and_weights_from_params(
            params, keys[0])
        model_targets, model_weights = _res

        # Each rank must have the SAME randkey in kdescent
        model_k, data_k, err_k = self.kcalc.compare_kde_counts(
            keys[1], model_targets, model_weights, return_err=True)
        model_f, data_f, err_f = self.kcalc.compare_fourier_counts(
            keys[2], model_targets, model_weights, return_err=True)
        model_mz, data_mz, err_mz = self.mag_z_kcalc.compare_kde_counts(
            keys[3], model_targets[:, :2], model_weights, return_err=True)

        # Summed over ranks
        sumstats = jnp.concatenate([model_k, model_f, model_mz])

        # Not summed over ranks
        sumstats_aux = jnp.concatenate([
            data_k, err_k, data_f, err_f, data_mz, err_mz])

        return sumstats, sumstats_aux

    @partial(jax.jit, static_argnums=[0])
    def loss_from_sumstats(self, sumstats, sumstats_aux):
        nk = self.num_kernels
        nf = self.num_fourier_positions
        nmz = self.num_mag_z_kernels
        model_k, model_f, model_mz = jnp.split(
            sumstats, np.cumsum([nk, nf]))
        data_k, err_k, data_f, err_f, data_mz, err_mz = jnp.split(
            sumstats_aux, np.cumsum([nk, nk, nf, nf, nmz]))

        if self.log_loss:
            eps = 1e-10
            model_k = jnp.log10(model_k.real + eps)
            data_k = jnp.log10(data_k.real + eps)
            model_mz = jnp.log10(model_mz.real + eps)
            data_mz = jnp.log10(data_mz.real + eps)

            # Use constant log errors so that fractional difference is
            # weighted evenly across high- and low-density regions of the PDF
            err_k = err_mz = 1.0

            # Don't take log of Fourier counts

        normalized_residuals = jnp.concatenate([
            (model_k.real - data_k.real) / err_k.real,
            (model_f.real - data_f.real) / err_f.real,
            (model_f.imag - data_f.imag) / err_f.imag,
            (model_mz.real - data_mz.real) / err_mz.real,
        ])
        reduced_chisq = jnp.mean(normalized_residuals**2)
        return reduced_chisq

    @dataclass
    class MultiGradModel(multigrad.OnePointModel):
        aux_data: dict
        sumstats_func_has_aux: bool = True

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
