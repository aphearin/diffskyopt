import os
import numpy as np
from dataclasses import dataclass
from functools import partial
from mpi4py import MPI

import jax
import jax.numpy as jnp

from diffopt import kdescent
from diffopt import multigrad

from cosmos20_colors import load_cosmos20
from diffsky.param_utils import diffsky_param_wrapper as dpw

from ..diffsky_model import (
    generate_lc_data,
    compute_targets_and_weights,
    downsample_upweight_lc_data,
    lc_data_slice,
    cosmos_mags_to_colors,
    FILTER_NAMES,
    I_BAND_IND,
)

u_param_collection = dpw.get_u_param_collection_from_param_collection(
    *dpw.DEFAULT_PARAM_COLLECTION)
COSMOS_SKY_AREA = 1.21

SIZE, RANK = MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank


class CosmosFit:
    default_u_param_arr = dpw.unroll_u_param_collection_into_flat_array(
        *u_param_collection)

    def __init__(self, zmin=0.4, zmax=2.0, lgmp_min=10.5, sky_area_degsq=0.1,
                 num_kernels=40, num_fourier_positions=20, i_thresh=25.0,
                 hmf_calibration=None, log_loss=False, num_mag_z_kernels=20,
                 max_n_halos_per_bin=1000, n_halo_weight_bins=10,
                 kde_idw_power=0.0, seed=0):
        self.zmin = zmin
        self.zmax = zmax
        self.lgmp_min = lgmp_min
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

        # --- Load and mask COSMOS data ---
        os.environ["COSMOS20_DRN"] = "/lcrc/project/halotools/COSMOS/"
        cat = load_cosmos20()

        # Mask out NaNs
        nan_msk_keys = ("photoz", *FILTER_NAMES)
        msk_no_nan = np.ones(len(cat), dtype=bool)
        for key in nan_msk_keys:
            msk_no_nan &= ~np.isnan(np.array(cat[key]))
        cosmos = cat[msk_no_nan]

        photoz_arr = np.array(cosmos["photoz"])
        msk_redshift = (photoz_arr > self.zmin) & (photoz_arr < self.zmax)
        cosmos = cosmos[msk_redshift]

        # Prepare data targets
        _res = self._prepare_data_targets_and_weights(
            cosmos, i_band_thresh=self.i_thresh)
        self.data_targets, self.data_weights = _res

        ran_keys = jax.random.split(jax.random.key(seed), 3)
        if RANK == 0:
            # Generate full lightcone data ONLY on rank 0
            full_lc_data = generate_lc_data(
                self.zmin, self.zmax, lgmp_min=self.lgmp_min,
                sky_area_degsq=self.sky_area_degsq, ran_key=ran_keys[0],
                hmf_calibration=self.hmf_calibration)
            _res = downsample_upweight_lc_data(
                full_lc_data, lgmp_min=self.lgmp_min, ran_key=ran_keys[1],
                max_n_halos_per_bin=self.max_n_halos_per_bin,
                n_halo_weight_bins=self.n_halo_weight_bins)
            full_downsampled_lc_data, full_halo_upweights = _res

            # Split lightcone data ONLY on rank 0
            indices = np.array_split(np.arange(full_lc_data.z_obs.size), SIZE)
            lc_data_slices = [lc_data_slice(
                full_lc_data, idx) for idx in indices]
            indices = np.array_split(np.arange(
                full_downsampled_lc_data.z_obs.size), SIZE)
            downsampled_lc_data_slices = [lc_data_slice(
                full_downsampled_lc_data, idx) for idx in indices]
            halo_upweights_slices = [
                full_halo_upweights[idx] for idx in indices]
        else:
            lc_data_slices = None
            downsampled_lc_data_slices = None
            halo_upweights_slices = None

        # Distribute lc_data and downsampled_lc_data across MPI ranks
        self.lc_data = MPI.COMM_WORLD.scatter(
            lc_data_slices, root=0)
        self.downsampled_lc_data = MPI.COMM_WORLD.scatter(
            downsampled_lc_data_slices, root=0)
        self.halo_upweights = MPI.COMM_WORLD.scatter(
            halo_upweights_slices, root=0)

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
                num_pretrain_kernels=100*self.num_kernels,
                num_pretrain_fourier_positions=100*self.num_fourier_positions,
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
                num_pretrain_kernels=100*self.num_mag_z_kernels,
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

    def _prepare_data_targets_and_weights(self, cosmos, i_band_thresh=23.0,
                                          thresh_softening=0.1,
                                          min_weight=1e-3):
        # Target data is (N, 9): [i, g-r, r-i, i-z, z-y, Y-J, J-H, H-Z, photoz]
        # Y-band taken from both HSC and UVISTA => no cross-filter colors
        mags = jnp.stack(jnp.array(
            [cosmos[name] for name in FILTER_NAMES]), axis=1)
        colors = cosmos_mags_to_colors(mags)
        i_mag = mags[:, I_BAND_IND]
        photoz = cosmos["photoz"]
        targets = np.stack([i_mag, photoz, *colors.T], axis=1)
        weights = jax.nn.sigmoid(
            # sigmoid weights instead of sharp i-band threshold
            (i_band_thresh - i_mag) / thresh_softening)

        msk_good_colors = np.ones(len(cosmos)).astype(bool)
        for color in colors.T:
            color_lo, color_hi = np.percentile(color, (0.5, 99.5))
            msk_good_colors &= (color > color_lo) & (color < color_hi)

        good_weights = weights > min_weight
        targets = jnp.array(targets[msk_good_colors & good_weights])
        weights = jnp.array(weights[msk_good_colors & good_weights])
        return targets, weights

    def get_multi_grad_calc(self, modelsamp=True):
        return self.MultiGradModel(aux_data=dict(
            fitobj=self, modelsamp=modelsamp))

    @partial(jax.jit, static_argnums=[0, 3])
    def targets_and_weights_from_params(self, params, randkey,
                                        modelsamp=False):
        data = self.downsampled_lc_data if modelsamp else self.lc_data
        weights = self.volume_factor_weight
        if modelsamp:
            weights = self.halo_upweights

        # Each rank must have a UNIQUE key to compute photometry
        targets, weights = compute_targets_and_weights(
            params, data, ran_key=jax.random.split(randkey, SIZE)[RANK],
            weights=weights, i_band_thresh=self.i_thresh)

        return targets, weights

    @partial(jax.jit, static_argnums=[0, 3])
    def sumstats_from_params(self, params, randkey, modelsamp=False):
        keys = jax.random.split(randkey, 4)
        _res = self.targets_and_weights_from_params(
            params, keys[0], modelsamp=modelsamp)
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

            # err_k = err_k / (data_k.real * np.log(10) + eps)
            # err_mz = err_mz / (data_mz.real * np.log(10) + eps)

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
            fitobj = self.aux_data["fitobj"]
            sumstats, sumstats_aux = fitobj.sumstats_from_params(
                params, randkey, modelsamp=self.aux_data["modelsamp"])
            return sumstats, sumstats_aux

        def calc_loss_from_sumstats(self, sumstats, sumstats_aux,
                                    randkey=None):
            del randkey
            fitobj = self.aux_data["fitobj"]
            loss = fitobj.loss_from_sumstats(sumstats, sumstats_aux)
            return loss
