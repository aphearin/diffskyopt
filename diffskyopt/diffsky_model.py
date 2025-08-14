import os
import pathlib
import warnings

from mpi4py import MPI
import numpy as np
import jax
import jax.numpy as jnp

from dsps import load_ssp_templates
from dsps.data_loaders.defaults import TransmissionCurve
from dsps.data_loaders import load_transmission_curve
from diffsky.experimental import lc_phot_kern
from diffsky.experimental.lc_phot_kern import mclh
from diffsky.experimental.mc_lightcone_halos import get_nhalo_weighted_lc_grid
from diffsky.mass_functions.fitting_utils.calibrations \
    import hacc_core_shmf_params
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffmah.diffmah_kernels import _log_mah_kern
from diffsky.mass_functions import mc_hosts
from diffsky.mass_functions.hmf_calibrations import \
    smdpl_hmf_subs, smdpl_hmf
from diffsky.ssp_err_model import ssp_err_model
from diffmah.diffmahpop_kernels.bimod_censat_params import \
    DEFAULT_DIFFMAHPOP_PARAMS
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.cosmology import flat_wcdm

from scipy.stats import qmc

if os.environ.get("DIFFSKYOPT_EXPAND_ZRANGE_SSP_ERR"):
    ssp_err_model.Z_INTERP_ABSCISSA = jnp.array([0.6, 1.6])
    # default = jnp.array([0.5, 0.75])

DATA_DIR = pathlib.Path("/lcrc/project/halotools/COSMOS/")
FILTERS_DIR = pathlib.Path("/home/apearl/data/cosmos_filters/")

ssp_files = [
    "ssp_data_fsps_v3.2_lgmet_age.h5",
    "ssp_data_continuum_fsps_v3.2_lgmet_age.h5",
    "ssp_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5",
    "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ-0.4.h5",
    "ssp_prsc_miles_chabrier_wNE_logGasU-2.5_logGasZ0.0.h5",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ-1.0.h5",
]
alt_ssp = os.environ.get("DIFFSKYOPT_ALT_SSP")
if alt_ssp:
    SSP_FILE = ssp_files[int(alt_ssp)]
else:
    SSP_FILE = ssp_files[0]
FILTER_FILES = [
    "g_HSC.txt",
    "r_HSC.txt",
    "i_HSC.txt",
    "z_HSC.txt",
    "y_HSC.txt",
    "Y_uv.res",
    "J_uv.res",
    "H_uv.res",
    "K_uv.res",
]

FILTER_NAMES = [
    "HSC_g_MAG",
    "HSC_r_MAG",
    "HSC_i_MAG",
    "HSC_z_MAG",
    "HSC_y_MAG",
    "UVISTA_Y_MAG",
    "UVISTA_J_MAG",
    "UVISTA_H_MAG",
    "UVISTA_Ks_MAG",
]

if os.environ.get("DIFFSKYOPT_REMOVE_G_FILTER"):
    FILTER_FILES.remove("g_HSC.txt")
    FILTER_NAMES.remove("HSC_g_MAG")

I_BAND_IND = FILTER_NAMES.index("HSC_i_MAG")

assert len(FILTER_FILES) == len(FILTER_NAMES)


def cosmos_mags_to_colors(mags):
    assert mags.shape[1] == len(FILTER_NAMES), \
        f"mags must be shape (N, {len(FILTER_NAMES)})"
    filter_set_names = list(dict.fromkeys(
        [x.split("_")[0] for x in FILTER_NAMES]))
    filter_set_inds = [
        [i for i, fname in enumerate(FILTER_NAMES) if fname.startswith(sname)]
        for sname in filter_set_names
    ]
    combined_mag_sets = [
        jnp.stack([mags[:, i] for i in inds], axis=1)
        for inds in filter_set_inds
    ]
    return jnp.concatenate([
        -jnp.diff(mag_set, axis=1) for mag_set in combined_mag_sets], axis=1)


def generate_lc_data_kern(
    ran_key,
    lgmp_min,
    z_min,
    z_max,
    sky_area_degsq,
    ssp_data,
    cosmo_params,
    tcurves,
    z_phot_table,
    logmp_cutoff=0.0,
    hmf_calibration=None,
):
    mclh_args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)
    mclh_kwargs = dict()
    if hmf_calibration == "smdpl_hmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf.HMF_PARAMS
    elif hmf_calibration == "smdpl_shmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf_subs.HMF_PARAMS
    elif hmf_calibration == "hacc_shmf":
        mclh_kwargs["hmf_params"] = hacc_core_shmf_params.HMF_PARAMS
    else:
        assert hmf_calibration is None, f"Unrecognized {hmf_calibration=}"

    lc_halopop = mclh.mc_lightcone_host_halo_diffmah(
        *mclh_args, logmp_cutoff=logmp_cutoff, **mclh_kwargs)  # type: ignore

    t0 = lc_phot_kern.flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(lc_phot_kern.T_TABLE_MIN,
                           t0, lc_phot_kern.N_SFH_TABLE)

    precomputed_ssp_mag_table = \
        mclh.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table
        )
    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    lc_data = lc_phot_kern.LCData(
        lc_halopop["z_obs"],
        lc_halopop["t_obs"],
        lc_halopop["mah_params"],
        lc_halopop["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
    )
    return lc_data


def generate_lc_data(z_min, z_max, lgmp_min, sky_area_degsq,
                     ran_key=None, n_z_phot_table=15, logmp_cutoff=0.0,
                     hmf_calibration=None):
    if ran_key is None:
        ran_key = jax.random.key(0)

    ssp_data = load_ssp_templates(DATA_DIR / SSP_FILE)

    ffiles = [FILTERS_DIR / fn for fn in FILTER_FILES]
    tcurves = [load_transmission_curve(fn) if str(fn).endswith(".h5") else
               TransmissionCurve(*np.loadtxt(fn).T) for fn in ffiles]

    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    lc_data = generate_lc_data_kern(
        ran_key,
        lgmp_min,
        z_min,
        z_max,
        sky_area_degsq,
        ssp_data,
        DEFAULT_COSMOLOGY,
        tcurves,
        z_phot_table,
        logmp_cutoff=logmp_cutoff,
        hmf_calibration=hmf_calibration,
    )
    return lc_data


def lc_data_slice(lc_data, selection):
    downsampled_mah_params = lc_data.mah_params._make(
        [x[selection] for x in lc_data.mah_params])
    return lc_data._make([
        lc_data.z_obs[selection],
        lc_data.t_obs[selection],
        downsampled_mah_params,
        lc_data.logmp0[selection],
        *lc_data[4:]
    ])


def downsample_upweight_lc_data(lc_data, lgmp_min, n_halo_weight_bins=10,
                                max_n_halos_per_bin=100, ran_key=None):
    if ran_key is None:
        ran_key = jax.random.key(1)
    logmp0_bin_edges = jnp.linspace(
        lgmp_min-0.01, lc_data.logmp0.max()+0.01, n_halo_weight_bins)
    hist, _ = jnp.histogram(lc_data.logmp0, bins=logmp0_bin_edges)
    bin_ind = jnp.digitize(lc_data.logmp0, logmp0_bin_edges) - 1

    downsampled_halo_indices = []
    downsampled_halo_weights = []
    for bin_i in range(len(logmp0_bin_edges) - 1):
        if hist[bin_i] < max_n_halos_per_bin:
            downsampled_halo_indices.append(jnp.where(bin_ind == bin_i)[0])
            downsampled_halo_weights.append(jnp.ones(int(hist[bin_i])))
        else:
            # Randomly sample max_n_halos_per_bin halos from this bin
            bin_halo_indices = jnp.where(bin_ind == bin_i)[0]
            bin_halo_weights = jnp.full(
                (max_n_halos_per_bin,), hist[bin_i] / max_n_halos_per_bin)
            downsampled_halo_indices.append(
                jax.random.choice(ran_key, bin_halo_indices,
                                  shape=(max_n_halos_per_bin,), replace=False))
            downsampled_halo_weights.append(bin_halo_weights)
    downsampled_halo_indices = jnp.concatenate(downsampled_halo_indices)
    downsampled_halo_weights = jnp.concatenate(downsampled_halo_weights)

    downsampled_lc_data = lc_data_slice(lc_data, downsampled_halo_indices)
    return downsampled_lc_data, downsampled_halo_weights


def generate_weighted_grid_lc_data_kern(
    ran_key,
    lgmp_min,
    lgmp_max,
    num_m_grid,
    z_min,
    z_max,
    num_z_grid,
    sky_area_degsq,
    ssp_data,
    cosmo_params,
    tcurves,
    z_phot_table,
    logmp_cutoff=0.0,
    hmf_calibration=None,
):
    lgmp_grid = jnp.linspace(lgmp_min, lgmp_max, num_m_grid)
    z_grid = jnp.linspace(z_min, z_max, num_z_grid)
    mclh_args = (ran_key, lgmp_grid, z_grid, sky_area_degsq)
    mclh_kwargs = dict()
    if hmf_calibration == "smdpl_hmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf.HMF_PARAMS
    elif hmf_calibration == "smdpl_shmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf_subs.HMF_PARAMS
    elif hmf_calibration == "hacc_shmf":
        mclh_kwargs["hmf_params"] = hacc_core_shmf_params.HMF_PARAMS
    else:
        assert hmf_calibration is None, f"Unrecognized {hmf_calibration=}"

    res = lc_phot_kern.mclh.get_weighted_lightcone_grid_host_halo_diffmah(
        *mclh_args, logmp_cutoff=logmp_cutoff, **mclh_kwargs)  # type: ignore

    t0 = lc_phot_kern.flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(lc_phot_kern.T_TABLE_MIN,
                           t0, lc_phot_kern.N_SFH_TABLE)

    precomputed_ssp_mag_table = \
        lc_phot_kern.mclh.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table
        )
    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    lc_data = lc_phot_kern.LCData(
        res["z_obs"],
        res["t_obs"],
        res["mah_params"],
        res["logmp0"],
        t_table,
        ssp_data,
        precomputed_ssp_mag_table,
        z_phot_table,
        wave_eff_table,
    )
    return lc_data, res["nhalos"]


def generate_weighted_grid_lc_data(z_min, z_max, num_z_grid,
                                   lgmp_min, lgmp_max, num_m_grid,
                                   sky_area_degsq, ran_key=None,
                                   n_z_phot_table=15, logmp_cutoff=0.0,
                                   hmf_calibration=None):
    if ran_key is None:
        ran_key = jax.random.key(0)

    ssp_data = load_ssp_templates(DATA_DIR / SSP_FILE)

    tcurves = [load_transmission_curve(fn) if str(fn).endswith(".h5") else
               TransmissionCurve(*np.loadtxt(fn).T) for fn in FILTER_FILES]

    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    lc_data, halo_weights = generate_weighted_grid_lc_data_kern(
        ran_key,
        lgmp_min,
        lgmp_max,
        num_m_grid,
        z_min,
        z_max,
        num_z_grid,
        sky_area_degsq,
        ssp_data,
        DEFAULT_COSMOLOGY,
        tcurves,
        z_phot_table,
        logmp_cutoff=logmp_cutoff,
        hmf_calibration=hmf_calibration,
    )
    return lc_data, halo_weights


@jax.jit
def compute_targets_and_weights(
        u_param_arr, lc_data, i_band_thresh=23.0,
        thresh_softening=0.1, weights=None, ran_key=None):
    # For mag bands = g, r, i, z -> i_band_ind = 2

    if ran_key is None:
        ran_key = jax.random.key(2)
    if weights is None:
        weights = 1

    lc_phot = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data)
    m_msb, m_mss, m_q, w_msb, w_mss, w_q = lc_phot

    # Multiple weights by halo upweights AND i-band threshold weights
    w_msb *= weights * jax.nn.sigmoid(
        (i_band_thresh - m_msb[:, I_BAND_IND]) / thresh_softening)
    w_mss *= weights * jax.nn.sigmoid(
        (i_band_thresh - m_mss[:, I_BAND_IND]) / thresh_softening)
    w_q *= weights * jax.nn.sigmoid(
        (i_band_thresh - m_q[:, I_BAND_IND]) / thresh_softening)

    combined_mags = jnp.concatenate(
        (m_msb, m_mss, m_q), axis=0)
    combined_weights = jnp.concatenate(
        (w_msb, w_mss, w_q), axis=0)
    z_obs = jnp.tile(lc_data.z_obs[:, None], (3, 1))

    # Compute colors without cross-filtering between HSC and UVISTA
    combined_colors = cosmos_mags_to_colors(combined_mags)

    # Target is the i-band mag + colors between all bands + redshift
    combined_targets = jnp.concatenate(
        [combined_mags[:, I_BAND_IND, None], z_obs,
         combined_colors], axis=1)

    return combined_targets, combined_weights
