import os
import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from diffmah.diffmah_kernels import _log_mah_kern
from diffmah.diffmahpop_kernels.bimod_censat_params import DEFAULT_DIFFMAHPOP_PARAMS
from diffmah.diffmahpop_kernels.mc_bimod_cens import mc_cenpop
from diffsky.experimental import lc_phot_kern
from diffsky.experimental.lc_phot_kern import mclh
from diffsky.experimental.mc_lightcone_halos import get_nhalo_weighted_lc_grid
from diffsky.mass_functions import mc_hosts
from diffsky.mass_functions.fitting_utils.calibrations import hacc_core_shmf_params
from diffsky.mass_functions.hmf_calibrations import smdpl_hmf, smdpl_hmf_subs
from diffsky.ssp_err_model import ssp_err_model
from dsps import load_ssp_templates
from dsps.cosmology import flat_wcdm
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_transmission_curve
from dsps.data_loaders.defaults import TransmissionCurve
from mpi4py import MPI
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


def cosmos_mags_to_colors(mags, filter_names=FILTER_NAMES):
    assert mags.shape[1] == len(filter_names), \
        f"mags must be shape (N, {len(filter_names)})"
    filter_set_names = list(dict.fromkeys(
        [x.split("_")[0] for x in filter_names]))
    filter_set_inds = [
        [i for i, fname in enumerate(filter_names) if fname.startswith(sname)]
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


def get_nhalo_from_grid_interp(tot_num_halos, z_obs, logmp_obs_mf,
                               z_min, z_max, lgmp_min, lgmp_max,
                               sky_area_degsq, hmf_params, cosmo_params):
    ngrid_z = 200
    ngrid_m = 200
    ngrid_tot = ngrid_z * ngrid_m
    z_grid = jnp.linspace(z_min, z_max, ngrid_z)
    lgmp_grid = jnp.linspace(lgmp_min, lgmp_max, ngrid_m)
    nhalo_grid = get_nhalo_weighted_lc_grid(
        lgmp_grid, z_grid, sky_area_degsq, hmf_params, cosmo_params)

    interpolator = jax.scipy.interpolate.RegularGridInterpolator(
        (z_grid, lgmp_grid), nhalo_grid,
        bounds_error=False, fill_value=None)  # type: ignore

    interp = interpolator(jnp.column_stack([z_obs, logmp_obs_mf]))
    return interp * ngrid_tot / tot_num_halos


def get_weighted_lightcone_sobol_host_halo_diffmah(
    ran_key,
    tot_num_halos,
    z_obs,
    logmp_obs_mf,
    z_min,
    z_max,
    lgmp_min,
    lgmp_max,
    sky_area_degsq,
    cosmo_params=flat_wcdm.PLANCK15,
    hmf_params=mc_hosts.DEFAULT_HMF_PARAMS,
    diffmahpop_params=DEFAULT_DIFFMAHPOP_PARAMS,
    logmp_cutoff=0.0,
):
    """
    Compute the number of halos on the input halo mass and redshift points
    """

    nhalo_weights = get_nhalo_from_grid_interp(
        tot_num_halos, z_obs, logmp_obs_mf,
        z_min, z_max, lgmp_min, lgmp_max,
        sky_area_degsq, hmf_params=hmf_params, cosmo_params=cosmo_params,
    )
    # nhalo_weights = nhalo_weighted_lc_grid.flatten()
    # z_obs = np.repeat(z_grid, lgmp_grid.size)
    # logmp_obs_mf = np.tile(lgmp_grid, z_grid.size)

    t_obs = flat_wcdm.age_at_z(z_obs, *cosmo_params)
    t_0 = flat_wcdm.age_at_z0(*cosmo_params)
    lgt0 = jnp.log10(t_0)

    logmp_obs_mf_clipped = np.clip(logmp_obs_mf, logmp_cutoff, np.inf)

    tarr = np.array((10**lgt0,))
    args = (diffmahpop_params, tarr, logmp_obs_mf_clipped,
            t_obs, ran_key, lgt0)
    mah_params_uncorrected = mc_cenpop(*args)[0]  # mah_params, dmhdt, log_mah

    logmp_obs_orig = _log_mah_kern(mah_params_uncorrected, t_obs, lgt0)
    delta_logmh_clip = logmp_obs_orig - logmp_obs_mf
    mah_params = mah_params_uncorrected._replace(
        logm0=mah_params_uncorrected.logm0 - delta_logmh_clip
    )

    logmp0 = _log_mah_kern(mah_params, 10**lgt0, lgt0)
    logmp_obs = _log_mah_kern(mah_params, t_obs, lgt0)

    fields = ("z_obs", "t_obs", "logmp_obs", "mah_params", "logmp0")
    values = (z_obs, t_obs, logmp_obs, mah_params, logmp0)
    cenpop_out = dict()
    for key, value in zip(fields, values):
        cenpop_out[key] = value
    cenpop_out["nhalos"] = nhalo_weights

    return cenpop_out


def generate_weighted_sobol_lc_data(num_halos, z_min, z_max,
                                    lgmp_min, lgmp_max,
                                    sky_area_degsq, ran_key=None,
                                    n_z_phot_table=15, logmp_cutoff=0.0,
                                    hmf_calibration=None, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    if ran_key is None:
        ran_key = jax.random.key(0)
    ran_key, ran_key_sobol = jax.random.split(ran_key, 2)

    # ONLY generate the halos necessary on this rank
    num_halos_on_rank = num_halos // comm.size + \
        (1 if comm.rank < num_halos % comm.size else 0)
    starting_index = comm.rank * \
        (num_halos // comm.size) + min(comm.rank, num_halos % comm.size)

    # Generate Sobol sequence for halo masses and redshifts
    seed = int(jax.random.randint(ran_key_sobol, (), 0, 2**31 - 1))
    bits = None
    if num_halos > 1e9:
        # 64-bit sequence required to generate over 2^30 halos
        bits = 64
    sampler = qmc.Sobol(d=2, scramble=True, rng=seed, bits=bits)
    if starting_index > 0:
        sampler.fast_forward(starting_index)

    with warnings.catch_warnings():
        # Ignore warning about Sobol sequences not being fully balanced
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="scipy.stats")
        sample = sampler.random(num_halos_on_rank)
        z_obs, logmp_obs_mf = qmc.scale(
            sample, (z_min, lgmp_min), (z_max, lgmp_max)).T

    ssp_data = load_ssp_templates(DATA_DIR / SSP_FILE)

    ffiles = [FILTERS_DIR / fn for fn in FILTER_FILES]
    tcurves = [load_transmission_curve(fn) if str(fn).endswith(".h5") else
               TransmissionCurve(*np.loadtxt(fn).T) for fn in ffiles]

    z_phot_table = np.linspace(z_min, z_max, n_z_phot_table)

    mclh_args = (ran_key, num_halos, z_obs, logmp_obs_mf, z_min, z_max,
                 lgmp_min, lgmp_max, sky_area_degsq)
    mclh_kwargs = dict()
    if hmf_calibration == "smdpl_hmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf.HMF_PARAMS
    elif hmf_calibration == "smdpl_shmf":
        mclh_kwargs["hmf_params"] = smdpl_hmf_subs.HMF_PARAMS
    elif hmf_calibration == "hacc_shmf":
        mclh_kwargs["hmf_params"] = hacc_core_shmf_params.HMF_PARAMS
    else:
        assert hmf_calibration is None, f"Unrecognized {hmf_calibration=}"

    res = get_weighted_lightcone_sobol_host_halo_diffmah(
        *mclh_args, logmp_cutoff=logmp_cutoff, **mclh_kwargs)  # type: ignore

    t0 = lc_phot_kern.flat_wcdm.age_at_z0(*DEFAULT_COSMOLOGY)
    t_table = jnp.linspace(lc_phot_kern.T_TABLE_MIN,
                           t0, lc_phot_kern.N_SFH_TABLE)

    precomputed_ssp_mag_table = \
        mclh.get_precompute_ssp_mag_redshift_table(
            tcurves, ssp_data, z_phot_table
        )
    wave_eff_table = lc_phot_kern.get_wave_eff_table(z_phot_table, tcurves)

    lc_data = lc_phot_kern.LCData(
        res["nhalos"],
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
    return lc_data


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
