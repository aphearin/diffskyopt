import numpy as np

import jax
import jax.numpy as jnp

from dsps.data_loaders import retrieve_fake_fsps_data
from dsps.data_loaders.defaults import TransmissionCurve
from diffsky.experimental import lc_phot_kern
from diffsky.mass_functions.fitting_utils.calibrations \
    import hacc_core_shmf_params as hcshmf
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY


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
    hacc_core_params=True,
):
    mclh_args = (ran_key, lgmp_min, z_min, z_max, sky_area_degsq)

    kw = dict(hmf_params=hcshmf.HMF_PARAMS) if hacc_core_params else {}
    lc_halopop = lc_phot_kern.mclh.mc_lightcone_host_halo_diffmah(
        *mclh_args, logmp_cutoff=logmp_cutoff, **kw)

    t0 = lc_phot_kern.flat_wcdm.age_at_z0(*cosmo_params)
    t_table = jnp.linspace(lc_phot_kern.T_TABLE_MIN,
                           t0, lc_phot_kern.N_SFH_TABLE)

    precomputed_ssp_mag_table = \
        lc_phot_kern.mclh.get_precompute_ssp_mag_redshift_table(
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
                     hacc_core_params=True):
    if ran_key is None:
        ran_key = jax.random.key(0)

    ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()

    _res = retrieve_fake_fsps_data.load_fake_filter_transmission_curves()
    wave, u, g, r, i, z, y = _res

    tcurves = [TransmissionCurve(wave, x)  # type: ignore
               for x in (u, g, r, i, z, y)]

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
        hacc_core_params=hacc_core_params,
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


@jax.jit
def compute_targets_and_weights(
        u_param_arr, lc_data, i_band_thresh=23.0,
        i_band_ind=3, thresh_softening=0.1,
        halo_upweights=None, ran_key=None):
    # For mag bands = u, g, r, i, z, y -> i_band_ind = 3

    if ran_key is None:
        ran_key = jax.random.key(2)
    if halo_upweights is None:
        halo_upweights = jnp.ones(len(lc_data.logmp0))

    lc_phot = lc_phot_kern.multiband_lc_phot_kern_u_param_arr(
        u_param_arr, ran_key, lc_data)
    m_msb, m_mss, m_q, w_msb, w_mss, w_q = lc_phot

    # Multiple weights by halo upweights AND i-band threshold weights
    w_msb *= halo_upweights * jax.nn.sigmoid(
        (i_band_thresh - m_msb[:, i_band_ind]) / thresh_softening)
    w_mss *= halo_upweights * jax.nn.sigmoid(
        (i_band_thresh - m_mss[:, i_band_ind]) / thresh_softening)
    w_q *= halo_upweights * jax.nn.sigmoid(
        (i_band_thresh - m_q[:, i_band_ind]) / thresh_softening)

    combined_mags = jnp.concatenate(
        (m_msb, m_mss, m_q), axis=0)
    combined_weights = jnp.concatenate(
        (w_msb, w_mss, w_q), axis=0)
    z_obs = jnp.tile(lc_data.z_obs[:, None], (3, 1))

    # Target is the i-band mag + colors between all bands + redshift
    combined_targets = jnp.concatenate(
        [combined_mags[:, i_band_ind, None],
         -jnp.diff(combined_mags, axis=1), z_obs], axis=1)

    return combined_targets, combined_weights
