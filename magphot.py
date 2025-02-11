#!/usr/bin/env python3
"""
Take photometry_*.fits tables and do relative photometry,
output relphot_*.fits and binned_*.fits tables
"""

import astropy.units as u
from astropy import table
from newport import *
from astroquery.utils.tap.core import TapPlus

SIMBAD_TAP = TapPlus(url="http://simbad.u-strasbg.fr/simbad/sim-tap")

# Percentile of non-NaN observations required for column to be considered valid
CRITERION = 0.70
N_COL_HEAD = 5  # TODO replace with str.isdigit()
COMBINED_KEY = 'combined'


def combine_mags(mags):
    mags = np.array([_ for _ in mag_dict.values()])
    if np.isnan(mags).any():
        raise ValueError('NaN is in the input.')

    fluxes = 10 ** (mags / -2.5)
    total_flux = np.sum(fluxes)
    if total_flux <= 0 or np.isnan(total_flux):
        raise RuntimeError('Total flux is 0 or NaN.')

    return -2.5 * np.log10(total_flux)


for fn in TARGET_FN:
    if '86226' not in fn:
        continue

    phot_table_all_band = table.Table.read(f'tables/phot/photometry_{fn}.fits')

    for band in ['B', 'V', 'R', 'I']:
        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        valid_colnames = []
        for colname in phot_table.colnames[N_COL_HEAD:]:
            if (
                    isinstance(phot_table[colname], table.Table.MaskedColumn)
                    and phot_table[colname].mask.sum() / len(phot_table) < (1 - CRITERION)
                    and colname != TARGET_GAIA_DR3[fn]
            ):
                valid_colnames.append(colname)
        phot_table = phot_table.filled(np.nan)

        print(f'{fn}\t{band}\tvalid\t{len(valid_colnames)}')  # TODO DEV

        # valid_table = phot_table[valid_colnames]

        mag_dict = {}
        for colname in valid_colnames:
            mag = SIMBAD_TAP.launch_job(
                f"SELECT {band} FROM allfluxes JOIN ident USING(oidref) WHERE id = 'Gaia DR3 {colname}';"
            ).get_results()[band]
            if len(mag) > 0 and not mag.mask:
                mag_dict[colname] = mag[0]

        print(f'{fn}\t{band}\twith mag\t{len(mag_dict)}')  # TODO DEV

        combined_counts = np.array([0.0] * len(phot_table))
        onerel_table = table.Table()  # names=valid_table.colnames
        for colname in mag_dict.keys():
            combined_counts += phot_table[colname]
            onerel_table[colname] = phot_table[TARGET_GAIA_DR3[fn]] / phot_table[colname]
        onerel_table[COMBINED_KEY] = phot_table[TARGET_GAIA_DR3[fn]] / combined_counts
        mag_dict[COMBINED_KEY] = combine_mags(mag_dict.values())

        mag_table = table.Table()  # names=valid_table.colnames
        for colname, mag in mag_dict.items():
            target_mag = -2.5 * np.log10(onerel_table[colname]) + mag
            mag_table[colname] = target_mag * u.mag

        # add mean column
        mean_mag_array = []
        for col in mag_table.colnames:
            if col != COMBINED_KEY:
                mean_mag_array.append(mag_table[col])
        mag_table['mean'] = np.mean(mean_mag_array, axis=0)

        mag_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]], mag_table])
        mag_table.write(f'tables/mag/mag_{fn}_{band}.fits', overwrite=False)
