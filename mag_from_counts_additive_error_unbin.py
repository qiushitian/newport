#!/usr/bin/env python3
"""
Run this first before `_bin`

from `mag_from_counts.py`
"""

from astropy import table
import astropy.units as u
from pathlib import Path
import scipy.stats
from newport import *

N_COL_HEAD = 6  # TODO replace with str.isdigit()
READ_DIR = Path('tables/phot/nobd-sqrt-full')
WRITE_DIR = Path('tables/mag_from_counts/nobd-sqrt-full')
REQUIRE_ROW_COMPLETE = True


def delete_incomplete_rows(t: table.Table, t2: table.Table):
    """
    Remove rows from an Astropy Table that contain any NaN values.

    Parameters:
        t: Main input table, based on which the mask is determined.
        t2: Ancillary table to which the mask is also applied. Must have same number of rows as `t`.

    Returns:
        (table.Table, table.Table): New tables with rows containing NaNs removed.
    """
    mask = np.all(np.isfinite([t[col] for col in t.colnames[N_COL_HEAD:]]), axis=0)
    return t[mask], t2[mask]


def keep_only_mode_exptime_all_bands(t: table.Table, t2: table.Table):
    """
    Remove rows whose exposure time is not the mode of the exposure times of all exposures in the band.

    **IMPORTANT: THIS FUNCTION IS FOR DEV ONLY AND IS NOT TESTED.**

    Parameters:
        t: Main input table, based on which the mask is determined.
        t2: Ancillary table to which the mask is also applied. Must have same number of rows as `t`.

    Returns:
        (table.Table, table.Table): New tables with only mode-exposure time rows.
    """
    mask = np.full((len(t), ), True)
    for band in ['B', 'V', 'R', 'I']:
        band_select = t['band'] == band
        mode_time = scipy.stats.mode(t[band_select]['exptime'])
        mask[np.logical_and(band_select, t['exptime'] != mode_time)] = False
    return t[mask], t2[mask]


def keep_only_mode_exptime(t: table.Table, t2: table.Table):
    """
    Remove rows whose exposure time is not the mode of the exposure times of all exposures in the band.

    **IMPORTANT: THIS FUNCTION IS FOR DEV ONLY AND IS NOT TESTED.**

    Parameters:
        t: Main input table, based on which the mask is determined.
        t2: Ancillary table to which the mask is also applied. Must have same number of rows as `t`.

    Returns:
        (table.Table, table.Table): New tables with only mode-exposure time rows.
    """
    mask = t['exptime'] == scipy.stats.mode(t['exptime'])[0]
    return t[mask], t2[mask]


if REQUIRE_ROW_COMPLETE:
    print('Fully complete rows only. '
          'An exposure is disqualified if it misses any `COMPARISON_STAR` '
          'and/or the target star `TARGET_GAIA_DR3` defined in `newport.py`.')

for fn in TARGET_FN:
    if '86226' not in fn:
        continue

    phot_table_all_band = table.Table.read(READ_DIR / f'phot_w_err_{fn}.fits')
    err_table_all_band = table.Table.read(READ_DIR / f'err_{fn}.fits')
    phot_target = TARGET_GAIA_DR3[fn]  # TODO allow targets outside of SPACE program targets, e.g. a random comp star

    for band in ['B', 'V', 'R', 'I']:
        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]
        err_table = err_table_all_band[err_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        comp_star_list = COMPARISON_STAR[fn][band]

        phot_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]],
                                   phot_table[comp_star_list + [phot_target]]])
        err_table = table.hstack([err_table[err_table.colnames[: N_COL_HEAD]],
                                   err_table[comp_star_list + [phot_target]]])

        phot_table, err_table = delete_incomplete_rows(phot_table, err_table)
        phot_table, err_table = keep_only_mode_exptime(phot_table, err_table)

        super_mag = get_super_mag(get_comp_mags(comp_star_list, band))

        # <-------> -> v -> unbin_comb_mag_...[super]
        super_count = np.sum([phot_table[_] for _ in comp_star_list], axis=0)
        super_err_squared = np.sum([err_table[_] ** 2 for _ in comp_star_list], axis=0)
        unbin_mags = -2.5 * np.log10(phot_table[phot_target] / super_count) + super_mag
        unbin_mag_errs = np.sqrt(
            (-2.5 / np.log(10) / phot_table[phot_target] * err_table[phot_target]) ** 2
            + (2.5 / np.log(10) / super_count) ** 2 * super_err_squared
        )

        unbin_table = phot_table[phot_table.colnames[: N_COL_HEAD]].copy()
        unbin_table['super'] = unbin_mags * u.mag
        unbin_table['super_err'] = unbin_mag_errs * u.mag
        unbin_table.write(WRITE_DIR / f'unbin_comb_mag_{fn}_{band}.fits', overwrite=False)

        # TODO ^
        # TODO |
        # TODO |
        # TODO |
        # TODO |
        # TODO v
        # TODO
        # TODO |
        # TODO v
        # TODO
        # TODO v -> bin_comb_mag_...[comp_star_colname]

        # \
        #  \
        #   \
        #    \
        #     \  # this can use results from either one above... which? bin_comb_COUNT[comp_star_colname]
        #                                                             # no, actually from super count
        #       v -> bin_comb_mag_...[super]
        _night_super_target = phot_table[['night']]
        _night_super_target['super'] = super_count
        _night_super_target['target'] = phot_table[phot_target]
        _night_super_target['super_err_squared'] = super_err_squared
        _night_super_target['target_err_squared'] = err_table[phot_target] ** 2
        _aggregated = _night_super_target.group_by('night').groups.aggregate(np.sum)
        bin_mags = -2.5 * np.log10(_aggregated['target'] / _aggregated['super']) + super_mag
        bin_mag_errs = np.sqrt(
            (-2.5 / np.log(10) / _aggregated['target']) ** 2 * _aggregated['target_err_squared']
            + (2.5 / np.log(10) / _aggregated['super']) ** 2 * _aggregated['super_err_squared']
        )

        bin_table = phot_table[['night', 'jd', 'airmass']].copy()
        bin_table = bin_table.group_by('night').groups.aggregate(np.mean)
        bin_table['super'] = bin_mags * u.mag
        bin_table['super_err'] = bin_mag_errs * u.mag
        bin_table.write(WRITE_DIR / f'bin_comb_mag_{fn}_{band}.fits', overwrite=False)
