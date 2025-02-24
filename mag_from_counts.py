#!/usr/bin/env python3
"""

from `onerel.py`
"""

from astropy import table
import astropy.units as u
from newport import *

N_COL_HEAD = 5  # TODO replace with str.isdigit()
REQUIRE_ROW_COMPLETE = True


def delete_incomplete_rows(t: table.Table):
    """
    Remove rows from an Astropy Table that contain any NaN values.

    Parameters:
        t: Input table.

    Returns:
        table.Table: A new table with rows containing NaNs removed.
    """
    mask = np.all(np.isfinite([t[col] for col in t.colnames[N_COL_HEAD:]]), axis=0)
    return t[mask]


if REQUIRE_ROW_COMPLETE:
    print('Fully complete rows only. '
          'An exposure is disqualified if it misses any `COMPARISON_STAR` '
          'and/or the target star `TARGET_GAIA_DR3` defined in `newport.py`.')

for fn in TARGET_FN:
    if '86226' not in fn:
        continue

    phot_table_all_band = table.Table.read(f'tables/phot/photometry_{fn}.fits')
    phot_target = TARGET_GAIA_DR3[fn]  # TODO allow different targets

    for band in ['B', 'V', 'R', 'I']:
        if band == 'B':
            continue

        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        comp_star_list = COMPARISON_STAR[fn][band]

        phot_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]],
                                   phot_table[comp_star_list + [phot_target]]])

        phot_table = delete_incomplete_rows(phot_table)

        super_mag = get_super_mag(get_comp_mags(comp_star_list, band))

        # <-------> -> v -> unbin_comb_mag_...[super]
        super_count = np.sum([phot_table[_] for _ in comp_star_list], axis=0)
        unbin_mags = -2.5 * np.log10(phot_table[phot_target] / super_count) + super_mag
        unbin_table = phot_table[phot_table.colnames[: N_COL_HEAD]].copy()
        unbin_table['super'] = unbin_mags * u.mag
        unbin_table.write(
            f'tables/mag_from_counts/unbin_comb_mag_{fn}_{band}.fits',
            overwrite=False)

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
        _aggregated = _night_super_target.group_by('night').groups.aggregate(np.sum)
        bin_mags = -2.5 * np.log10(_aggregated['target'] / _aggregated['super']) + super_mag

        bin_table = phot_table[['night', 'jd', 'airmass']].copy()
        bin_table = bin_table.group_by('night').groups.aggregate(np.mean)
        bin_table['super'] = bin_mags * u.mag
        bin_table.write(
            f'tables/mag_from_counts/bin_comb_mag_{fn}_{band}.fits',
            overwrite=False)
