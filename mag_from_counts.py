#!/usr/bin/env python3
"""

from `onerel.py`
"""

from astropy import table
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
    print('Fully complete rows only.'
          'An exposure is disqualified if it misses any `COMPARISON_STAR` defined in `newport.py`.')

for fn in TARGET_FN:
    if '86226' not in fn:
        continue

    phot_table_all_band = table.Table.read(f'tables/phot/photometry_{fn}.fits')
    phot_target = TARGET_GAIA_DR3[fn]  # TODO

    for band in ['B', 'V', 'R', 'I']:
        if band != 'B':
            continue

        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        phot_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]],
                                   phot_table[COMPARISON_STAR[fn][band]]])

        phot_table = delete_incomplete_rows(phot_table)

        # TODO now, this is what happens:
        # TODO <-------> -> v -> unbin_comb_mag_...[super]

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

        # TODO \
        # TODO  \
        # TODO   \
        # TODO    \
        # TODO     \  # this can use results from either one above... which? bin_comb_COUNT[comp_star_colname]
        # TODO
        # TODO       v -> bin_comb_mag_...[combined]


        # filtered_table = table.Table()
        # for colname in COMPARISON_STAR[fn][band]:
        #     if colname != phot_target:
        #         filtered_table[colname] = phot_table[phot_target] / phot_table[colname]

        # onerel_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]], onerel_table])
        # onerel_table.write(f'tables/onerel/{fn}_field/onerel_field={fn}_target={phot_target}_band={band}.fits',
        #                    overwrite=False)
