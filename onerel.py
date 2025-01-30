#!/usr/bin/env python3
"""
Take photometry_*.fits tables and do relative photometry,
output relphot_*.fits and binned_*.fits tables
"""

# import numpy as np
from astropy import table
from newport import *
# from astroquery.utils.tap.core import TapPlus
#
# SIMBAD_TAP = TapPlus(url="http://simbad.u-strasbg.fr/simbad/sim-tap")

# Percentile of non-NaN observations required for column to be considered valid
# CRITERION = 70
N_COL_HEAD = 5  # TODO replace with str.isdigit()

for fn in TARGET_FN:
    # if '191939' in fn or '86226' in fn:
    #     continue

    phot_table_all_band = table.Table.read(f'tables/phot/photometry_{fn}.fits')

    for band in ['B', 'V', 'R', 'I']:
        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        # body_cols = phot_table.columns[N_COL_HEAD:]
        # valid_counts = [np.sum(~ np.isnan(col)) for col in body_cols]
        # # for col in body_cols:
        # #     valid_counts.append(np.sum(~np.isnan(col)))
        # valid_counts_threshold = np.percentile(valid_counts, CRITERION)
        # valid_col_mask = np.array(valid_counts) > valid_counts_threshold
        # valid_table = body_cols[valid_col_mask]
        #
        # valid_col = []
        # for colname in phot_table.colnames[N_COL_HEAD:]:
        #     col_data = phot_table[colname]
        #     valid_fraction = np.sum(~np.isnan(col_data)) / len(col_data)
        #     if valid_fraction > CRITERION:
        #         valid_col.append(colname)
        #     else:
        #         valid_col.append(colname)

        # 2
        valid_colnames = []
        for colname in phot_table.colnames[N_COL_HEAD:]:
            if colname == TARGET_GAIA_DR3[fn] \
                    or not isinstance(phot_table[colname], table.Table.MaskedColumn):
                valid_colnames.append(colname)

        print(f'{fn}\t{band}\t{len(valid_colnames)}')  # DEV

        # valid_table = phot_table[valid_colnames]

        onerel_table = table.Table()  # names=valid_table.colnames
        for colname in valid_colnames:
            onerel_table.add_column(phot_table[TARGET_GAIA_DR3[fn]] / phot_table[colname], name=colname)

        onerel_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]], onerel_table])

        gpb = onerel_table.group_by(['night', 'band'])
        binned_onerel = gpb.groups.aggregate(np.median)
        # TODO add std or mad_std here and save it

        # band-night normalization
        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO WHAT THE FUC IS GOING ON HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for colname in binned_onerel.colnames:
            if colname.isdigit():
                med = np.nanmedian(binned_onerel[colname])
                binned_onerel[colname] /= med
                onerel_table[colname] /= med

        binned_onerel.write(f'tables/onerel/binned_onerel_{fn}_{band}.fits', overwrite=False)
        onerel_table.write(f'tables/onerel/onerel_{fn}_{band}.fits', overwrite=False)
