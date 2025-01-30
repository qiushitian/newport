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
# CRITERION = 70
N_COL_HEAD = 5  # TODO replace with str.isdigit()

for fn in TARGET_FN:
    if '86226' not in fn:
        continue

    for band in ['B', 'V', 'R', 'I']:
        # (f'tables/onerel/binned_onerel_{fn}_{band}.fits')
        onerel_table = table.Table.read(f'tables/onerel/onerel_{fn}_{band}.fits')

        valid_colnames = []
        for colname in onerel_table.colnames[N_COL_HEAD:]:
            if colname != TARGET_GAIA_DR3[fn]:
                valid_colnames.append(colname)

        print(f'{fn}\t{band}\t{len(valid_colnames)}')  # DEV

        # valid_table = phot_table[valid_colnames]

        mag_table = table.Table()  # names=valid_table.colnames
        for colname in valid_colnames:
            mag = SIMBAD_TAP.launch_job(
                f"SELECT {band} FROM allfluxes JOIN ident USING(oidref) WHERE id = 'Gaia DR3 {colname}';"
            ).get_results()[band]
            if len(mag) > 0 and mag is not np.ma.masked:
                mag_table.add_column((- 2.5 * np.log10(onerel_table[colname]) + mag[0]) * u.mag, name=colname)

        mag_table = table.hstack([onerel_table[onerel_table.colnames[: N_COL_HEAD]], mag_table])

        # gpb = onerel_table.group_by(['night', 'band'])
        # binned_onerel = gpb.groups.aggregate(np.median)
        # # TODO add std or mad_std here and save it

        # band-night normalization
        # for colname in binned_onerel.colnames:
        #     if colname.isdigit():
        #         med = np.nanmedian(binned_onerel[colname])
        #         binned_onerel[colname] /= med
        #         onerel_table[colname] /= med
        #
        # binned_onerel.write(f'tables/onerel/binned_onerel_{fn}_{band}.fits', overwrite=False)
        mag_table.write(f'tables/mag/mag_{fn}_{band}.fits', overwrite=True)
