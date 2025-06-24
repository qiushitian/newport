#!/usr/bin/env python3
"""
Calculate relative photometry of a target
against one comparison star at a time.

from `old_magphot.py`
"""
import numpy as np
from astropy import table
from pathlib import Path
import newport

READ_DIR = Path('tables/list_runs/191939/phot_gaia_run')
WRITE_DIR = Path('tables/list_runs/191939/onerel')

# Percentile of non-NaN observations required for column to be considered valid
CRITERION = 0.01  # was 0.7
N_COL_HEAD = 5  # TODO replace with str.isdigit()

if __name__ == '__main__':
    WRITE_DIR.mkdir(parents=True, exist_ok=True)

    for fn in newport.TARGET_FN:
        if '191939' not in fn:
            continue

        phot_table_all_band = table.Table.read(READ_DIR / f'phot_w_err_{fn}.fits')
        phot_target = newport.TARGET_GAIA_DR3[fn]  # TODO

        for band in ['B', 'V', 'R', 'I']:
            phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

            if len(phot_table) < 1:
                print(f'{fn}\t{band}\t skipped')
                continue

            valid_colnames = []
            for colname in phot_table.colnames[N_COL_HEAD:]:
                if (
                        isinstance(phot_table[colname], table.Table.MaskedColumn)
                        and phot_table[colname].mask.sum() / len(phot_table) > (1 - CRITERION)
                ):
                    continue
                valid_colnames.append(colname)
            phot_table = phot_table.filled(np.nan)

            print(f'{fn}\t{band}\tvalid\t{len(valid_colnames)}')  # TODO DEV

            onerel_table = table.Table()
            for colname in valid_colnames:
                if colname != phot_target:
                    onerel_table[colname] = phot_table[phot_target] / phot_table[colname]

            onerel_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]], onerel_table])
            onerel_table.write(
                WRITE_DIR / f'onerel_field={fn}_target={phot_target}_band={band}.fits',
                overwrite=False
            )
