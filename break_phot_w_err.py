"""
Break photometry_w_err.fits into photometry.fits and err.fits

Created: 20250306
Updated: 20250306
Author: Qiushi (Chris) Tian
"""

from astropy import table
import astropy.units as u
from pathlib import Path
import newport


READ_PATH = Path('tables/phot_w_err')


for fn in newport.TARGET_FN:
    if '86226' not in fn:
        continue

    combined_table = table.Table.read(READ_PATH / f'photometry_w_err_{fn}.fits')

    kept_colnames = []
    err_colnames = []
    for colname in combined_table.colnames:
        if colname.startswith('err_'):
            err_colnames.append(colname)
        else:
            kept_colnames.append(colname)

    phot_table = combined_table[kept_colnames]

    err_table = combined_table[kept_colnames]
    for colname in err_colnames:
        err_table[colname.replace('err_', '')] = combined_table[colname]

    phot_table.write(READ_PATH / f'photometry_{fn}.fits')
    err_table.write(READ_PATH / f'err_{fn}.fits')
