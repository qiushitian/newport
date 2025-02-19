#!/usr/bin/env python3
"""
groupby binning
"""

from astropy import table
from pathlib import Path
from newport import *


DIR = Path('tables/mag')
TABLE_FN = 'mag'
GROUP_BY_KEY = 'night'  # ['night', 'band']


for fn in TARGET_FN:
    if '86226' not in fn:  # TODO DEV
        continue
    for band in ['B', 'V', 'R', 'I']:
        ori_table = table.Table.read(DIR / f'{TABLE_FN}_{fn}_{band}.fits').filled(np.nan)

        # TODO jd/airmass inaccuracy: median of mag is not necessarily median of jd/airmss
        # TODO replace with sigclipped mean
        grp_by = ori_table.group_by(GROUP_BY_KEY)
        binned_tbl = grp_by.groups.aggregate(np.nanmedian)
        binned_tbl.write(DIR / f'binned_{TABLE_FN}_{fn}_{band}.fits', overwrite=True)
