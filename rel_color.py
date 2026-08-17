from tqdm import tqdm
from pathlib import Path
from astropy import table
import numpy as np


OVERWRITE = False
TABLE_DIR = Path("tables/opt_comp_stars/HD_191939/two_ss")
COLORS = [
    ('B','V'),
    ('B','R'),
    ('B','I'),
    ('V','R'),
    ('V','I'),
    ('R','I')
]
BTAB_NAME = 'bin_results_all.fits'
CTAB_NAME = 'colors.fits'


if __name__ == "__main__":
    btab = table.Table.read(TABLE_DIR / BTAB_NAME)
    sigclip_colname = f'within_{btab.meta["SIGCLIP"]}_sig'

    overwrite = OVERWRITE
    prompt_overwrite = 'Color(s) already exist. Overwrite? [y/*]: '
    rm_i = []
    for i in range(len(btab)):
        if '/' in btab[i]['band'] or '-' in btab[i]['band']:
            if overwrite:
                rm_i.append(i)
            elif input(prompt_overwrite).lower().startswith('y'):
                overwrite = True
                rm_i.append(i)
            else:
                print('Color(s) already exist and no overwrite. Aborting.')
                import sys
                sys.exit()
    btab.remove_rows(rm_i)
    
    groups = btab.group_by('night').groups
    n_night_any = len(groups)

    ctab = table.Table(dtype=btab.dtype)

    for g in tqdm(groups, desc='Calculating colors for night'):
        for b, r in COLORS:
            tb, tr = g[g['band'] == b], g[g['band'] == r]
            if len(tb) == 1 and len(tr) == 1:
                ctab.add_row([
                    tb['night'],
                    np.mean([tb['jd'], tr['jd']]),
                    tb['flux'] / tr['flux'],
                    np.sqrt((tb['error']/tb['flux'])**2 + (tr['error']/tr['flux'])**2) * tb['flux'] / tr['flux'],  # TODO verify
                    np.mean([tb['airmass'], tr['airmass']]),
                    tb['exptime_sum'] + tr['exptime_sum'],
                    tb['n_exp'] + tr['n_exp'],
                    np.sqrt((tb['intraday_std']/tb['flux'])**2 + (tr['intraday_std']/tr['flux'])**2) * tb['flux'] / tr['flux'],  # TODO verify
                    tb[sigclip_colname] & tr[sigclip_colname],
                    f'{b}/{r}'
                ])
            elif len(tb) > 1:
                raise ValueError(f'There are {len(tb)} {b}-band measurements on {g["night"][0]}')
            elif len(tr) > 1:
                raise ValueError(f'There are {len(tr)} {r}-band measurements on {g["night"][0]}')
    

    ctab[sigclip_colname].description = f'Whether both bands data are within {btab.meta["SIGCLIP"]} sigma'
    ctab.meta['N_NIGHTS'] = n_night_any
    ctab.write(TABLE_DIR / CTAB_NAME, overwrite=True)

    btab = table.vstack([btab, ctab])
    btab.meta['N_NIGHTS'] = n_night_any
    btab.write(TABLE_DIR / BTAB_NAME, overwrite=True)
