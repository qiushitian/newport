from pathlib import Path
from astropy import table
import numpy as np
from optimize_rel_phot import *

if __name__ == "__main__":
    TARGET = "HD_191939"
    TARGET_ID = "2248126315275354496"
    INPUT_PATH = Path('tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939.fits')
    BASE_PATH = Path('tables/opt_comp_stars/HD_191939/six/results')

    full_table = table.Table.read(INPUT_PATH)

    for crit in np.arange(0.75, 0.89, 0.01)[::-1]:
        print(f'\n{crit:.2f}')
        for band in ['B', 'V', 'R', 'I']:
            band_table = full_table[full_table['band'] == band]
            best_ensemble = get_comps(band_table, TARGET_ID, criterion=crit)

            # print(f'\t{band}\t{len(best_ensemble)}')

            engine = RelativePhotometryEngine(band_table, TARGET_ID)
            t = engine.get_binned(best_ensemble)[0]

            print(f'\t{band}\t{len(t)}')
