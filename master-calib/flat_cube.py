import numpy as np
from astropy.io import fits
from pathlib import Path
from tqdm import tqdm


CALIB_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib')
DATE_RANGE_START = 20230314
DATE_RANGE_END = 20230801

if __name__ == '__main__':
    date_list = list(CALIB_PATH.glob('[!.]*'))
    cube = None
    i = 1
    for date_path in tqdm(date_list):
        date = date_path.name

        try:
            date_int = int(date)
        except ValueError:
            continue

        if date_int < DATE_RANGE_START or date_int > DATE_RANGE_END:
            continue

        try:
            flat_hdu = fits.open(CALIB_PATH / date / 'master_flat_bdcorrected_R.fit', output_verify='ignore')[0].data
        except FileNotFoundError:
            continue

        print(i, date)

        if cube is None:
            cube = flat_hdu[np.newaxis, :, :]
        else:
            cube = np.vstack([cube, flat_hdu[np.newaxis, :, :]])

        i += 1

    hdu = fits.PrimaryHDU(cube)
    # fits.HDUList([hdu]).writeto('flat-cube-r.fits')  # , overwrite=True)
