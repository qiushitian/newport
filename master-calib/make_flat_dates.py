"""
Make flat of specific dates

From: make_mastercalib.py
Author: Qiushi (Chris) Tian
Created: 2025-03-21
"""
import numpy as np
from pathlib import Path
import ccdproc as ccdp
from astropy.io import fits
import astropy.units as u
from astropy.nddata import CCDData, StdDevUncertainty
from convenience_functions import show_image

DATES = ['20230515', '20231102']

ROOT_PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw/calib')
SAVE_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib-2025-no_flat')

# need to be expanded
keywords = [
    'imagetyp','filter', 'date-obs', 'exposure', 'naxis1', 'naxis2', 'xbinning', 'ybinning',
    'xorgsubf', 'yorgsubf', 'ccd-temp', 'readoutm', 'gain', 'readout'
]


def inv_nanmedian(a):
    return 1 / np.nanmedian(a)


if __name__ == '__main__':
    for date in DATES:
        bias_hdul = fits.open(SAVE_PATH / date / 'master_bias.fit')
        combined_bias = CCDData(
            bias_hdul[0].data, StdDevUncertainty(bias_hdul[2].data, unit='adu'), bias_hdul[1].data,
            meta=bias_hdul[0].header, unit='adu'
        )

        dark_collection = ccdp.ImageFileCollection(
            SAVE_PATH / date, glob_include='master_dark_bias-subtracted_*s.fit', keywords=['exposure']
        )
        dark_collection.sort('exposure')
        dark_hdul = fits.open(SAVE_PATH / date / dark_collection.files[-1])
        combined_dark = CCDData(
            dark_hdul[0].data, StdDevUncertainty(dark_hdul[2].data, unit='adu'), dark_hdul[1].data,
            meta=dark_hdul[0].header, unit='adu'
        )

        try:
            flat_collection = ccdp.ImageFileCollection(ROOT_PATH / date / 'AutoFlat', glob_exclude='.*', keywords=keywords)
        except FileNotFoundError:
            continue

        flat_collection = flat_collection.filter(xbinning=1, ybinning=1)

        flat_filters = set(flat_collection.summary['filter'])
        for filt in flat_filters:
            # bd correction
            single_filter_flats = []
            for file in flat_collection.filter(filter=filt).files:
                ccd_data = CCDData.read(file, unit='adu', output_verify='ignore')
                ccd_data = ccdp.subtract_bias(ccd_data, combined_bias)
                ccd_data = ccdp.subtract_dark(
                    ccd_data, combined_dark, exposure_time='exposure', exposure_unit=u.s, scale=True
                )
                ccd_data.header['history'] = 'Bias and dark subtracted with NewPORT, by Qiushi (Chris) Tian'
                single_filter_flats.append(ccd_data)

            # averaging
            combined_flat = ccdp.combine(
                single_filter_flats, unit='adu', method='median', output_verify='ignore'  # scale=inv_nanmedian,
                # sigma_clip=True, sigma_clip_low_thresh=0.5, sigma_clip_high_thresh=0.5,
                # sigma_clip_func='median', signma_clip_dev_func='mad_std'  # , mem_limit=8e9
            )

            combined_flat.data /= np.nanmedian(combined_flat.data)

            combined_flat.meta['combined'] = True  # meta ###
            combined_flat.header['history'] = 'Master flat build with NewPORT, by Qiushi (Chris) Tian'
            # combined_flat.header['history'] = 'MJD-OBS fixed by astropy datfix.'

            # (SAVE_PATH / date).mkdir(exist_ok=True)
            combined_flat.write(
                SAVE_PATH / date / f'master_flat_bdcorrected_{filt}_median.fit', overwrite=True, output_verify='ignore'
            )

            # show
            show_image(combined_flat, cmap='gray', figsize=(4, 8))
