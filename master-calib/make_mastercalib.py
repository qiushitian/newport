"""
from /Volumes/emlaf/qtian/station/vvophot-eres/try-calib.ipynb
"""

import numpy as np
from pathlib import Path
import ccdproc as ccdp
from datetime import datetime, timedelta
# from astropy.io import fits
import astropy.units as u
from astropy.nddata import CCDData#, StdDevUncertainty
from tqdm import tqdm
from convenience_functions import show_image

# preserved - DO NOT MODIFY
ROOT_PATH = Path('/opt/westep/qiushi/mountpoint/space-raw/calib/')
SAVE_PATH = Path('/opt/westep/qiushi/mastercalib/')

# HERE modify
ROOT_PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw/calib')
SAVE_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib-2025-no_flat')

# need to be expanded
keywords = ['imagetyp','filter', 'date-obs',
            'exposure',
            'naxis1', 'naxis2',
            'xbinning', 'ybinning',
            'xorgsubf', 'yorgsubf',
            'ccd-temp', 'readoutm', 'gain', 'readout']
bias_filter = {'imagetyp': 'Bias Frame',
               'exposure': 0.0,
               # 'naxis1': 2048, 'naxis2': 2048,
               'xbinning': 1, 'ybinning': 1,
               'xorgsubf': 0, 'yorgsubf': 0,
               'ccd-temp': -20.0, 'readoutm': '2.0 MHz', 'gain': 1.85, 'readout': 13}


DARK_EXP_TIME = 30.0

dark_filter = {'imagetyp': 'Dark Frame',
               # 'exposure': DARK_EXP_TIME,
               # 'naxis1': 2048, 'naxis2': 2048,
               'xbinning': 1, 'ybinning': 1,
               'xorgsubf': 0, 'yorgsubf': 0,
               'ccd-temp': -20.0, 'readoutm': '2.0 MHz', 'gain': 1.85, 'readout': 13}


def inv_nanmedian(a):
    return 1 / np.nanmedian(a)


date_list = list(ROOT_PATH.glob('[!.]*'))
for date_path in tqdm(date_list):
    # night date string
    date = date_path.stem

    # open collection
    try:
        bd_collection = ccdp.ImageFileCollection(date_path / 'Calibration', glob_exclude='.*', keywords=keywords)
    except FileNotFoundError:
        print(f'\n{date} no bias-dark found\n')
        continue

    # ----- DEV skip dates ----- #
    date_datetime = datetime.strptime(date, '%Y%m%d')
    if datetime.strptime('20230710', '%Y%m%d') <= date_datetime <= datetime.strptime('20230720', '%Y%m%d'):
        # print(date)
        continue
    # ----- END ----- #

    # BIAS
    # get bias
    bias_collection = bd_collection.filter(**bias_filter)
    bias_files = bias_collection.files

    # TODO skip days without bias entirely - NO DARK OR FLAT EITHER
    if len(bias_files) < 1:
        continue

    # build master bias
    combined_bias = ccdp.combine(bias_files,
                                 clip_extrema=True, nlow=1, nhigh=1,
                                 sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
                                 sigma_clip_func='median', signma_clip_dev_func='mad_std',
                                 unit='adu', output_verify='silentignore')
    combined_bias.meta['combined'] = True
    combined_bias.header['history'] = 'Master built by vvophot/calib pipeline'
    # combined_bias.header['history'] = 'MJD-OBS fixed by astropy/datfix.'

    # save master bias
    (SAVE_PATH / date).mkdir(exist_ok=True)
    combined_bias.write(SAVE_PATH / date / 'master_bias.fit', overwrite=True, output_verify='ignore')

    # # show
    # show_image(combined_bias, cmap='gray', figsize=(4, 8))  ###

    # DARK
    # get dark
    dark_collection = bd_collection.filter(**dark_filter)
    dark_files = dark_collection.files
    dark_times = set(dark_collection.summary['exposure'])

    # build master dark
    for exp_time in sorted(dark_times, reverse=True):
        single_time_dark_files = bd_collection.filter(exposure=exp_time).files
        combined_dark = ccdp.combine(single_time_dark_files,
                                     sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
                                     sigma_clip_func='median', signma_clip_dev_func='mad_std',
                                     unit='adu', output_verify='ignore')
        combined_dark.header['combined'] = True
        combined_dark.header['history'] = 'Master build by vvophot/calib pipeline'
        # combined_dark.header['history'] = 'MJD-OBS fixed by astropy datfix.'

        combined_dark = ccdp.subtract_bias(combined_dark, combined_bias)
        combined_dark.header['history'] = 'Bias subtracted by vvophot/calib pipeline'

        # save bias-subtracted master dark
        combined_dark.write(SAVE_PATH / date / f'master_dark_bias-subtracted_{int(exp_time)}s.fit', overwrite=True,
                            output_verify='ignore')

        # # show
        # show_image(combined_dark, cmap='gray', figsize=(4, 8))  ###

    # # FLAT
    # try:
    #     flat_collection = ccdp.ImageFileCollection(ROOT_PATH / date / 'AutoFlat', glob_exclude='.*', keywords=keywords)
    #     flat_collection = flat_collection.filter(naxis1=2048, naxis2=2048, xbinning=1, ybinning=1)
    # except FileNotFoundError:
    #     continue
    #
    # flat_filters = set(flat_collection.summary['filter'])
    # for filt in flat_filters:
    #     # bd correction
    #     single_filter_flats = []
    #     for file in flat_collection.filter(filter=filt).files:
    #         ccd_data = CCDData.read(file, unit='adu', output_verify='ignore')
    #         ccd_data = ccdp.subtract_bias(ccd_data, combined_bias)
    #         ccd_data = ccdp.subtract_dark(ccd_data, combined_dark, exposure_time='exposure', exposure_unit=u.s,
    #                                       scale=True)
    #         ccd_data.header['history'] = 'Bias and dark correction by vvophot/calib pipeline'
    #         single_filter_flats.append(ccd_data)
    #
    #     # averaging
    #     combined_flat = ccdp.combine(single_filter_flats,
    #                                  method='average', scale=inv_nanmedian,
    #                                  sigma_clip=True, sigma_clip_low_thresh=0.5, sigma_clip_high_thresh=0.5,
    #                                  sigma_clip_func='median', signma_clip_dev_func='mad_std',
    #                                  mem_limit=8e9,
    #                                  unit='adu', output_verify='ignore')
    #
    #     combined_flat.meta['combined'] = True  # meta ###
    #     combined_flat.header['history'] = 'Master build by vvophot/calib pipeline'
    #     # combined_flat.header['history'] = 'MJD-OBS fixed by astropy datfix.'
    #
    #     # (SAVE_PATH / date).mkdir(exist_ok=True)
    #     combined_flat.write(SAVE_PATH / date / f'master_flat_bdcorrected_{filt}.fit', overwrite=True,
    #                         output_verify='ignore')  ###
    #
    #     # show
    #     show_image(combined_flat, cmap='gray', figsize=(4, 8))