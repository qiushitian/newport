#
"""
Do photometry on sources listed in a Gaia star list

Author: Qiushi (Chris) Tian
"""
import newport
from pathlib import Path
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAnnulus, SkyCircularAnnulus, CircularAperture, SkyCircularAperture
from astropy.io import fits
import numpy as np
from astropy.coordinates import ICRS
from photutils.aperture import ApertureStats
from astropy.stats import SigmaClip
from photutils.utils import calc_total_error
import aswcs
from tqdm import tqdm
from astropy.table import QTable
import ccdproc
from astropy.nddata import CCDData, StdDevUncertainty
import concurrent.futures


FIELD = 'TOI-561'

APER_SIZE_FACTOR = 1.4
APSIZE = np.array([14, 50, 90]) * 0.699  # tuple values in PIXEL (converted to arcsec)

BANDS = ['B', 'V', 'R', 'I']

SCI_PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw/TOI 561')
CALIB_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib-2025-no_flat')
WRITE_PATH = Path('tables/list_runs/TOI-561/phot_list_run')
WRITE_PATH.mkdir(parents=True, exist_ok=True)

FLAT_NO_OVERSCAN_DATE, FLAT_WITH_OVERSCAN_DATE = '20230515', '20231102'

DATE_RANGE_START = '20010101'  # 20230314
DATE_RANGE_END = '21230501'  # 20230801

FIRST_OVERSCAN = '20230721'

NANMIN_MAX_TIME = 60  # s


def error_func(data, gain=1.85):
    # return np.sqrt(data)
    # return np.sqrt(data / gain)
    return calc_total_error(data.astype(float), 0, gain)


def get_fwhm_nanmin(aperture_stats: ApertureStats):
    return np.nanmin(aperture_stats.fwhm)


if __name__ == '__main__':
    comp_star_all_bands = set()
    comp_star_all_bands.add(newport.TARGET_GAIA_DR3[FIELD])
    for band in ['B', 'V', 'R', 'I']:
        for star in newport.get_comparison_star_list(FIELD, band):
            comp_star_all_bands.add(star)

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # Execute the query
    job = Gaia.launch_job(
        "SELECT source_id, ra, dec FROM gaiadr3.gaia_source "
        f"WHERE source_id IN ({','.join(map(str, comp_star_all_bands))})"
    )
    gaia_table = job.get_results()

    positions = SkyCoord(gaia_table['ra'], gaia_table['dec'], frame=ICRS)
    aperture = SkyCircularAperture(positions, r=APSIZE[0] * u.arcsec)
    annuli = SkyCircularAnnulus(positions, r_in=APSIZE[1] * u.arcsec, r_out=APSIZE[2] * u.arcsec)

    phot_table = QTable(
        names=('band', 'night', 'time', 'jd', 'airmass', 'exptime', 'skytemp'),
        dtype=('S', 'i', 'S', 'f8', 'f8', 'f8', 'f8')
    )

    for source_id in gaia_table['SOURCE_ID']:
        phot_table[str(source_id)] = [None] * len(phot_table)

    err_table = phot_table.copy()

    for source_id in gaia_table['SOURCE_ID']:
        phot_table['err_' + str(source_id)] = [None] * len(phot_table)

    # pixel_table = phot_table.copy()

    # list of dates
    date_list = list(SCI_PATH.glob('[!.]*'))
    date_list.sort()

    # keep only dates in range
    date_list = [date_path for date_path in date_list if DATE_RANGE_START < date_path.name < DATE_RANGE_END]

    # Initialize alternative progress bar
    date_count = 0

    # aper_size skip count
    aper_size_skip_count = 0

    for date_path in tqdm(date_list):
    # for date_path in date_list:  # no tqdm
        # date string
        date = date_path.name

        # determine flat with overscan or not
        flat_date = FLAT_NO_OVERSCAN_DATE if date < FIRST_OVERSCAN else FLAT_WITH_OVERSCAN_DATE

        # alternative progress bar
        date_count += 1
        # print(f'{date_count}\t{date}')

        # open calibration files
        try:
            with fits.open(
                    CALIB_PATH / date / 'master_bias.fit', output_verify='ignore'
            ) as b:
                ccd_bias_data = CCDData(  # TODO confirm stddecuncert is the right uncert
                    b[0].data, StdDevUncertainty(b[2].data, unit='adu'), b[1].data, meta=b[0].header, unit='adu'
                )
        except (ValueError, FileNotFoundError):
            print(f'\n{date} dropped because of missing bias.')
            continue

        ccd_dark_data = None
        for dark_t in range(600, 0, -1):
            try:
                with fits.open(
                        CALIB_PATH / date / f'master_dark_bias-subtracted_{dark_t}s.fit', output_verify='ignore'
                ) as d:
                    ccd_dark_data = CCDData(
                        d[0].data, StdDevUncertainty(d[2].data, unit='adu'), d[1].data, meta=d[0].header, unit='adu'
                    )
                # print(f'\n{date}\t{dark_t}')
                break
            except FileNotFoundError:
                # print(f'\n{date}\t{dark_t}dark not found')
                pass

        try:
            if ccd_dark_data is None:
                raise NameError
        except NameError:
            print(f'\n{date} dropped because of missing dark')
            continue

        file_list = list(date_path.glob("[!._]*.wcs"))

        # for file in tqdm(file_list, desc=f'{date_count}\t/ {len(date_list)}\t{date}'):
        for file in file_list:  # no tqdm
            # print(file)

            # read WCS
            try:
                wcs = aswcs.ini_to_wcs(file.with_suffix('.ini'))
                with fits.open(file.with_suffix('.fts'), output_verify='warn') as hdul:
                    data = CCDData(
                        hdul[0].data, unit='adu', wcs=wcs, meta=hdul[0].header,
                        uncertainty=StdDevUncertainty(error_func(hdul[0].data), unit='adu')
                    )

                    # bias
                    # print(f'Bias\t{date}\t{file.name}')
                    # start_time = time.perf_counter()
                    data = ccdproc.subtract_bias(data, ccd_bias_data)
                    # end_time = time.perf_counter()
                    # print(f"\t\t\t{end_time - start_time} s\n")

                    # dark
                    # print(f'Dark\t{date}\t{file.name}')
                    # start_time = time.perf_counter()
                    data = ccdproc.subtract_dark(
                        data, ccd_dark_data, scale=True, exposure_time='EXPTIME', exposure_unit=u.s
                    )
                    # end_time = time.perf_counter()
                    # print(f"\t\t\t{end_time - start_time} s\n")

                    # flat
                    with fits.open(
                            CALIB_PATH / flat_date /
                            f'master_flat_bdcorrected_{data.meta["FILTER"]}_median_noNorm.fit',
                            output_verify='ignore'
                    ) as flat_hdul:
                        ccd_flat_data = CCDData(
                            flat_hdul[0].data, StdDevUncertainty(flat_hdul[2].data, unit='adu'),
                            flat_hdul[1].data, unit='adu'
                        )
                    data = ccdproc.flat_correct(data, ccd_flat_data)

                    # centroiding and creating invalid aperture mask
                    raw_aperstats = ApertureStats(data, aperture)
                    centroids = raw_aperstats.centroid
                    invalid_aper = np.isnan(centroids[:, 0])

                    # skip if all are invalid
                    if np.all(invalid_aper):
                        print(f"\nSkipping item {file} because all apertures are invalid.\n")
                        continue

                    # setting real aperture size
                    aper_size = 2
                    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    #     future = executor.submit(get_fwhm_nanmin, raw_aperstats[~invalid_aper])
                    #     try:
                    #         aper_size = future.result(timeout=NANMIN_MAX_TIME)
                    #     except concurrent.futures.TimeoutError:
                    #         print(f"Skipping item {file} due to timeout.")
                    #         aper_size_skip_count += 1
                    #         continue
                    aper_size = get_fwhm_nanmin(raw_aperstats[~invalid_aper])  # non-parallel version
                    if not np.isfinite(aper_size):
                        continue
                    aper_size = aper_size.to(u.pixel).value
                    aper_size = 2 if aper_size < 2 else aper_size
                    aper_size *= APER_SIZE_FACTOR
                    print(f'aper_size = {aper_size}')  # TODO DEV

                    # get background
                    sigclip = SigmaClip(sigma=3.0, maxiters=10)
                    bkg_stats = ApertureStats(
                        data.data,
                        CircularAnnulus(
                            centroids[~invalid_aper], r_in=aper_size * 1.7, r_out=aper_size * 1.7 * 3 / 2 + .0001
                        ),
                        sigma_clip=sigclip,
                        error=data.uncertainty.array  # data_err
                    )

                    bkg = bkg_stats.mean
                    # bkg_err = bkg_stats.sum_err.value / bkg_stats.sum_aper_area.value
                    bkg_invalid = ~np.isfinite(bkg)
                    bkg[bkg_invalid] = 0.0  # `local_bkg` doesn't allow inf or NaN
                    # bkg_err[bkg_invalid] = 0.0

                    # photometry
                    aper_stats_bkgsub = ApertureStats(
                        data.data,
                        CircularAperture(centroids[~invalid_aper], r=aper_size),
                        local_bkg=bkg,
                        error=data.uncertainty.array  # data_err
                    )

                    # TODO ... todo what?
                    phot_result_list = []
                    # pixel_pos_list = []
                    phot_err_list = []
                    for ap_stats, ap_stats_bkg in zip(aper_stats_bkgsub, bkg_stats):
                        phot_result_list.append(ap_stats.sum)
                        # pixel_pos_list.append(i.centroid)
                        phot_err_list.append(
                            np.sqrt(
                                ap_stats.sum_err ** 2
                                + (- ap_stats.sum_aper_area / ap_stats_bkg.sum_aper_area * ap_stats_bkg.sum_err) ** 2
                            )
                        )
                        # sq_count = np.sqrt(ap_stats.sum)  # DEV
                        # if sq_count > ap_stats.sum_err:
                        #     print(ap_stats.id, np.sqrt(ap_stats.sum), ap_stats.sum_err)

                    phot_result_list = np.array(phot_result_list)
                    phot_err_list = np.array(phot_err_list)

                    # mask off apertures without measurable background
                    phot_result_list[bkg_invalid] = np.nan
                    phot_err_list[bkg_invalid] = np.nan

                    # reject saturation TODO replace after calib is implemented
                    saturated_mask = aper_stats_bkgsub.max + bkg > 65535 - 1000
                    phot_result_list[saturated_mask] = np.nan
                    phot_err_list[saturated_mask] = np.nan  # TODO is this needed?

                    # reshape
                    phot_result_reshape = np.full(invalid_aper.shape, np.nan)
                    phot_result_reshape[~invalid_aper] = phot_result_list
                    phot_err_reshape = np.full(invalid_aper.shape, np.nan)
                    phot_err_reshape[~invalid_aper] = phot_err_list

                    # save to table
                    table_leader = [
                        data.meta['FILTER'],
                        date,
                        data.meta['DATE-OBS'],
                        data.meta['JD'],
                        data.meta['AIRMASS'],
                        data.meta['EXPOSURE'],
                        data.meta['SKYTEMP'],
                    ]
                    phot_table.add_row(table_leader + phot_result_reshape.tolist() + phot_err_reshape.tolist())
                    err_table.add_row(table_leader + phot_err_reshape.tolist())
                    # pixel_table.add_row(table_leader + pixel_pos_list)

            except Exception as e:
                print(f'\n\n\033[91mTerminating on: {file}\033[0m\n')
                raise e  # This line is good for debug because it terminates the whole thing when 1 file fails

                # These two lines below are good for actual running because it doesn't terminate the whole thing;
                # instead, it prints which file(s) fail(s)
                # print(str(e))
                # print(f'{file} dropped\n')

            del wcs
        del ccd_bias_data
        del ccd_dark_data

    # print skip count dur to ApertureStats timeout
    print(f'aper_size_skip_count = {aper_size_skip_count}')

    target_name = SCI_PATH.name.replace('HD ', 'HD_').replace('TOI ', 'TOI-')
    try:
        phot_table.write(WRITE_PATH / f'phot_w_err_{target_name}.fits')
        err_table.write(WRITE_PATH / f'err_{target_name}.fits')
        # pixel_table.write(WRITE_PATH / f'pixel_position_{target_name}.fits')
    except OSError:
        phot_table.write(WRITE_PATH / f'phot_w_err_{target_name}_2.fits')
        err_table.write(WRITE_PATH / f'err_{target_name}_2.fits')
        # pixel_table.write(WRITE_PATH / f'pixel_position_{target_name}_2.fits')
