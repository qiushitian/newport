#
"""

Author: Qiushi (Chris) Tian
Created: 2025 May 21 from phot_gaia_run.py
"""
import numpy as np
from photutils.aperture import ApertureStats
from photutils.utils import calc_total_error


def get_fwhm_nanmin(aperture_stats: ApertureStats):
    return np.nanmin(aperture_stats.fwhm)


def error_func(data, gain=1.85):
    # return np.sqrt(data)
    # return np.sqrt(data / gain)
    return calc_total_error(data.astype(float), 0, gain)


if __name__ == "__main__":
    from pathlib import Path
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.gaia import Gaia
    from astropy.coordinates import SkyCoord
    from photutils.aperture import CircularAnnulus, SkyCircularAnnulus, CircularAperture, SkyCircularAperture
    from astropy.io import fits
    from astropy.coordinates import FK5, ICRS
    from astropy.stats import SigmaClip, mad_std
    import aswcs
    from tqdm import tqdm
    from astropy.table import QTable
    import ccdproc as ccdp
    from astropy.nddata import CCDData, StdDevUncertainty
    from astroquery.utils.tap.core import TapPlus
    import concurrent.futures
    import newport

    NANMIN_MAX_TIME = 200  # s

    coord = newport.TARGET_SKYCOORD['HD_191939']
    SCI_PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw/HD 191939')
    CALIB_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib-2025-no_flat')
    WRITE_PATH = Path('tables/list_runs/191939/phot_gaia_run')
    WRITE_PATH.mkdir(parents=True, exist_ok=True)

    FLAT_NO_OVERSCAN_DATE, FLAT_WITH_OVERSCAN_DATE = '20230515', '20231102'
    FIRST_OVERSCAN = '20230721'

    DATE_RANGE_START = 20010101  # 20230314
    DATE_RANGE_END = 21000101  # 20230801

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    SIMBAD_TAP = TapPlus(url="http://simbad.u-strasbg.fr/simbad/sim-tap")

    # Set the search parameters
    search_radius = 25 * u.arcmin  # Replace with your desired radius
    mag_limit = 15  # Replace with your desired magnitude limit

    # Query Gaia around the coord within the radius
    query = f"""
    SELECT TOP 1000
        source_id, ra, dec, phot_g_mean_mag,
        phot_bp_mean_mag, phot_rp_mean_mag, 
        bp_rp, bp_g, g_rp, -- Precomputed color indices
        teff_gspphot,  -- Effective temperature
        phot_variable_flag,
        classprob_dsc_combmod_star
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {search_radius.to(u.deg).value})
    ) = 1
    AND phot_g_mean_mag < {mag_limit}
    AND (classprob_dsc_combmod_star > 0.9) -- Probability it's a star
    """

    # Execute the query
    job = Gaia.launch_job(query)
    gaia_table = job.get_results()

    # keep only comp stars with mag
    mask_gaia_table = np.zeros((len(gaia_table)), dtype=bool)
    # mask_gaia_table[[10, 15, 100]] = True  # TODO DEV
    for i, row in enumerate(tqdm(gaia_table, desc='Checking mags')):
        one_mag_table = SIMBAD_TAP.launch_job(
            f"SELECT B,V,R,I FROM allfluxes JOIN ident USING(oidref) WHERE id = 'Gaia DR3 {row['SOURCE_ID']}';"
        ).get_results()
        if len(one_mag_table) > 0:
            mask_gaia_table[i] = True

    apsize = np.array([14, 50, 90]) * 0.699 # tuple values in PIXEL (converted to arcsec)

    gaia_table = gaia_table[mask_gaia_table]
    print(f'\nNumber of comp stars with mags: {len(gaia_table)}\n')  # TODO DEV

    positions = SkyCoord(gaia_table['ra'], gaia_table['dec'], frame=ICRS)
    aperture = SkyCircularAperture(positions, r=apsize[0] * u.arcsec)
    annuli = SkyCircularAnnulus(positions, r_in=apsize[1] * u.arcsec, r_out=apsize[2] * u.arcsec)

    APER_SIZE_FACTOR = 1.4
    # BIAS = 940
    # BANDS = ['B', 'V', 'R', 'I']


    phot_table = QTable(names=('band', 'night', 'time', 'jd', 'airmass', 'exptime'),
                        dtype=('S', 'i', 'S', 'f8', 'f8', 'f2'))
    for source_id in gaia_table['SOURCE_ID']:
        phot_table[str(source_id)] = [None] * len(phot_table)

    err_table = phot_table.copy()

    for source_id in gaia_table['SOURCE_ID']:
        phot_table['err_' + str(source_id)] = [None] * len(phot_table)

    # pixel_table = phot_table.copy()


    # list of dates
    date_list = list(SCI_PATH.glob('[!.]*'))
    date_count = 0

    # aper_size skip count
    aper_size_skip_count = 0

    for date_path in tqdm(date_list, desc='phot run'):
    # for date_path in date_list:  # no tqdm
        # date string
        date = date_path.name

        # determine flat with overscan or not
        flat_date = FLAT_NO_OVERSCAN_DATE if date < FIRST_OVERSCAN else FLAT_WITH_OVERSCAN_DATE

        # alternative progress bar
        date_count += 1
        # print(f'{date_count}\t{date}')

        # date int and skip
        try:
            date_int = int(date)
        except ValueError:  # if not int date
            continue
        if date_int < DATE_RANGE_START or date_int > DATE_RANGE_END:
            continue

        # open calibration files
        try:
            with fits.open(
                    CALIB_PATH / date / 'master_bias.fit', output_verify='ignore'
            ) as b:
                ccd_bias_data = CCDData(  # TODO confirm stddecuncert is the right uncert
                    b[0].data, StdDevUncertainty(b[2].data, unit='adu'), b[1].data, meta=b[0].header, unit='adu'
                )
        except (ValueError, FileNotFoundError):
            print(f'\n{date} dropped because of missing bias.\n')
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
            # read WCS
            try:
                wcs = aswcs.ini_to_wcs(file.with_suffix('.ini'))
                with fits.open(file.with_suffix('.fts'), output_verify='warn') as hdul:
                    hdu = hdul[0]
                    data_err = error_func(hdu.data)
                    data = CCDData(
                        hdu.data, unit='adu', wcs=wcs, meta=hdu.header, uncertainty=StdDevUncertainty(data_err, unit='adu')
                    )

                    # print(f'Bias\t{date}\t{file.name}')
                    # start_time = time.perf_counter()
                    data = ccdp.subtract_bias(data, ccd_bias_data)
                    # end_time = time.perf_counter()
                    # print(f"\t\t\t{end_time - start_time} s\n")

                    # print(f'Dark\t{date}\t{file.name}')
                    # start_time = time.perf_counter()
                    data = ccdp.subtract_dark(data, ccd_dark_data, scale=True, exposure_time='EXPTIME', exposure_unit=u.s)
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
                    data = ccdp.flat_correct(data, ccd_flat_data)

                    data_err = data.uncertainty.array

                    # centroiding and invalid aper
                    raw_aperstats = ApertureStats(data, aperture)
                    centroids = raw_aperstats.centroid
                    invalid_aper = np.isnan(centroids[:, 0])

                    # skip if all are invalid
                    if np.all(invalid_aper):
                        print(f"\nSkipping item {file} because all apertures are invalid.\n")
                        continue

                    # setting aperture size with parallelization
                    aper_size = 2
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(get_fwhm_nanmin, raw_aperstats[~invalid_aper])
                        try:
                            aper_size = future.result(timeout=NANMIN_MAX_TIME)
                        except concurrent.futures.TimeoutError:
                            print(f"Skipping item {file} due to timeout.")
                            aper_size_skip_count += 1
                            continue
                    # aper_size = get_nanmin(raw_aperstats[~invalid_aper].fwhm)  # non-parallel version
                    if not np.isfinite(aper_size):
                        print(f"\nSkipping item {file} because of infinite/NaN aperture size.\n")
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
                        error=data_err
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
                        error=data_err
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
                    phot_result_list[aper_stats_bkgsub.max + bkg > 64000] = np.nan
                    phot_err_list[aper_stats_bkgsub.max + bkg > 64000] = np.nan  # TODO is this needed?

                    # reshape
                    phot_result_reshape = np.full(invalid_aper.shape, np.nan)
                    phot_result_reshape[~invalid_aper] = phot_result_list
                    phot_err_reshape = np.full(invalid_aper.shape, np.nan)
                    phot_err_reshape[~invalid_aper] = phot_err_list

                    # save to table
                    table_leader = [
                        hdu.header['FILTER'],
                        date,
                        hdu.header['DATE-OBS'],
                        hdu.header['JD'],
                        hdu.header['AIRMASS'],
                        hdu.header['EXPOSURE'],
                    ]
                    phot_table.add_row(table_leader + phot_result_reshape.tolist() + phot_err_reshape.tolist())
                    err_table.add_row(table_leader + phot_err_reshape.tolist())
                    # pixel_table.add_row(table_leader + pixel_pos_list)

            except Exception as e:
                print(f'\n\n\033[91mTerminating on: {file}\033[0m\n')
                print(f'aper_size_skip_count = {aper_size_skip_count}\n')
                # if input('Write table?').lower().startswith('y'):
                #     target_name = SCI_PATH.name.replace('HD ', 'HD_').replace('TOI ', 'TOI-')
                #     phot_table.write(WRITE_PATH / f'phot_w_err_{target_name}.fits')
                #     err_table.write(WRITE_PATH / f'err_{target_name}.fits')
                raise e  # This line is good for debug because it terminates the whole thing when 1 file fails
                # print(str(e))  # These two lines are good for actual running because it doesn't terminal the whole thing
                # print(f'{file} dropped\n')  # Instead, it prints which file(s) fail(s)

            del wcs
        del ccd_bias_data
        del ccd_dark_data

    # print skip count dur to ApertureStats timeout
    print(f'aper_size_skip_count = {aper_size_skip_count}')

    target_name = SCI_PATH.name.replace('HD ', 'HD_').replace('TOI ', 'TOI-')
    phot_table.write(WRITE_PATH / f'phot_w_err_{target_name}.fits')
    err_table.write(WRITE_PATH / f'err_{target_name}.fits')
    # pixel_table.write(f'pixel_position_{target_name}.fits')
