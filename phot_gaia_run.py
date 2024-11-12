from pathlib import Path
from astropy.coordinates import SkyCoord

coord = SkyCoord(ra='326.851662d', dec='62.753816d')
PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw/TOI 178/')


import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from photutils.aperture import CircularAnnulus, SkyCircularAnnulus, CircularAperture, SkyCircularAperture
from astropy.coordinates import Angle

from astropy.io import fits
from astropy.io import ascii
from astropy.wcs import WCS

from astropy.visualization import simple_norm

import matplotlib.pyplot as plt

import numpy as np

from astropy.coordinates import FK5, ICRS

from photutils.aperture import ApertureStats
from photutils.aperture import aperture_photometry

from astropy.stats import SigmaClip, mad_std

from photutils.utils import calc_total_error

import aswcs


from tqdm import tqdm

from astropy.time import Time
from astropy.table import QTable

import ccdproc as ccdp
from astropy.nddata import CCDData, StdDevUncertainty

from astropy.timeseries import LombScargle

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"


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

apsize = np.array([14, 50, 90]) * 0.699 # tuple values in PIXEL (converted to arcsec)

positions = SkyCoord(gaia_table['ra'], gaia_table['dec'], frame=ICRS)
aperture = SkyCircularAperture(positions, r=apsize[0] * u.arcsec)
annuli = SkyCircularAnnulus(positions, r_in=apsize[1] * u.arcsec, r_out=apsize[2] * u.arcsec)


APER_SIZE_FACTOR = 1.4
BIAS = 940
BANDS = ['B', 'V', 'R', 'I']

phot_table = QTable(names=('band', 'night', 'time', 'jd', 'airmass'),
                    dtype=('S', 'i', 'S', 'f8', 'f8'))
for source_id in gaia_table['SOURCE_ID']:
    phot_table[str(source_id)] = [None] * len(phot_table)
pixel_table = phot_table.copy()

# multi band - exec cell

# list of dates
date_list = list(PATH.glob('[!.]*'))

# Initialize the progress bar
progress_bar = tqdm(total=len(date_list))

for date_path in date_list:
    # date string
    date = date_path.name

    # # open calibration files
    # try:
    #     b = fits.open(BIAS_PATH / date / 'master_bias.fit', output_verify='warn')
    #     ccd_bias_data = CCDData(b[0].data,
    #                             StdDevUncertainty(b[2].data, unit='adu'),
    #                             b[1].data,
    #                             unit='adu')
    # except (ValueError, FileNotFoundError):
    #     progress_bar.update(1)
    #     continue

    for file in date_path.glob("[!._]*.wcs"):
        # read WCS
        try:
            wcs = aswcs.ini_to_wcs(file.with_suffix('.ini'))
        
            with fits.open(file.with_suffix('.fts'), output_verify='warn') as f:
                # # get data and bias subtraction
                # data = ccdp.subtract_bias(CCDData(f[0].data, unit='adu'), ccd_bias_data).data
                data = CCDData(f[0].data, unit='adu', wcs=wcs)
                
                # centroiding and setting aperture size
                raw_aperstats = ApertureStats(data, aperture)
                centroids = raw_aperstats.centroid
                invalid_aper = np.isnan(centroids[:, 0])
                aper_size = np.nanmin(raw_aperstats.fwhm.to(u.pixel).value) * APER_SIZE_FACTOR
            
                # get background
                sigclip = SigmaClip(sigma=3.0, maxiters=10)
                bkg_stats = ApertureStats(data,
                                          CircularAnnulus(centroids[~invalid_aper],
                                                          r_in=aper_size * 1.7,
                                                          r_out=aper_size * 1.7 * 3 / 2 + .0001),
                                          sigma_clip=sigclip)
                
                bkg = bkg_stats.mean.value
                bkg_invalid = ~np.isfinite(bkg)
                bkg[bkg_invalid] = 0.0
                
                # photometry
                aper_stats_bkgsub = ApertureStats(
                    data.data,
                    CircularAperture(centroids[~invalid_aper], r=aper_size),
                    local_bkg=bkg,
                    error=calc_total_error(f[0].data.astype(np.float64), 0, 1.85)
                )
                
                # TODO
                phot_result_list = []
                # pixel_pos_list = []
                for i in aper_stats_bkgsub:
                    phot_result_list.append(i.sum)
                    # pixel_pos_list.append(i.centroid)
                    
                phot_result_list = np.array(phot_result_list)
                
                phot_result_list[bkg_invalid] = np.nan
                                
                # reject saturation
                phot_result_list[aper_stats_bkgsub.max > 64000] = np.nan
                
                # reshape
                phot_result_reshape = np.full(invalid_aper.shape, np.nan)
                phot_result_reshape[~invalid_aper] = phot_result_list
                
                # save to table
                table_leader = [f[0].header['FILTER'],
                                date,
                                f[0].header['DATE-OBS'],
                                f[0].header['JD'],
                                f[0].header['AIRMASS']]
                phot_table.add_row(table_leader + phot_result_reshape.tolist())
                # pixel_table.add_row(table_leader + pixel_pos_list)

        except Exception as e:
            # raise e
            print(str(e))
            print(f'{file} dropped\n')
            
    progress_bar.update(1)
progress_bar.close()

target_name = PATH.name.replace(' ', '_')
phot_table.write(f'photometry_{target_name}.fits')
# pixel_table.write(f'pixel_position_{target_name}.fits')
