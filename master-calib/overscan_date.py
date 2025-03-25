#!
"""
Find which date overscan got turned on

From: flat_skytemp.py
Author: Qiushi (Chris) Tian
Created: 2025-03-21
"""
from pathlib import Path
import ccdproc as ccdp
# from tqdm import tqdm
# from astropy.table import Table
# from datetime import datetime

SCI_PATH = Path('/Volumes/emlaf/westep-transfer/mountpoint/space-raw')
BD_PATH = Path('/Volumes/emlaf/westep-transfer/mastercalib-2025-no_flat')

sci_keywords = ['imagetyp', 'jd', 'date-obs', 'skytemp', 'naxis1', 'naxis2']


if __name__ == '__main__':
    # t = Table(names=['jd', 'skytemp'])

    date_list = sorted(list(BD_PATH.glob('202*')))

    for date_path in date_list:
        # night date string
        date = date_path.stem

        # ----- DEV skip dates ----- #
        # date_datetime = datetime.strptime(date, '%Y%m%d')
        # if datetime.strptime('20230710', '%Y%m%d') <= date_datetime <= datetime.strptime('20230720', '%Y%m%d'):
        #     print(date)
        #     continue
        # ----- END ----- #

        # open sci images for skytemp
        try:
            sci_collection = ccdp.ImageFileCollection(
                filenames=[f for f in SCI_PATH.glob(f'*/{date}/*.fts') if not f.name.startswith('.')],
                location=SCI_PATH, keywords=sci_keywords
            )
        except FileNotFoundError as e:
            raise e

        if len(sci_collection.files) > 0:
            # sci_collection.sort('jd')
            # t.add_row([datetime.strptime(date, "%Y%m%d").date(), sci_collection.values('skytemp')[0]])
            # t.add_row([sci_collection.values('jd')[0], sci_collection.values('skytemp')[0]])
            print(date, sci_collection.values('naxis1')[0])

    # t.write('flat_skytemp.fits')
