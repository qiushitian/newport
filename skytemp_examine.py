#!/usr/bin/env python3
"""
Sky Temperature Examiner: Plots 'SKYTEMP' against 'OBS-TIME' from FITS headers.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import numpy as np

PATH = Path("/Volumes/emlaf/westep-transfer/mountpoint/space-raw/HD 191939/20230627")

if __name__ == "__main__":
    fts_files = sorted(list(PATH.glob('*.wcs')))
    fts_files = [f for f in fts_files if not f.name.startswith('.')]

    times = []
    temps = []

    print(f"Processing {len(fts_files)} files...")
    for f in fts_files:
        with fits.open(f.with_suffix('.fts')) as hdul:
            header = hdul[0].header
            obs_time = header['DATE-OBS']
            sky_temp = header['SKYTEMP']
            
            t = Time(obs_time)
            times.append(t.datetime)
            temps.append(float(sky_temp))

    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    temps = np.array(temps)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(times, temps, 'o-', markersize=4, alpha=0.7)
    plt.xlabel('Observation Time')
    plt.ylabel('Sky Temperature (SKYTEMP)')
    plt.title(f'Sky Temperature over Time\nDirectory: {PATH.name}')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    # filename = "skytemp_plot.png"
    # plt.savefig(filename, dpi=150, bbox_inches='tight')
    # print(f"Plot saved to {filename}")
    plt.show()
