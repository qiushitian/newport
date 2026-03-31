#!/usr/bin/env python3
import numpy as np
from astropy import table
from pathlib import Path
from astropy.time import Time
import pytz
from datetime import datetime, timedelta
from newport import *
import matplotlib.pyplot as plt
from target_phot import OUTPUT_DIR


if __name__ == '__main__':
    print(f"Cadence analysis for {OUTPUT_DIR}\n")
    print(f"{'Band':<5} {'Median (d)':<12} {'Mean (d)':<12} {'Min (d)':<10} {'Max (d)':<10} {'N_obs':<6}")
    print("-" * 65)

    all_diffs = {}

    for band in ['B', 'V', 'R', 'I']:
        try:
            binned_table = table.Table.read(OUTPUT_DIR / f'bin_results_{band}.fits')
        except FileNotFoundError:
            continue
            
        # Round JD to nearest 00:00 (12am) Eastern Time
        et = pytz.timezone('US/Eastern')
        
        jd_rounded = []
        for j in binned_table['jd'].data:
            t_utc = Time(j, format='jd')
            dt_et = t_utc.to_datetime(timezone=et)
            
            # Round to nearest midnight
            if dt_et.hour >= 12:
                dt_rounded = datetime(dt_et.year, dt_et.month, dt_et.day) + timedelta(days=1)
            else:
                dt_rounded = datetime(dt_et.year, dt_et.month, dt_et.day)
            
            # Convert back to JD (handles DST correctly)
            jd_rounded.append(Time(et.localize(dt_rounded)).jd)
            
        jd = np.sort(np.unique(jd_rounded))
        diff = np.diff(jd)
        
        if len(diff) == 0:
            print(f"{band:<5} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {len(jd):<6}")
            continue

        all_diffs[band] = diff
        med = np.median(diff)
        avg = np.mean(diff)
        mini = np.min(diff)
        maxi = np.max(diff)

        print(f"{band:<5} {med:<12.2f} {avg:<12.2f} {mini:<10.2f} {maxi:<10.2f} {len(jd):<6}")

    # Plotting
    if all_diffs:
        fig, axs = plt.subplots(nrows=len(all_diffs), figsize=(6, 2 * len(all_diffs)), sharex=True, sharey=True)
        if len(all_diffs) == 1: axs = [axs]
        
        for i, (band, diff) in enumerate(all_diffs.items()):
            color = COLORS.get(band, 'gray')
            axs[i].hist(diff, bins=80, color=color, alpha=0.7, label=f'{band} band')
            axs[i].axvline(np.median(diff), color='black', linestyle='--', alpha=0.5, label=f'Med: {np.median(diff):.2f}d')
            axs[i].set_ylabel('Frequency')
            axs[i].legend(loc='upper right', fontsize=9)
            axs[i].grid(True, alpha=0.3)

        plt.xlabel('Interval between observations (days)')
        plt.tight_layout()
        
        pdf_path = OUTPUT_DIR / 'cadence_histogram.pdf'
        plt.savefig(pdf_path)
        print(f"\nPlot saved to {pdf_path}")
        plt.show()
