#!/usr/bin/env python3
"""
Plotting ASAS-SN SkyPatrol v1 Saturated Stars light curves.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table
import numpy as np
from astropy.time import Time


def bin_every_n(x, y, n, bin_func=np.median, debug_print=False):
    """
    Bins data by every n points.
    """
    _sort_idx = np.argsort(x)
    x = x[_sort_idx]
    y = y[_sort_idx]

    i = 0
    bx, by = [], []
    while True:
        this_bin_start = n * i
        next_bin_start = n * (i + 1)
        if next_bin_start > len(y):
            break
        bx.append(bin_func(x[this_bin_start : next_bin_start]))
        by.append(bin_func(y[this_bin_start : next_bin_start]))
        i += 1
    return np.array(bx), np.array(by)


if __name__ == "__main__":
    # # Plot all csv files in the current directory
    # csv_files = list(Path('./asassn').glob('*.csv'))
    # for csv_file in csv_files:
    #     print(f"Processing {csv_file}...")
    #     plot_asassn(csv_file)

    PATH = Path('asassn/light_curve_afcf75ed-3b1c-4391-a4f4-e20c924cc338.csv')

    table = Table.read(PATH, comment='#')

    plt.figure(figsize=(10, 6))

    table = table[table['Filter'] == 'g']

    # for camera in np.unique(table['Camera']):
    #     mask = table['Camera'] == camera
    plt.errorbar(
        Time(table['HJD'], format='jd').to_datetime(), table['mag'],
        # yerr=table['mag_err'],
        fmt='o', markersize=2, alpha=0.2  # , label=f'Camera {camera}'
    )

    # bin
    x, y = bin_every_n(table['HJD'], table['mag'], 8)
    plt.errorbar(Time(x, format='jd').to_datetime(), y, fmt='o', markersize=5, alpha=0.8, markeredgecolor='none')

    _ylim = plt.ylim()
    plt.ylim(_ylim[1], _ylim[0])

    plt.xlabel('Time')
    plt.ylabel('g mag')
    plt.title(f'ASAS-SN Light Curve: {PATH.name}')
    # plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_pdf = PATH.with_suffix('.pdf')
    plt.savefig(output_pdf)
    print(f"Saved plot to {output_pdf}")
    plt.show()
