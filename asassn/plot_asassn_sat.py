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
    PATH2 = Path('asassn/comp120_sat.csv')

    table = Table.read(PATH, comment='#')
    table2 = Table.read(PATH2, comment='#')

    plt.figure(figsize=(10, 6))

    table = table[table['Filter'] == 'g']
    table2 = table2[table2['Filter'] == 'g']

    # cameras = set(np.unique(table['Camera']))
    # cameras.update(np.unique(table2['Camera']))

    # for camera in cameras:
    #     mask = table['Camera'] == camera
    #     mask2 = table2['Camera'] == camera
    #     plt.errorbar(
    #         Time(table['HJD'][mask], format='jd').to_datetime(),
    #         table['mag'][mask],
    #         yerr=table['mag_err'][mask],
    #         fmt='o', markersize=2, alpha=0.2,
    #         label=camera
    #     )
    #     plt.errorbar(
    #         Time(table2['HJD'][mask2], format='jd').to_datetime(),
    #         table2['mag'][mask2],
    #         yerr=table2['mag_err'][mask2],
    #         fmt='o', markersize=2, alpha=0.2,
    #         label=camera
    #     )

    plt.scatter(
        Time(table['HJD'], format='jd').to_datetime(),
        table['mag'],
        s=2, alpha=0.2,
        label='The real star'
    )
    plt.scatter(
        Time(table2['HJD'], format='jd').to_datetime(),
        table2['mag'],
        s=2, alpha=0.2,
        label='A nearby star'
    )

    # # bin
    # x, y = bin_every_n(table['HJD'], table['mag'], 8)
    # plt.errorbar(Time(x, format='jd').to_datetime(), y, fmt='o', markersize=5, alpha=0.8, markeredgecolor='none')

    _ylim = plt.ylim()
    plt.ylim(_ylim[1], _ylim[0])

    plt.xlabel('Time')
    plt.ylabel('g mag')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_pdf = PATH.with_suffix('.pdf')
    # plt.savefig(output_pdf)
    # print(f"Saved plot to {output_pdf}")
    plt.show()
