#!/usr/bin/env python3
"""
Take photometry_*.fits tables and do relative photometry,
output relphot_*.fits and binned_*.fits tables
"""

import matplotlib.pyplot as plt
import astropy.table as table
import astropy.stats as stats
from newport import *

COLORS = {'B': 'C0', 'V': 'C2', 'R': 'C3', 'I': 'maroon'}
MARKERS = {'B': 'o', 'V': 'x', 'R': 's', 'I': 'D'}

PLOT_HST = False
SUFFIX = ''
SUFFIX = '_rms_nonvar'

if PLOT_HST:
    _, hst, _ = hst_visits()

for fn in target_fn:
    plotted = False
    fig, axs = plt.subplots(nrows=4, figsize=(10, 10), sharex=True, dpi=300)
    for i, band in enumerate(['B', 'V', 'R', 'I']):
        try:
            binned_table = table.Table.read(f'tables/binned/binned_{fn}_{band}{SUFFIX}.fits')
            unbinned_table = table.Table.read(f'tables/relphot/relphot_{fn}_{band}{SUFFIX}.fits')
        except FileNotFoundError as e:
            print(str(e))
            continue

        # after = binned_table['jd'] > 2460180
        # after = binned['jd'] > 0
        # ax.plot(binned[after]['jd'] - 2400000.5, binned[after]['phot'] / binned_norm, MARKERS[band], c=COLORS[band])
        axs[i].grid(axis='y', linestyle=':', alpha=0.5)
        axs[i].errorbar(binned_table['jd'] - 2400000.5,
                    binned_table[target_gaia_dr3[fn]],
                    fmt=MARKERS[band], c=COLORS[band], label=band)
        axs[i].plot(unbinned_table['jd'] - 2400000.5,
                    unbinned_table[target_gaia_dr3[fn]],
                    MARKERS[band], c=COLORS[band], ms=1, alpha=0.3)
        clipped_std = np.nanstd(
            stats.sigma_clip(binned_table[target_gaia_dr3[fn]], sigma=5, maxiters=None, masked=False)
        )
        axs[i].set_ylim(1 - clipped_std * 5, 1 + clipped_std * 5)
        _xlim = axs[i].get_xlim()
        if PLOT_HST:
            plot_hst(fn.replace('_', ''), hst, axs[i], 'C7')
        axs[i].set_xlim(_xlim)

        plotted = True

    if not plotted:
        continue
    fig.supxlabel('MJD', y=0.06)
    fig.supylabel('Relative Flux', x=0.05)
    fig.suptitle(fn, y=0.91)
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9))
    # ax.set_ylim(0.96, 1.03)

    plt.savefig(f'fig/{fn}{SUFFIX}.pdf')
    plt.savefig(f'fig/png/{fn}{SUFFIX}.png')
    # plt.show()
    #
    # break


        # fig, ax = plt.subplots(figsize=(6.4 * 1.5, 3.7 * 1.5), dpi=150)
        #
        # for n, band in enumerate(BANDS):
        #     data = table[table['band'] == band]
        #     norm = np.median(data['phot'])
        #     after = data['jd'] > 2460024
        #     # after = data['jd'] > 0
        #     ax.plot(data[after]['jd'] - 2400000.5, data[after]['phot'] / norm - n * 0.1, MARKERS[band], alpha=0.05,
        #             markersize=4, c=COLORS[band])
        #
        #     # plot binned
        #     binned = dat_binned[dat_binned['band'] == band]
        #     binned_norm = np.median(binned['phot'])
        #     after = binned['jd'] > 2460024
        #     # after = binned['jd'] > 0
        #     ax.errorbar(binned[after]['jd'] - 2400000.5,
        #                 binned[after]['phot'] / binned_norm - n * 0.1,
        #                 binned[after]['photerr'] / binned_norm, elinewidth=1.5,
        #                 fmt=MARKERS[band], linewidth=0.3, markersize=5, c=COLORS[band], label=band)
        # ax.set_xlim(ax.get_xlim())
        # plot_hst('TOI-1201', hst, ax, 'C7')
        # ax.set_xlabel('MJD')
        # ax.set_ylabel('Relative Flux')
        # ax.set_title(r'HD 191939 $BVRI$ Photometry')
        # ax.legend(ncol=2, bbox_to_anchor=(0.75, 0.87))
        # # ax.set_xlim(60015,60110)
        # # ax.set_yticks(np.arange(0.3, 1.2, 0.1))
        # ax.set_ylim(0.55, 1.2)
        # ax.grid(axis='y', linestyle=':', alpha=0.5, which='both')
        # # plt.minorticks_on()
        #
        # plt.show()