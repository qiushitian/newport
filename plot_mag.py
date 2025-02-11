#!/usr/bin/env python3
"""
Take photometry_*.fits tables and do relative photometry,
output relphot_*.fits and binned_*.fits tables
"""

import matplotlib.pyplot as plt
import astropy.table as table
import astropy.stats as stats
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.patches as patches
from newport import *

COLORS = {'B': 'C0', 'V': 'C2', 'R': 'C3', 'I': 'maroon'}
MARKERS = {'B': 'o', 'V': '>', 'R': 's', 'I': 'D'}

PLOT_HST = True
CUTOFF = Time('2023-03-14')
N_SIG = 3
SUFFIX = ''


for fn in TARGET_FN:
    # TODO DEV
    if fn != 'HD_86226':
        continue

    if PLOT_HST:
        wfc3, stis = get_hst(fn, path='xml/HST-17192-visit-status-20250205.xml')

    plotted = False
    fig, axs = plt.subplots(nrows=4, figsize=(6, 6.5), sharex='all', dpi=300)
    for i, band in enumerate(['B', 'V', 'R', 'I']):
        try:
            binned_table = table.Table.read(f'tables/mag/binned_mag_{fn}_{band}{SUFFIX}.fits')
            unbinned_table = table.Table.read(f'tables/mag/mag_{fn}_{band}{SUFFIX}.fits')
        except FileNotFoundError as e:
            print(str(e))
            continue

        # cutoff
        time_binned = Time(binned_table['jd'], format='jd')
        time_unbinned = Time(unbinned_table['jd'], format='jd')

        after_binned = time_binned > CUTOFF
        after_unbinned = time_unbinned > CUTOFF

        time_binned = time_binned[after_binned]
        time_unbinned = time_unbinned[after_unbinned]

        data_binned = binned_table['combined'].data[after_binned]
        data_unbinned = unbinned_table['combined'].data[after_unbinned]
        # end of cutoff

        axs[i].errorbar(
            time_binned.to_datetime(), data_binned, yerr=0.02,
            fmt=MARKERS[band], c=COLORS[band], ms=6, alpha=0.5, label=f'{band} band'
        )
        axs[i].plot(
            time_unbinned.to_datetime(), data_unbinned, MARKERS[band], c='grey', ms=4, alpha=0.2
        )

        # unbinned std
        unbinned_table_after = unbinned_table[after_unbinned]
        # gpb = unbinned_table.group_by('night')
        # groups_data = [gpb.groups[i] for i in range(len(gpb.groups))]
        # # Make sure to create a writable copy of the column
        # std_per_night = [np.nanstd(group['combined'].data.copy()) for group in groups_data]

        # for j, n in enumerate(np.unique(unbinned_table['night'])):
        #     print(j, n, np.std(unbinned_table[unbinned_table['night'] == n]['combined']))
        unique_nights = np.unique(unbinned_table_after['night'])
        nightly_table_list = [unbinned_table_after[unbinned_table_after['night'] == n] for n in unique_nights]
        nightly_n_exp, nightly_std = [], []
        for t in nightly_table_list:
            nightly_n_exp.append(len(t))
            nightly_std.append(np.std(t['combined']))
        print(f"{band}\t"
              f"n_nights = {len(unique_nights)}\t"
              f"med = {np.nanmedian(nightly_std):.5f}\t"
              f"max_n_exp = {max(nightly_n_exp)}\t"
              f"min_n_exp = {min(nightly_n_exp)}")

        clipped_data = stats.sigma_clip(data_binned, sigma=5, maxiters=None, masked=False)
        clipped_mean = np.nanmean(clipped_data)
        clipped_std = np.nanstd(clipped_data)
        raw_std = np.nanstd(data_binned)
        min_data, max_data = np.nanmin(data_binned), np.nanmax(data_binned)
        print(f'{band}\t{clipped_mean:.3f} +/- {raw_std:.3f} '
              f'({max_data:.3f} ~ {min_data:.3f} = {np.max([abs(max_data - clipped_mean), abs(clipped_mean - min_data)]):.3f})', end='\n\n')

        axs[i].axhline(clipped_mean, color=COLORS[band], lw=1, alpha=0.4)
        # x1, x2 = datetime(2022, 1, 1), datetime(2025, 12, 31)
        # y1, y2 = clipped_mean - raw_std * N_SIG, clipped_mean + raw_std * N_SIG
        # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=0.1, color='gray', zorder=0)
        # axs[i].add_patch(rect)

        # Place ticks inside and labels outside
        axs[i].tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
        # axs[i].tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False)  # Labels outside

        axs[i].set_ylim(clipped_mean + clipped_std * 5, clipped_mean - clipped_std * 5)
        axs[i].set_ylabel(f'$m_{band}$')
        axs[i].tick_params(axis='y', direction='in')
        # axs[i].yaxis.set_major_locator(MaxNLocator(format='%.1f'))  # TODO new version can work with this
        axs[i].yaxis.set_major_locator(MultipleLocator(0.2))
        axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
        # axs[i].legend(loc='upper right')
        axs[i].grid(axis='y', linestyle=':', alpha=0.5)

        if PLOT_HST:
            for _ in wfc3:
                wfc3_line = axs[i].axvline(_.to_datetime(), c='C7', linewidth=1, alpha=0.8)
            for _ in stis:
                stis_line = axs[i].axvline(_.to_datetime(), ls='--', c='C1', linewidth=2, alpha=0.9)

        plotted = True

    if plotted:
        axs[3].set_xlim(datetime(2023, 3, 1), datetime(2024, 5, 25))

        # Set x-axis ticks for "YYYY-MM, -MM"
        # axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        # axs[3].xaxis.set_major_locator(mdates.YearLocator())
        # axs[3].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        # axs[3].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        fig.autofmt_xdate()

        fig.supxlabel('Time of observation', y=0.03)
        # fig.suptitle(fn.replace('_', ' '))

        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        if PLOT_HST:
            handles.extend([wfc3_line, stis_line])
            labels.extend(['HST WFC3 planetary transit obs.', 'HST STIS host star observation'])

        fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.54, 1.009),
                   handles=handles, labels=labels)  # , ['WFC3 transit visits', 'STIS UV visit'])
        fig.tight_layout()
        fig.subplots_adjust(top=0.915)

        plt.savefig(f'fig/paper/monitoring-20250205-same.pdf')
        # plt.savefig(f'fig/png/{fn}{SUFFIX}.png')
        # plt.show()

    break


# B	n_nights = 58	med = 0.03108	max_n_exp = 20	min_n_exp = 1
# B	8.644 +/- 0.086 (9.015 ~ 8.462 = 0.371)
#
# V	n_nights = 69	med = 0.02908	max_n_exp = 20	min_n_exp = 2
# V	7.912 +/- 0.038 (8.051 ~ 7.802 = 0.138)
#
# R	n_nights = 62	med = 0.03319	max_n_exp = 20	min_n_exp = 1
# R	7.900 +/- 0.054 (8.270 ~ 7.827 = 0.370)
#
# I	n_nights = 64	med = 0.03737	max_n_exp = 24	min_n_exp = 1
# I	7.174 +/- 0.066 (7.461 ~ 6.899 = 0.287)
