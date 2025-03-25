#!/usr/bin/env python3
"""

"""

import matplotlib.pyplot as plt
import astropy.table as table
import astropy.stats as stats
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.patches as patches
from pathlib import Path
from newport import *

COLORS = {'B': 'C0', 'V': 'C2', 'R': 'C3', 'I': 'maroon'}
MARKERS = {'B': 'o', 'V': '>', 'R': 's', 'I': 'D'}

PLOT_HST = True
READ_DIR = Path('tables/list_runs/with_flat/mag')
CUTOFF = Time('2023-03-14')
# CUTOFF_END = Time('2023-08-01')
CUTOFF_END = Time('2025-08-01')
N_SIG = 1
SUFFIX = ''
PLOT_ERR = False


if __name__ == '__main__':
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
                binned_table = table.Table.read(READ_DIR / f'bin_comb_mag_{fn}_{band}{SUFFIX}.fits')
                unbinned_table = table.Table.read(READ_DIR / f'unbin_comb_mag_{fn}_{band}{SUFFIX}.fits')
            except FileNotFoundError as e:
                print(str(e))
                continue

            # cutoff
            time_binned = Time(binned_table['jd'], format='jd')
            time_unbinned = Time(unbinned_table['jd'], format='jd')

            after_binned = np.logical_and(time_binned > CUTOFF, time_binned < CUTOFF_END)
            after_unbinned = np.logical_and(time_unbinned > CUTOFF, time_unbinned < CUTOFF_END)

            time_binned = time_binned[after_binned]
            time_unbinned = time_unbinned[after_unbinned]

            data_binned = binned_table['super'].data[after_binned]
            data_unbinned = unbinned_table['super'].data[after_unbinned]
            err_binned = binned_table['super_err'].data[after_binned]
            err_unbinned = unbinned_table['super_err'].data[after_unbinned]
            # end of cutoff

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
                nightly_std.append(np.std(t['super']))
            print(f"{band}\t"
                  f"n_nights = {len(unique_nights)}\t"
                  f"med_of_std = {np.nanmean(nightly_std):.5f}\t"  # TODO mean of std
                  f"max_n_exp = {max(nightly_n_exp)}\t"
                  f"min_n_exp = {min(nightly_n_exp)}")

            # clipping low nightly_n_exp nights
            # nightly_n_exp > 5  # TODO

            clipped_data = stats.sigma_clip(data_binned, sigma=5, maxiters=None, masked=False)
            clipped_mean = np.nanmean(clipped_data)
            clipped_std = np.nanstd(clipped_data)
            raw_mean = np.nanmean(data_binned)
            raw_std = np.nanstd(data_binned)
            min_data, max_data = np.nanmin(data_binned), np.nanmax(data_binned)
            mean_err = np.nanmean(err_binned)
            print(f'{band}\t{clipped_mean:.3f} +/- {raw_std:.3f}\t({max_data:.3f} ~ {min_data:.3f} = {np.max([abs(max_data - clipped_mean), abs(clipped_mean - min_data)]):.3f})', end='\n')
            print(f'{band}\t{raw_mean:.3f} +/- {raw_std:.3f}\t{mean_err}\t(... = {np.max([abs(max_data - raw_mean), abs(raw_mean - min_data)]):.3f})', end='\n\n')

            if PLOT_ERR:
                axs[i].errorbar(
                    time_unbinned.to_datetime(), data_unbinned, yerr=err_unbinned, fmt=MARKERS[band],
                    c='silver', ms=4, alpha=0.3, markeredgewidth=0, ecolor='lightgrey', elinewidth=1
                )
                axs[i].errorbar(
                    time_binned.to_datetime(), data_binned, yerr=err_binned, fmt=MARKERS[band],
                    c=COLORS[band], ms=7, alpha=0.7, markeredgewidth=0, zorder=2.10, label=f'{band} band'
                )
            else:
                axs[i].plot(
                    time_unbinned.to_datetime(), data_unbinned, MARKERS[band],
                    c='lightgrey', ms=4, alpha=0.5, markeredgewidth=0
                )
                axs[i].plot(
                    time_binned.to_datetime(), data_binned, MARKERS[band],
                    c=COLORS[band], ms=7, alpha=0.7, markeredgewidth=0, label=f'{band} band'
                )

            axs[i].axhline(clipped_mean, color=COLORS[band], lw=1, alpha=0.4)
            # axs[i].axhline(clipped_mean - raw_std * N_SIG, color=COLORS[band], lw=1, alpha=0.4, ls=':')
            # axs[i].axhline(clipped_mean + raw_std * N_SIG, color=COLORS[band], lw=1, alpha=0.4, ls=':')
            x1, x2 = datetime(2022, 1, 1), datetime(2025, 12, 31)
            y1, y2 = clipped_mean - raw_std * N_SIG, clipped_mean + raw_std * N_SIG
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=0.05, color=COLORS[band], lw=0, zorder=0)
            axs[i].add_patch(rect)

            # Place ticks inside and labels outside
            axs[i].tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
            # axs[i].tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False)  # Labels outside

            axs[i].set_ylim(clipped_mean + clipped_std * 5, clipped_mean - clipped_std * 5)
            axs[i].set_ylabel(f'$m_{band}$')
            axs[i].tick_params(axis='y', direction='in')
            # axs[i].yaxis.set_major_locator(MaxNLocator(format='%.1f'))  # TODO new version can work with this
            # axs[i].yaxis.set_major_locator(MultipleLocator(0.2))
            # axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
            # axs[i].legend(loc='upper right')
            # axs[i].grid(axis='y', linestyle=':', alpha=0.5)

            if PLOT_HST:
                for _ in wfc3:
                    wfc3_line = axs[i].axvline(_.to_datetime(), c='C7', linewidth=1, alpha=0.4)
                for _ in stis:
                    stis_line = axs[i].axvline(_.to_datetime(), ls='--', c='C1', linewidth=2, alpha=0.5)

            plotted = True

        if plotted:
            axs[3].set_xlim(datetime(2023, 3, 2), datetime(2024, 5, 25))

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

            plt.savefig(f'fig/paper/monitoring_with_flat_no_nExpNight_clip.pdf')
            # plt.savefig(f'fig/png/{fn}{SUFFIX}.png')
            # plt.show()

        break  # TODO dev


# with flat
# med nightly std
# B	n_nights = 47	med_of_std = 0.01572	max_n_exp = 20	min_n_exp = 1
# B	8.590 +/- 0.023	(8.631 ~ 8.516 = 0.073)
# B	8.590 +/- 0.023	4.404500825496091	(... = 0.073)
# V	n_nights = 53	med_of_std = 0.01945	max_n_exp = 20	min_n_exp = 7
# V	7.908 +/- 0.050	(8.027 ~ 7.619 = 0.290)
# V	7.903 +/- 0.050	8.664040053627733	(... = 0.284)
# R	n_nights = 46	med_of_std = 0.02752	max_n_exp = 20	min_n_exp = 1
# R	7.734 +/- 0.030	(7.778 ~ 7.561 = 0.172)
# R	7.730 +/- 0.030	4.472983720069519	(... = 0.168)
# I	n_nights = 48	med_of_std = 0.03745	max_n_exp = 20	min_n_exp = 1
# I	7.179 +/- 0.045	(7.352 ~ 7.036 = 0.173)
# I	7.179 +/- 0.045	44.26558362011497	(... = 0.173)

# mean nightly std
# B	n_nights = 47	med_of_std = 0.03060	max_n_exp = 20	min_n_exp = 1
# B	8.590 +/- 0.023	(8.631 ~ 8.516 = 0.073)
# B	8.590 +/- 0.023	4.404500825496091	(... = 0.073)
# V	n_nights = 53	med_of_std = 0.07791	max_n_exp = 20	min_n_exp = 7
# V	7.908 +/- 0.050	(8.027 ~ 7.619 = 0.290)
# V	7.903 +/- 0.050	8.664040053627733	(... = 0.284)
# R	n_nights = 46	med_of_std = 0.03818	max_n_exp = 20	min_n_exp = 1
# R	7.734 +/- 0.030	(7.778 ~ 7.561 = 0.172)
# R	7.730 +/- 0.030	4.472983720069519	(... = 0.168)
# I	n_nights = 48	med_of_std = 0.08446	max_n_exp = 20	min_n_exp = 1
# I	7.179 +/- 0.045	(7.352 ~ 7.036 = 0.173)
# I	7.179 +/- 0.045	44.26558362011497	(... = 0.173)



# list run wo bd simple sqrt
# B	n_nights = 48	med_of_std = 0.01627	max_n_exp = 20	min_n_exp = 1
# B	8.575 +/- 0.125	(8.918 ~ 7.798 = 0.777)
# B	8.566 +/- 0.125	0.010585227095765036	(... = 0.768)
# V	n_nights = 53	med_of_std = 0.02005	max_n_exp = 20	min_n_exp = 7
# V	7.904 +/- 0.038	(8.010 ~ 7.709 = 0.195)
# V	7.900 +/- 0.038	0.004803166317355516	(... = 0.191)
# R	n_nights = 46	med_of_std = 0.02859	max_n_exp = 20	min_n_exp = 1
# R	7.722 +/- 0.031	(7.784 ~ 7.544 = 0.179)
# R	7.718 +/- 0.031	0.009885796615380475	(... = 0.175)
# I	n_nights = 48	med_of_std = 0.04084	max_n_exp = 20	min_n_exp = 1
# I	7.174 +/- 0.147	(8.116 ~ 7.122 = 0.942)
# I	7.202 +/- 0.147	0.03000972349909076	(... = 0.914)

# list run with bd simple sqrt
# B	n_nights = 48	med_of_std = 0.01594	max_n_exp = 20	min_n_exp = 1
# B	8.562 +/- 0.031	(8.619 ~ 8.442 = 0.119)
# B	8.562 +/- 0.031	0.007800401748420398	(... = 0.119)
# V	n_nights = 53	med_of_std = 0.01927	max_n_exp = 20	min_n_exp = 7
# V	7.897 +/- 0.047	(8.014 ~ 7.621 = 0.276)
# V	7.891 +/- 0.047	0.0038637439299297085	(... = 0.271)
# R	n_nights = 46	med_of_std = 0.02750	max_n_exp = 20	min_n_exp = 1
# R	7.716 +/- 0.030	(7.763 ~ 7.543 = 0.172)
# R	7.712 +/- 0.030	0.007435725850545447	(... = 0.169)
# I	n_nights = 48	med_of_std = 0.03777	max_n_exp = 20	min_n_exp = 1
# I	7.179 +/- 0.140	(8.114 ~ 7.037 = 0.936)
# I	7.198 +/- 0.140	0.022919679735488658	(... = 0.916)


# -------------


# bd fake
# B	n_nights = 11	med_of_std = 0.02122	max_n_exp = 20	min_n_exp = 3
# B	8.563 +/- 0.020	(8.605 ~ 8.535 = 0.042)
# B	8.563 +/- 0.020	0.0069423009291949	(... = 0.042)
# V	n_nights = 11	med_of_std = 0.01960	max_n_exp = 20	min_n_exp = 8
# V	7.894 +/- 0.026	(7.955 ~ 7.850 = 0.061)
# V	7.894 +/- 0.026	0.0027518825091734337	(... = 0.061)
# R	n_nights = 7	med_of_std = 0.02286	max_n_exp = 20	min_n_exp = 20
# R	7.716 +/- 0.018	(7.757 ~ 7.696 = 0.041)
# R	7.716 +/- 0.018	0.004352914402180803	(... = 0.041)
# I	n_nights = 8	med_of_std = 0.06087	max_n_exp = 20	min_n_exp = 6
# I	7.181 +/- 0.032	(7.233 ~ 7.112 = 0.069)
# I	7.181 +/- 0.032	0.02745287799953399	(... = 0.069)

# bd
# B	n_nights = 11	med_of_std = 0.01834	max_n_exp = 20	min_n_exp = 3
# B	8.556 +/- 0.018	(8.587 ~ 8.531 = 0.031)
# B	8.556 +/- 0.018	0.00606165742649702	(... = 0.031)
# V	n_nights = 11	med_of_std = 0.01553	max_n_exp = 20	min_n_exp = 8
# V	7.891 +/- 0.024	(7.945 ~ 7.846 = 0.055)
# V	7.891 +/- 0.024	0.0024712434059965035	(... = 0.055)
# R	n_nights = 7	med_of_std = 0.02341	max_n_exp = 20	min_n_exp = 20
# R	7.708 +/- 0.015	(7.741 ~ 7.692 = 0.033)
# R	7.708 +/- 0.015	0.0035384677347200166	(... = 0.033)
# I	n_nights = 8	med_of_std = 0.04433	max_n_exp = 20	min_n_exp = 6
# I	7.168 +/- 0.051	(7.221 ~ 7.037 = 0.130)
# I	7.168 +/- 0.051	0.024270033660317325	(... = 0.130)



# -------------


# w poisson w bkg
# B	n_nights = 57	med_of_std = 0.01767	max_n_exp = 20	min_n_exp = 1
# B	8.564 +/- 0.050	(8.881 ~ 8.462 = 0.316)
# B	8.570 +/- 0.050	0.003460394570519026	(... = 0.311)
# V	n_nights = 68	med_of_std = 0.02453	max_n_exp = 20	min_n_exp = 1
# V	7.898 +/- 0.061	(7.992 ~ 7.467 = 0.431)
# V	7.892 +/- 0.061	0.0016817538259516563	(... = 0.425)
# R	n_nights = 61	med_of_std = 0.03276	max_n_exp = 20	min_n_exp = 1
# R	7.719 +/- 0.036	(7.811 ~ 7.544 = 0.175)
# R	7.719 +/- 0.036	0.003712652564834233	(... = 0.175)
# I	n_nights = 63	med_of_std = 0.04393	max_n_exp = 24	min_n_exp = 1
# I	7.194 +/- 0.049	(7.379 ~ 7.067 = 0.185)
# I	7.194 +/- 0.049	0.006175186367289447	(... = 0.185)


# new new with poission w/o bkg
# B	n_nights = 57	med_of_std = 0.01767	max_n_exp = 20	min_n_exp = 1
# B	8.564 +/- 0.050	(8.881 ~ 8.462 = 0.316)
# B	8.570 +/- 0.050	0.0030719169400332644	(... = 0.311)
# V	n_nights = 68	med_of_std = 0.02453	max_n_exp = 20	min_n_exp = 1
# V	7.898 +/- 0.061	(7.992 ~ 7.467 = 0.431)
# V	7.892 +/- 0.061	0.001499600975507814	(... = 0.425)
# R	n_nights = 61	med_of_std = 0.03276	max_n_exp = 20	min_n_exp = 1
# R	7.719 +/- 0.036	(7.811 ~ 7.544 = 0.175)
# R	7.719 +/- 0.036	0.003307837158301661	(... = 0.175)
# I	n_nights = 63	med_of_std = 0.04393	max_n_exp = 24	min_n_exp = 1
# I	7.194 +/- 0.049	(7.379 ~ 7.067 = 0.185)
# I	7.194 +/- 0.049	0.005493648254879462	(... = 0.185)


# new mag_from_counts
# B	n_nights = 57	med = 0.01767	max_n_exp = 20	min_n_exp = 1
# B	8.564 +/- 0.050 (8.881 ~ 8.462 = 0.316)
# V	n_nights = 68	med = 0.02453	max_n_exp = 20	min_n_exp = 1
# V	7.898 +/- 0.062 (7.992 ~ 7.467 = 0.430)
# R	n_nights = 62	med = 0.03287	max_n_exp = 20	min_n_exp = 1
# R	7.719 +/- 0.059 (8.092 ~ 7.544 = 0.373)
# I	n_nights = 64	med = 0.04555	max_n_exp = 24	min_n_exp = 1
# I	7.194 +/- 0.062 (7.498 ~ 7.067 = 0.304)


# old
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
