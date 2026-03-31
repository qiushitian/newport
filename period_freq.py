#!/usr/bin/env python3
"""
Lomb-Scargle periodogram in period on frequency scale
"""
import matplotlib.pyplot as plt
import astropy.table as table
import astropy.stats as stats
import numpy as np
from astropy.timeseries import LombScargle
import astropy.units as u
from pathlib import Path
from newport import *
from target_phot import OUTPUT_DIR

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Verdana"]

READ_DIR = OUTPUT_DIR
WRITE_DIR = OUTPUT_DIR
WRITE_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF = Time('2013-03-14')
SUFFIX = ''
# YLIM_UPPER = 0.7
ERROR_TYPE = 'error'

# long period
MIN_PERIOD, MIN_FREQ = 15, -0.001
XTICKS = np.array([20, 30, 40, 50, 70, 100, 200, 600])  # [2, 3, 4, 5, 7, 9, 20, 300])
RANGES = np.array([
    [40, 55],
    [60, 70],
    [80, 120],
    [150, 180],
    [200, 600]
])
FAP = 1

# # 20-ish day period
# MIN_PERIOD, MIN_FREQ = 17, 1 / 55
# XTICKS = np.array([20, 25, 30, 50])  # [2, 3, 4, 5, 7, 9, 20, 300])
# RANGES = np.array([
#     [26, 30],
#     [40, 55]
# ])
# FAP = 15

# # unconstrained
# MIN_PERIOD, MIN_FREQ = 1.8, -0.01
# XTICKS = np.array([2, 3, 4, 5, 7, 9, 20, 300])
# RANGES = np.array([])
# FAP = 1

# Stellar parameters for P_rot limit from v sin i
# HD 191939: R_star ~ 0.94 R_sun, vsini ~ 1 km/s (upper limit)
VSINI = 1.6    # km/s
R_STAR = 0.94 # R_sun

# P_max = 2 * pi * R / (v sin i)
P_MAX_VSINI = (2 * np.pi * R_STAR * 6.957e5) / (VSINI * 86400)
print(f"Max rotation period from vsini ({VSINI} km/s, R={R_STAR} R_sun): {P_MAX_VSINI:.2f} days")


if __name__ == '__main__':
    for fn in TARGET_FN:
        # TODO DEV
        if fn != 'HD_191939':
            continue

        fig, axs = plt.subplots(nrows=4, figsize=(5, 6), sharex=True, sharey=True)
        ylim0, ylim1 = 0, 0

        for i, band in enumerate(['B', 'V', 'R', 'I']):
            try:
                binned_table = table.Table.read(READ_DIR / f'bin_results_{band}.fits')
            except FileNotFoundError as e:
                print(str(e))
                continue

            # cutoff
            time_binned = Time(binned_table['jd'], format='jd')

            after_binned = time_binned > CUTOFF

            time_binned = time_binned[after_binned]

            data_binned = binned_table['flux'].data[after_binned]

            err_binned = binned_table[ERROR_TYPE].data[after_binned]
            # end of cutoff

            not_nan_mask = ~np.isnan(data_binned)
            ls = LombScargle(time_binned[not_nan_mask], data_binned[not_nan_mask], err_binned[not_nan_mask])
            frequency, power = ls.autopower()
            # print(frequency, power)

            mask = (frequency >= MIN_FREQ / u.d) & (frequency <= 1 / MIN_PERIOD / u.d)
            axs[i].plot(
                frequency[mask], power[mask], c=COLORS[band], label=f'{band} band'
            )

            # --- Peer into specific ranges and plot maxima ---
            for r in RANGES:
                f_min, f_max = 1 / (r[1] * u.d), 1 / (r[0] * u.d)
                m_range = (frequency >= f_min) & (frequency <= f_max)
                if np.any(m_range):
                    idx_peak = np.argmax(power[m_range])
                    f_peak = frequency[m_range][idx_peak]
                    
                    # Vertical indicator
                    axs[i].axvline(
                        f_peak.value, color='gray', linestyle='--', 
                        alpha=0.4, linewidth=0.8, zorder=0
                    )
                    # Period label near top
                    axs[i].text(
                        f_peak.value, 0.95, f'{1/f_peak.value:.1f}d', 
                        transform=axs[i].get_xaxis_transform(),
                        rotation=90, fontsize=7, ha='right', va='top', color='0.3'
                    )

            # false alarm
            false_alarm_power = ls.false_alarm_level(FAP / 100)
            print(f'{band}-band power={false_alarm_power} @ {FAP}% FAP')
            axs[i].axhline(
                false_alarm_power, color='gray', linestyle=':', alpha=0.8,
                label=f'{FAP}% false-alarm probability'  # FAP'
            )

            # Max rotation period from v sin i
            label_vsini = r'Max $P_{\rm rot}$'
            label_vsini = r'Max $P_{\rm rot}$ from $v\,\sin{i}$'
            label_vsini = r'Max $P_{\rm rot}$ from $v\,\sin{i} = ' + str(VSINI) + r'\,\rm{km/s}$'
            axs[i].axvline(
                1 / P_MAX_VSINI, color='gray', linestyle='--', alpha=0.8,
                label=label_vsini
            )

            # ### BLOCK stats ###
            # # unbinned std
            # unbinned_table_after = unbinned_table[after_unbinned]
            # # gpb = unbinned_table.group_by('night')
            # # groups_data = [gpb.groups[i] for i in range(len(gpb.groups))]
            # # # Make sure to create a writable copy of the column
            # # std_per_night = [np.nanstd(group['combined'].data.copy()) for group in groups_data]

            # # for j, n in enumerate(np.unique(unbinned_table['night'])):
            # #     print(j, n, np.std(unbinned_table[unbinned_table['night'] == n]['combined']))
            # unique_nights = np.unique(unbinned_table_after['night'])
            # nightly_table_list = [unbinned_table_after[unbinned_table_after['night'] == n] for n in unique_nights]
            # nightly_n_exp, nightly_std = [], []
            # for t in nightly_table_list:
            #     nightly_n_exp.append(len(t))
            #     nightly_std.append(np.std(t['flux']))
            # print(f"{band}\t"
            #       f"n_nights = {len(unique_nights)}\t"
            #       f"med = {np.nanmedian(nightly_std):.5f}\t"
            #       f"max_n_exp = {max(nightly_n_exp)}\t"
            #       f"min_n_exp = {min(nightly_n_exp)}")

            # clipped_data = stats.sigma_clip(data_binned, sigma=5, masked=False)  # maxiters=None
            # clipped_mean = np.nanmean(clipped_data)
            # clipped_std = np.nanstd(clipped_data)
            # raw_std = np.nanstd(data_binned)
            # min_data, max_data = np.nanmin(data_binned), np.nanmax(data_binned)
            # print(f'{band}\t{clipped_mean:.3f} +/- {clipped_std:.3f} [{raw_std:.3f}]'
            #       f'({max_data:.3f} ~ {min_data:.3f} = {np.max([abs(max_data - clipped_mean), abs(clipped_mean - min_data)]):.3f})', end='\n\n')
            # ### END BLOCK ###

            # Place ticks inside and labels outside
            axs[i].tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
            # axs[i].tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False)  # Labels outside

            # axs[i].set_ylim()
            # axs[i].set_ylabel(f'{band} band power')
            axs[i].tick_params(axis='y', direction='in')
            # axs[i].yaxis.set_major_locator(MaxNLocator(format='%.1f'))  # TODO new version can work with this
            # axs[i].yaxis.set_major_locator(MultipleLocator(0.2))
            # axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))

            # _yl = axs[i].get_ylim()
            # ylim0 = min(ylim0, _yl[0])
            # ylim1 = max(ylim1, _yl[1])
            # # print(ylim0, ylim1)

            # axs[i].grid(axis='y', linestyle=':', alpha=0.5)

        axs[-1].set_xticks(1 / XTICKS, XTICKS)
        # _xl0, _xl1 = axs[-1].get_xlim()
        # axs[-1].set_xlim(_xl1, _xl0)
        axs[-1].set_xlim(1 / MIN_PERIOD, MIN_FREQ)
        # axs[-1].set_ylim(ylim0, YLIM_UPPER)

        fig.supxlabel('Period (day)', y=0.03)
        fig.supylabel("Lomb-Scargle power", x=0.04)
        # fig.suptitle(fn.replace('_', ' '))

        # Deduplicate legend handles and labels
        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    if 'band' in label:
                        handles.append(handle)
                        labels.append(label)
                    else:
                        handles.insert(0, handle)
                        labels.insert(0, label)

        fig.legend(
            ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1),
            # fontsize=8,
            handles=handles, labels=labels
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.89)

        plt.savefig(WRITE_DIR / 
            f'periodogram_freq_p{MIN_PERIOD}_fap{FAP}_{ERROR_TYPE}.pdf'
        )
        # plt.savefig(f'fig/png/{fn}{SUFFIX}.png')
        plt.show()


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
