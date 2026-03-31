#!/usr/bin/env python3
"""
Lomb-Scargle periodogram with a manual high-resolution frequency grid.
Used for precise peak identification beyond the autopower() heuristic.
"""
import matplotlib.pyplot as plt
import astropy.table as table
import numpy as np
from astropy.timeseries import LombScargle, LombScargleMultiband
import astropy.units as u
from pathlib import Path
from newport import *
from target_phot import OUTPUT_DIR, TARGET

# --- Configuration ---
# Script will use a dense linear frequency grid
MIN_PERIOD = 4  # days
MAX_PERIOD = 100000  # days
N_SAMPLES = int(1e4)

# Plotting constants matching period_freq.py
XTICKS = np.array(
    # [12, 15, 20, 30, 40, 50, 100, 400]  # 12 / 14
    [4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 400]
)
RANGES = np.array([
    # [5, 6],
    # [6, 7],
    # [10, 15],
    # [20, 40]
])
FAP = 1  # False alarm probability percentage

# Stellar parameters for P_rot limit from v sin i
# HD 191939: R_star ~ 0.94 R_sun, vsini ~ 1.6 km/s (upper limit)
VSINI = 1.6    # km/s
R_STAR = 0.94  # R_sun

CUTOFF = Time('2013-03-14')
LW = 1.2
BAND_MAP = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
# ---------------------

if __name__ == "__main__":
    print(f"Generating high-resolution manual periodogram for {TARGET}...")
    print(f"Grid: {MIN_PERIOD} to {MAX_PERIOD} days ({N_SAMPLES} samples)")
    
    # Linear frequency grid (uniform spacing on the frequency-scaled x-axis)
    f_min = 1.0 / MAX_PERIOD
    f_max = 1.0 / MIN_PERIOD
    freq_grid = np.linspace(f_min, f_max, N_SAMPLES) / u.d

    # P_max = 2 * pi * R / (v sin i)
    p_rot_max_vsini = (2 * np.pi * R_STAR * 6.957e5) / (VSINI * 86400)
    print(f"Max P_rot from vsini ({VSINI} km/s, R={R_STAR} R_sun): {p_rot_max_vsini:.2f} days")

    fig, axs = plt.subplots(5, 1, figsize=(5, 6), sharex=True)
    plt.subplots_adjust(hspace=0.08, top=0.91, bottom=0.07)

    legend_handles = []
    legend_labels = []

    for i, band in enumerate(['B', 'V', 'R', 'I']):
        # Try both root and 'results' subfolder
        found = False
        for sub in ['', 'results/']:
            path = OUTPUT_DIR / f"{sub}bin_results_{band}.fits"
            if path.exists():
                binned_table = table.Table.read(path)
                found = True
                break
        
        if not found:
            print(f"Skipping {band} band: bin_results_{band}.fits not found in {OUTPUT_DIR}")
            continue

        # Data prep
        time = Time(binned_table['jd'], format='jd')
        # Filter for recent data (CUTOFF) and non-NaNs
        mask = (time > CUTOFF) & (~np.isnan(binned_table['flux']))
        
        t_data = time[mask]
        y_data = binned_table['flux'].data[mask]
        dy_data = binned_table['intraday_std'].data[mask]

        if len(t_data) < 5:
            print(f"Skipping {band}: insufficient data points.")
            continue

        # Lomb-Scargle
        ls = LombScargle(t_data, y_data, dy_data)
        power = ls.power(freq_grid)

        # Plotting
        axs[i].plot(freq_grid.value, power, c=COLORS[band], lw=LW, label=f'{band} band')

        # --- Automated Peak Tagging ---
        for r in RANGES:
            # f_min_r, f_max_r in frequency
            f_range_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
            if np.any(f_range_mask):
                idx_peak = np.argmax(power[f_range_mask])
                f_peak = freq_grid.value[f_range_mask][idx_peak]
                p_peak = power[f_range_mask][idx_peak]
                
                axs[i].axvline(f_peak, color='gray', ls='--', alpha=0.3, lw=0.8)
                axs[i].text(f_peak, 0.95, f'{1/f_peak:.2f}d', 
                            transform=axs[i].get_xaxis_transform(),
                            rotation=90, fontsize=7, ha='right', va='top', color='0.4')

        # Global FAP threshold
        fap_level = ls.false_alarm_level(FAP / 100)
        fap_line = axs[i].axhline(fap_level, color='gray', ls=':', alpha=0.6, label=f'{FAP}% FAP')
        
        # Max rotation period from v sin i
        label_vsini = rf'Max $P_{{\rm rot}}$ from $v\,\sin{{i}} = {VSINI}$ km/s'
        vsini_line = axs[i].axvline(
            1 / p_rot_max_vsini, color='gray', linestyle='--', alpha=0.8,
            label=label_vsini
        )
        
        # Panel Title
        axs[i].text(
            0.02, 0.95, f"{band} band", 
            transform=axs[i].transAxes, 
            fontweight='bold', va='top', color=COLORS[band],
            # fontsize=9
        )

        if i == 0:
            legend_handles.extend([fap_line, vsini_line])
            legend_labels.extend([fap_line.get_label(), vsini_line.get_label()])

        axs[i].tick_params(axis='both', direction='in')  # , labelsize=9)

    # --- 5. Multiband Analysis ---
    all_table = table.Table.read(OUTPUT_DIR / "bin_results_all.fits")
    mask = (Time(all_table['jd'], format='jd') > CUTOFF) & (~np.isnan(all_table['flux']))
    
    t_all = Time(all_table['jd'][mask], format='jd')
    y_all = all_table['flux'].data[mask]
    dy_all = all_table['intraday_std'].data[mask]
    b_all = np.array([BAND_MAP[b] for b in all_table['band'][mask]])

    ls_multi = LombScargleMultiband(t_all, y_all, b_all, dy_all)
    power_multi = ls_multi.power(freq_grid)

    ax_m = axs[4]
    ax_m.plot(freq_grid.value, power_multi, c='k', lw=LW, label='Joint Multiband')

    # --- Automated Peak Tagging ---
    for r in RANGES:
        # f_min_r, f_max_r in frequency
        f_range_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
        if np.any(f_range_mask):
            idx_peak = np.argmax(power_multi[f_range_mask])
            f_peak = freq_grid.value[f_range_mask][idx_peak]
            p_peak = power_multi[f_range_mask][idx_peak]

            print(f'Multiband peak {r}\t{1/f_peak}')
            
            ax_m.axvline(f_peak, color='gray', ls='--', alpha=0.3, lw=0.8)
            ax_m.text(
                f_peak, 0.95, f'{1/f_peak:.2f}d', 
                transform=ax_m.get_xaxis_transform(),
                rotation=90, fontsize=7, ha='right', va='top', color='0.4'
            )
    
    # Max P_rot from vsini line
    ax_m.axvline(1 / p_rot_max_vsini, color='gray', linestyle='--', alpha=0.8)

    # NB: FAP for multiband is not implemented
    # fap_m = ls_multi.false_alarm_level(FAP / 100)
    # ax_m.axhline(fap_m, color='gray', ls=':', alpha=0.6)
    
    ax_m.text(
        0.02, 0.95, "Multiband", transform=ax_m.transAxes, 
        fontweight='bold', va='top', color='k'
    )
    ax_m.tick_params(axis='both', direction='in')

    # X-axis (Period labeling)
    axs[4].set_xticks(1.0 / XTICKS)
    axs[4].set_xticklabels(XTICKS)
    axs[4].set_xlim(f_min, f_max)
    axs[4].set_xlabel("Period (days)")  # , fontsize=11)
    
    # Invert freq so period increases to the right
    axs[4].invert_xaxis()

    fig.supylabel("Lomb-Scargle Power", x=0.01)
    
    # Global Figure Legend
    fig.legend(
        legend_handles, legend_labels, loc='upper center', 
        bbox_to_anchor=(0.52, 0.98), ncol=2
    )

    # Use tight_layout with rect to make room for the manual subplots_adjust
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUTPUT_DIR / f"periodogram_manual_{TARGET}_{MIN_PERIOD}-{MAX_PERIOD}_n{N_SAMPLES}.pdf"
    plt.savefig(save_path)  # , dpi=200)
    print(f"Manual high-res periodogram saved to {save_path}")
    # plt.show()
    plt.close()
