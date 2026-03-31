#!/usr/bin/env python3
"""
"Null Hypothesis" Periodogram using mock noise data.
Preserves real observation timestamps but replaces flux with synthetic noise.
"""
import matplotlib.pyplot as plt
import astropy.table as table
import numpy as np
from astropy.timeseries import LombScargle, LombScargleMultiband
import astropy.units as u
from pathlib import Path

# --- Import Configuration from existing scripts ---
from target_phot import OUTPUT_DIR, TARGET
from period_manual import (
    MIN_PERIOD, MAX_PERIOD, N_SAMPLES, XTICKS, FAP, 
    VSINI, R_STAR, CUTOFF, LW, BAND_MAP
)
from newport import COLORS, Time

# --- Mock Configuration ---
MODE = 'constant'  # Options: 'gaussian', 'uniform', 'constant', 'shuffle'
RANGES = np.array([
    # Add [min_p, max_p] ranges here for independent peak tagging on mock data
])
# ---------------------

if __name__ == "__main__":
    print(f"Generating Mock Periodogram (mode: {MODE}) for {TARGET}...")
    print(f"Grid: {MIN_PERIOD} to {MAX_PERIOD} days ({N_SAMPLES} samples)")
    
    # frequency grid
    f_min = 1.0 / MAX_PERIOD
    f_max = 1.0 / MIN_PERIOD
    freq_grid = np.linspace(f_min, f_max, N_SAMPLES) / u.d

    # P_max from vsini for reference
    p_rot_max_vsini = (2 * np.pi * R_STAR * 6.957e5) / (VSINI * 86400)

    fig, axs = plt.subplots(5, 1, figsize=(5, 6), sharex=True)
    plt.subplots_adjust(hspace=0.08, top=0.91, bottom=0.07)

    legend_handles = []
    legend_labels = []

    # Load all-band data for reference timestamps and multiband
    all_path = OUTPUT_DIR / "bin_results_all.fits"
    if not all_path.exists():
        raise FileNotFoundError(f"Required file {all_path} not found. Run target_phot.py first.")
    
    all_table = table.Table.read(all_path)
    # Global mask for multiband
    global_mask = (Time(all_table['jd'], format='jd') > CUTOFF) & (~np.isnan(all_table['flux']))
    all_table = all_table[global_mask]

    # Pre-generate mock data for the joint multiband panel
    # We'll store it by band to keep logic consistent
    mock_data = {}

    for i, band in enumerate(['B', 'V', 'R', 'I']):
        # Filter all_table for this band
        band_mask = (all_table['band'] == band)
        band_data = all_table[band_mask]
        
        if len(band_data) < 5:
            print(f"Skipping {band} (Mock): insufficient data points.")
            continue

        t_real = Time(band_data['jd'], format='jd')
        # Use real error median as noise scale
        dy_real = band_data['intraday_std'].data
        sigma_noise = np.nanmedian(dy_real)
        
        # Generator
        dy_mock = np.full(len(t_real), sigma_noise)
        if MODE == 'gaussian':
            y_mock = np.random.normal(1.0, sigma_noise, len(t_real))
        elif MODE == 'uniform':
            # Matching variance (sigma^2 = (b-a)^2 / 12) => range = sigma * sqrt(12)
            half_width = sigma_noise * np.sqrt(3)
            y_mock = np.random.uniform(1.0 - half_width, 1.0 + half_width, len(t_real))
        elif MODE == 'constant':
            y_mock = np.ones(len(t_real))
            dy_mock = np.full(len(t_real), sigma_noise * 0.1)
        elif MODE == 'shuffle':
            y_real = band_data['flux'].data
            # Shuffle indices to break temporal correlation
            idx = np.random.permutation(len(y_real))
            y_mock = y_real[idx]
            dy_mock = dy_real[idx]
        else:
            raise ValueError(f"Unknown mock mode: {MODE}")
        
        # Store for multiband later
        mock_data[band] = {'t': t_real, 'y': y_mock, 'dy': dy_mock}

        # Individual Lomb-Scargle
        ls = LombScargle(t_real, y_mock, dy_mock)
        power = ls.power(freq_grid)

        # Plotting
        axs[i].plot(freq_grid.value, power, c=COLORS[band], lw=LW, label=f'{band} (mock - {MODE})')

        # Peak tagging
        for r in RANGES:
            f_r_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
            if np.any(f_r_mask):
                idx_p = np.argmax(power[f_r_mask])
                f_p = freq_grid.value[f_r_mask][idx_p]
                axs[i].axvline(f_p, color='gray', ls='--', alpha=0.3, lw=0.8)
                axs[i].text(f_p, 0.95, f'{1/f_p:.2f}d', transform=axs[i].get_xaxis_transform(),
                            rotation=90, fontsize=7, ha='right', va='top', color='0.4')

        # Thresholds
        fap_level = ls.false_alarm_level(FAP / 100)
        fap_line = axs[i].axhline(fap_level, color='gray', ls=':', alpha=0.6, label=f'{FAP}% FAP')
        label_vsini = rf'Max $P_{{\rm rot}}$ from $v\,\sin{{i}}$'
        vsini_line = axs[i].axvline(1 / p_rot_max_vsini, color='gray', linestyle='--', alpha=0.8, label=label_vsini)
        
        if i == 0:
            legend_handles.extend([fap_line, vsini_line])
            legend_labels.extend([fap_line.get_label(), vsini_line.get_label()])

        # Labels
        axs[i].text(0.02, 0.95, f"{band} band (mock - {MODE})", transform=axs[i].transAxes, 
                    fontweight='bold', va='top', color=COLORS[band])
        axs[i].tick_params(axis='both', direction='in')

        if MODE == 'constant':
            axs[i].set_ylim(-0.02, 0.35)

    # --- Multiband (Mock) ---
    if MODE != 'constant':
        print(f"Computing Multiband Periodogram (Mock)...")
        # Combine the mock data we just generated
        t_multi = []
        y_multi = []
        dy_multi = []
        b_multi = []
        
        for band, d in mock_data.items():
            t_multi.append(d['t'].jd)
            y_multi.append(d['y'])
            dy_multi.append(d['dy'])
            b_multi.append(np.full(len(d['y']), BAND_MAP[band]))
        
        if t_multi:
            t_multi = Time(np.concatenate(t_multi), format='jd')
            y_multi = np.concatenate(y_multi)
            dy_multi = np.concatenate(dy_multi)
            b_multi = np.concatenate(b_multi)

            ls_m = LombScargleMultiband(t_multi, y_multi, b_multi, dy_multi)
            power_m = ls_m.power(freq_grid)

            axs[4].plot(freq_grid.value, power_m, c='k', lw=LW, label=f'Joint Multiband (mock - {MODE})')
            axs[4].axvline(1 / p_rot_max_vsini, color='gray', linestyle='--', alpha=0.8)
            axs[4].text(0.02, 0.95, f"Multiband (mock - {MODE})", transform=axs[4].transAxes, 
                        fontweight='bold', va='top', color='k')
            
            # Peak tagging for multiband
            for r in RANGES:
                f_r_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
                if np.any(f_r_mask):
                    idx_p = np.argmax(power_m[f_r_mask])
                    axs[4].axvline(
                        freq_grid.value[f_r_mask][idx_p],
                        color='gray', ls='--', alpha=0.3, lw=0.8
                    )

    # X-axis
    axs[4].set_xticks(1.0 / XTICKS)
    axs[4].set_xticklabels(XTICKS)
    axs[4].set_xlim(f_min, f_max)
    axs[4].set_xlabel("Period (days)")
    axs[4].invert_xaxis()
    axs[4].tick_params(axis='both', direction='in')

    fig.supylabel("Lomb-Scargle Power", x=0.01)
    fig.legend(
        legend_handles, legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.52, 0.98), 
        ncol=2
    )

    save_path = OUTPUT_DIR / \
        f"mock_periodogram_{TARGET}_{MIN_PERIOD}-{MAX_PERIOD}_{MODE}.pdf"
    plt.savefig(save_path)
    print(f"Mock periodogram saved to {save_path}")
    plt.show()
    plt.close()
