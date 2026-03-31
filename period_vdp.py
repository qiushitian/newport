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

# mfg = np.linspace(-0.1, 3.2, 200) / u.d; plt.plot(mfg, _ls.power(mfg)); plt.xscale('linear'); plt.xlabel('freq (day$^{-1}$)'); plt.ylabel('L-S power'); plt.title('Multiband Window Fuction'); plt.show()

# --- Configuration ---
# Script will use a dense linear frequency grid
MIN_PERIOD = 4  # days
MAX_PERIOD = 10000000  # days
N_0 = 5
MANUAL_TIME_RANGE = 378.72
PERIOD_GRID_RANGE = (150, 400)
SHOW_WINDOW = False

FAP = 1  # False alarm probability percentage
XTICKS = np.array(
    # [12, 15, 20, 30, 40, 50, 100, 400]  # 12 / 14
    [4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 400]
)
RANGES = np.array([
    # [5, 6],
    # [6, 7],
    # [7, 8],
    # [8, 10],
    # [10, 15],
    # [15, 20],
    # [20, 32],
    # [32, 40],
    # # [40, 200],
    # [200, 400]
])
PRIMARY_ALIAS_RANGES = np.array([
    # [20, 32],
    # [32, 40],
    # [100, 400]
])
ALIAS_TAG_COLORS = ['C9', 'C8', 'C1']

# Stellar parameters for P_rot limit from v sin i
# HD 191939: R_star ~ 0.94 R_sun, vsini ~ 1.6 km/s (upper limit)
VSINI = 1.6    # km/s
R_STAR = 0.94  # R_sun

CUTOFF = Time('2013-03-14')
LW = 1.2
BAND_MAP = {'B': 0, 'V': 1, 'R': 2, 'I': 3}
# ---------------------


def get_aliases(freq, time_range, n=5, masking_func=None):
    if isinstance(time_range, u.Quantity):
        time_range = time_range.value
    df = 1 / time_range
    aliases = freq + df * np.arange(-n, n + 1, 1)
    if masking_func:
        mask = masking_func(aliases)
    else:
        mask = aliases > 0
    return aliases[mask]

# def alias_period(period, df=1/(t_all.max()-t_all.min()).value, n=5, masking=None):
#     f = 1 / period
#     a = f + df * np.arange(-n, n+1, 1)
#     ap = 1 / a
#     if masking:
#         mask = masking(ap)
#     else:
#         mask = a > 0
#     return ap[mask]

def compute_and_plot_ls(
    ax, freq_grid, t, y, dy, label, color,
    p_rot_max_vsini=None, 
    is_multiband=False, band_indices=None, fap_p=None, 
    ranges=None, show_window=True, show_text=True
):
    """
    Compute and plot Lomb-Scargle periodogram + optional window function.
    Handles both standard and multiband analysis.
    """
    time_range = (t.max() - t.min()).value
    
    # 1. Periodogram Calculation
    if is_multiband:
        ls = LombScargleMultiband(t, y, band_indices, dy)
    else:
        ls = LombScargle(t, y, dy)
    power = ls.power(freq_grid)
    
    # 2. Window Function Calculation
    ls_w = None
    if show_window:
        y_window = np.ones(len(t))
        # Window is always standard LS even for multiband
        ls_w = LombScargle(t, y_window, dy, fit_mean=False, center_data=False)
        power_window = ls_w.power(freq_grid)

    # 3. Plotting
    line_p = ax.plot(freq_grid.value, power, c=color, lw=LW, 
                     label='L-S periodogram' if is_multiband else None)
    
    line_w = None
    if show_window:
        _ylim = ax.get_ylim()
        line_w = ax.plot(freq_grid.value, power_window, c=color, lw=LW - 0.5, 
                         ls='--', alpha=0.5, label='Window function' if is_multiband else None)
        ax.set_ylim(_ylim)

    # 4.1. Peak Tagging
    freq_peaks = []
    if ranges is not None:
        for r in ranges:
            f_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
            if np.any(f_mask):
                idx_p = np.argmax(power[f_mask])
                f_p = freq_grid.value[f_mask][idx_p]
                freq_peaks.append(f_p)
                
                ax.axvline(f_p, color='gray', ls='--', alpha=0.3, lw=0.9)
                ax.text(f_p, 0.95, f'{1/f_p:.2f}d', 
                        transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=7, ha='right', va='top', color='0.4')

    # 4.2. Alias Tagging
    if PRIMARY_ALIAS_RANGES is not None:
        for i, r in enumerate(PRIMARY_ALIAS_RANGES):
            f_mask = (freq_grid.value >= 1/r[1]) & (freq_grid.value <= 1/r[0])
            if np.any(f_mask):
                idx_p = np.argmax(power[f_mask])
                f_p = freq_grid.value[f_mask][idx_p]
                period_text = round(1 / f_p)
                aliases = get_aliases(f_p, time_range, n=150, masking_func=None)
                for alias in aliases:
                    for tagged_freq in freq_peaks:
                        if np.isclose(alias, tagged_freq, atol=0.0008):
                            ax.axvline(
                                alias, ls='-', alpha=0.3, lw=0.7,
                                color=ALIAS_TAG_COLORS[i]
                            )
                            ax.text(
                                alias, 0.95, period_text,
                                fontsize=9 if alias == f_p else 7,
                                weight='bold' if alias == f_p else 'normal',
                                style='italic' if alias != f_p else 'normal',
                                transform=ax.get_xaxis_transform(),
                                color=ALIAS_TAG_COLORS[i],
                                ha='center'
                            )

    # 5. Thresholds & Markers
    fap_line = None
    if fap_p is not None:
        try: # LombScargleMultiband doesn't support FAP yet
            f_level = ls.false_alarm_level(fap_p / 100)
            fap_line = ax.axhline(
                f_level, color='gray', ls=':', alpha=0.6, 
                label=f'{fap_p}% ' + ('false-alarm probability' if show_window else 'FAP')
            )
        except AttributeError:
            pass
    
    if p_rot_max_vsini:
        vsini_line = ax.axvline(
            1 / p_rot_max_vsini, color='gray', ls='--', alpha=0.8,
            label=rf'Max $P_{{\rm rot}}$ from $v\,\sin{{i}} = {VSINI}$ km/s'
        )
    else:
        vsini_line = None

    # 6. Aesthetics
    if show_text:
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontweight='bold', va='top', color=color)
    ax.tick_params(axis='both', direction='in')

    return line_p[0], (line_w[0] if line_w else None), fap_line, vsini_line, ls, ls_w


if __name__ == "__main__":
    print(f"Generating high-resolution manual periodogram for {TARGET}...")
    print(f"Grid: {MIN_PERIOD} to {MAX_PERIOD} days")
    
    f_min, f_max = 1.0 / MAX_PERIOD, 1.0 / MIN_PERIOD
    grid_step = 1 / (N_0 * MANUAL_TIME_RANGE)
    f_switch = 1 / np.array(PERIOD_GRID_RANGE[::-1])
    
    freq_grid = np.concatenate([
        np.arange(f_min, f_switch[0], grid_step),
        1 / np.arange(PERIOD_GRID_RANGE[1], PERIOD_GRID_RANGE[0], -1),
        np.arange(f_switch[1], f_max, grid_step)
    ]) / u.d

    # P_max = 2 * pi * R / (v sin i)
    p_rot_max_vsini = (2 * np.pi * R_STAR * 6.957e5) / (VSINI * 86400)
    print(f"Max P_rot from vsini ({VSINI} km/s, R={R_STAR} R_sun): {p_rot_max_vsini:.2f} days")

    fig, axs = plt.subplots(5, 1, figsize=(4.5, 5), sharex=True, sharey=True)
    plt.subplots_adjust(
        left=0.15,
        right=0.95,
        hspace=0.08,
        top=0.89 if SHOW_WINDOW else 0.915,
        bottom=0.09
    )

    legend_handles, legend_labels = [], []

    ls, ls_window = dict(), dict()

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
            print(f"Skipping {band} band: bin_results_{band}.fits not found.")
            continue

        time = Time(binned_table['jd'], format='jd')
        mask = (time > CUTOFF) & (~np.isnan(binned_table['flux']))
        
        t_data, y_data = time[mask], binned_table['flux'].data[mask]
        dy_data = binned_table['intraday_std'].data[mask]

        if len(t_data) < 5:
            print(f"Skipping {band}: insufficient data points.")
            continue

        print(f"Time range for {band} band: {t_data.max() - t_data.min()} days.")

        l_p, l_w, l_fap, l_vsini, ls[band], ls_window[band] = compute_and_plot_ls(
            axs[i], freq_grid, t_data, y_data, dy_data, f"{band} band", 
            COLORS[band],
            p_rot_max_vsini,
            fap_p=FAP,
            ranges=RANGES,
            show_window=SHOW_WINDOW,
            # show_text=False
        )

        if i == 0:
            legend_handles.extend([l_fap, l_vsini])
            legend_labels.extend([l_fap.get_label(), l_vsini.get_label()])

    # --- 5. Multiband Analysis ---
    all_table = table.Table.read(OUTPUT_DIR / "bin_results_all.fits")
    mask = (Time(all_table['jd'], format='jd') > CUTOFF) & (~np.isnan(all_table['flux']))
    
    t_all = Time(all_table['jd'][mask], format='jd')
    y_all, dy_all = all_table['flux'].data[mask], all_table['intraday_std'].data[mask]
    b_all = np.array([BAND_MAP[b] for b in all_table['band'][mask]])

    l_p, l_w, l_fap, l_vsini, ls['multi'], ls_window['multi'] = compute_and_plot_ls(
        axs[4], freq_grid, t_all, y_all, dy_all, "Multiband", 'k', 
        p_rot_max_vsini,
        is_multiband=True, band_indices=b_all, ranges=RANGES,
        show_window=SHOW_WINDOW,
        # show_text=False
    )

    if SHOW_WINDOW:
        legend_handles.extend([l_p, l_w])
        legend_labels.extend([l_p.get_label(), l_w.get_label()])

    # Formatting
    axs[4].set_xticks(1.0 / XTICKS)
    axs[4].set_xticklabels(XTICKS)
    axs[4].set_xlim(1.0 / MAX_PERIOD, 1.0 / MIN_PERIOD)
    _ylb, _ylu = axs[4].get_ylim()
    axs[4].set_ylim(_ylb, _ylu + 0.03)
    axs[4].invert_xaxis()
    fig.supxlabel("Period (days)", y=0.005)
    fig.supylabel("Lomb-Scargle Power", x=0.03)
    fig.legend(
        legend_handles, legend_labels,
        loc='upper center', bbox_to_anchor=(0.53, 1), ncol=2
    )

    save_fn = "periodogram_manual_"
    save_fn += f"{TARGET}_{MIN_PERIOD}-{MAX_PERIOD}"
    save_fn += "_window" if SHOW_WINDOW else ""
    save_fn += "_labeled" if len(RANGES) > 0 else ""
    save_fn += ".pdf"
    plt.savefig(OUTPUT_DIR / save_fn)
    print(f"Manual high-res periodogram saved to {OUTPUT_DIR / save_fn}")
    plt.show()
    plt.close(fig)
