import numpy as np
from astropy import table
import itertools
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches, ticker, dates as mdates
from datetime import datetime
from astropy.time import Time
import newport
from astropy.timeseries import LombScargle
import astropy.units as u

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Verdana"]


TARGET = "HD_191939"
target_name = "HD_191939"
OUTPUT_DIR = Path('tables/opt_comp_stars/HD_191939/ppm')
INPUT_PATH = Path(f'tables/opt_comp_stars/{TARGET}/phot_w_err_{TARGET}_ss.fits')
base_path = Path('tables/opt_comp_stars/HD_191939/two_ss') / "results"


if __name__ == "__main__":
    savefig_path = OUTPUT_DIR / "monitoring.pdf"
    n_std_mid = 9
    bands = ['B', 'V', 'R', 'I']

    N_SIG = 1

    wfc3, stis = newport.get_hst(
        target_name,
        path='xml/HST-17192-visit-status_20260216.xml'
    )

    fig, axs = plt.subplots(
        nrows=len(bands), figsize=(8, 1.7 * len(bands)),
        sharex=True, sharey=True
    )
    if len(bands) == 1: axs = [axs]

    # --- Aesthetic Configuration ---
    use_panel_titles = True  # Toggle between panel titles and per-band legends
    color_wfc3 = 'peru' # Previous: C5
    color_stis = 'olive'   # Previous: C1

    std_mid = 0

    wfc3_line, stis_line = [], []
    
    for i, band in enumerate(bands):
        ax = axs[i]
        bin_path = base_path.parent / f"bin_{base_path.stem}_{band}.fits"
        unbin_path = base_path.parent / f"unbin_{base_path.stem}_{band}.fits"
        
        bin_table = table.Table.read(bin_path)
        unbin_table = table.Table.read(unbin_path)

        # within_sig_colname = [
        #     _ for _ in bin_table.colnames 
        #     if _.startswith('within_') and _.endswith('_sig')
        # ][0]
        # bin_table = bin_table[bin_table[within_sig_colname]]

        print(f'{band}\t{len(bin_table)}\t{len(unbin_table)}')
            
        # Time conversion
        t_unbin = Time(unbin_table['jd'], format='jd').to_datetime()
        t_bin = Time(bin_table['jd'], format='jd').to_datetime()

        # Convert to percent
        unbin_table['flux'] *= 1e2
        bin_table['flux'] *= 1e2
        bin_table['intraday_std'] *= 1e2
        unbin_table['error'] *= 1e2
        bin_table.meta['RMSDAY'] *= 1e2
        bin_table.meta['RMSINTRA'] *= 1e2

        # Calculate shift
        pctile_95 = np.nanpercentile(bin_table['flux'], 95)
        pctile_5 = np.nanpercentile(bin_table['flux'], 5)
        shift = (pctile_95 + pctile_5) / 2
        pctile_95 -= shift
        pctile_5 -= shift
        unbinned_flux = unbin_table['flux'] - shift
        binned_flux = bin_table['flux'] - shift

        binned_error = bin_table['intraday_std']
        unbinned_error = unbin_table['error']

        rms_day = bin_table.meta['RMSDAY']
        rms_intra = bin_table.meta['RMSINTRA']
        
        # Unbinned points in background
        unbin_artist = ax.errorbar(
            t_unbin, unbinned_flux,
            # yerr=unbinned_error,
            fmt='o', color='silver',
            ms=5.5, alpha=0.2, markeredgewidth=0, ecolor='lightgrey', elinewidth=1,
            label='Unbinned'
        )
        
        # Binned points with error bars
        binned_artist = ax.errorbar(
            t_bin, binned_flux,
            yerr=binned_error,
            fmt=newport.MARKERS[band], color=newport.COLORS[band],
            alpha=0.7, markeredgewidth=0,
            ms=7, capsize=6, label=f'{band} band'
        )

        # Plot HST
        for _ in wfc3:
            wfc3_line = ax.axvline(
                _.to_datetime(),
                zorder=4, ls='--', c=color_wfc3, linewidth=1.8, alpha=0.7,
                label='HST/WFC3 planetary transit visits'
            )
        for _ in stis:
            stis_line = ax.axvline(
                _.to_datetime(),
                zorder=4, ls=':', c=color_stis, linewidth=1.8, alpha=0.8,
                label='HST/STIS host star observation'
            )
        
        # high / low lines
        ax.axhline(pctile_95, color='teal', ls='-', lw=0.6, alpha=0.6)
        pctile_line = ax.axhline(
            pctile_5, color='teal', ls='-', lw=0.6, alpha=0.6,
            label='5th/95th percentile'
        )
        
        # Panel Title (Color matched)
        if use_panel_titles:
            ax.text(
                0.06, 0.91,
                rf"{band} band $(A={pctile_95:.2f}~\%)$",
                backgroundcolor=(0.95, 0.95, 0.95, 0.6), zorder=5,
                transform=ax.transAxes, 
                fontweight='bold', 
                fontsize=11,
                va='top',
                color=newport.COLORS[band]
            )
            # ax.legend(
            #     handles=[binned_artist, unbin_artist],
            #     labels=[binned_artist.get_label(), unbin_artist.get_label()],
            #     loc='upper left', bbox_to_anchor=(0.18, 0.95), ncol=1
            # )
        else:
            ax.text(
                datetime(2022, 12, 10), pctile_95, f"95th percentile: {pctile_95:.2f} %",
                va='center', ha='left', fontsize=8, style='italic'
            )
            ax.text(
                datetime(2022, 12, 10), pctile_5, f"5th percentile: {pctile_5:.2f} %",
                va='center', ha='left', fontsize=8, style='italic'
            )

        # # mean line and std patch
        # x1, x2 = ax.get_xlim()
        # # x1, x2 = datetime(2022, 1, 1), datetime(2025, 12, 31)
        # y1, y2 = - rms_day * N_SIG, rms_day * N_SIG
        # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=0.1, color=newport.COLORS[band], lw=0, zorder=0)
        # ax.add_patch(rect)
        # ax.set_xlim(x1, x2)

        # Calculate ylim
        p25, p75 = np.nanpercentile(binned_flux, [25, 75])
        mid_mask = (binned_flux >= p25) & (binned_flux <= p75)
        _std_mid = np.nanstd(binned_flux[mid_mask])
        std_mid = max(std_mid, _std_mid)
        
        ax.grid(True, 'major', alpha=0.3)
        ax.grid(True, 'minor', alpha=0.1)
        ax.tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
        ax.tick_params(axis='y', direction='in', which='both')

        # Limit x-axis tick density and snap to months
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # More y ticks
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        # # Metadata labels
        # ax.set_title(f"Daily RMS: {rms_day:.6f} | Intraday RMS: {rms_intra:.6f}", fontsize=10)
        # ax.legend(loc='upper right', fontsize=8)

    # # Rotation for bottom row
    # plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')

    # Set ylim
    axs[-1].set_ylim(- n_std_mid * std_mid, n_std_mid * std_mid)

    fig.supxlabel("Time of observation", x=0.53, y=0.03)
    fig.supylabel(
        r"$\Delta F$",
        # 'Change in relative flux (%)',
        x=0.025, y=0.5
    )
    
    # # Global Legend matching plot_mag.py style
    # handles, labels = [], []
    # for ax in axs:
    #     h, l = ax.get_legend_handles_labels()
    #     handles.extend(h)
    #     labels.extend(l)

    # handles.extend([
    #     wfc3_line,
    #     # stis_line
    # ])

    # # --- Layered Legend ---
    # # Row 1: Filter Bands (Deduplicated)
    # all_h, all_l = [], []
    # for _ax in axs:
    #     _h, _l = _ax.get_legend_handles_labels()
    #     all_h.extend(_h)
    #     all_l.extend(_l)
    
    # band_handles, band_labels = [], []
    # for h, l in zip(all_h, all_l):
    #     if l in bands and l not in band_labels:
    #         band_handles.append(h)
    #         band_labels.append(l)
    
    # # Sort bands for consistency (B, V, R, I)
    # sorted_indices = np.argsort([['B', 'V', 'R', 'I'].index(b) for b in band_labels])
    # band_handles = [band_handles[i] for i in sorted_indices]
    # band_labels = [band_labels[i] for i in sorted_indices]

    # # Row 2: HST & Others
    # hst_handles, hst_labels = [], []
    # for h, l in zip(all_h, all_l):
    #     if 'HST' in l and l not in band_labels and l not in hst_labels:
    #         hst_handles.append(h)
    #         hst_labels.append(l)

    # # Place Band Legend
    # leg1 = fig.legend(band_handles, band_labels, ncol=len(band_labels), 
    #                   loc='upper center', bbox_to_anchor=(0.54, 1.003), fontsize=9)
    # fig.add_artist(leg1)
    
    # # Place HST Legend (spanning row)
    # if hst_handles:
    #     leg2 = fig.legend(hst_handles, hst_labels, ncol=1, 
    #                loc='upper center', bbox_to_anchor=(0.54, 0.965), fontsize=9)
    #     fig.add_artist(leg2)

    handles, labels = [], []
    # for ax in axs:
    #     h, l = ax.get_legend_handles_labels()
    #     for hh, ll in zip(h, l):
    #         # If using panel titles, exclude band labels from the legend
    #         label_condition = (ll not in labels)
    #         if use_panel_titles:
    #             label_condition = (ll not in labels and 'band' not in ll)
            
    #         if label_condition:
    #             handles.append(hh)
    #             labels.append(ll)

    # if hasattr(wfc3_line, "__getitem__"):
    #     wfc3_line = wfc3_line[0]
    # if hasattr(stis_line, "__getitem__"):
    #     stis_line = stis_line[0]

    # if handles and labels:  # TODO verify?
    handles = [wfc3_line, stis_line, pctile_line]
    labels = [l.get_label() for l in handles]

    print(f'this is handles: {handles}')
    print(f'this is labels: {labels}')

    fig.legend(
        ncol=3,  # 2 if use_panel_titles else 3,
        loc='upper center', bbox_to_anchor=(0.5, 1),
        handles=handles, labels=labels
    )
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)  # Make room for dual legends
    
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight') # TODO verify if bbox tight is needed
        print(f"Target plot saved to {savefig_path}")
    plt.show()
    plt.close()
