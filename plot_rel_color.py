#!/usr/bin/env python3
"""
Plotting (relative) color made by rel_color.py
adapted from optimize_rel_phot.plot_target_ppm

Created: 2026-06-22
"""
import numpy as np
from astropy import table
import matplotlib.pyplot as plt
from matplotlib import patches, dates as mdates
from astropy.time import Time
import newport
from rel_color import TABLE_DIR, CTAB_NAME


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Verdana"]

TARGET_NAME = 'HD_191939'
PLOT_NAME = 'colors.pdf'


if __name__ == '__main__':
    N_SIG = 1
    N_STD_MID = 12

    ctab = table.Table.read(TABLE_DIR / CTAB_NAME)
    colors = np.unique(ctab['band'])

    wfc3, stis = newport.get_hst(
        TARGET_NAME,
        path='xml/HST-17192-visit-status_20260216.xml'
    )

    fig, axs = plt.subplots(
        nrows=len(colors), figsize=(8, 1.7 * len(colors)),
        sharex=True, sharey=True
    )
    if len(colors) == 1: axs = [axs]

    # --- Aesthetic Configuration ---
    use_panel_titles = True  # Toggle between panel titles and per-band legends
    color_wfc3 = 'peru' # Previous: C5
    color_stis = 'olive'   # Previous: C1

    std_mid = 0

    wfc3_line, stis_line = [], []
    
    for i, color in enumerate(colors):
        ax = axs[i]

        ctab_i = ctab[ctab['band'] == color]
            
        # Time conversion
        times = Time(ctab_i['jd'], format='jd').to_datetime()

        # Normalize
        ctab_i['flux'] /= np.nanmedian(ctab_i['flux'])
        ctab_i['error'] /= np.nanmedian(ctab_i['flux'])
        std = np.nanstd(ctab_i['flux'])
        
        # Plot with error bars
        artist = ax.errorbar(
            times, ctab_i['flux'],
            yerr=ctab_i['error'],
            fmt='o',
            alpha=0.7, markeredgewidth=0,
            ms=7, capsize=6, label=color
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
        
        # Panel Title (Color matched)
        if use_panel_titles:
            ax.text(
                0.18, 0.92, color, transform=ax.transAxes, 
                fontweight='bold', va='top',
                # color=newport.COLORS[band],
                fontsize=11
            )
            # ax.legend(
            #     handles=artist,
            #     labels=artist.get_label(),
            #     loc='upper left', bbox_to_anchor=(0.18, 0.95), ncol=1
            # )

        # mean line and std patch
        x1, x2 = ax.get_xlim()
        # x1, x2 = datetime(2022, 1, 1), datetime(2025, 12, 31)
        y1, y2 = 1 - std * N_SIG, 1 + std * N_SIG
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=0.1, color='gray', lw=0, zorder=0)
        ax.add_patch(rect)
        ax.set_xlim(x1, x2)

        # Calculate ylim
        p25, p75 = np.nanpercentile(ctab_i['flux'], [25, 75])
        mid_mask = (ctab_i['flux'] >= p25) & (ctab_i['flux'] <= p75)
        _std_mid = np.nanstd(ctab_i['flux'][mid_mask])
        std_mid = max(std_mid, _std_mid)
        
        ax.grid(True, 'major', alpha=0.3)
        ax.grid(True, 'minor', alpha=0.1)
        ax.tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
        ax.tick_params(axis='y', direction='in')

        # Limit x-axis tick density and snap to months
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # # Metadata labels
        # std = bin_table.meta.get('RMS_DAY', 0)
        # rms_intra = bin_table.meta.get('RMS_INTRA', 0)
        # ax.set_title(f"Daily RMS: {std:.6f} | Intraday RMS: {rms_intra:.6f}", fontsize=10)
        # ax.legend(loc='upper right', fontsize=8)

    # # Rotation for bottom row
    # plt.setp(axs[-1].get_xticklabels(), rotation=45, ha='right')

    # Set ylim
    axs[-1].set_ylim(1 - N_STD_MID * std_mid, 1 + N_STD_MID * std_mid)

    fig.supxlabel("Time of observation", x=0.53, y=0.03)
    fig.supylabel("Color", x=0.025, y=0.5)
    
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
    handles = [wfc3_line, stis_line]
    labels = [wfc3_line.get_label(), stis_line.get_label()]

    print(f'this is handles: {handles}')
    print(f'this is labels: {labels}')

    fig.legend(
        ncol=2 if use_panel_titles else 3,
        loc='upper center', bbox_to_anchor=(0.5, 1),
        handles=handles, labels=labels
    )
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.94)  # Make room for dual legends
    
    save_path = TABLE_DIR / PLOT_NAME
    plt.savefig(save_path, bbox_inches='tight') # TODO verify if bbox tight is needed
    print(f"Target plot saved to {save_path}")

    plt.show()

    plt.close()
