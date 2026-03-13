#!/usr/bin/env python3
"""
Big Plot: Two-section diagnostic figure for examining anomalous nights.

Top section:  4 stacked BVRI panels (no vertical gaps) with full light curve.
Bottom section: 6-column grid of zoomed-in panels for flagged nights.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import dates as mdates, ticker
from astropy import table
from astropy.time import Time
from pathlib import Path
from zoneinfo import ZoneInfo
import newport

EASTERN = ZoneInfo('US/Eastern')

# ── Configuration ──────────────────────────────────────────────
READ_DIR = Path('tables/opt_comp_stars/HD_191939/two')
BANDS = ['B', 'V', 'R', 'I']

HIGH_STD = 0.08    # Flag night if intraday_std / flux > this in any band
HIGH_DEVI = 2.5     # Flag night if |flux - 1| > this many sigmas in any band

N_COLS = 4        # Number of columns in the bottom grid
MAX_ROWS = 10      # Maximum rows in the bottom grid (caps at N_COLS * MAX_ROWS nights)

N_STD_MID = 22
SAVEFIG_NAME = 'anomalous_nights.pdf'

PLOT_UNBINNED = False
PLOT_AIRMASS = False  # Show airmass on twin y-axis in bottom panels


def load_tables(read_dir, bands):
    """Load binned and unbinned FITS tables for all bands."""
    bin_tables, unbin_tables = {}, {}
    for band in bands:
        bin_path = read_dir / f'bin_results_{band}.fits'
        unbin_path = read_dir / f'unbin_results_{band}.fits'
        if bin_path.exists() and unbin_path.exists():
            bin_tables[band] = table.Table.read(bin_path)
            unbin_tables[band] = table.Table.read(unbin_path)
    return bin_tables, unbin_tables


def find_flagged_nights(bin_tables, bands, high_std, high_devi):
    """
    Identify nights that need attention.

    A night is flagged if, in any band:
      1) intraday_std / flux > high_std, OR
      2) |flux - 1| > high_devi * sigma (sigma = std of all binned flux in that band)
    """
    flagged = set()

    for band in bands:
        bt = bin_tables[band]

        # Full light curve sigma for this band
        sigma = bt['flux'].std()

        for row in bt:
            night = row['night']
            flux = row['flux']
            intraday_std = row['intraday_std']

            # Criterion 1: high intraday scatter relative to flux
            if intraday_std / flux > high_std:
                flagged.add(night)

            # Criterion 2: binned flux deviates from 1
            if abs(flux - 1) > high_devi * sigma:
                flagged.add(night)

    return sorted(flagged)


def night_to_date_str(night_int, mmdd=False):
    """Convert YYYYMMDD integer to YYYY-MM-DD string."""
    s = str(int(night_int))
    return f"{s[4:6]}-{s[6:8]}" if mmdd else f"{s[:4]}-{s[4:6]}-{s[6:8]}"


if __name__ == '__main__':
    bin_tables, unbin_tables = load_tables(READ_DIR, BANDS)
    if not bin_tables:
        print("No tables found.")
        exit()

    # ── Find flagged nights ──
    flagged_nights = find_flagged_nights(bin_tables, BANDS, HIGH_STD, HIGH_DEVI)
    n_flagged = len(flagged_nights)
    print(f"Flagged {n_flagged} nights: {flagged_nights}")

    if n_flagged == 0:
        print("No anomalous nights found. Nothing to plot in bottom section.")

    # Cap to grid capacity
    n_flagged = min(n_flagged, N_COLS * MAX_ROWS)
    flagged_nights = flagged_nights[:n_flagged]
    n_bottom_rows = max(1, int(np.ceil(n_flagged / N_COLS))) if n_flagged > 0 else 0

    # ── Figure layout ──
    n_top_rows = len(BANDS)

    fig = plt.figure(
        figsize=(10, 1.4 * n_top_rows + 1.6 * n_bottom_rows),
        # dpi=200
    )

    # Outer gridspec: top section vs bottom section
    outer_gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[n_top_rows * 1.4, n_bottom_rows * 1.6] if n_bottom_rows > 0 else [1, 0.001],
        hspace=0.12
    )

    # ── TOP SECTION: stacked BVRI ──
    top_gs = gridspec.GridSpecFromSubplotSpec(
        n_top_rows, 1, subplot_spec=outer_gs[0], hspace=0
    )
    top_axs = []
    for i in range(n_top_rows):
        share_x = top_axs[0] if i > 0 else None
        share_y = top_axs[0] if i > 0 else None
        ax = fig.add_subplot(top_gs[i], sharex=share_x, sharey=share_y)
        top_axs.append(ax)
        if i < n_top_rows - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    # Calculate global ylim from all bands
    std_mid = 0
    for band in BANDS:
        if band not in bin_tables:
            continue
        flux = bin_tables[band]['flux']
        valid = np.isfinite(flux)
        p25, p75 = np.nanpercentile(flux[valid], [25, 75])
        mid_mask = (flux >= p25) & (flux <= p75)
        _std = np.nanstd(flux[mid_mask]) if np.any(mid_mask) else np.nanstd(flux[valid])
        std_mid = max(std_mid, _std)

    flagged_set = set(flagged_nights)

    for i, band in enumerate(BANDS):
        ax = top_axs[i]
        if band not in bin_tables:
            continue
        bt = bin_tables[band]
        ut = unbin_tables[band]

        t_unbin = Time(ut['jd'], format='jd').to_datetime()
        t_bin = Time(bt['jd'], format='jd').to_datetime()

        # Unbinned (faint background)
        if PLOT_UNBINNED:
            ax.plot(
                t_unbin, ut['flux'], 'o', color='silver',
                ms=3, alpha=0.3, markeredgewidth=0
            )

        # Binned: split into flagged (full alpha) vs normal (dimmed)
        is_flagged = np.array([n in flagged_set for n in bt['night']])
        yerr = bt['intraday_std'] if 'intraday_std' in bt.colnames else None

        # Normal points (dimmed)
        if np.any(~is_flagged):
            ax.errorbar(
                np.array(t_bin)[~is_flagged], bt['flux'][~is_flagged],
                yerr=yerr[~is_flagged] if yerr is not None else None,
                fmt=newport.MARKERS[band], color=newport.COLORS[band],
                alpha=0.2, markeredgewidth=0, ms=7
            )

        # Flagged points (full alpha)
        if np.any(is_flagged):
            ax.errorbar(
                np.array(t_bin)[is_flagged], bt['flux'][is_flagged],
                yerr=yerr[is_flagged] if yerr is not None else None,
                fmt=newport.MARKERS[band], color=newport.COLORS[band],
                alpha=0.9, markeredgewidth=0, ms=7,
                # label=f'{band} band'
            )

        # Add MM-DD text labels for flagged nights (only on the bottom axis)
        if i == len(BANDS) - 1:
            labeled_nights = set()
            for row_idx in range(len(bt)):
                night = bt['night'][row_idx]
                if night in flagged_set and night not in labeled_nights:
                    labeled_nights.add(night)
                    t_pt = t_bin[row_idx]
                    y_pt = bt['flux'][row_idx]
                    ax.annotate(
                        night_to_date_str(night, mmdd=True),
                        (t_pt, y_pt),
                        textcoords='offset points',
                        xytext=(0, -18 if y_pt > 0.95 else 15),
                        fontsize=7, ha='center', va='top',
                        rotation=0, color='0.3'
                    )

        ax.tick_params(axis='both', direction='in', labelsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.set_title(f'{band} band', loc='left', x=0.02, y=0.75)
        ax.grid(True, alpha=0.2)

    top_axs[0].set_ylim(1 - N_STD_MID * std_mid, 1 + N_STD_MID * std_mid)

    # # Top legend
    # handles, labels = [], []
    # for ax in top_axs:
    #     h, l = ax.get_legend_handles_labels()
    #     for hh, ll in zip(h, l):
    #         if ll not in labels:
    #             handles.append(hh)
    #             labels.append(ll)
    # fig.legend(
    #     handles, labels, ncol=len(labels),
    #     loc='upper center', bbox_to_anchor=(0.5, 1),
    #     # fontsize=9
    # )

    # ── BOTTOM SECTION: per-night zoom-ins ──
    if n_flagged > 0:
        bot_gs = gridspec.GridSpecFromSubplotSpec(
            n_bottom_rows, N_COLS, subplot_spec=outer_gs[1],
            hspace=0.4, wspace=0.3
        )

        bot_axs = []
        airmass_ref_ax = None
        for idx, night in enumerate(flagged_nights):
            # Row-first ordering: fill across rows first, then columns
            col = idx // n_bottom_rows
            row = idx % n_bottom_rows
            ax = fig.add_subplot(bot_gs[row, col])  # , sharey=share_y)
            bot_axs.append(ax)

            # Collect all data for this night across bands
            for band in BANDS:
                if band not in bin_tables:
                    continue
                bt = bin_tables[band]
                ut = unbin_tables[band]

                # Unbinned data for this night (with error bars)
                u_mask = ut['night'] == night
                if np.any(u_mask):
                    t_u = Time(ut['jd'][u_mask], format='jd').to_datetime(
                        timezone=EASTERN
                    )
                    ax.errorbar(
                        t_u, ut['flux'][u_mask],
                        yerr=ut['error'][u_mask] if 'error' in ut.colnames else None,
                        fmt=newport.MARKERS[band], color=newport.COLORS[band],
                        ms=4, alpha=0.4, markeredgewidth=0
                    )

                # Binned data for this night (reference marker with intraday_std)
                b_mask = bt['night'] == night
                if np.any(b_mask):
                    t_b = Time(bt['jd'][b_mask], format='jd').to_datetime(
                        timezone=EASTERN
                    )
                    ax.errorbar(
                        t_b, bt['flux'][b_mask],
                        yerr=bt['intraday_std'][b_mask] if 'intraday_std' in bt.colnames else None,
                        fmt=newport.MARKERS[band], color=newport.COLORS[band],
                        ms=7, alpha=0.8,
                        markeredgewidth=1, markeredgecolor=(0, 0, 0, 0.4)
                    )

            title = night_to_date_str(night)
            ax.tick_params(axis='both', direction='in', labelsize=7)
            ax.grid(True, alpha=0.2)

            # X formatting: zoom to night, show hours in Eastern Time
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # plt.setp(
            #     ax.get_xticklabels(),
            #     rotation=45, ha='right', fontsize=6
            # )

            # Collect airmass from unbinned data across all bands for this night
            airmass_t, airmass_v = [], []
            for band in BANDS:
                if band not in unbin_tables:
                    continue
                ut = unbin_tables[band]
                u_mask = ut['night'] == night
                if np.any(u_mask):
                    airmass_t.extend(
                        Time(ut['jd'][u_mask], format='jd').to_datetime(
                            timezone=EASTERN
                        )
                    )
                    airmass_v.extend(ut['airmass'][u_mask])

            if airmass_t:
                if PLOT_AIRMASS:
                    # Sort by time for a connected line
                    sort_idx = np.argsort(airmass_t)
                    airmass_t = np.array(airmass_t)[sort_idx]
                    airmass_v = np.array(airmass_v)[sort_idx]

                    ax2 = ax.twinx()
                    ax2.plot(
                        airmass_t, airmass_v, '-',
                        color='C1', lw=0.5, alpha=0.8
                    )
                    if col == N_COLS - 1:
                        ax2.set_ylabel(
                            'Airmass', x=1.03, fontsize=5, color='tan', rotation=270
                        )
                    ax2.tick_params(
                        axis='y', labelsize=5, colors='tan',
                        direction='in'
                    )
                    # Share airmass y-axis across all bottom panels
                    if airmass_ref_ax is None:
                        airmass_ref_ax = ax2
                    else:
                        ax2.sharey(airmass_ref_ax)
                else:
                    title += rf', $\langle \mathrm{{AM}} \rangle = {np.mean(airmass_v):.1f}$'

            ax.set_title(title, y=0.95, fontsize=8)

        # Hide unused bottom panels
        for idx in range(n_flagged, n_bottom_rows * N_COLS):
            col = idx // n_bottom_rows
            row = idx % n_bottom_rows
            ax = fig.add_subplot(bot_gs[row, col])
            ax.set_visible(False)

    fig.supxlabel("Time of observation", y=0.07)
    fig.supylabel("Relative flux", x=0.07)
    fig.subplots_adjust(top=0.96)

    if SAVEFIG_NAME:
        savefig_path = READ_DIR / SAVEFIG_NAME
        plt.savefig(savefig_path, bbox_inches='tight')
        print(f"Saved to {savefig_path}")

    # plt.show()
    plt.close(fig)
