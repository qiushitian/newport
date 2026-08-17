#!/usr/bin/env python3
"""
Plotting cmd
adapted from plot_rel_color.py

Created: 2026-06-22
"""
import numpy as np
from astropy import table
import matplotlib.pyplot as plt
from matplotlib import patches, dates as mdates
from astropy.time import Time
import newport
from rel_color import TABLE_DIR, CTAB_NAME


# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = ["Verdana"]

plt.rcParams["font.family"] = "serif"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

TAB_NAME = 'bin_results_all.fits'
LC_BAND = 'R'
PLOT_NAME = 'cmd.pdf'


if __name__ == '__main__':
    tab = table.Table.read(TABLE_DIR / TAB_NAME)
    tab['datenum'] = mdates.date2num(Time(tab['jd'], format='jd').datetime)

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax_dict = fig.subplot_mosaic([
        ['B/V', 'lc', 'lc'],
        ['B/R', 'V/R', 'c'],
        ['B/I', 'V/I', 'R/I']
    ])

    min_t = tab['datenum'].min()
    max_t = tab['datenum'].max()

    for color, ax in ax_dict.items():
        if color == 'lc':
            btab = tab[tab['band'] == LC_BAND]
            s = ax.scatter(
                Time(btab['jd'], format='jd').datetime,
                btab['flux'],
                c=btab['datenum'], cmap='viridis',
                vmin=min_t, vmax=max_t, s=60, ec='none', alpha=0.7
            )

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.set_xlabel('Time of observation')

            ax.yaxis.set_major_formatter('{x:.2f}')
            ax.set_ylabel(LC_BAND)
        elif color != 'c':
            ctab = tab[tab['band'] == color]
            ctab.sort('jd')
            ftab = tab[tab['band'] == color[-1]]
            
            c, m, t = [], [], []
            for r in ctab:
                c.append(r['flux'])
                m.append(ftab[ftab['night'] == r['night']]['flux'])
                t.append(r['datenum'])

            ax.scatter(
                c, m, c=t, cmap='viridis',
                vmin=min_t, vmax=max_t, s=60, ec='none', alpha=0.7
            )

            ax.invert_xaxis()
            ax.set_xlabel(color)
            ax.set_ylabel(color[-1])

    cbar = fig.colorbar(
        s,
        cax=ax_dict['c'],
        fraction=0.35,
        shrink=0.75,
        label='Time of observation'
    )

    ax_dict['c'].set_yticks(ax_dict['lc'].get_xticks())
    ax_dict['c'].yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    pos = ax_dict['c'].get_position()
    ax_dict['c'].set_position([
        pos.x0 + pos.width * 0.4,
        pos.y0 + pos.height * 0.2,
        pos.width * 0.15,
        pos.height
    ])

    pos = ax_dict['lc'].get_position()
    ax_dict['lc'].set_position([
        pos.x0 + pos.width * 0.2,
        pos.y0 + pos.height * 0.6,
        pos.width * 0.8,
        pos.height * 0.7
    ])

    fig.supxlabel('blue ←    Relative color    → red')
    fig.supylabel('dim ←    Relative flux    → bright')
    fig.suptitle('Color–Magnitude Diagrams')
    
    save_path = TABLE_DIR / PLOT_NAME
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Target plot saved to {save_path}")

    plt.show()

    plt.close()
