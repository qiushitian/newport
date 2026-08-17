#!/usr/bin/env python3
"""
Plotting ccd
adapted from cmd.py

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

TARGET_NAME = 'HD_191939'
CTAB_NAME = 'colors.fits'
PLOT_NAME = 'ccd.pdf'

AX_FLAVOR = {
    '1': ('B/V', 'V/R'),
    '2': ('B/V', 'V/I'),
    '3': ('B/R', 'R/I'),
    '4': ('B/V', 'R/I'),
    '5': ('V/R', 'R/I')
}


if __name__ == '__main__':
    ctab = table.Table.read(TABLE_DIR / CTAB_NAME)

    fig = plt.figure(figsize=(5, 9), constrained_layout=True)
    ax_dict = fig.subplot_mosaic(
        '''
        1c
        2c
        3c
        45
        '''
    )

    min_t = mdates.date2num(Time(ctab['jd'].min(), format='jd').datetime)
    max_t = mdates.date2num(Time(ctab['jd'].max(), format='jd').datetime)

    for ax_id, flavor in AX_FLAVOR.items():
        ax = ax_dict[ax_id]
            
        btab = ctab[ctab['band'] == flavor[0]]
        btab.sort('jd')
        rtab = ctab[ctab['band'] == flavor[1]]
        rtab.sort('jd')
        
        b, r, t = [], [], []
        for row in btab:
            r_idx = np.where(rtab['night'] == row['night'])[0]
            if len(r_idx) == 1:
                rrow = rtab[r_idx[0]]
                b.append(row['flux'])
                r.append(rrow['flux'])
                t.append(
                    Time(np.mean([row['jd'], rrow['jd']]), format='jd').datetime
                )
            elif len(r_idx) > 1:
                raise ValueError(
                    f'There are {len(r_idx)} {flavor[1]}-band measurements on {row["night"]} while there is only 1 {flavor[0]}-band measurement.'
                )
            

        s = ax.scatter(
            r, b,
            c=mdates.date2num(t), cmap='viridis', vmin=min_t, vmax=max_t,
            s=60, ec='none', alpha=0.7
        )

        ax.invert_xaxis()
        ax.set_xlabel(flavor[1])
        ax.set_ylabel(flavor[0])

    cbar = fig.colorbar(s, cax=ax_dict['c'], label='Time of observation')
    ax_dict['c'].yaxis.set_major_locator(mdates.MonthLocator())
    ax_dict['c'].yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    pos = ax_dict['c'].get_position()
    ax_dict['c'].set_position([
        pos.x0 + pos.width * 0.4,
        pos.y0 + pos.height * 0.2,
        pos.width * 0.1,
        pos.height * 0.8
    ])

    fig.supxlabel('blue ←    Relative color    → red')
    fig.supylabel('red ←    Relative color    → blue')
    fig.suptitle('Color–Color Diagrams')
    
    save_path = TABLE_DIR / PLOT_NAME
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Target plot saved to {save_path}")

    plt.show()

    plt.close()
