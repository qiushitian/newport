#!/usr/bin/env python3
"""
Plotting color evolution
adapted from cmd.py

Created: 2026-06-23
"""
from tqdm import tqdm
import numpy as np
from astropy import table
import matplotlib.pyplot as plt
from matplotlib import patches, dates as mdates
from astropy.time import Time
from rel_color import TABLE_DIR, CTAB_NAME, COLORS


plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.sans-serif"] = ["Verdana"]

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

TAB_NAME = 'bin_results_all.fits'

TIME_DIV = mdates.date2num(
    Time(['2023-05-20', '2023-09-01', '2024-02-01']).datetime
)
DIV_ROMAN = ['i)', 'ii)', 'iii)', 'iv)']


def get_mask(i, t):
    if i == 0:
        return t < TIME_DIV[0]
    elif i == len(TIME_DIV):
        return t >= TIME_DIV[-1]
    return (t >= TIME_DIV[i - 1]) & (t < TIME_DIV[i])


if __name__ == '__main__':
    tab = table.Table.read(TABLE_DIR / TAB_NAME)

    tab['datenum'] = mdates.date2num(Time(tab['jd'], format='jd').datetime)
    min_t = tab['datenum'].min()
    max_t = tab['datenum'].max()

    for b_band, r_band in tqdm(COLORS, desc='Making CMD evo plot'):
        plot_name = f'color_evo_{b_band}{r_band}.pdf'
        color = f'{b_band}/{r_band}'

        ctab = tab[tab['band'] == color]
        ctab.sort('jd')
        mtab = tab[tab['band'] == r_band]

        c, m, t = [], [], []
        for row in ctab:
            c.append(row['flux'])
            m.append(mtab[mtab['night'] == row['night']]['flux'])
            t.append(row['datenum'])
        c, m, t = np.array(c), np.array(m), np.array(t)

        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        subfigs = fig.subfigures(2, 1)

        ##### Top panel: color-color plot with different time bins #####
        axs = subfigs[0].subplots(1, 4, sharex=True, sharey=True)
        for i, ax in enumerate(axs):
            mask = get_mask(i, t)
            ax.scatter(
                c[~mask], m[~mask], c='gray', s=60, ec='none', alpha=0.2
            )
            ax.scatter(
                c[mask], m[mask], c=t[mask], cmap='viridis',
                vmin=min_t, vmax=max_t, s=60, ec='none'
            )
            ax.xaxis.set_major_formatter('{x:0.2f}')
            ax.set_title(
                DIV_ROMAN[i],
                loc='left', fontweight='bold', x=0.05, y=0.01, fontsize=10
            )
        axs[-1].invert_xaxis()
        subfigs[0].supxlabel(
            f'blue ←       Relative {color} color       → red', fontsize=11
        )

        ##### Bottom panel: light curve of the reference filter #####
        ax = subfigs[1].subplots()
        s = ax.scatter(
            Time(mtab['jd'], format='jd').datetime,
            mtab['flux'],
            c=mtab['datenum'], cmap='viridis', vmin=min_t, vmax=max_t,
            s=60, ec='none'
        )

        for d in TIME_DIV:
            ax.axvline(d, color='gray', linestyle='--')

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_major_formatter('{x:.2f}')

        y0 = ax.get_ylim()[0] + 0.005
        for d, text in zip(np.append(ax.get_xlim()[0], TIME_DIV), DIV_ROMAN):
            ax.text(d + 6, y0, text, fontweight='bold', fontsize=10)

        subfigs[1].supxlabel('Time of observation', fontsize=11)

        ##### Whole figure info #####
        fig.supylabel(f'Relative {r_band}-band flux')
        fig.suptitle(f'{color} Color–Magnitude Diagram Evolution')
        
        save_path = TABLE_DIR / plot_name
        plt.savefig(save_path, bbox_inches='tight')
        
        if len(COLORS) < 2:
            print(f"Target plot saved to {save_path}")
            plt.show()

        plt.close()
