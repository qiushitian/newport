#!/usr/bin/env python3
"""

Made from old_magphot.py
"""

import numpy as np
import astropy.units as u
from astropy import table
from astropy.stats import sigma_clipped_stats
import newport
import matplotlib.pyplot as plt
from astroquery.utils.tap.core import TapPlus

SIMBAD_TAP = TapPlus(url="http://simbad.u-strasbg.fr/simbad/sim-tap")

N_COL_HEAD = 5  # TODO replace with str.isdigit()
N_SIGMA = 5


for fn in newport.TARGET_FN:
    if '86226' not in fn:
        continue

    phot_target = newport.TARGET_GAIA_DR3[fn]  # TODO

    for band in ['B', 'V', 'R', 'I']:
        # if band != 'V':
        #     continue

        onerel_table = table.Table.read(
            f'tables/onerel/{fn}_field/onerel_field={fn}_target={phot_target}_band={band}.fits')

        mag_dict = {}
        for colname in onerel_table.colnames[N_COL_HEAD:]:
            mag = SIMBAD_TAP.launch_job(
                f"SELECT {band} FROM allfluxes JOIN ident USING(oidref) WHERE id = 'Gaia DR3 {colname}';"
            ).get_results()[band]
            if len(mag) > 0 and not mag.mask:
                mag_dict[colname] = mag[0]

        print(f'{fn}\t{band}\twith mag\t{len(mag_dict)}')  # TODO DEV

        mag_table = table.Table()
        for colname, mag in mag_dict.items():
            target_mag = -2.5 * np.log10(onerel_table[colname]) + mag
            mag_table[colname] = target_mag * u.mag

        mag_table = table.hstack([onerel_table[onerel_table.colnames[: N_COL_HEAD]], mag_table])
        mag_table.write(
            f'tables/mag_from_onerel/mag_from_onerel_unmasked_field={fn}_target={phot_target}_band={band}.fits',
            overwrite=False)

        plt.figure()
        mag_table = table.Table(mag_table, masked=True)  # Convert to a MaskedTable
        for i, colname in enumerate(mag_table.colnames[N_COL_HEAD:]):
            col = mag_table[colname]
            # if np.issubdtype(col.dtype, np.number):
            col.mask += col < 0
            center_value = np.nanmedian(col)
            std = np.nanstd(col)
            # center_value, median, std = sigma_clipped_stats(col, sigma=n_sigma)
            col.mask += np.abs(col - center_value) > N_SIGMA * std

            plt.hist(col, bins=20, color=f'C{i}', histtype='stepfilled', alpha=0.6, label=f'Gaia DR3 {colname}')
            plt.axvline(center_value, color=f'C{i}', lw=1,
                        label=f'{fn.replace("_", " ")} $ = ({center_value:.2f} \\pm {std:.2f})$ mag')

        _lit_mag = newport.LITERATURE_MAG[fn][band]
        plt.axvline(_lit_mag, color='k', lw=1, ls='--', label=f'{fn.replace("_", " ")} literature mag = {_lit_mag}')
        _xlim = plt.xlim()
        plt.xlim(_xlim[1], _xlim[0])
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
        plt.xlabel('Magnitude')
        plt.ylabel('Number of exposures')
        # plt.title(f"{band} band Gaia DR3 {phot_target} in {fn.replace('_', ' ')} field")
        plt.title(f"{band} band {fn.replace('_', ' ')} predicted by...")
        plt.savefig(f'tables/mag_from_onerel/diagnostic_plot_field={fn}_target={phot_target}_band={band}.pdf',
                    bbox_inches='tight')

        mag_table.write(
            f'tables/mag_from_onerel/mag_from_onerel_masked_field={fn}_target={phot_target}_band={band}.fits',
            overwrite=False)
