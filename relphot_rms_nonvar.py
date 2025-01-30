#!/usr/bin/env python3
"""
Take photometry_*.fits tables and do relative photometry,
output relphot_*.fits and binned_*.fits tables
"""

import numpy as np
from astropy import table
from newport import *
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt

# Percentile of non-NaN observations required for column to be considered valid
# CRITERION = 70
N_COL_HEAD = 5  # TODO replace with str.isdigit()

# for fn in target_fn:
for fn in ['TOI-1759', 'TOI-1410']:
#     if '191939' in fn or '86226' in fn:
#     if '86226' in fn:
#         continue

    phot_table_all_band = table.Table.read(f'tables/phot/photometry_{fn}.fits')
    fig, axs = plt.subplots(nrows=4, figsize=(10, 10), sharex=True, dpi=300)
    for i, band in enumerate(['B', 'V', 'R', 'I']):
        phot_table = phot_table_all_band[phot_table_all_band['band'] == band]

        if len(phot_table) < 1:
            print(f'{fn}\t{band}\t skipped')
            continue

        # body_cols = phot_table.columns[N_COL_HEAD:]
        # valid_counts = [np.sum(~ np.isnan(col)) for col in body_cols]
        # # for col in body_cols:
        # #     valid_counts.append(np.sum(~np.isnan(col)))
        # valid_counts_threshold = np.percentile(valid_counts, CRITERION)
        # valid_col_mask = np.array(valid_counts) > valid_counts_threshold
        # valid_table = body_cols[valid_col_mask]
        #
        # valid_col = []
        # for colname in phot_table.colnames[N_COL_HEAD:]:
        #     col_data = phot_table[colname]
        #     valid_fraction = np.sum(~np.isnan(col_data)) / len(col_data)
        #     if valid_fraction > CRITERION:
        #         valid_col.append(colname)
        #     else:
        #         valid_col.append(colname)

        # 2
        valid_colnames = []
        gaia_g_mags = []
        for colname in phot_table.colnames[N_COL_HEAD:]:
            if colname == TARGET_GAIA_DR3[fn]:
                query = f"""
                        SELECT phot_g_mean_mag
                        FROM gaiadr3.gaia_source
                        WHERE source_id = {colname}
                        """
                job = Gaia.launch_job_async(query)
                result = job.get_results()
                gaia_g_mags.append(result["phot_g_mean_mag"][0])
                valid_colnames.append(colname)
            elif not isinstance(phot_table[colname], table.Table.MaskedColumn):
                query = f"""
                        SELECT phot_g_mean_mag, phot_variable_flag
                        FROM gaiadr3.gaia_source
                        WHERE source_id = {colname}
                        """
                job = Gaia.launch_job_async(query)
                result = job.get_results()
                if len(result) > 0 and not result["phot_variable_flag"][0] == "VARIABLE":
                    gaia_g_mags.append(result["phot_g_mean_mag"][0])
                    valid_colnames.append(colname)

        print(f'{fn}\t{band}\t{len(valid_colnames)}')  # DEV

        valid_table = phot_table[valid_colnames]

        relphot_table = table.Table(names=valid_table.colnames)
        for row in valid_table:
            # relphot_row = []
            row_as_array = np.array(list(row))
            row_sum = np.nansum(row_as_array)
            relphot_table.add_row(row_as_array / (row_sum - row_as_array))  # TODO
            # for i, value in enumerate(row):
            #     relphot_row.append(value / (row_sum - value))
            # relphot_table_no_head.add_row(relphot_row)

        # relphot_table = table.hstack([phot_table[phot_table.colnames[ : N_COL_HEAD]], relphot_table_no_head])
        relphot_table = table.hstack([phot_table[phot_table.colnames[: N_COL_HEAD]], relphot_table])

        gpb = relphot_table.group_by(['night', 'band'])
        binned_relphot = gpb.groups.aggregate(np.median)
        # TODO add std or mad_std here and save it

        # band-night normalization
        for colname in binned_relphot.colnames:
            if colname.isdigit():
                med = np.nanmedian(binned_relphot[colname])
                binned_relphot[colname] /= med
                relphot_table[colname] /= med

        # plot rms
        rmses = []
        target_index = -1
        for col_index, col in enumerate(valid_colnames):
            data = binned_relphot[col]
            if np.ma.isMaskedArray(data):
                rms = np.sqrt(np.mean(data.compressed() ** 2))  # For masked arrays
            else:
                rms = np.sqrt(np.nanmean(data ** 2))  # For regular arrays with possible NaNs
            rmses.append(rms)
            if col == TARGET_GAIA_DR3[fn]:
                target_index = col_index
        axs[i].plot(gaia_g_mags[: target_index], rmses[: target_index], MARKERS[band], c=COLORS[band])
        try:
            axs[i].plot(gaia_g_mags[target_index + 1:], rmses[target_index + 1:], MARKERS[band], c=COLORS[band])
        except IndexError as e:
            print(str(e))
        axs[i].plot(gaia_g_mags[target_index], rmses[target_index], '*', ms=16, c=COLORS[band], label=band)

        binned_relphot.write(f'tables/binned/binned_{fn}_{band}_rms_nonvar.fits', overwrite=True)
        relphot_table.write(f'tables/relphot/relphot_{fn}_{band}_rms_nonvar.fits', overwrite=True)

    fig.supxlabel('Gaia G mag', y=0.06)
    fig.supylabel('Light curve RMS', x=0.05)
    fig.suptitle(fn, y=0.91)
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.9))
    plt.savefig(f'fig/rms/{fn}.pdf')
    plt.savefig(f'fig/rms/{fn}.png')