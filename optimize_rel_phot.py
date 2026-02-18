#!/usr/bin/env python3
"""
Relative Photometry Optimization Engine.
Performs exhaustive search for the optimal comparison star ensemble.
"""

import numpy as np
from astropy import table
import itertools
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
from astropy.time import Time
import newport

class RelativePhotometryEngine:
    def __init__(self, input_table, target_id):
        # Standardize table: ensure it's not masked and fills gaps with NaNs
        # TODO maybe we just make sure phot_[gaia|list]_run never produces MaskedColumn
        if input_table.has_masked_columns:
            self.table = input_table.filled(np.nan)
        else:
            self.table = input_table.copy()
        
        self.target_id = str(target_id)
        
    def _rel_phot(self, f_t, e_t, f_e, v_e):
        """Internal math for relative flux and error propagation."""
        rel_flux = f_t / f_e
        # sigma_rel = rel_flux * sqrt( (sig_t/f_t)**2 + (sig_e/f_e)**2 )
        # where var_target = e_t**2, var_ensemble = v_e
        rel_err = rel_flux * np.sqrt((e_t / f_t)**2 + (v_e / f_e**2))
        return rel_flux, rel_err

    def get_unbinned(self, comp_ids):
        """
        Calculates image-by-image relative flux and propagated error.
        
        Args:
            comp_ids (list): List of comparison star IDs.
        
        Returns:
            astropy.table.Table: Table with night, jd, airmass, exptime, flux, error.
        """
        f_target = self.table[self.target_id]
        err_target = self.table[f"err_{self.target_id}"]
        
        f_ens = np.sum([self.table[cid] for cid in comp_ids], axis=0)
        var_ens = np.sum([self.table[f"err_{cid}"]**2 for cid in comp_ids], axis=0)
        
        fluxes, errors = self._rel_phot(f_target, err_target, f_ens, var_ens)
        
        # Normalize by median
        valid_fluxes = fluxes[np.isfinite(fluxes)]
        if len(valid_fluxes) > 0:
            norm = np.median(valid_fluxes)
            fluxes /= norm
            errors /= norm
            
        res = table.Table()  # TODO move this part to `save` method
        res['night'] = self.table['night']
        res['jd'] = self.table['jd']
        res['airmass'] = self.table['airmass']
        res['exptime'] = self.table['exptime']
        res['flux'] = fluxes
        res['error'] = errors
        return res

    def get_binned(self, comp_ids, min_exp_per_night=4, sig_clip=5, diagnostics=False):
        """
        Calculates daily binned flux and propagated error.
        
        Args:
            comp_ids (list): List of comparison star IDs.
            min_exp_per_night (int): Min exposures per night.
            sig_clip (float): Sigma level for clipping. 0 to disable.
            diagnostics (bool): If True, calculates extra stats (airmass, exptime, intraday_std).
        
        Returns:
            tuple: (astropy.table.Table, daily_rms)
        """
        # 1. Identify rows where target and ALL comp stars are present
        valid_row_mask = np.isfinite(self.table[self.target_id])
        for cid in comp_ids:
            valid_row_mask &= np.isfinite(self.table[cid])

        # 2. Filter data
        valid_table = self.table[valid_row_mask]
        if not valid_table:
            return table.Table(), np.inf
            
        # Efficiently group by night
        grouped = valid_table.group_by('night')
        nights_list = []
        mean_jds = []
        binned_fluxes = []
        binned_errors = []
        
        # Diagnostic lists
        mean_airmasses = []
        exptime_sums = []
        n_exps = []
        intraday_stds = []
        
        for group in grouped.groups:
            n_exp = len(group)
            if n_exp < min_exp_per_night:
                continue
                
            f_target_sum = np.sum(group[self.target_id])
            var_target_sum = np.sum(group[f"err_{self.target_id}"]**2)
            
            f_ens_sum = 0
            var_ens_sum = 0
            for cid in comp_ids:
                f_ens_sum += np.sum(group[cid])  # TODO can we use this part for unbin/intraday rms?
                var_ens_sum += np.sum(group[f"err_{cid}"]**2)
            
            if f_ens_sum > 0:
                nights_list.append(group['night'][0])
                mean_jds.append(np.mean(group['jd']))
                rel_flux, rel_err = self._rel_phot(f_target_sum, np.sqrt(var_target_sum), f_ens_sum, var_ens_sum)
                binned_fluxes.append(rel_flux)
                binned_errors.append(rel_err)
                
                if diagnostics:
                    mean_airmasses.append(np.mean(group['airmass']))
                    exptime_sums.append(np.sum(group['exptime']))
                    n_exps.append(n_exp)
                    
                    # Calculate intraday std for this night
                    # Use local ensemble series to avoid redundant full-table grouping
                    night_target = group[self.target_id]
                    night_err = group[f"err_{self.target_id}"]
                    night_ens = np.sum([group[cid] for cid in comp_ids], axis=0)
                    night_var_ens = np.sum([group[f"err_{cid}"]**2 for cid in comp_ids], axis=0)
                    
                    n_flux, _ = self._rel_phot(night_target, night_err, night_ens, night_var_ens)
                    valid = np.isfinite(n_flux)
                    if np.sum(valid) > 2:
                        intraday_stds.append(np.std(n_flux[valid] / np.median(n_flux[valid])))
                    else:
                        intraday_stds.append(np.nan)

        res = table.Table()
        res['night'] = np.array(nights_list)
        res['jd'] = np.array(mean_jds)
        res['flux'] = np.array(binned_fluxes)
        res['error'] = np.array(binned_errors)
        
        if diagnostics:
            res['airmass'] = np.array(mean_airmasses)
            res['exptime_sum'] = np.array(exptime_sums)
            res['n_exp'] = np.array(n_exps)
            res['intraday_std'] = np.array(intraday_stds)

        # 3. Sigma Clipping
        if sig_clip > 0 and len(res['flux']) > 2:
            med = np.nanmedian(res['flux'])
            std = np.nanstd(res['flux'])
            inliers = np.abs(res['flux'] - med) < sig_clip * std
            res[f'within_{sig_clip}_sig'] = inliers # TODO save sig_clip in meta?
        
        # 4. Normalize by median
        norm = np.median(res['flux'][inliers])
        res['flux'] /= norm
        res['error'] /= norm
        daily_rms = np.std(res['flux'][inliers])
            
        return res, daily_rms

    def calculate_rms_metric(self, comp_ids, metric_type='daily'):
        """
        Computes various RMS metrics for a given ensemble.
        'daily': Night-to-night stability of binned flux.
        'intraday': Average stability within each night.
        'total': Overall time-series stability.
        """
        if metric_type == 'daily':
            _, daily_rms = self.get_binned(comp_ids, diagnostics=False)
            return daily_rms

        elif metric_type == 'amplitude':
            bin_table, _ = self.get_binned(comp_ids, diagnostics=False)
            fluxes = bin_table['flux']
            valid = np.isfinite(fluxes)
            if np.sum(valid) < 2:
                return np.inf
            # Difference between representative high (95%) and low (5%)
            return np.nanpercentile(fluxes, 95) - np.nanpercentile(fluxes, 5)
        
        elif metric_type == 'intraday':  # TODO this reruns phot
            bin_table, _ = self.get_binned(comp_ids, diagnostics=True)
            stds = bin_table['intraday_std']
            valid = np.isfinite(stds)
            return np.median(stds[valid]) if np.sum(valid) > 0 else np.inf
            
        elif metric_type == 'total':  # TODO this is nonsense now
            unbin_table = self.get_unbinned(comp_ids)
            rel_flux = unbin_table['flux']
            valid = np.isfinite(rel_flux)
            if np.sum(valid) < 2: return np.inf
            normalized = rel_flux[valid] / np.nanmedian(rel_flux[valid])
            return np.nanstd(normalized)
        
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    def save(self, comp_ids, base_filename, sig_clip=5, overwrite=False):
        """
        Saves both binned and unbinned results to FITS tables with metadata.
        
        Args:
            comp_ids (list): Ensemble comparison stars.
            base_filename (str or Path): Output filename base (e.g., 'HD_191939_V').
            sig_clip (float): Sigma clipping value.
        """
        # check overwrite
        base_path = Path(base_filename)
        bin_path = base_path.parent / f"bin_{base_path.with_suffix('.fits').name}"
        unbin_path = base_path.parent / f"unbin_{base_path.with_suffix('.fits').name}"

        if not overwrite and (bin_path.exists() or unbin_path.exists()):
            response = input(f"Files already exist for {base_path.name}. Overwrite? (y/n): ")
            if not response.lower().startswith('y'):
                return

        # 1. Generate Tables
        bin_table, daily_rms = self.get_binned(comp_ids, sig_clip=sig_clip, diagnostics=True)
        unbin_table = self.get_unbinned(comp_ids)
        
        # 2. Calculate Aggregate Metrics
        amp = self.calculate_rms_metric(comp_ids, metric_type='amplitude')  # TODO these two lines rerun issue
        intra_rms = self.calculate_rms_metric(comp_ids, metric_type='intraday')
        
        # 3. Attach Metadata
        meta = {
            'COMP_IDS': ",".join(map(str, comp_ids)),
            'RMS_DAY': daily_rms,
            'RMS_INTRA': intra_rms,
            'AMP': amp,
            'TARGET': self.target_id,
            'SIG_CLIP': sig_clip
        }
        
        bin_table.meta.update(meta)
        unbin_table.meta.update(meta)
        
        # 5. Save
        bin_table.write(bin_path, overwrite=True)
        unbin_table.write(unbin_path, overwrite=True)

def plot_multi_band(base_path, target_name, bands, savefig_path=None):
    """
    Generates a multi-band diagnostic plot similar to plot_mag.py.
    """
    N_SIG = 1

    mpl.rc('font', family='serif')

    base_path = Path(base_path)

    wfc3, stis = newport.get_hst(
        target_name,
        path='xml/HST-17192-visit-status_20260216.xml'
    )

    fig, axs = plt.subplots(nrows=len(bands), figsize=(6, 1.5 * len(bands)), sharex=True)
    if len(bands) == 1: axs = [axs]

    wfc3_line, stis_line = [], []
    
    for i, band in enumerate(bands):
        ax = axs[i]
        bin_path = base_path.parent / f"bin_{base_path.stem}_{band}.fits"
        unbin_path = base_path.parent / f"unbin_{base_path.stem}_{band}.fits"
        
        bin_table = table.Table.read(bin_path)
        unbin_table = table.Table.read(unbin_path)

        rms_day = bin_table.meta['RMS_DAY']
        # rms_intra = bin_table.meta['RMS_INTRA']

        bin_table = bin_table[bin_table['within_3_sig']]
            
        # Time conversion
        t_unbin = Time(unbin_table['jd'], format='jd').to_datetime()
        t_bin = Time(bin_table['jd'], format='jd').to_datetime()
        
        # Unbinned points in background
        ax.errorbar(
            t_unbin, unbin_table['flux'], # yerr=unbin_table['error'],
            fmt='o', color='silver',
            ms=4, alpha=0.3, markeredgewidth=0, ecolor='lightgrey', elinewidth=1
        )
        
        # Binned points with error bars
        ax.errorbar(t_bin, bin_table['flux'], yerr=bin_table['intraday_std'], 
                    fmt=newport.MARKERS[band], color=newport.COLORS[band],
                    alpha=0.7, markeredgewidth=0, 
                    ms=7, capsize=3, label=f'{band} band')

        # mean line and std patch
        _xl = ax.get_xlim()
        x1, x2 = datetime(2022, 1, 1), datetime(2025, 12, 31)
        y1, y2 = 1 - rms_day * N_SIG, 1 + rms_day * N_SIG
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=0.1, color=newport.COLORS[band], lw=0, zorder=0)
        ax.add_patch(rect)
        ax.set_xlim(_xl)
        
        # Plot HST
        for _ in wfc3:
            wfc3_line = ax.axvline(_.to_datetime(), ls='--', c='C1', linewidth=2, alpha=0.55)
        for _ in stis:
            stis_line = ax.axvline(_.to_datetime(), c='C7', linewidth=2, alpha=0.5)
        
        # Set ylim
        flux = bin_table['flux']
        p25, p75 = np.nanpercentile(flux, [25, 75])
        mid_mask = (flux >= p25) & (flux <= p75)
        std_mid = np.nanstd(flux[mid_mask])
        ax.set_ylim(1 - 11 * std_mid, 1 + 11 * std_mid)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', direction='in', which='both', labelsize=10)  # Ticks inside
        ax.tick_params(axis='y', direction='in')
        
        # # Metadata labels
        # rms_day = bin_table.meta.get('RMS_DAY', 0)
        # rms_intra = bin_table.meta.get('RMS_INTRA', 0)
        # ax.set_title(f"Daily RMS: {rms_day:.6f} | Intraday RMS: {rms_intra:.6f}", fontsize=10)
        # ax.legend(loc='upper right', fontsize=8)

    fig.supxlabel("Time of observation", y=0.03)
    fig.supylabel("Relative flux", x=0.035)
    
    # Global Legend matching plot_mag.py style
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    handles.extend([
        wfc3_line,
        # stis_line
    ])
    labels.extend([
        'HST WFC3 planetary transit obs.',
        # 'HST STIS host star observation'
    ])

    fig.legend(handles, labels, ncol=3, loc='upper center', 
               bbox_to_anchor=(0.54, 1.003), fontsize=9)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.91) # Make room for legend
    
    if savefig_path:
        plt.savefig(savefig_path)
        print(f"Figure saved to {savefig_path}")
    
    # plt.show()
    plt.close()
            

def optimize_ensemble(phot_table, target_id, metric='daily', criterion=0.7, max_comps=None):
    """Performs exhaustive combinations search."""
    # Identify potential comparison stars (digit columns, not target)
    comp_ids = [col for col in phot_table.colnames if col.isdigit() and col != str(target_id)]
    
    # Pre-filter by valid fraction
    qualified_comps = []
    for cid in comp_ids:
        # Handle both regular columns (NaNs) and MaskedColumns
        col_data = phot_table[cid]
        if hasattr(col_data, 'filled'):
            valid_frac = np.sum(~col_data.mask) / len(phot_table)
        else:
            valid_frac = np.sum(np.isfinite(col_data)) / len(phot_table)
            
        if valid_frac >= criterion:
            qualified_comps.append(cid)
            
    print(f"Qualified comparison stars: {len(qualified_comps)}. (2^N combinations).")
    
    if max_comps:
        qualified_comps = qualified_comps[:max_comps] # Safety cap for dev

    engine = RelativePhotometryEngine(phot_table, target_id)
    best_rms = np.inf
    best_ensemble = None
    
    total_combinations = sum(1 for r in range(1, len(qualified_comps) + 1) for _ in itertools.combinations(qualified_comps, r))
    
    with tqdm(total=total_combinations, desc=f"Optimizing ({metric})") as pbar:
        for r in range(1, len(qualified_comps) + 1):
            for subset in itertools.combinations(qualified_comps, r):
                rms = engine.calculate_rms_metric(subset, metric_type=metric)
                if rms < best_rms:
                    best_rms = rms
                    best_ensemble = list(subset)
                pbar.update(1)
                
    return best_ensemble, best_rms


if __name__ == "__main__":
    # TARGET = "HD_191939"
    # TARGET_ID = "2248126315275354496"
    # INPUT_PATH = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939.fits')
    # OUTPUT_DIR = Path(f"tables/opt_comp_stars/{TARGET}/plottimg")
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # full_table = table.Table.read(INPUT_PATH)
    # bt = full_table[full_table['band'] == 'B']
    # vt = full_table[full_table['band'] == 'V']
    # rt = full_table[full_table['band'] == 'R']
    # it = full_table[full_table['band'] == 'I']

    # bc = ["2248131366156908288","2248136313959256960"]
    # vc = [
    #     "2248135317526853120",
    #     "2248136313959256960",
    #     "2248137653989048320"
    # ]
    # rc = [
    #     "2248135317526853120",
    #     "2248137653989048320"
    # ]
    # ic = [
    #     "2248131366156908288",
    #     "2248137653989048320"
    # ]

    # be = RelativePhotometryEngine(bt, TARGET_ID)
    # ve = RelativePhotometryEngine(vt, TARGET_ID)
    # re = RelativePhotometryEngine(rt, TARGET_ID)
    # ie = RelativePhotometryEngine(it, TARGET_ID)

    # be.save(bc, 'tables/opt_comp_stars/HD_191939/plottimg/result_B', 3, overwrite=True)
    # ve.save(vc, 'tables/opt_comp_stars/HD_191939/plottimg/result_V', 3, overwrite=True)
    # re.save(rc, 'tables/opt_comp_stars/HD_191939/plottimg/result_R', 3, overwrite=True)
    # ie.save(ic, 'tables/opt_comp_stars/HD_191939/plottimg/result_I', 3, overwrite=True)

    plot_multi_band(
        'tables/opt_comp_stars/HD_191939/plottimg/result',
        'HD 191939',
        ['B', 'V', 'R', 'I'],
        'tables/opt_comp_stars/HD_191939/plottimg/intraday_std_ebar.pdf'
    )


# if __name__ == "__main__":
#     TARGET = "HD_191939"
#     TARGET_ID = "2248126315275354496"
#     INPUT_PATH = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939.fits')
#     OUTPUT_DIR = Path(f"tables/opt_comp_stars/{TARGET}/amplitude_89")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     full_table = table.Table.read(INPUT_PATH)
#     bands = np.unique(full_table['band'])
    
#     fig, axs = plt.subplots(nrows=len(bands), figsize=(8, 2 * len(bands)), sharex=True)
#     if len(bands) == 1: axs = [axs]

#     for i, band in enumerate(bands):
#         print(f"\n--- Band {band} ---")
#         band_table = full_table[full_table['band'] == band]
        
#         # 1. Optimize
#         best_ensemble, best_rms = optimize_ensemble(band_table, TARGET_ID, metric='amplitude', criterion=0.89)
        
#         # 2. Save results (includes binned/unbinned + metadata)
#         engine = RelativePhotometryEngine(band_table, TARGET_ID)
#         output_fn_base = OUTPUT_DIR / f"results_{band}"
#         engine.save(best_ensemble, output_fn_base, overwrite=True)
        
#         # 3. Save optimization summary (temporarily commented out JSON saving if requested, but let's just keep as requested)
#         # summary = {
#         #     "target": TARGET,
#         #     "target_id": TARGET_ID,
#         #     "band": band,
#         #     "best_ensemble": best_ensemble,
#         #     "achieved_rms": float(best_rms) if np.isfinite(best_rms) else None,
#         # }
#         # output_json = OUTPUT_DIR / f"opt_ensemble_{band}.json"
#         # with open(output_json, "w") as f:
#         #     json.dump(summary, f, indent=4)
#         # print(f"Summary saved to {output_json}")

#     # 4. Multi-band plot from saved tables
#     plot_multi_band(OUTPUT_DIR, TARGET, bands, savefig_path=OUTPUT_DIR / "multi_band_diagnostic.png")

# criteria: BR 0.88, V 0.89, I 0.87
