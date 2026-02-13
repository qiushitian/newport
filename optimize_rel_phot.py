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
import matplotlib.pyplot as plt
from datetime import datetime

class RelativePhotometryEngine:  # TODO handle bands
    def __init__(self, input_table, target_id):
        # Standardize table: ensure it's not masked and fills gaps with NaNs
        # TODO maybe we just make sure phot_[gaia|list]_run never produces MaskedColumn
        if input_table.has_masked_columns:
            self.table = input_table.filled(np.nan)
        else:
            self.table = input_table.copy()
        
        self.target_id = str(target_id)
        
    def get_relative_flux_series(self, comp_ids):
        """Calculates image-by-image relative flux."""
        super_flux = np.nansum([self.table[cid] for cid in comp_ids], axis=0)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_flux = self.table[self.target_id] / super_flux
        return rel_flux

    def get_binned(self, comp_ids, sig_clip=5):
        """Calculates daily binned flux based on summed counts per night."""
        # 1. Identify rows where target and ALL comp stars are present
        valid_row_mask = np.isfinite(self.table[self.target_id])
        for cid in comp_ids:
            valid_row_mask &= np.isfinite(self.table[cid])

        # 2. Filter data
        valid_table = self.table[valid_row_mask]
        if not valid_table:
            return np.array([]), np.array([])
            
        # Efficiently group by night
        grouped = valid_table.group_by('night')
        unique_nights = []
        binned_fluxes = []
        
        for group in grouped.groups:
            night = group['night'][0]
            target_sum = np.sum(group[self.target_id])
            
            super_sum = 0
            for cid in comp_ids:
                super_sum += np.sum(group[cid])
            
            if super_sum > 0:
                unique_nights.append(night)
                binned_fluxes.append(target_sum / super_sum)
        
        nights = np.array(unique_nights)
        binned_fluxes = np.array(binned_fluxes)
        
        # 3. Sigma Clipping
        if sig_clip > 0 and len(binned_fluxes) > 2:
            med = np.nanmedian(binned_fluxes)
            std = np.nanstd(binned_fluxes)
            outliers = np.abs(binned_fluxes - med) > sig_clip * std
            binned_fluxes[outliers] = np.nan
        
        # 4. Normalize by median
        valid_binned = binned_fluxes[np.isfinite(binned_fluxes)]
        if len(valid_binned) > 0:
            binned_fluxes /= np.median(valid_binned)
            
        return nights, binned_fluxes

    def calculate_rms_metric(self, comp_ids, metric_type='daily'):
        """
        Computes various RMS metrics for a given ensemble.
        'daily': Night-to-night stability of binned flux.
        'intraday': Average stability within each night.
        'total': Overall time-series stability.
        """
        if metric_type == 'daily':
            _, fluxes = self.get_binned(comp_ids)
            if np.sum(np.isfinite(fluxes)) < 2:
                return np.inf
            return np.nanstd(fluxes)

        else:
            raise NotImplementedError("intraday and total metrics pending verification")
        
        # elif metric_type == 'intraday':
        #     rel_flux = self.get_relative_flux_series(comp_ids)
        #     night_stds = []
        #     for night in np.unique(self.nights):
        #         mask = self.nights == night
        #         night_data = rel_flux[mask]
        #         valid = np.isfinite(night_data)
        #         if np.sum(valid) > 2:
        #             normalized = night_data[valid] / np.nanmedian(night_data[valid])
        #             night_stds.append(np.nanstd(normalized))
        #     return np.nanmean(night_stds) if night_stds else np.inf
            
        # elif metric_type == 'total':
        #     rel_flux = self.get_relative_flux_series(comp_ids)
        #     valid = np.isfinite(rel_flux)
        #     if np.sum(valid) < 2: return np.inf
        #     normalized = rel_flux[valid] / np.nanmedian(rel_flux[valid])
        #     return np.nanstd(normalized)
        
        # else:
        #     raise ValueError(f"Invalid metric type: {metric_type}")

    def plot_binned(self, comp_ids=None, criterion=0.7, ax=None, sig_clip=5):
        """Generates a diagnostic plot of the binned relative flux."""
        if comp_ids is None:
            # Auto-identify comparison stars based on criterion
            all_comps = [col for col in self.table.colnames if col.isdigit() and col != self.target_id]
            comp_ids = []
            for cid in all_comps:
                valid_frac = np.mean(np.isfinite(self.table[cid]))
                if valid_frac >= criterion:
                    comp_ids.append(cid)
        
        nights, binned_flux = self.get_binned(comp_ids, sig_clip=sig_clip)
        
        if len(nights) == 0:
            print("Warning: No valid data for plotting.")
            return ax

        # Convert nights YYYYMMDD to datetime objects
        nights_dt = [datetime.strptime(str(int(n)), '%Y%m%d') for n in nights]
        
        show_at_end = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            show_at_end = True
            
        std = np.nanstd(binned_flux)
        rms = self.calculate_rms_metric(comp_ids, metric_type='daily')
        
        ax.plot(nights_dt, binned_flux, 'o')
        ax.set_ylim(1 - 3 * std, 1 + 3 * std)
        ax.set_title(
            f"{len(comp_ids)} comps, {len(nights)}/{len(np.unique(self.table['night']))} nights\n"
            rf"$\sigma={std:.6f}$, "
            f"Daily RMS={rms:.6f}"
        )
        ax.set_ylabel("Rel Flux")
        ax.grid(True, alpha=0.3)
        
        if show_at_end:
            plt.tight_layout()
            plt.show()
        else:
            return ax
            

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
    TARGET = "HD_191939"
    TARGET_ID = "2248126315275354496"
    INPUT_PATH = Path(f"tables/list_runs/191939/phot_gaia_run/phot_w_err_{TARGET}.fits")
    OUTPUT_DIR = Path(f"tables/opt_comp_stars/{TARGET}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    full_table = table.Table.read(INPUT_PATH)
    bands = np.unique(full_table['band'])
    
    fig, axs = plt.subplots(nrows=len(bands), figsize=(8, 2 * len(bands)), sharex=True)
    if len(bands) == 1: axs = [axs]

    for i, band in enumerate(bands):
        print(f"\n--- Band {band} ---")
        band_table = full_table[full_table['band'] == band]
        
        # 1. Optimize
        best_ensemble, best_rms = optimize_ensemble(band_table, TARGET_ID, metric='daily', criterion=0.8)
        
        # 2. Plot results on provided Axis
        engine = RelativePhotometryEngine(band_table, TARGET_ID)
        engine.plot_binned(best_ensemble, ax=axs[i])
        axs[i].set_ylabel(band)

        # 3. Save result
        result = {
            "target": TARGET,
            "target_id": TARGET_ID,
            "band": band,
            "best_ensemble": best_ensemble,
            "achieved_rms": float(best_rms) if np.isfinite(best_rms) else None,
        }
        output_fn = OUTPUT_DIR / f"opt_ensemble_{band}.json"
        with open(output_fn, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved to {output_fn}")

    fig.suptitle(f"Optimization Diagnostic: {TARGET}")
    plt.xlabel("Night")
    plt.tight_layout()
    plt.show()

# criteria: BR 0.88, V 0.89, I 0.87
