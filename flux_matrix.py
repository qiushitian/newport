#!/usr/bin/env python3
"""
Generate flux ratio stability matrices for target and comparison stars.
"""
from pathlib import Path
from astropy import table
from optimize_rel_phot import plot_flux_matrix
from target_phot import TARGET, TARGET_ID, INPUT_PATH, OUTPUT_DIR
from comp_phot import ALL_COMPS

if __name__ == "__main__":
    print(f"Generating star flux matrices for {TARGET}...")
    
    # Load standardized photometry table
    phot_table = table.Table.read(INPUT_PATH)
    
    # Full list of stars to compare (Target + All Comparisons)
    star_ids = [TARGET_ID] + ALL_COMPS
    
    comp_diag_dir = OUTPUT_DIR / "comp_diag"
    comp_diag_dir.mkdir(parents=True, exist_ok=True)
    
    for band in ['B', 'V', 'R', 'I']:
        for metric in ['std', 'amplitude']:
            save_path = comp_diag_dir / f"flux_matrix_{metric}_{band}.pdf"
            print(f"Processing {band} band with {metric} metric...")
            plot_flux_matrix(
                phot_table, star_ids, 
                band=band, metric=metric, 
                savefig_path=save_path
            )
