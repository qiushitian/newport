#!/usr/bin/env python3
"""
Grid of periodograms for comparison stars.
"""
from pathlib import Path
import numpy as np
from optimize_rel_phot import plot_comp_periodogram
from comp_phot import ALL_COMPS
from target_phot import TARGET, OUTPUT_DIR, USED_COMPS

# --- Configuration for long periods ---
MIN_PERIOD = 25
MIN_FREQ = -0.001
XTICKS = np.array([30, 40, 50, 70, 100, 200, 600]) # 3, 4, 5, 7, 10, 300]) # 20, 25, 30, 50])
RANGES = np.array([
    [200, 1900]
])
FAP = 15
# ------------------------------------------

# unconstrained
MIN_PERIOD, MIN_FREQ = 2.2, -0.01
XTICKS = np.array([3, 4, 5, 7, 9, 20, 300])
RANGES = np.array([])
FAP = 15

if __name__ == "__main__":
    print(f"Generating comparison star periodograms for {TARGET}...")
    
    comp_diag_dir = OUTPUT_DIR / "comp_diag"
    
    plot_comp_periodogram(
        base_dir=comp_diag_dir,
        all_comps=ALL_COMPS,
        used_comps=USED_COMPS,
        target_name=TARGET,
        savefig_path=comp_diag_dir / f"comp_periodograms_p{MIN_PERIOD}.pdf",
        min_period=MIN_PERIOD,
        min_freq=MIN_FREQ,
        xticks=XTICKS,
        ranges=RANGES,
        fap=FAP
    )
