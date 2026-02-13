"""
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from pathlib import Path
from optimize_rel_phot import RelativePhotometryEngine
from datetime import datetime

# Use same constants as optimize_rel_phot
TARGET = "HD_191939"
TARGET_ID = "2248126315275354496"
INPUT_PATH = Path(f"tables/list_runs/191939/phot_gaia_run/phot_w_err_{TARGET}.fits")

full_table = table.Table.read(INPUT_PATH)
bands = np.unique(full_table['band'])
fig, axs = plt.subplots(nrows=len(bands), figsize=(8, 2 * len(bands)), sharex=True)

if len(bands) == 1:
    axs = [axs]

for i, band in enumerate(bands):
    ax = axs[i]
    band_table = full_table[full_table['band'] == band]
    engine = RelativePhotometryEngine(band_table, TARGET_ID)
    
    # Identify qualified comparison stars (>70% valid)
    comp_ids = [col for col in band_table.colnames if col.isdigit() and col != TARGET_ID]
    qualified_ids = [cid for cid in comp_ids if np.mean(np.isfinite(band_table[cid])) >= 1]
    
    if not qualified_ids:
        continue
        
    nights, binned_flux = engine.get_binned(qualified_ids)
    nights_dt = [datetime.strptime(str(n), '%Y%m%d') for n in nights]
    
    # Plot
    std = np.nanstd(binned_flux)
    
    ax.plot(nights_dt, binned_flux, 'o')
    ax.set_ylim(1 - 3 * std, 1 + 3 * std)
    ax.set_ylabel(band)
    ax.set_title(
        f"{len(qualified_ids)} comps, "
        rf"$\sigma={std}, "
        f"{engine.calculate_rms_metric(qualified_ids)}$"
    )
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Diagnostic: Nightly Binned Relative Flux - {TARGET}")
plt.xlabel("Night")
plt.tight_layout()
plt.show()
