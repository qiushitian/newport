from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *
from newport import TARGET_GAIA_DR3, get_input_path, get_excluded_comp


TARGET = "TOI-1759"
TARGET_ID = TARGET_GAIA_DR3[TARGET]

# 1759 only
USED_COMPS = [
    '2216414888108779520', '2216416743536122752', '2216419698472018560',
    '2216431445197473152', '2216432475989623040', '2216433953458370176',
    '2216441031566760960', '2216445876287582848'
]

OUTPUT_DIR = Path(
    f"tables/opt_comp_stars/{TARGET}/{'_'.join(c[-3:] for c in USED_COMPS)}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERWRITE = False


if __name__ == "__main__":
    full_table = table.Table.read(get_input_path(TARGET))

    read_saved_table_base = OUTPUT_DIR / "results"

    bands = np.unique(full_table['band'])  # Only 766452769615859712 has no V band data
    all_binned = []
    all_unbinned = []

    for i, band in enumerate(bands):
        print(f"\n--- Band {band} ---")
        band_table = full_table[full_table['band'] == band]

        best_ensemble = USED_COMPS
        
        # 2. Save results (includes binned/unbinned + metadata)
        engine = RelativePhotometryEngine(band_table, TARGET_ID)
        output_fn_base = OUTPUT_DIR / f"results_{band}"
        engine.save(best_ensemble, output_fn_base, sig_clip=3, overwrite=OVERWRITE)
        
        # Collect for stacking
        bin_t = table.Table.read(OUTPUT_DIR / f"bin_results_{band}.fits")
        unbin_t = table.Table.read(OUTPUT_DIR / f"unbin_results_{band}.fits")
        bin_t['band'] = band
        unbin_t['band'] = band
        all_binned.append(bin_t)
        all_unbinned.append(unbin_t)

        # 3. Save optimization summary
        summary = {
            "target": TARGET,
            "target_id": TARGET_ID,
            "band": band,
            "best_ensemble": best_ensemble,
        }
        output_json = OUTPUT_DIR / f"comps_{band}.json"
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Summary saved to {output_json}")
        print(best_ensemble)

    # 4. Save consolidated tables
    if all_binned:
        table.vstack(all_binned).write(OUTPUT_DIR / "bin_results_all.fits", overwrite=OVERWRITE)
        table.vstack(all_unbinned).write(OUTPUT_DIR / "unbin_results_all.fits", overwrite=OVERWRITE)
        print(f"\nConsolidated results saved to {OUTPUT_DIR}/[bin|unbin]_results_all.fits")

    # 5. Multi-band plot from saved tables
    plot_target(
        read_saved_table_base, TARGET, # yrange=12,
        savefig_path=OUTPUT_DIR / "monitoring.pdf"
    )
