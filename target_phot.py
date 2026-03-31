from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *


TARGET = "HD_191939"
TARGET_ID = "2248126315275354496"
INPUT_PATH = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939_ss.fits')
OUTPUT_DIR = Path(f"tables/opt_comp_stars/{TARGET}/two_ss")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIAG_DIR = OUTPUT_DIR / "comp_diag"
CRIT = 0.86
OVERWRITE = True

USED_COMPS = [
    # '2248124184971495936',
    '2248135317526853120',
    '2248136313959256960'
]


if __name__ == "__main__":
    full_table = table.Table.read(INPUT_PATH)

    # ### BLOCK: Print out comparison stars for comp diag ###
    # COMP_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # comp_set = set()
    # for band in ['B', 'V', 'R', 'I']:
    #     json_path = OUTPUT_DIR / f"opt_ensemble_{band}.json"
    #     comp_set.update(load_optimized_json(json_path))

    # for cid in comp_set:
    #     print(f"Gaia DR3 {cid}:")
    #     for band in ['B', 'V', 'R', 'I']:
    #         t = table.Table.read(COMP_DIAG_DIR / f"bin_diag_{cid}_{band}.fits")
    #         e = t.meta['COMPIDS']
    #         print(f"  {band}: {e}")
    # ### END BLOCK ###

    ### BLOCK: Run ensembles ###
    read_saved_table_base = OUTPUT_DIR / "results"

    # bands = ['B', 'V', 'R', 'I']  # np.unique(full_table['band'])
    # all_binned = []
    # all_unbinned = []

    # for i, band in enumerate(bands):
    #     print(f"\n--- Band {band} ---")
    #     band_table = full_table[full_table['band'] == band]
        
    #     # # 1. Optimize
    #     # best_ensemble = get_comps(
    #     #     band_table, TARGET_ID, criterion=CRIT,
    #     #     # exclude_ids=['2248137653989048320', '2248131366156908288']
    #     # )

    #     best_ensemble = USED_COMPS
        
    #     # 2. Save results (includes binned/unbinned + metadata)
    #     engine = RelativePhotometryEngine(band_table, TARGET_ID)
    #     output_fn_base = OUTPUT_DIR / f"results_{band}"
    #     engine.save(best_ensemble, output_fn_base, sig_clip=3, overwrite=OVERWRITE)
        
    #     # Collect for stacking
    #     bin_t = table.Table.read(OUTPUT_DIR / f"bin_results_{band}.fits")
    #     unbin_t = table.Table.read(OUTPUT_DIR / f"unbin_results_{band}.fits")
    #     bin_t['band'] = band
    #     unbin_t['band'] = band
    #     all_binned.append(bin_t)
    #     all_unbinned.append(unbin_t)

    #     # 3. Save optimization summary
    #     summary = {
    #         "target": TARGET,
    #         "target_id": TARGET_ID,
    #         "band": band,
    #         "best_ensemble": best_ensemble,
    #     }
    #     output_json = OUTPUT_DIR / f"opt_ensemble_{band}.json"
    #     with open(output_json, "w") as f:
    #         json.dump(summary, f, indent=4)
    #     print(f"Summary saved to {output_json}")
    #     print(best_ensemble)

    # # 4. Save consolidated tables
    # if all_binned:
    #     table.vstack(all_binned).write(OUTPUT_DIR / "bin_results_all.fits", overwrite=OVERWRITE)
    #     table.vstack(all_unbinned).write(OUTPUT_DIR / "unbin_results_all.fits", overwrite=OVERWRITE)
    #     print(f"\nConsolidated results saved to {OUTPUT_DIR}/[bin|unbin]_results_all.fits")

    # 5. Multi-band plot from saved tables
    plot_target(
        read_saved_table_base, TARGET, # n_std_mid=12,
        savefig_path=OUTPUT_DIR / "monitoring.pdf"
    )
    ### END BLOCK ###
