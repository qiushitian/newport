from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *

if __name__ == "__main__":
    TARGET = "HD_191939"
    TARGET_ID = "2248126315275354496"
    INPUT_PATH = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939.fits')
    OUTPUT_DIR = Path(f"tables/opt_comp_stars/{TARGET}/two")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CRIT = 0.86

    full_table = table.Table.read(INPUT_PATH)

    # ### BLOCK: Print out comparison stars for comp diag ###
    # COMP_DIAG_DIR = Path(f"tables/opt_comp_stars/{TARGET}/comp_diag")
    # COMP_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    # comp_set = set()
    # for band in ['B', 'V', 'R', 'I']:
    #     json_path = JSON_DIR / f"opt_ensemble_{band}.json"
    #     comp_set.update(load_optimized_json(json_path))

    # for cid in comp_set:
    #     print(f"Gaia DR3 {cid}:")
    #     for band in ['B', 'V', 'R', 'I']:
    #         t = table.Table.read(COMP_DIAG_DIR / f"bin_diag_{cid}_{band}.fits")
    #         e = t.meta['COMPIDS']
    #         print(f"  {band}: {e}")
    # ### END BLOCK ###

    # ### BLOCK: Run ensembles ###
    read_saved_table_base = OUTPUT_DIR / "results"

    bands = ['B', 'V', 'R', 'I']  # np.unique(full_table['band'])

    for i, band in enumerate(bands):
        print(f"\n--- Band {band} ---")
        band_table = full_table[full_table['band'] == band]
        
        # 1. Optimize
        best_ensemble = get_comps(
            band_table, TARGET_ID, criterion=CRIT,
            # exclude_ids=['2248137653989048320', '2248131366156908288']
        )

        best_ensemble = [
            # '2248124184971495936',
            '2248135317526853120',
            '2248136313959256960',
            # '2248137653989048320'
        ]
        
        # # 2. Save results (includes binned/unbinned + metadata)
        # engine = RelativePhotometryEngine(band_table, TARGET_ID)
        # output_fn_base = OUTPUT_DIR / f"results_{band}"
        # engine.save(best_ensemble, output_fn_base, sig_clip=3, overwrite=True)
        
        # 3. Save optimization summary (temporarily commented out JSON saving if requested, but let's just keep as requested)
        # summary = {
        #     "target": TARGET,
        #     "target_id": TARGET_ID,
        #     "band": band,
        #     "best_ensemble": best_ensemble,
        # }
        # output_json = OUTPUT_DIR / f"opt_ensemble_{band}.json"
        # with open(output_json, "w") as f:
        #     json.dump(summary, f, indent=4)
        # print(f"Summary saved to {output_json}")
        print(best_ensemble)

    # 4. Multi-band plot from saved tables
    plot_target(
        read_saved_table_base, TARGET, # n_std_mid=12,
        savefig_path=OUTPUT_DIR / "monitoring.pdf"
    )
    ### END BLOCK ###

    ##########################################################################

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

    # plot_multi_band(
    #     'tables/opt_comp_stars/HD_191939/plottimg/result',
    #     'HD 191939',
    #     ['B', 'V', 'R', 'I'],
    #     'tables/opt_comp_stars/HD_191939/plottimg/intraday_std_ebar.pdf'
    # )