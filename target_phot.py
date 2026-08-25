from pathlib import Path
from astropy import table
import numpy as np
import json
from datetime import datetime
from optimize_rel_phot import *
from newport import TARGET_GAIA_DR3, get_excluded_comp, get_input_path

from dep import plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["serif"]


TARGET = "TOI-1410"
TARGET_ID = TARGET_GAIA_DR3[TARGET]

OUTPUT_DIR = Path(f"data/tables/opt_comp_stars/{TARGET}/c15")
if __name__ == "__main__":
    print(f'Newport photometry for target {TARGET} to {OUTPUT_DIR}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # COMP_DIAG_DIR = OUTPUT_DIR / "comp_diag"
    # CRIT = 0.86

    OVERWRITE = False

    USED_COMPS = [
        "1958533300622189184",
        "1958534434491983232",
        "1958534468851721856",
        "1958535740162028032",
        "1958535976382244480",
        "1958536599157070592",
        "1958537561228353152",
        "1958555703170244608",
        "1958561200728431872",
        "1958582671266686080",
        "1958583603279110528",
        "1958586867452688768",
        "1958588860317508224",
        "1958589410073513344",
        "1958608720246495360"
    ]

    all_comps = [
        "1958514712002236160",
        "1958515330477513728",
        "1958515570995685248",
        "1958517323342115584",
        "1958517701299221248",
        "1958517701299226112",
        "1958518113616081152",
        "1958518933952065408",
        "1958523508095004032",
        "1958523542454743168",
        "1958524538887143936",
        "1958525638394734080",
        "1958526634830953344",
        "1958529727209311360",
        "1958530208244184832",
        "1958530753702017152",
        "1958530964158427904",
        "1958531273397589376",
        "1958531784497511040",
        "1958532853945554816",
        "1958533300622189184",
        "1958534159613862400",
        "1958534262693295616",
        "1958534434491983232",
        "1958534468851721856",
        "1958535740162028032",
        "1958535976382244480",
        "1958536599157070592",
        "1958537561228353152",
        "1958539966410102784",
        "1958542199793256960",
        "1958543711621565312",
        "1958203515852455296",
        "1958544879852674304",
        "1958545463968392320",
        "1958545876285244416",
        "1958546838357909888",
        "1958552816952547968",
        "1958553194909669632",
        "1958555703170244608",
        "1958556287285793024",
        "1958556321645530368",
        "1958556424724742016",
        "1958557519938241792",
        "1958557936553229696",
        "1958558073992353664",
        "1958558589388428160",
        "1958561200728431872",
        "1958562506398582144",
        "1958563056154391168",
        "1958566384750731648",
        "1958569240907453312",
        "1958573776391974528",
        "1958574635385431424",
        "1958576422092037760",
        "1958576834408678656",
        "1958578479377170688",
        "1958580201663234432",
        "1958582671266686080",
        "1958583603279110528",
        "1958586867452688768",
        "1958588860317508224",
        "1958589410073513344",
        "1958589513152724864",
        "1958591024980607872",
        "1958591609098046592",
        "1958592571164942080",
        "1958594151718733184",
        "1958594460954618624",
        "1958595182509117696",
        "1958595560466053760",
        "1958599202598489984",
        "1958599855433768576",
        "1958603978602548736",
        "1958604184760953856",
        "1958605112473708032",
        "1958606040186636928",
        "1958608720246495360",
        "1958610300794623360",
        "1958616279389078272",
        "1958616412528336256",
        "1958617305891059200",
        "1958627515023326080",
        "1958627652462456448",
        "1958627789901413376",
        "1958785020063452288",
        "1958787047288006144",
        "1958787425245128320",
        "1958787528324503808",
        "1958793575638453120",
        "1958793781796883840",
        "1958793884876098688",
        "1958795396704738560"
    ]

    full_table = table.Table.read(get_input_path(TARGET))

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

    #     # Only 766452769615859712 has no V band data
    #     if len(band_table) == 0:
    #         print(f"No data for band {band}. Skipping.")
    #         continue
        
    #     # # 1. Optimize
    #     # best_ensemble, all_comps = get_comps(
    #     #     band_table, TARGET_ID, criterion=CRIT,
    #     #     exclude_ids=get_excluded_comp(TARGET, band)
    #     # )

    #     best_ensemble = USED_COMPS  # Only used if not optimizing
        
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
    #         # "best_ensemble": best_ensemble,
    #         "forced_comps": USED_COMPS,
    #         "all_comps": all_comps,
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
        read_saved_table_base,
        TARGET,
        # n_std_mid=50,  # normally 12
        xlim=(datetime(2023, 7, 2), datetime(2024, 3, 20)),
        x_title=0.1,
        y_title=0.94,
        fig_width=6,
        fig_height_per_panel=1.3,
        sp_adj_top=0.92,
        savefig_path=OUTPUT_DIR / f"monitoring_{TARGET}.pdf"
    )
    ### END BLOCK ###
