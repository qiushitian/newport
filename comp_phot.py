from pathlib import Path
from astropy import table
import numpy as np
import json
from newport import get_input_path
from target_phot import TARGET, TARGET_ID, OUTPUT_DIR
from optimize_rel_phot import phot_comp, plot_comp, load_from_json


if __name__ == "__main__":
    print(f'Newport comparison star photometry for {TARGET} to {OUTPUT_DIR}')
    
    comp_diag_dir = OUTPUT_DIR / "comp_diag"
    comp_diag_dir.mkdir(parents=True, exist_ok=True)

    # Read json to find all comp stars
    comp_phot_full_table = table.Table.read(get_input_path(TARGET))
    comp_phot_bands_avail = \
        np.unique(comp_phot_full_table['band']).astype('str')

    used_comps, all_comps = set(), set()
    for band in comp_phot_bands_avail:
        json_path = OUTPUT_DIR / f"comps_{band}.json"
        used_comps.update(
            load_from_json(json_path, ['forced_comps', 'best_ensemble'])
        )
        all_comps.update(
            load_from_json(json_path, ['ref_comps', 'all_comps'])
        )

    # # 191939
    # all_comps = {
    #     '2248119198511385984', '2248124184971495936',
    #     '2248131366156908288', '2248134939569731840',
    #     '2248135317526853120', '2248136313959256960',
    #     '2248136825057166720', '2248137653989048320'
    # }

    # # 1759
    # all_comps = {
    #     # '2216368566879908608',
    #     '2216414888108779520', '2216416743536122752', '2216417808686469376', '2216419698472018560', '2216431445197473152', '2216432475989623040', '2216433953458370176', '2216441031566760960', '2216445876287582848'
    # }

    try:
        from target_phot import USED_COMPS
        used_comps = set(USED_COMPS)
        print(f"Using `USED_COMPS` from target_phot.py")
    except ImportError:
        pass

    ### BLOCK: Run comp diagnostics ###
    phot_comp(
        comp_phot_full_table, TARGET_ID, all_comps, comp_diag_dir,
        # comp_phot_full_table, TARGET_ID, used_comps, comp_diag_dir,
        bands=comp_phot_bands_avail
    )
    ### END BLOCK ###

    ### BLOCK: Plot comp diagnostics ###
    plot_comp(
        comp_diag_dir, all_comps, used_comps, TARGET,
        # comp_diag_dir, used_comps, used_comps, TARGET,
        n_std_mid=15, savefig_path=comp_diag_dir / "comp_diagnostics.pdf",
        bands=comp_phot_bands_avail
    )
    ### END BLOCK ###

else:
    raise RuntimeError('`comp_phot.py` is intended to be run as a script.')
