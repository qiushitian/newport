from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *
from target_phot import TARGET, TARGET_ID, OUTPUT_DIR
from newport import get_input_path


print(f'Newport comparison star photometry for {TARGET} to {OUTPUT_DIR}')

# Read json to find all comp stars
comp_phot_full_table = table.Table.read(get_input_path(TARGET))
comp_phot_bands_avail = np.unique(comp_phot_full_table['band'])
ALL_COMPS = set()
for band in comp_phot_bands_avail:
    json_path = OUTPUT_DIR / f"opt_ensemble_{band}.json"
    ALL_COMPS.update(load_optimized_json(json_path))

# # 191939
# ALL_COMPS = {
#     '2248119198511385984', '2248124184971495936',
#     '2248131366156908288', '2248134939569731840',
#     '2248135317526853120', '2248136313959256960',
#     '2248136825057166720', '2248137653989048320'
# }

# # 1759
# ALL_COMPS = {
#     # '2216368566879908608',
#     '2216414888108779520', '2216416743536122752', '2216417808686469376', '2216419698472018560', '2216431445197473152', '2216432475989623040', '2216433953458370176', '2216441031566760960', '2216445876287582848'
# }

# Note: USED_COMPS are plotted in color while ALL_COMPS are grayed out. 
# If no USED_COMPS are defined in `target_phot.py`, then ALL_COMPS are plotted in color.
try:
    from target_phot import USED_COMPS
except ImportError:
    USED_COMPS = ALL_COMPS


if __name__ == "__main__":
    comp_diag_dir = OUTPUT_DIR / "comp_diag"
    comp_diag_dir.mkdir(parents=True, exist_ok=True)

    ### BLOCK: Run comp diagnostics ###
    phot_comp(
        comp_phot_full_table, TARGET_ID, ALL_COMPS, comp_diag_dir,
        bands=comp_phot_bands_avail
    )
    ### END BLOCK ###

    ### BLOCK: Plot comp diagnostics ###
    plot_comp(
        comp_diag_dir, ALL_COMPS, USED_COMPS, TARGET,
        n_std_mid=15, savefig_path=comp_diag_dir / "comp_diagnostics.pdf",
        bands=comp_phot_bands_avail
    )
    ### END BLOCK ###
