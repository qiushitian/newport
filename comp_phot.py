from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *
from target_phot import TARGET, TARGET_ID, INPUT_PATH, OUTPUT_DIR, USED_COMPS


ALL_COMPS = [
    '2248119198511385984', '2248124184971495936',
    '2248131366156908288', '2248134939569731840',
    '2248135317526853120', '2248136313959256960',
    '2248136825057166720', '2248137653989048320'
]


if __name__ == "__main__":
    full_table = table.Table.read(INPUT_PATH)

    comp_diag_dir = OUTPUT_DIR / "comp_diag"
    comp_diag_dir.mkdir(parents=True, exist_ok=True)

    ### BLOCK: Run comp diagnostics ###
    phot_comp(full_table, TARGET_ID, ALL_COMPS, comp_diag_dir)
    ### END BLOCK ###

    ### BLOCK: Plot comp diagnostics ###
    plot_comp(
        comp_diag_dir, ALL_COMPS, USED_COMPS, TARGET,
        n_std_mid=15, savefig_path=comp_diag_dir / "comp_diagnostics.pdf"
    )
    ### END BLOCK ###
