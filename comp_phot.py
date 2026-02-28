from pathlib import Path
from astropy import table
import numpy as np
import json
from optimize_rel_phot import *

if __name__ == "__main__":
    TARGET = "HD_191939"
    TARGET_ID = "2248126315275354496"
    INPUT_PATH = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/phot_w_err_HD_191939.fits')
    READ_DIR = Path('/Users/tqs/GitHub/newport/tables/opt_comp_stars/HD_191939/eight_86')

    ALL_COMPS = ['2248119198511385984', '2248124184971495936', '2248131366156908288', '2248134939569731840', '2248135317526853120', '2248136313959256960', '2248136825057166720', '2248137653989048320']

    USED_COMPS = [
        '2248124184971495936',
        '2248135317526853120',
        '2248136313959256960'
    ]

    full_table = table.Table.read(INPUT_PATH)

    comp_diag_dir = READ_DIR / "comp_diag"
    comp_diag_dir.mkdir(parents=True, exist_ok=True)

    # ### BLOCK: Run comp diagnostics ###
    # phot_comp(full_table, TARGET_ID, ALL_COMPS, comp_diag_dir)
    # ### END BLOCK ###

    ### BLOCK: Plot comp diagnostics ###
    plot_comp(
        comp_diag_dir, ALL_COMPS, USED_COMPS, TARGET,
        n_std_mid=15, savefig_path=comp_diag_dir / "comp_diagnostics.pdf"
    )
    ### END BLOCK ###
