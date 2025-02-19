#!/usr/bin/env python3
"""

"""

from newport import *
import json
from astroquery.utils.tap.core import TapPlus

SIMBAD_TAP = TapPlus(url="http://simbad.u-strasbg.fr/simbad/sim-tap")

if __name__ == '__main__':
    comp_star_mags = {}
    for target, comp_star_per_band_dict in COMPARISON_STAR.items():
        comp_star_mags[target] = {}
        for band, comp_star_list in comp_star_per_band_dict.items():
            comp_star_mags[target][band] = {}
            for comp_star in comp_star_list:
                mag = SIMBAD_TAP.launch_job(
                    f"SELECT {band} FROM allfluxes JOIN ident USING(oidref) WHERE id = 'Gaia DR3 {comp_star}';"
                ).get_results()[band]
                if len(mag) > 0 and not mag.mask:
                    comp_star_mags[target][band][comp_star] = mag[0]
                else:
                    comp_star_mags[target][band][comp_star] = np.nan
    with open('comp-mag.json', 'w') as f:
        json.dump(comp_star_mags, f, indent=4)  # indent for pretty-printing
