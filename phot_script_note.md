# `phot_[gaia|list]_run_*.py` Which Are Which?

Author: Qiushi Chris Tian

Created: 2026-10-05

Updated: 

## "List" Runs (New, Fast Option)

The idea for the "list" run option is that instead of running source finding for comparison stars, it takes a _list_ of Gaia DR3 IDs as comparison stars. The list comes from `newport.get_comparison_star_list()`, which is a getter/wrapper for `newport.COMPARISON_STAR`.

The only two "list" scripts are `phot_list_run.py` and `phot_list_run_nights.py`. The only difference (other than some dev printings) between the two is that **`_nights` includes a list of nights to _exclude_** from photometry.

## "Gaia" Runs (The OG)

There are four options, `phot_gaia_run.py`, `phot_gaia_mag_run.py`, `phot_gaia_mag_run_async.py`, and `phot_gaia_mag_run_2.py`. This part gets confusing, so please read very carefully:

`phot_gaia_run.py` seems  to be completely obsolete. It does not even do `if __name__ == '__main__'` and just runs things in the open. I doubt that there will be anything in it that is not in some of the newer files.

Both `phot_gaia_mag_run.py` and `phot_gaia_mag_run_2.py` import `concurrent.futures` and runs `get_fwhm_nanmin()` with a `ProcessPoolExecutor`. A difference (among others) is that `phot_gaia_mag_run.py` does not have giant Gaia table mask (`mask_gaia_table`), but `phot_gaia_mag_run_2.py` does.

`phot_gaia_mag_run_async.py` does ***not*** import `concurrent.futures`. Compared to `phot_gaia_mag_run_2.py`, it also has the giant Gaia table mask (`mask_gaia_table`) turn off and replaced with an all-zero mask, and uses SIMABD TAP to search for magnitudes and turns specific elements to one.

Aother thing to note is that none of the Gaia runs saves SKYTEMP, but both the of list runs do.