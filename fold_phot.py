#!/usr/bin/env python3
"""
Wrapper script for generating folded multi-band light curves using plot_fold.
"""
from pathlib import Path
from optimize_rel_phot import plot_fold
from target_phot import OUTPUT_DIR


TARGET = "HD_191939"
PERIODS = [
    # 28.7,

    # 29.4,
    # 46.8,
    # 49.2,
    # 50.5,
    # 46.8,
    # 49.2,
    # 50.5,
    # 64.2,
    # 66.4,
    # 92.4,
    # 97.1,
    # 151.5,
    # 164.7,
    # 344.3,
    # 420.8

    5.646666860479654,
    6.173040721695691,
    12.871725626868102,
    33.044944396155444
]
T0 = 0


if __name__ == "__main__":
    output_dir = OUTPUT_DIR / 'folded'
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in PERIODS:
        output_path = output_dir / f"folded_{str(p).replace('.', 'p')}d.pdf"
        print(f"Generating folded light curve for {TARGET} (P={p} days)...")
        
        # plot_fold expects base_path so that bin_{base_path.stem}_{band}.fits exists
        # If using bin_results_B.fits, base_path should be INPUT_DIR / 'results'
        plot_fold(
            base_path=OUTPUT_DIR / "results",
            target_name=TARGET,
            period=p,
            t0=T0,
            # n_std_mid=15,
            savefig_path=output_path,
            bands=['B', 'V', 'R', 'I'],
            show=False
        )
