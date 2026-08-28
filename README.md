# The New Pipeline for Optical Robotic Telescopes (NewPORT)

Photometric pipeline made for the automated 24-inch telescope at Van Vleck Observatory (VVO), Wesleyan University

- Author: Qiushi Chris Tian
- Updated: 2026-08-17

## Installing NewPORT

The best way to install NewPORT as of now is to clone the repo with Git. Alternatively, you can also download a ZIP of the repo by clicking on the "Code" button on GitHub. 

### Setting Up ASTAP

Please refer to [ASTAP website](https://www.hnsky.org/astap) and its [SourceForge page](https://sourceforge.net/projects/astap-program/). The command line version should suffice, but there's no harm in using a full version with GUI. If you use the command line version, what you download is the executable, and you will need the path to this executable. If you use the GUI version, the path to the executable is usually /Applications/ASTAP.app/Contents/MacOS/astap on a Mac.

A star database with the right size corresponding to the images' FOV is required, as illustrated on the [ASTAP website](https://www.hnsky.org/astap#:~:text=Star%20databases%20usability:). For the VVO 24-inch, the FOV is 0.7". Therefore, the default choice is D50, but D80 might give better results(?), although it requires more storage space on the computer. If storage is limited, D20 or D05 is also acceptable, but not other ones.

#### Code Signing for Macs with Apple Silicon (M1 and M2 chips, etc.)

As adapted from [ASTAP Mac installer SourceForge doc](https://sourceforge.net/projects/astap-program/files/macOS%20installer/#:~:text=Open%20a%20terminal%20windows%20and%20copy%20paste%20and%20execute%20the%20following%20command):

  "Open a terminal tab and copy paste and execute the following command: `codesign --force -s - [path_to_executable]`
  
  "The code signing is required only once. An update doesn't require code signing."

#### Security Override in Macs

macOS will prompt that the ASTAP program and the star database installer "cannot be opened because the developer cannot be identified." To override this, go to macOS's System Settings – Privacy & Security, scroll down to the "Security" section, find the message about ASTAP or the star database, and click "Open anyway." When prompt again, click "Open."

## Changelog

Commit [`c24678a`](https://github.com/qiushitian/newport/commit/c24678a3038488c4e17e566a4f2ed11e8410bbf7) is the closest representation of the state of the code at the time of Chris's thesis.

## Quick Start

### Analysis Scripts

- target_phot.py
- comp_phot.py
- period_vdp.py
- inspect_nights.py

optimize_rel_phot.py contains the object-oriented framework for performing relative photometry.

## Dependencies

The full functionalities of NewPORT require the following Python packages:
- NumPy
- SciPy
- Astropy
- Matplotlib
- [VisualAstro](https://github.com/elkogerville/VisualAstro)
- ccdproc
- photutils

It is possible to run NewPORT without certain packages, provided that you don't use the related functionalities. For example, if you don't do photometry on images and only post-process tables of data, you might not need ccdproc and photutils.

## Citing NewPORT

The following two papers include data and plots made with NewPORT:
- [K. A. Kahle et al., 2025, A&A 701 A184](https://ui.adsabs.harvard.edu/abs/2025arXiv250713439K/)
- [C. Gapp et al., 2026, arXiv:2608.05962](https://ui.adsabs.harvard.edu/abs/2026arXiv260805962G)

You can also link to this repo and/or refer to [Chris's thesis](https://doi.org/10.14418/wes01.1.2867).

The latest version of NewPORT uses [VisualAstro](https://github.com/elkogerville/VisualAstro) during plotting. You should also [cite VisualAstro](https://github.com/elkogerville/VisualAstro/blob/main/README.md#citing-visualastro) and other dependencies.

## Acknowledgments

### ASTAP - *Gaia*

ASTAP uses *Gaia* data to do photometric solving. Considering acknowledging *Gaia*. See [ASTAP star database doc](https://sourceforge.net/projects/astap-program/files/star_databases/) and [*Gaia* Data Credits and Acknowledgemnts](https://www.cosmos.esa.int/web/gaia-users/credits). (Note: I'm not sure if ASTAP uses *Gaia* DR3 or EDR3...)

##

Warning: Do not modify the name of this repository because it is linked in Chris's thesis.
