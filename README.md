# The New Pipeline for Optical Robotic Telescopes (NewPORT)

Photometric pipeline made for the automated 24-inch telescope at Van Vleck Observatory (VVO), Wesleyan University

## Getting Started

### Setting Up ASTAP

Please refer to [ASTAP website](https://www.hnsky.org/astap) and its [SourceForge page](https://sourceforge.net/projects/astap-program/). The command line version should suffice, but there's no harm in using a full version with GUI. If you use the command line version, what you download is the executable, and you will need the path to this executable. If you use the GUI version, the path to the executable is usually /Applications/ASTAP.app/Contents/MacOS/astap on a Mac.

A star database with the right size corresponding to the images' FOV is required, as illustrated on the [ASTAP website](https://www.hnsky.org/astap#:~:text=Star%20databases%20usability:). For the VVO 24-inch, the FOV is 0.7". Therefore, the default choice is D50, but D80 might give better results(?), although it requires more storage space on the computer. If storage is limited, D20 or D05 is also acceptable, but not other ones.

#### Code Signing for Mac with Apple Silicon (M1 and M2 chips, etc.)

As adapted from [ASTAP Mac installer SourceForge doc](https://sourceforge.net/projects/astap-program/files/macOS%20installer/#:~:text=Open%20a%20terminal%20windows%20and%20copy%20paste%20and%20execute%20the%20following%20command):

  "Open a terminal tab and copy paste and execute the following command: `codesign --force -s - [path_to_executable]`
  
  "The code signing is required only once. An update doesn't require code signing."

## Acknowledgment

### ASTAP - *Gaia*

ASTAP uses *Gaia* data to do photometric solving. Considering acknowledging *Gaia*. See [ASTAP star database doc](https://sourceforge.net/projects/astap-program/files/star_databases/) and [*Gaia* Data Credits and Acknowledgemnts](https://www.cosmos.esa.int/web/gaia-users/credits). (Note: I'm not sure if ASTAP uses *Gaia* DR3 or EDR3...)

Warning: do not modify the name of this repository because it is linked in Chris's thesis.
