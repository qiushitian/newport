def solve(read_path, solver_path, ra=None, dec=None): #, save_path): TODO implementation allowing saving solving results elsewhere is canceled
    '''
    Parameters
    -----
    read_path : path-like
        Directory containing FITS to be solve. Results will also be save here. 
    '''
    
    # TODO add progress bar

    from pathlib import Path
    read_path = Path(read_path)
    # save_path = Path(save_path) TODO

    for suffix in [".fit", ".fits", ".fts", ".FIT", ".FITS", ".FTS"]:
        for file in read_path.rglob("[!._]*" + suffix):
            if not file.with_suffix('.ini').exists():
                print(file.resolve(), flush=True) # show progress
                _solve(file, file.with_suffix(''), solver_path, ra, dec)
            

def _solve(fits_path, output_path, solver_path, ra, dec):
    '''
    Parameters
    -----
    fits_path : path-like
        Path to FITS file to be solved
    output_path : path-like
        Path to which solved .wcs and .ini files are saved
    solver_path : path-like
        Path to the solver's command line executable (e.g. ASTAP)

    Returns : None
    '''
    from astropy.io import fits
    hdu = fits.open(fits_path, output_verify='warn')[0] # NOTE try-except abandoned

    from astropy.coordinates import Angle
    if ra is None:
        ra = Angle(hdu.header['RA'] + ' hours').to_value()
    else:
        ra = Angle(ra).to_value()
    if dec is None:
        dec = Angle(hdu.header['DEC'] + ' degrees').to_value()
    else:
        dec = Angle(dec).to_value()
        

    
    from subprocess import run, DEVNULL
    run([solver_path, '-f', fits_path, '-o', output_path, '-ra', str(ra), '-spd', str(dec + 90)], stdout=DEVNULL)


def ini_to_wcs(ini_path):
    '''
    Parameters
    ini : path-like
        Path to .ini file

    Returns : astropy.wcs.WCS
    '''
    if not str(ini_path).endswith('.ini'):
        raise ValueError('Input must be a .ini file.')
    
    # Read the contents of the INI file
    with open(ini_path, 'r') as file:
        contents = file.read()

    # Add a default section header
    contents = '[WCS]\n' + contents

    # Create a ConfigParser instance and parse the contents
    from configparser import ConfigParser
    config = ConfigParser()
    config.read_string(contents)

    # abort if not solved
    if config.get('WCS', 'pltsolvd') == 'F':
        raise ValueError('Input not solved.')

    # Extract the relevant fields
    header = {
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'CRVAL1': config.getfloat('WCS', 'crval1'),
        'CRVAL2': config.getfloat('WCS', 'crval2'),
        'CRPIX1': config.getfloat('WCS', 'crpix1'),
        'CRPIX2': config.getfloat('WCS', 'crpix2'),
        'CD1_1': config.getfloat('WCS', 'cd1_1'),
        'CD1_2': config.getfloat('WCS', 'cd1_2'),
        'CD2_1': config.getfloat('WCS', 'cd2_1'),
        'CD2_2': config.getfloat('WCS', 'cd2_2'),
        'RADESYS': 'FK5',
        'COMMENT': 'ASTAP cmd: ' + config.get('WCS', 'cmdline')
    }

    # Return the WCS object
    from astropy.wcs import WCS
    return WCS(header)
