"""
Newport functional methods
"""

import numpy as np
import astropy
from datetime import datetime
from astropy.time import Time
import urllib.request
import xml.etree as etree
import matplotlib
import json

print(f'Numpy {np.version.version}, Astropy {astropy.version.version}, Matplotlib {matplotlib.__version__}')

TARGET_FN = [
    'HD_191939',
    'HD_86226',
    'TOI-431',
    'TOI-561',
    'TOI-1201',
    'TOI-178',
    'TOI-1759',
    'TOI-1410',
]

TARGET_GAIA_DR3 = {'HD_86226': '5660492297395345408',
                   'HD_191939': '2248126315275354496',
                   'TOI-1759': '2216420110788943744',
                   'TOI-178': '2318295979126499200',
                   'TOI-431': '2908664557091200768',
                   'TOI-561': '3850421005290172416',
                   'TOI-1410': '1958584565350234752',
                   'TOI-1201': '5157183324996790272'}

LITERATURE_MAG = {'HD_86226': {'B': 8.56, 'V': 7.93, 'R': 7.71, 'I': np.nan}}

COMPARISON_STAR = {
    'HD_86226': {
        'B': [
            '5660477110391333376', '5660490510688955904', '5660494320324102272', '5660496214405409152',
            '5660501673307548800', '5660503082056820608', '5660516447995041664'
        ],
        'V': [
            '5660477110391333376', '5660490510688955904', '5660501673307548800',
            '5660503082056820608', '5660516447995041664'
        ],
        'R': [
            '5660490510688955904', '5660516447995041664'  # can add 5660503082056820608
        ],
        'I': [
            '5660516447995041664'  # can add 5660501673307548800, 5660503082056820608
        ]
    }
}

EXCLUDED_COMP_STAR = {
    'HD_86226': {
        'B': ['5660517654882236800', '5660474529114240512', '5660494320324102272', '5660496214405409152'],
        'V': ['5660517654882236800', '5660474529114240512', '5660494320324102272', '5660496214405409152'],
        'R': ['5660501673307548800', '5660503082056820608'],  # can keep 608
        'I': ['5660501673307548800', '5660503082056820608']  # can keep 800 and 608
    }
}

COLORS = {'B': 'C0', 'V': 'C2', 'R': 'C3', 'I': 'maroon'}
MARKERS = {'B': 'o', 'V': 'x', 'R': 's', 'I': 'D'}

with open("comp-mag.json", "r") as f:
    comp_star_mags = json.load(f)


def get_comp_mag(star: str, band: str, target: str = None):
    """

    Parameters
    ----------
    star
    band
    target

    Returns
    -------
        float
    """
    mags = {}
    for target, comp_mag_target in comp_star_mags.items():
        if star in comp_mag_target[band]:
            mags[target] = comp_mag_target[band][star]
    if len(mags) == 1:
        return list(mags.values())[0]
    elif len(mags) == 0:
        raise ValueError(f'{star} not found in {band} band.')
    else:
        if target:
            return mags[target]
        else:
            raise ValueError(f'{star} found in {band} band for targets {mags.keys()}. '
                             f'Call with `get_comp_mag(band, star, target)`')


def super_mag(mags):
    """
    Calculate the "super" magnitude of a fictional star that has the combined fluxes
    of all stars in the input list/array of magnitudes.

    Parameters:
        mags: np.ndarray or list – Magnitudes of individual stars.

    Returns:
        float: The magnitude of the "super" star.
    """
    fluxes = 10 ** (-0.4 * np.array(mags))  # Convert magnitudes to fluxes (flux = 10^(-0.4 * magnitude))
    total_flux = np.sum(fluxes)  # Sum the fluxes
    return -2.5 * np.log10(total_flux)  # Convert the total flux back to magnitude


def get_hst(target: str, path=None, url=None, future=False):
    """
    Returns:
        (wfc3 mid time, stis mid time)
    """
    if path:
        with open(path, 'r') as f:
            xml_string = f.read()
    elif url:
        if url == '17192' or url == 17192:
            url = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17192&markupFormat=xml&observatory=HST'
        elif url == '17414' or url == 17414:
            url = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17414&markupFormat=xml&observatory=HST'
        with urllib.request.urlopen(url) as response:
            xml_string = response.read().decode('utf-8')
    else:
        raise ValueError('`path` and `url` cannot be both None.')

    root = etree.ElementTree.fromstring(xml_string)

    target = target.replace('_', '-').replace(' ', '-')

    # iter around visits
    wfc3_mid_time = []
    stis_mid_time = []
    for v in root.iterfind('visit'):
        # visit status screening
        status = v.find('status').text
        if not future and status != 'Executed':
            continue

        # visit target screening
        visit_target = v.find('target').text
        if visit_target != target:
            continue

        # find mid time
        start_time = datetime.strptime(v.find('startTime').text, '%b %d, %Y %H:%M:%S')
        end_time = datetime.strptime(v.find('endTime').text, '%b %d, %Y %H:%M:%S')
        midpoint = Time(start_time + (end_time - start_time) / 2)

        # goes into wfc3 or stis mid time list
        config = v.find('configuration').text
        if config.startswith('WFC3'):
            wfc3_mid_time.append(midpoint)
        elif config.startswith('STIS'):
            stis_mid_time.append(midpoint)
        else:
            raise ValueError(f'Unknown configuration: {config}')

    print(f'HST visit report time: {root.find("reportTime").text}')

    return wfc3_mid_time, stis_mid_time


# --- Below are old `hst_visits` and `plot_hst` that works together.     --- #
# --- They are both obsolete and `hst_visits` is replaced with `get_hst` --- #

# def hst_visits():
#     """
#     Process StSci HST GO 17192 XML status
#
#     :return: time at which XML is fetched, dict of past visits, dict of future visits
#     """
#     archived = {}
#     future = {}
#
#     url = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17192&markupFormat=xml&observatory=HST'
#     with urllib.request.urlopen(url) as response:
#         xml_string = response.read().decode('utf-8')
#     root = etree.ElementTree.fromstring(xml_string)
#
#     # iter around visits
#     for v in root.iterfind('visit'):
#         status = v.find('status').text
#         target = v.find('target').text
#
#         # keep only WFC3 observation
#         config = v.find('configuration').text
#         if not config.startswith('WFC3'):
#             continue
#         print(target, config)
#
#         # find archived (done) visits
#         starts = []
#         for t in v.iterfind('startTime'):
#             starts.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
#         ends = []
#         for t in v.iterfind('endTime'):
#             ends.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
#         for s, e in zip(starts, ends):
#             if target not in archived:
#                 archived[target] = []
#             archived[target].append((s, e))
#
#         # find future plan windows
#         for t in v.iterfind('planWindow'):
#             start_end = t.text[:-22].split(' - ')
#             start = datetime.strptime(start_end[0], '%b %d, %Y')
#             end = datetime.strptime(start_end[1], '%b %d, %Y')
#             if target not in future:
#                 future[target] = []
#             future[target].append((start, end))
#
#     url2 = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17414&markupFormat=xml&observatory=HST'
#     with urllib.request.urlopen(url2) as response:
#         xml_string2 = response.read().decode('utf-8')
#     root2 = etree.ElementTree.fromstring(xml_string2)
#
#     # iter around visits
#     for v in root2.iterfind('visit'):
#         status = v.find('status').text
#         target = v.find('target').text
#
#         # keep only WFC3 observation
#         config = v.find('configuration').text
#         if not config.startswith('WFC3'):
#             continue
#         print(target, config)
#
#         # find archived (done) visits
#         starts = []
#         for t in v.iterfind('startTime'):
#             starts.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
#         ends = []
#         for t in v.iterfind('endTime'):
#             ends.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
#         for s, e in zip(starts, ends):
#             if target not in archived:
#                 archived[target] = []
#             archived[target].append((s, e))
#
#         # find future plan windows
#         for t in v.iterfind('planWindow'):
#             start_end = t.text[:-22].split(' - ')
#             start = datetime.strptime(start_end[0], '%b %d, %Y')
#             end = datetime.strptime(start_end[1], '%b %d, %Y')
#             if target not in future:
#                 future[target] = []
#             future[target].append((start, end))
#
#     # get time at which the information is fetched
#     report_time = datetime.strptime(root2.find('reportTime').text, '%a %b %d %H:%M:%S %Z %Y')
#
#     # print(report_time.strftime("%m/%d/%Y %H:%M"))
#
#     return report_time, archived, future
#
#
# def plot_hst(id, dict, ax, c):
#     times = np.array(dict.get(id.replace(' ', '-')))
#     if len(times.shape) > 1:
#         for s, e in times:
#             mid = s + (e - s) / 2
#             ax.axvline(Time(mid).mjd, c=c, linewidth=0.3, alpha=1)
#             # ax.axvline(Time(s).mjd, c=c, linewidth=0.1, alpha=0.8)
#             # ax.axvline(Time(e).mjd, c=c, linewidth=0.1, alpha=0.8)
