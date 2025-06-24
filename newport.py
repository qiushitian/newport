"""
Newport functional methods
"""
import ccdproc.version
import numpy as np
import astropy
from datetime import datetime
from photutils.aperture import ApertureStats
from photutils.utils import calc_total_error
import photutils.version
from astropy.time import Time
from astropy.coordinates import SkyCoord
import urllib.request
import xml.etree as etree
import matplotlib
import json

print(f'Numpy {np.version.version}, Astropy {astropy.version.version}, photutils {photutils.version.version}, '
      f'ccdproc {ccdproc.version.version}, Matplotlib {matplotlib.__version__}')

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
                   'TOI-1201': '5157183324996790272'}  # 5157183324996789760

TARGET_SKYCOORD = {
    'TOI-1201': SkyCoord('02h48m59.45s -14d32m14.22s'),
    'HD_86226': SkyCoord(ra='149.123515d', dec='-24.099186d'),
    'HD_191939': SkyCoord(ra='302.025625d', dec='66.850301d'),
    'TOI-1759': SkyCoord(ra='326.851662d', dec='62.753816d'),
    'TOI-1410': SkyCoord(ra='334.882849d', dec='42.560314d'),
    'TOI-561': SkyCoord(ra='148.185152d', dec='6.216103d'),
    'TOI-431': SkyCoord(ra='83.26925d', dec='-26.72387d'),
    'TOI-178': SkyCoord(ra='7.302011d', dec='-30.454116d'),
}

LITERATURE_MAG = {
    'HD_86226': {'B': 8.56, 'V': 7.93, 'R': 7.71, 'I': np.nan},
    'TOI-1201': {'B': np.nan, 'V': np.nan, 'R': 12.677, 'I': 10.72},
    'TOI-1410': {'B': 12.26, 'V': 11.11, 'R': np.nan, 'I': np.nan, 'G': 10.875941, 'J': 9.251},
    'TOI-1759': {'B': np.nan, 'V': np.nan, 'R': 11.002, 'I': np.nan},
    'TOI-561': {'B': 10.909, 'V': 10.243, 'R': 10.065, 'I': 9.843, 'i': 9.843},  # I is i
    'TOI-431': {'B': 10.17, 'V': 9.13, 'R': np.nan, 'I': np.nan},
    'TOI-178': {'B': 13.05, 'V': 11.95, 'R': np.nan, 'I': np.nan, 'G': 11.157550, 'J': 9.372},
    'HD_191939': {'B': np.nan, 'V': np.nan, 'R': 8.5, 'I': np.nan},
}

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
            # can add 5660503082056820608,  can add (less good) 5660501673307548800
            '5660490510688955904', '5660516447995041664'
        ],
        'I': [
            # can add 5660503082056820608, but 5660501673307548800 seems to be bad (increase night-to-night std)
            '5660516447995041664', '5660503082056820608'
        ]
    },
    'TOI-1201': {
        'B': ['5157179060094260736', '5157184007896584960'],
        'V': ['5157179060094260736', '5157184007896584960'],
        'R': ['5157179060094260736', '5157184007896584960'],
        'I': ['5157179060094260736', '5157184007896584960'],
    },
    'TOI-1410': {
        'B': ['1958534468851721856', '1958561200728431872', '1958588413640899712', '1958589410073513344'],
        'V': ['1958534468851721856', '1958561200728431872', '1958588413640899712', '1958589410073513344'],
        'R': ['1958534468851721856', '1958561200728431872', '1958588413640899712', '1958589410073513344'],
        'I': ['1958534468851721856', '1958561200728431872', '1958588413640899712', '1958589410073513344']
    },
    'TOI-178': {
        'V': ['2318344495077073920', '2318389815572034816'],
        'R': ['2318344495077073920'],
        'I': ['2318344495077073920']
    },
    'TOI-1759': {
        'B': [
            '2216367437308515072', '2216392038881729408', '2216393069673870592',  # '2216391695286437888',
            '2216414888108779520', '2216416743536122752', '2216417808686469376',  # '2216425161670352128',
            '2216431445197473152', '2216432475989623040', '2216440176876564736', '2216441031566760960',
            '2216445876287582848'
        ],
        'V': [
            '2216367437308515072', '2216392038881729408', '2216393069673870592',
            '2216414888108779520', '2216416743536122752', '2216417808686469376',
            # '2216419698472018560', '2216425161670352128',
            '2216431445197473152', '2216432475989623040', '2216440176876564736', '2216441031566760960',
            '2216445876287582848'
        ],
        'R': ['2216393069673870592', '2216413543776087808']  # '2216368566879908608' is a lp variable
    },
    'TOI-431': {
        'B': [
            '2908637241099225600', '2908637172379753856', #'2908681049765615616', '2908663491939308416',
            # '2908676205042514688', '2908638959086136192', '2908658715935682176', '2908659506207715584',
            # '2908681874399329280'
        ],
        'V': [
            '2908637241099225600', '2908637172379753856', #'2908681049765615616', '2908663491939308416',
            # '2908676205042514688', '2908638959086136192', '2908658715935682176', '2908659506207715584'
        ],
        'R': ['2908637241099225600', '2908681874399329280'],  # , '2908669156998939264'
        # 'I': ['2908669156998939264']
    },
    'TOI-561': {
        'B': ['3850423268738291072', '3850420657398178176', '3850424406904271488', '3850405298595129856',
              '3850428255194995712'],
        'V': ['3850423268738291072', '3850420657398178176', '3850424406904271488', '3853427787340315136'],
        'R': ['3850423268738291072', '3850396365063154560', '3850405298595129856']
    },
    'HD_191939': {
        'B': ['2248124184971495936', '2248131366156908288', '2248134939569731840', '2248135317526853120',
              '2248136313959256960', '2248136825057166720'],  # , '2248137653989048320'
        'V': ['2248124184971495936', '2248131366156908288', '2248134939569731840', '2248135317526853120',
              '2248136313959256960', '2248136825057166720'],  # , '2248137653989048320'
        'R': ['2248111678026733440', '2248136825057166720']
    },
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


def get_comparison_star_list(target, band):
    target_comp_star_dict = COMPARISON_STAR.get(target)
    if target_comp_star_dict:
        comp_star_list = target_comp_star_dict.get(band)
        if comp_star_list:
            return comp_star_list
    return []


def get_comp_mags(star, band: str, target: str = None):
    """
    What?

    Parameters
    ----------
    star: (str or list) – comp star Gaia DR3 ID(s)
    band: d
    target: d

    Returns
    -------
        float
    """
    is_scalar = False
    if not isinstance(star, list):
        star = [star]
        is_scalar = True

    with open("comp-mag.json", "r") as f:
        comp_star_mags = json.load(f)

    mags = {}
    for _target, comp_mag_target in comp_star_mags.items():
        mag_list = []
        for i_star in star:
            if i_star in comp_mag_target.get(band, {}):
                mag_list.append(comp_mag_target[band][i_star])
            # else:
            #     raise ValueError(f'{i_star} not found in {band} band.')
        if len(mag_list) > 0:
            if is_scalar:
                mags[_target] = mag_list[0]
            else:
                mags[_target] = mag_list

    if len(mags) == 1:
        return list(mags.values())[0]
    elif len(mags) > 1:
        if target:
            return mags[target]
        else:
            raise ValueError(f'Stars {star} found in {band} band for targets {mags.keys()}. '
                             f'Call with `get_comp_mag(star, band, target)`')
    else:  # i.e. len(mags) == 0
        return []
        # raise RuntimeError('Impossible error???')


def get_super_mag(mags):
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


def get_fwhm_nanmin(aperture_stats: ApertureStats):
    return np.nanmin(aperture_stats.fwhm)


def error_func(data, gain=1.85):
    # return np.sqrt(data)
    # return np.sqrt(data / gain)
    return calc_total_error(data.astype(float), 0, gain)
