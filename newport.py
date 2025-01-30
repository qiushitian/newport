"""
Newport functional methods
"""

import numpy as np
import astropy.time as time
import urllib.request
import xml.etree as etree
from datetime import datetime

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

TARGET_GAIA_DR3 = {'HD_86226' : '5660492297395345408',
                   'HD_191939' : '2248126315275354496',
                   'TOI-1759' : '2216420110788943744',
                   'TOI-178' : '2318295979126499200',
                   'TOI-431' : '2908664557091200768',
                   'TOI-561' : '3850421005290172416',
                   'TOI-1410' : '1958584565350234752',
                   'TOI-1201' : '5157183324996790272'}

COLORS = {'B': 'C0', 'V': 'C2', 'R': 'C3', 'I': 'maroon'}
MARKERS = {'B': 'o', 'V': 'x', 'R': 's', 'I': 'D'}

def hst_visits():
    """
    Process StSci HST GO 17192 XML status

    :return: time at which XML is fetched, dict of past visits, dict of future visits
    """
    archived = {}
    future = {}

    url = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17192&markupFormat=xml&observatory=HST'
    with urllib.request.urlopen(url) as response:
        xml_string = response.read().decode('utf-8')
    root = etree.ElementTree.fromstring(xml_string)

    # iter around visits
    for v in root.iterfind('visit'):
        status = v.find('status').text
        target = v.find('target').text

        # keep only WFC3 observation
        config = v.find('configuration').text
        if not config.startswith('WFC3'):
            continue
        print(target, config)

        # find archived (done) visits
        starts = []
        for t in v.iterfind('startTime'):
            starts.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
        ends = []
        for t in v.iterfind('endTime'):
            ends.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
        for s, e in zip(starts, ends):
            if target not in archived:
                archived[target] = []
            archived[target].append((s, e))

        # find future plan windows
        for t in v.iterfind('planWindow'):
            start_end = t.text[:-22].split(' - ')
            start = datetime.strptime(start_end[0], '%b %d, %Y')
            end = datetime.strptime(start_end[1], '%b %d, %Y')
            if target not in future:
                future[target] = []
            future[target].append((start, end))

    url2 = 'https://www.stsci.edu/cgi-bin/get-visit-status?id=17414&markupFormat=xml&observatory=HST'
    with urllib.request.urlopen(url2) as response:
        xml_string2 = response.read().decode('utf-8')
    root2 = etree.ElementTree.fromstring(xml_string2)

    # iter around visits
    for v in root2.iterfind('visit'):
        status = v.find('status').text
        target = v.find('target').text

        # keep only WFC3 observation
        config = v.find('configuration').text
        if not config.startswith('WFC3'):
            continue
        print(target, config)

        # find archived (done) visits
        starts = []
        for t in v.iterfind('startTime'):
            starts.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
        ends = []
        for t in v.iterfind('endTime'):
            ends.append(datetime.strptime(t.text, '%b %d, %Y %H:%M:%S'))
        for s, e in zip(starts, ends):
            if target not in archived:
                archived[target] = []
            archived[target].append((s, e))

        # find future plan windows
        for t in v.iterfind('planWindow'):
            start_end = t.text[:-22].split(' - ')
            start = datetime.strptime(start_end[0], '%b %d, %Y')
            end = datetime.strptime(start_end[1], '%b %d, %Y')
            if target not in future:
                future[target] = []
            future[target].append((start, end))

    # get time at which the information is fetched
    report_time = datetime.strptime(root2.find('reportTime').text, '%a %b %d %H:%M:%S %Z %Y')

    # print(report_time.strftime("%m/%d/%Y %H:%M"))

    return report_time, archived, future

def plot_hst(id, dict, ax, c):
    times = np.array(dict.get(id.replace(' ', '-')))
    if len(times.shape) > 1:
        for s, e in times:
            mid = s + (e - s) / 2
            ax.axvline(time.Time(mid).mjd, c=c, linewidth=0.3, alpha=1)
            # ax.axvline(Time(s).mjd, c=c, linewidth=0.1, alpha=0.8)
            # ax.axvline(Time(e).mjd, c=c, linewidth=0.1, alpha=0.8)
