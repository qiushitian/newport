"""
Handle dependent packages (i.e. imports) for NewPORT.

Author: Qiushi Chris Tian
Created: 2026-08-17
Updated: 2026-08-17
"""

STYLE = 'stix'


def __import(pkg: str) -> object:
    raise NotImplementedError


try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Cannot import Matplotlib. No plotting possible.")
else:
    try:
        import visualastro as va
    except ImportError:
        print("Cannot import VisualAstro. Plot with Matplotlib default.")
    else:
        plt.style.use(STYLE)


if __name__ == '__main__':
    print('dep.py is not supposed to be executed.')
