# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0


import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.dates as mpldates
from datetime import datetime
import locale
import numpy as np
from collections import OrderedDict


locale.setlocale(locale.LC_ALL,'en_US.utf8')

ls='-'
ms='o'
lw=1.2

fontsize=16

fontname = 'FreeSans'

mpl.rc('font',family=fontname)


#%% fake news colors

from palettable.cartocolors.diverging import Geyser_7_r as colormap
mediacolors = colormap.mpl_colors
mediacolors.insert(0, (0.5,0.5,0.5))
clinton_color = [c/255 for c in [38, 17, 127]]
trump_color = [c/255 for c in [217, 189, 33]]


def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1, 1, 1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent

def darker(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1, 1, 1)'''
    color = np.array(color)
    black = np.array([0, 0, 0])
    vector = black-color
    return color + vector * percent

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
#%%



def compute_discrete_CCDF(values, normalise=True):
    """ compute survival function 
    (Complementary Cummulative Distribution Function) of a set of values
        P(X>=x)"""
    
    x_CCDF, y_CCDF = np.unique(np.sort(values)[::-1], return_index=True)
    y_CCDF = np.asarray(y_CCDF, dtype=np.float64)

    y_CCDF = y_CCDF[:-1]    
    y_CCDF = np.hstack((len(values),y_CCDF))
    if normalise:
        y_CCDF /= y_CCDF[0]
    
    return x_CCDF, y_CCDF       
