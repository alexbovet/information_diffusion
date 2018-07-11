# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import matplotlib.pyplot as plt

import graph_tool.all as gt


plt.style.use('alex_paper')

media_types = ['fake', 'far_right', 'right',
             'lean_right', 
            'center', 'lean_left', 'left',
            'far_left']


save_dir =  '../data/urls/revisions/'


#%% 

period = 'june-nov'

Graphs_simple = dict()
for media_type in media_types:
    Graphs_simple[media_type] = dict()
    Graphs_simple[media_type][period] = gt.load_graph(os.path.join(save_dir, 
                 'retweet_graph_' + \
                 media_type + '_simple_' + period + '.gt'))
    
Glist_simple = [Graphs_simple[key][period] for key in media_types]
    
    


#%% make CCDF plot
from PlotUtils import compute_discrete_CCDF
from PlotUtils import colormap, darker
from collections import OrderedDict

rename_dict = {'pro_h' : 'pro-Clinton',
               'pro_t' : 'pro-Trump',
               'fake' : 'fake news',
               'far_right' : 'extreme bias (right)',
               'right' : 'right',
               'lean_right' : 'right leaning',
               'center' : 'center',
               'lean_left' : 'left leaning',
               'left' : 'left',
               'far_left' : 'extreme bias (left)'}

colors = [(0.5,0.5,0.5)] + colormap.mpl_colors
markers = ['o','v','^','<', '>','s','+','x']
linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),
     ('dashed',              (0, (5, 5))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
lw = 1.7

for G, col, mk, ls in zip(Glist_simple, colors, markers, linestyles.values()):
    k, ccdf = compute_discrete_CCDF(G.vp.k_out.a)

    ax1.plot(k, ccdf, color=darker(col, 0.3), linewidth=lw,
             label=rename_dict[G.gp.name], ls=ls)
    
    k, ccdf = compute_discrete_CCDF(G.vp.k_in.a)

    ax2.plot(k, ccdf, color=darker(col, 0.3), linewidth=lw,
             label=rename_dict[G.gp.name], ls=ls)
    
    
ax1.set_xscale('log')    
ax1.set_xlim((1,3e5))
ax1.set_yscale('log')
ax1.legend(fontsize=14)
ax1.set_ylabel(r'$P(K_{out} \geq k_{out})$')
ax1.set_xlabel(r'$k_{out}$ [out-degree]')

ax2.set_xscale('log')    
ax2.set_xlim((1,1500))
ax2.set_yscale('log')
ax2.legend(fontsize=14)
ax2.set_ylabel(r'$P(K_{in} \geq k_{in})$')
ax2.set_xlabel(r'$k_{in}$ [in-degree]')

ax1.text(-0.15,1.0, 'a', fontweight='bold', transform=ax1.transAxes)
ax2.text(-0.15,1.0, 'b', fontweight='bold', transform=ax2.transAxes)
    
