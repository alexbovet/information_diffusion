# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

import graph_tool.all as gt


save_dir = '../data/urls/revisions/'

raise Exception



#%% load results

#
res = pd.read_pickle(os.path.join(save_dir,
     'tigramite_res/CE_max_rcot2_june-nov_pc_alpha0.1_tau_max18_q00.05_maxcomb3.pickle'))


suffix = ''
df_CEmax = res['df_CEmax']
df_CEmax_mean = res['df_CEmax_mean']
df_CEmax_std = res['df_CEmax_std']
df_CEmax_tau = res['df_CEmax_tau']
parameters = res['parameters']

#%%

G = gt.Graph()


# i -> j
edge_list = []
for j in df_CEmax.columns:
    for i in df_CEmax.index:
        if not np.isnan(df_CEmax.loc[j,i]):
            edge_list.append((i,j, df_CEmax.loc[j,i], df_CEmax_mean.loc[j,i], 
                                   df_CEmax_std.loc[j,i], df_CEmax_tau.loc[j,i]))

G.ep['CEmax'] = G.new_edge_property('float')
G.ep['CEmax_mean'] = G.new_edge_property('float')
G.ep['CEmax_std'] = G.new_edge_property('float')
G.ep['CEmax_tau'] = G.new_edge_property('float')

G.vp['var_names'] = G.add_edge_list(edge_list, hashed=True,
                        eprops=[G.ep['CEmax'], G.ep['CEmax_mean'],
                                G.ep['CEmax_std'], G.ep['CEmax_tau']])

name2v_id = dict((G.vp['var_names'][v], G.vertex_index[v]) for v in  G.vertices())

gt.remove_self_loops(G)
#%% prepare graph plot
pos = G.new_vertex_property('vector<double>')

supps = ['pro_t','pro_h']

infls = ['infl_fake' ,
         'infl_far_right',
         'infl_right',
         'infl_lean_right',
         'infl_center',
         'infl_lean_left',
         'infl_left',
         'infl_far_left',]

num_infls = len(infls)
infl_y_pos = np.linspace(-4, 4, len(infls))
infl_x_pos = -2

supp_y_pos = np.linspace(-2, 2, 2)
supp_x_pos = 2




# set positions
for name, ypos in zip(supps, supp_y_pos):
    pos[name2v_id[name]] = [supp_x_pos, ypos]

for name, ypos in zip(infls, infl_y_pos):
    pos[name2v_id[name]] = [infl_x_pos, ypos]
    
x, y = pos[name2v_id['infl_right']]    
pos[name2v_id['infl_right']] = x + 1, y - 0.5
    
if 'infl_far_right' in name2v_id:
    x, y = pos[name2v_id['infl_far_right']]    
    pos[name2v_id['infl_far_right']] = x - 1, y 

    
x, y = pos[name2v_id['infl_lean_left']]    
pos[name2v_id['infl_lean_left']] = x + 1, y+0.5
    
x, y = pos[name2v_id['infl_left']]    
pos[name2v_id['infl_left']] = x - 1, y

# set node colors
from PlotUtils import mediacolors, trump_color, clinton_color, darker


vfillcolor = G.new_vertex_property('vector<double>')

mediacolors_bb = mediacolors[:]
mediacolors_bb.insert(1, mediacolors[1])
mediacolors_bb.insert(-1, mediacolors[-1])

for name, color in zip(infls,
        mediacolors_bb):
    vfillcolor[name2v_id[name]] = list(darker(color,0.2)) + [1.0]
#    
vfillcolor[name2v_id['pro_t']] = trump_color + [1.0]
vfillcolor[name2v_id['pro_h']] = clinton_color + [1.0]
        

infl_ids = [name2v_id[name] for name in infls]
supp_ids = [name2v_id[name] for name in supps]


edge_color = G.new_edge_property('vector<double>')
alpha = 0.8
for edge in G.edges():
    
    ecolor = vfillcolor[edge.source()].copy()
    ecolor[3] = alpha
    edge_color[edge] = ecolor 

# rescale edge width    
max_width = 50
min_width = 1
ewidth = G.new_edge_property('double')
ewidth.a = (G.ep.CEmax.a - G.ep.CEmax.a.min())/max(G.ep.CEmax.a - G.ep.CEmax.a.min())*(max_width-min_width)+\
        min_width

vsize = G.new_vertex_property('float')

# use autolinks for node size
for name in df_CEmax.columns:
    vsize[name2v_id[name]] = df_CEmax.loc[name,name]

max_vsize = 60
min_vsize = 20

vsize.a = (vsize.a - vsize.a.min())/(max((vsize.a - vsize.a.min())))*(max_vsize-min_vsize) +\
          min_vsize

#%% plot graph


def thresh_edges(e, thresh=0.05):
    return G.ep.CEmax[e] > thresh


thresh = 0.05

Gfilt = gt.GraphView(G, efilt=lambda e:thresh_edges(e, thresh))


rename_dict = {'infl_fake':'fake news',
             'infl_fake_not_aggr' : 'fake (no aggregators)',
             'infl_far_right':'far-right',
             'infl_far_right_not_aggr':'far-right (no aggregators)',
             'infl_breitbart' : 'breitbart',
             'infl_not_breitbart' : 'far-right except BB',
             'infl_right':'right',
             'infl_lean_right' : 'right leaning',
             'infl_lean_right_not_aggr':'right leaning (no aggregators)',
             'infl_center' : 'center',
             'infl_lean_left': 'left leaning',
             'infl_left' : 'left',
             'infl_far_left' : 'far-left',
             'infl_shareblue' : 'SB + BNR',
             'infl_not_shareblue' : 'far-left except SB + BNR',
             'infl_fake_no_staff':'fake news',
             'infl_far_right_no_staff':'far-right',
             'infl_right_no_staff':'right',
             'infl_lean_right_no_staff' : 'right leaning',
             'infl_center_no_staff' : 'center',
             'infl_lean_left_no_staff': 'left leaning',
             'infl_left_no_staff' : 'left',
             'infl_far_left_no_staff' : 'far-left',
             'pro_t' : 'pro-Trump',
             'pro_h' : 'pro-Clinton',
             'tot' : 'total activity'}          
          
names = Gfilt.new_vertex_property('string')
for v in Gfilt.vertices():
    names[v] = rename_dict[Gfilt.vp.var_names[v]]


graph_filename = 'figures/paper_figures/revisions/CE_graph_{condtest}_q0{q_0}_thresh{thresh}_max_comb{max_comb}_taumax{tau_max}_pc_alpha{pc_alpha}{suffix}.svg'.\
                      format(q_0=parameters['q_0'], thresh=thresh,
                             condtest=parameters['cond_ind_test'],
                             max_comb=parameters['max_combinations'],
                             tau_max=parameters['tau_max'],
                             pc_alpha=parameters['pc_alpha'],
                             suffix=suffix)
print(graph_filename)                   
if os.path.exists(graph_filename):
    graph_filename=None
    print('file exists, not saving')

gt.graph_draw(Gfilt, pos=pos,edge_pen_width=ewidth,
              vertex_halo=False,
              vertex_size=vsize,
              bg_color=[1,1,1,1], vertex_pen_width=0, 
              vertex_fill_color=vfillcolor,
              vertex_text=names,
              vertex_text_position=0,
              vertex_text_color=[0,0,0,1],
#              edge_marker_size=ewidth,
              edge_color=edge_color,
              vertex_font_family='Helvetica Neue LT Std',
              output=graph_filename,
              )


