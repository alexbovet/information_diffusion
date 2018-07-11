# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), 
                                                           os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta, datetime
import pickle
import graph_tool.all as gt


plt.style.use('alex_paper')



raise Exception

#%% load user and tweet list

save_dir = '../data/urls/revisions/host_counts/'


resample_freq = '15min'




#%% read group counts

#which counts to use
count_type = 'df_tid_count'

dfs_dict = dict()

for host_type in ['fake', 'fake_not_aggr', 'far_right', 'far_right_not_aggr', 
                  'breitbart', 'not_breitbart',
                  'right', 'lean_right', 'lean_right_not_aggr', 'center', 'lean_left', 'left', 
                  'far_left', 'shareblue', 'not_shareblue']:
    with open(save_dir + 'df_' + host_type + '_counts_5min.pickle', 'rb') as fopen:
        dfs_dict[host_type] = pickle.load(fopen)
        
#% dataframe with tid count without retweeted statuses
        
df_tid_no_ret = pd.DataFrame()

for media_type in dfs_dict.keys():

    df_tid_no_ret[media_type] = dfs_dict[media_type]['df_tid_count_no_retweet']

#% resample
df_tid_no_ret = df_tid_no_ret.resample(resample_freq).sum()

#%% load supporter activity
    
sup_jun_sep = pd.read_pickle("../data/complete_trump_vs_hillary/class_proba/stats_corrected_official_client_thresh0.5_rthresh0.5_june1_sep1_signi_final_2_5min_res.pickle")

sup_sep_nov = pd.read_pickle("../data/complete_trump_vs_hillary_sep-nov/class_proba/stats_corrected_official_client_thresh0.5_rthresh0.5_sep1_nov9_signi_final_5min_res.pickle")
    
df_num_tweets = pd.concat([sup_jun_sep['df_num_tweets'],
                          sup_sep_nov['df_num_tweets']])

df_num_tweets.index = pd.to_datetime(df_num_tweets.index)
    
df_tot_tweets = df_num_tweets.n_pro_h + df_num_tweets.n_pro_t

df_tot_tweets.index = pd.to_datetime(df_tot_tweets.index)

#% resample

df_num_tweets = df_num_tweets.resample(resample_freq).sum()
df_tot_tweets = df_tot_tweets.resample(resample_freq).sum()


    

start_date = datetime(2016,6,1)
stop_date = datetime(2016,11,9)
df_tid_no_ret.drop(df_tid_no_ret.loc[np.logical_or(df_tid_no_ret.index < start_date, 
                             df_tid_no_ret.index >= stop_date)].index,
                            inplace=True)
    
#raise Exception    
#%% 

#prepare data


df = df_tid_no_ret.copy()
df['tot'] = df_tot_tweets
df['pro_h'] = df_num_tweets.n_pro_h
df['pro_t'] = df_num_tweets.n_pro_t

# replace null values by nan
for col in df.columns:
    df[col].loc[df['tot'].isna()] = np.nan


# deal with missing values:
all_time_index = df.index
missing_ts = df.loc[df['tot'].isna()].index


def get_day_start_end(ts, cuthour=4, cutmin=0, cutsec=0):
    # get 4am before:
    if ts.hour >= 4:
        start = ts - timedelta(hours=ts.hour-cuthour,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
        end = ts - timedelta(hours=ts.hour-cuthour-24,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
    else:
        start = ts - timedelta(days=1,
                               hours=ts.hour-cuthour,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
        end = ts - timedelta(days=1,
                               hours=ts.hour-cuthour-24,
                               minutes=ts.minute-cutmin,
                               seconds=ts.second-cutsec)
    return (start, end)

# mask with values to remove
ts_mask = np.zeros_like(all_time_index, dtype=bool)
for ts in missing_ts:
    start, end = get_day_start_end(ts)
    ts_mask = np.logical_or(ts_mask,
                            np.logical_and(
                                    all_time_index >= start,
                                    all_time_index < end)
                            )
                            
df.loc[ts_mask] = np.nan



#%% remove nan values:

dfdrop = df.dropna()



#%% LOESS smoothing

from pyloess import stl

# second possibility, remove seasonality and trend

multiplier = {'15min': 4,
              'H' : 1,
              '5min' : 12 }
n_p = multiplier[resample_freq]*24
n_s = n_p - 1

stl_params = dict(np = n_p, # period of season
    ns = n_s, # seasonal smoothing
    nt = None, # trend smooting int((1.5*np)/(1-(1.5/ns)))
    nl = None, # low-pass filter leat odd integer >= np
    isdeg=1,
    itdeg=1,
    ildeg=1,
    robust=True,
    ni = 1,
    no = 5)



stl_res_dict = {}
for col in dfdrop:

    stl_res_dict[col] = stl(dfdrop[col].values,
                **stl_params)


#%% compute correlations

selected_cols = ['fake','far_right','right', 'lean_right',
                 'center','lean_left','left', 'far_left', 
                 'pro_t', 'pro_h']
    
df_residuals = pd.DataFrame(data=dict((key, stl_res_dict[key].residuals) for key in stl_res_dict.keys()),
                            columns=selected_cols)


    
df_residuals.rename({'fake':'fake news',
                     'fake_not_aggr' : 'fake (no aggregators)',
                     'far_right':'far-right',
                     'far_right_not_aggr':'far-right (no aggregators)',
                     'breitbart' : 'breitbart',
                     'not_breitbart' : 'far-right except BB',
                     'right':'right',
                     'lean_right' : 'right leaning',
                     'lean_right_not_aggr':'right leaning (no aggregators)',
                     'center' : 'center',
                     'lean_left': 'left leaning',
                     'left' : 'left',
                     'far_left' : 'far-left',
                     'shareblue' : 'SB + BNR',
                     'not_shareblue' : 'far-left except SB + BNR',
                     'pro_t' : 'pro-Trump',
                     'pro_h' : 'pro-Clinton',
                     'tot' : 'total activity'},
    axis='columns',
    inplace=True)
  
method = 'pearson'    


cols_to_plot = ['fake news','far-right','right','pro-Trump', 
                 'center','left leaning','left', 
                 'pro-Clinton']


#corr_matrix = dfdrop.corr(method=method)
corr_matrix = df_residuals[cols_to_plot].corr(method=method)


fig, ax = plt.subplots(1,1, figsize=(8,6))

plt.pcolor(corr_matrix)
plt.colorbar()

ax = plt.gca()

ax.set_xticks(np.arange(corr_matrix.shape[0])+0.5)
ax.set_yticks(np.arange(corr_matrix.shape[0])+0.5)

ax.set_xticklabels(corr_matrix.columns, rotation=80)
ax.set_yticklabels(corr_matrix.columns)

ax.set_aspect('equal')


#to latex
rename = {col: '{' + col + '}' for col in corr_matrix.columns}
corr_matrix = corr_matrix.rename(rename, axis=0)
corr_matrix = corr_matrix.rename(rename, axis=1)

print(corr_matrix.to_latex(float_format=(lambda x: '{:.2f}'.format(x)),
    escape=False))

#%% make graph

threshold = 0.49

edge_list = []
for i in corr_matrix.index:
        for j in corr_matrix.columns:
            if corr_matrix.loc[i,j] > threshold and i != j:
                edge_list.append((j,i,corr_matrix.loc[i,j]))

                 
G = gt.Graph(directed=False)

weights = G.new_edge_property('float')
names = G.add_edge_list(edge_list, hashed=True, string_vals=True, eprops=[weights])
gt.remove_parallel_edges(G)

G.vp['names'] = names
G.ep['weights'] = weights


name2v_id = dict((G.vp.names[v], G.vertex_index[v]) for v in  G.vertices())                    

#%%

pos = G.new_vertex_property('vector<float>')

pos[name2v_id['left leaning']] = [105,179-102]
pos[name2v_id['left']] = [56,162-102]
pos[name2v_id['center']] = [99,128-102]

pos[name2v_id['right']] = [pos[name2v_id['center']][0],
                                           -pos[name2v_id['center']][1]]
pos[name2v_id['fake news']] = [pos[name2v_id['left']][0],
                                           -pos[name2v_id['left']][1]]
pos[name2v_id['far-right']] = [pos[name2v_id['left leaning']][0],
                                           -pos[name2v_id['left leaning']][1]]

pos[name2v_id['pro-Clinton']] = [192,146-102]

pos[name2v_id['pro-Trump']] = [192,82-102]

colors = G.new_vertex_property('vector<float>')


from palettable.cartocolors.diverging import Geyser_2 as colormap
from palettable.palette import Palette
clinton_color = [c/255 for c in [38, 17, 127]]
trump_color = [c/255 for c in [217, 189, 33]]


colors[name2v_id['left leaning']] = colormap.mpl_colors[0]
colors[name2v_id['left']] = colormap.mpl_colors[0]
colors[name2v_id['center']] = colormap.mpl_colors[0]

colors[name2v_id['right']] = colormap.mpl_colors[1]
colors[name2v_id['fake news']] = colormap.mpl_colors[1]
colors[name2v_id['far-right']] = colormap.mpl_colors[1]

colors[name2v_id['pro-Clinton']] = clinton_color
colors[name2v_id['pro-Trump']] = trump_color

edge_widths = G.new_edge_property('float')

max_width = 40
min_width = 8
edge_widths.a = (G.ep.weights.a - G.ep.weights.a.min())/G.ep.weights.a.max()*max_width+min_width

gradient = G.new_edge_property('vector<float>')

alpha = 0.9
for e in G.edges():
    gradient[e] = np.concatenate(([0],colors[e.source()][0:3],[alpha],
                                  [1],colors[e.target()][0:3],[alpha]))
#%% draw

gt.graph_draw(G, pos=pos,
              bg_color=[1,1,1,1],
              vertex_fill_color=colors,
              vertex_pen_width=0,
              vertex_size=80,
              edge_pen_width=edge_widths,
              edge_gradient=gradient,
              inline=True)
