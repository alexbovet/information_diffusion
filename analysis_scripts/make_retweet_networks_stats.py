# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt


plt.style.use('alex_paper')

media_types = ['fake', 'breitbart',
             'far_right', 'right',
             'lean_right', 
            'center', 'lean_left', 'left',
            'far_left', 'shareblue', 'not_shareblue']

periods = ['june-nov']

save_dir =  '../data/urls/revisions/'


raise Exception



#%% =============================================================================
# Tables for paper
# =============================================================================
#%% compute global statistics 

period = 'june-nov'

Graphs_simple = dict()
Graphs_complete = dict()
for media_type in media_types:
    Graphs_simple[media_type] = dict()
    Graphs_simple[media_type][period] = gt.load_graph(os.path.join(save_dir, 
                 'retweet_graph_' + \
                 media_type + '_simple_' + period + '.gt'))
    
    Graphs_complete[media_type] = dict()
    Graphs_complete[media_type][period] = gt.load_graph(os.path.join(save_dir, 
                 'retweet_graph_' + \
                 media_type + '_complete_' + period + '.gt'))
    
    
Glist_simple = [Graphs_simple[key][period] for key in media_types]
Glist_complete = [Graphs_complete[key][period] for key in media_types]    
#%%    
    
df_stats_simple = pd.DataFrame(index = [G.gp['name'] for G in Glist_simple])

df_stats_complete = pd.DataFrame(index = [G.gp['name'] for G in Glist_complete])




df_stats_simple['{$N$ nodes}'] = [G.num_vertices() for G in Glist_simple]
df_stats_simple['{$N$ edges}'] = [G.num_edges() for G in Glist_simple]
df_stats_simple['{$<k>$}'] = [float(G.vp['k_out'].a.mean()) for G in Glist_simple]
df_stats_simple['{$\max(k_{\textrm{out}})$}'] = [int(G.vp['k_out'].a.max()) for G in Glist_simple]
df_stats_simple['{$\max(k_{\textrm{in}})$}'] = [int(G.vp['k_in'].a.max()) for G in Glist_simple]

df_stats_complete['{$N$ nodes}'] = [G.num_vertices() for G in Glist_complete]
df_stats_complete['{$N$ edges}'] = [G.num_edges() for G in Glist_complete]
df_stats_complete['{$<k>$}'] = [float(G.vp['k_out'].a.mean()) for G in Glist_complete]
df_stats_complete['{$\max(k_{\textrm{out}})$}'] = [int(G.vp['k_out'].a.max()) for G in Glist_complete]
df_stats_complete['{$\max(k_{\textrm{in}})$}'] = [int(G.vp['k_in'].a.max()) for G in Glist_complete]



#%% check size effect with bootstraping

def compute_sample_sigma(degree_sequence, sample_size=200000, num_bootstrap=10,
                         replace=True):
    
    sig_over_avg = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(degree_sequence,size=sample_size,
                                             replace=replace)
        sig_over_avg.append(sample.std()/sample.mean())

    return np.mean(sig_over_avg), np.std(sig_over_avg)

def compute_avg(degree_sequence, sample_size=200000, num_bootstrap=10,
                         replace=True):
    
    avg = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(degree_sequence,size=sample_size,
                                             replace=replace)
        avg.append(sample.mean())

    return np.mean(avg), np.std(avg)

def round_uncert(a, b):
    dec = 0
    while b < 10**-dec:
        dec +=1
        if dec >= 4:
            break
    ret_string = '{a:.' + str(dec) +  'f} \pm {b:.' + str(dec) +  'f}'
    return ret_string.format(a=a,b=b)

#%% for simple
avg_avgs = []
std_avgs = []
for gname in df_stats_simple.index:
    avg_avg, std_avg = compute_avg(Graphs_simple[gname][period].vp.k_out.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_avgs.append(avg_avg)
    std_avgs.append(std_avg)    
avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_avgs, std_avgs)]



avg_soas = []
std_soas = []
for gname in df_stats_simple.index:
    avg_soa, std_soa = compute_sample_sigma(Graphs_simple[gname][period].vp.k_out.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_soas.append(avg_soa)
    std_soas.append(std_soa)
    
avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_soas, std_soas)]
df_stats_simple['{$\sigma(k_{\textrm{out}})/<k>$}'] = avg_pm_std


avg_soas = []
std_soas = []
for gname in df_stats_simple.index:
    avg_soa, std_soa = compute_sample_sigma(Graphs_simple[gname][period].vp.k_in.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_soas.append(avg_soa)
    std_soas.append(std_soa)

avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_soas, std_soas)]
    
df_stats_simple['{$\sigma(k_{\textrm{in}})/<k>$}'] = avg_pm_std


#%% save

df_stats_simple.to_pickle(os.path.join(save_dir, 'df_stats_retweet_networks_simple_' + period + '.pickle'))

#df_stats_simple = pd.read_pickle('../data/urls/df_stats_retweet_networks_simple_june-nov.pickle')

#%% print table


rename_dict = {'far_right' : '{far-right}',
               'breitbart' : '{breitbart}',
               'not_breitbart' : '{far-right \textbackslash breitbart}',
               'far_left' : '{far-left}',
               'shareblue': '{SB+BNR}',
               'not_shareblue' : '{far-left \textbackslash (SB+BNR)}',
               'fake' : '{fake news}',
               'right' : '{right}',
               'lean_right' : '{right leaning}',
               'center' : '{center}',
               'lean_left' : '{left leaning}',
               'left' : '{left}',}

df = df_stats_simple[['{$N$ nodes}','{$N$ edges}','{$<k>$}','{$\sigma(k_{\textrm{out}})/<k>$}',
                       '{$\sigma(k_{\textrm{in}})/<k>$}',
                       '{$\max(k_{\textrm{out}})$}',
                       '{$\max(k_{\textrm{in}})$}']].rename(rename_dict, axis=0)
    
print(df.to_latex(index=True, 
                        float_format=lambda f: "{0:.2f}".format(f),
      escape=False))

#%% for complete

avg_avgs = []
std_avgs = []
for gname in df_stats_complete.index:
    avg_avg, std_avg = compute_avg(Graphs_complete[gname][period].vp.k_out.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_avgs.append(avg_avg)
    std_avgs.append(std_avg)    
avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_avgs, std_avgs)]



avg_soas = []
std_soas = []
for gname in df_stats_complete.index:
    avg_soa, std_soa = compute_sample_sigma(Graphs_complete[gname][period].vp.k_out.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_soas.append(avg_soa)
    std_soas.append(std_soa)
    
avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_soas, std_soas)]
df_stats_complete['{$\sigma(k_{\textrm{out}})/<k>$}'] = avg_pm_std

avg_soas = []
std_soas = []
for gname in df_stats_complete.index:
    avg_soa, std_soa = compute_sample_sigma(Graphs_complete[gname][period].vp.k_in.a,
                                            sample_size=78911,
                                            num_bootstrap=1000,
                                            replace=True)
    avg_soas.append(avg_soa)
    std_soas.append(std_soa)

avg_pm_std = [round_uncert(a,s) for a,s in zip(avg_soas, std_soas)]
    
df_stats_complete['{$\sigma(k_{\textrm{in}})/<k>$}'] = avg_pm_std

#%%
df_stats_complete.to_pickle(os.path.join(save_dir, 'df_stats_retweet_networks_complete_' + period + '.pickle'))
#%%

rename_dict = {'far_right' : '{far-right}',
               'breitbart' : '{breitbart}',
               'not_breitbart' : '{far-right \textbackslash breitbart}',
               'far_left' : '{far-left}',
               'shareblue': '{SB+BNR}',
               'not_shareblue' : '{far-left \textbackslash (SB+BNR)}',
               'fake' : '{fake news}',
               'right' : '{right}',
               'lean_right' : '{right leaning}',
               'center' : '{center}',
               'lean_left' : '{left leaning}',
               'left' : '{left}',}

df = df_stats_complete[['{$N$ nodes}','{$N$ edges}','{$<k>$}','{$\sigma(k_{\textrm{out}})/<k>$}',
                       '{$\sigma(k_{\textrm{in}})/<k>$}',
                       '{$\max(k_{\textrm{out}})$}',
                       '{$\max(k_{\textrm{in}})$}']].rename(rename_dict, axis=0)
    
print(df.to_latex(index=True, 
                        float_format=lambda f: "{0:.2f}".format(f),
      escape=False))
