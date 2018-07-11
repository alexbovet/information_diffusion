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




import pickle



plt.style.use('alex_paper')

media_types = ['fake', 'far_right', 'right', 'left', 
               'far_left', 
               'lean_left', 'center', 'lean_right']
periods = ['june-nov']

save_dir =  '../data/urls/revisions/'





#%% =============================================================================
# Tables for paper
# =============================================================================

#%% influencer table
#%% load results        
with open(os.path.join(save_dir, 'influencer_rankings_simple.pickle'), 'rb') as fopen:
    
    ranks_dict_simple = pickle.load(fopen)

    
#%%
period = 'june-nov'
ranker = 'CI_out'
top_num = 25
#ranks_dict_simple[media][period][ranker].head(top_num)

df_infls = pd.DataFrame(columns = media_types,
                        index = range(1,top_num+1))

for media in df_infls.columns:
    df_infls[media] = ranks_dict_simple[media][period][ranker].head(top_num)['screenname'].values
    for i, verif in ranks_dict_simple[media][period][ranker].head(top_num)['is_verified'].items():
        if verif:
            df_infls[media].iloc[i] = '\textcolor{ForestGreen}{' + df_infls[media].iloc[i] + '}\,\checkmark'
        else:
            df_infls[media].iloc[i] = '\textcolor{BrickRed}{' + df_infls[media].iloc[i] + '}'

#%%            
columns_to_print = ['fake','far_right','right', 'lean_right']
columns_to_print = ['center','lean_left', 'left', 'far_left']

columns_to_print = ['breitbart','not_breitbart', 'shareblue', 'not_shareblue']

columns_to_print = ['fake_not_aggr','far_right_not_aggr', 'lean_right_not_aggr']

pd.set_option('display.max_colwidth', 1000)
print(df_infls[columns_to_print].to_latex(index=True, escape=False))

#%% make bar plot of verif/unverif/deleted accounts
period = 'june-nov'
ranker = 'CI_out'
top_num = 100
#ranks_dict_simple[media][period][ranker].head(top_num)

line_plot = False

bar_plots = dict()

for media in ranks_dict_simple.keys():
    bar_plots[media] = dict()
    bar_plots[media]['num verified'] = ranks_dict_simple[media][period][ranker].is_verified[:top_num].sum()
    bar_plots[media]['num deleted'] = (ranks_dict_simple[media][period][ranker]['screenname'] == '@???????')[:top_num].sum()
    bar_plots[media]['num unverified'] = top_num - bar_plots[media]['num verified'] - bar_plots[media]['num deleted']
    

#from palettable.colorbrewer.qualitative import Dark2_3

#green = '#005911'
green = '#1b9e77'
#bluish_green = [c/255 for c in (0,158,115)]
red = '#a6001a'
#orange = [c/255 for c in (230,159,0)]
orange = '#d95f02'
black = '#000000'

alpha=0.5 if line_plot else 1.0

x_verif = []
h_verif = []
x_unverif = []
h_unverif = []
x_del = []
h_del = []

heights = []
colors = []
labels = []

media_to_plot = ['fake','far_right','right', 'lean_right',
                 'center','lean_left', 'left', 'far_left']

for i,media in enumerate(media_to_plot):
    heights.extend((bar_plots[media]['num verified'],
                    bar_plots[media]['num unverified'],
                    bar_plots[media]['num deleted']))
    colors.extend((green,red,black))
    
    x_verif.append(i+0.75)
    h_verif.append(bar_plots[media]['num verified'])

    x_unverif.append(i+1)
    h_unverif.append(bar_plots[media]['num unverified'])

    x_del.append(i+1.25)
    h_del.append(bar_plots[media]['num deleted'])
    
width = 0.25
fig, ax = plt.subplots(1,1, figsize=(6.5,5))    

bv = ax.bar(x_verif, h_verif, width, label=None if line_plot else 'verified',color=green,linewidth=0,
            alpha=alpha)
bu = ax.bar(x_unverif, h_unverif, width, label=None if line_plot else 'unverified',color=orange,linewidth=0,
            alpha=alpha)
bd = ax.bar(x_del, h_del, width, label=None if line_plot else 'deleted',color=black,linewidth=0,
            alpha=alpha)

if not line_plot:
    ax.legend()

ax.set_ylabel('Number of accounts')
ax.set_xticks(range(1,len(media_to_plot)+1))


ax.set_xticklabels(['fake\nnews', 'extreme bias\n(right)', 
                    'right', 'right\nleaning',
                   'center', 'left\nleaning','left',
                   'extreme bias\n(left)'],rotation=50, fontsize=16)

    

    
