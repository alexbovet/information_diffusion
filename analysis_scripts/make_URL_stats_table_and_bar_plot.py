# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import time

urls_db_file = '../databases_ssd/urls_db.sqlite'

plt.style.use('alex_paper')


official_twitter_clients = ['Twitter for iPhone',
'Twitter for Android',
'Twitter Web Client',
'Twitter for iPad',
'Mobile Web (M5)',
'TweetDeck',
'Facebook',
'Twitter for Windows',
'Mobile Web (M2)',
'Twitter for Windows Phone',
'Mobile Web',
'Google',
'Twitter for BlackBerry',
'Twitter for Android Tablets',
'Twitter for Mac',
'iOS',
'Twitter for BlackBerry®',
'OS X']


save_dir = '../data/urls/revisions/'

os.makedirs(save_dir, exist_ok=True)

#raise Exception
#%%


    # count total number of tweets pointing outside of twitter



with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

    
    c = conn.cursor()
    c.execute("""SELECT COUNT(DISTINCT tweet_id) FROM urls
                         WHERE final_hostname  != 'amp.twimg.com' 
                         AND final_hostname != 'twitter.com'""")
    tot_num_tweets = c.fetchall()
    print(tot_num_tweets)
    
    c.execute("""SELECT COUNT(DISTINCT user_id) FROM urls
                         WHERE final_hostname  != 'amp.twimg.com' 
                         AND final_hostname != 'twitter.com'""")
    tot_num_users = c.fetchall()
    print(tot_num_users)
    


#%% count total number of tweets pointing to selected newswebsites
    
sql_condition_dict = {}
                                     
for media_type in ['fake', 'far_right', 'right', 'lean_right', 'center', 'lean_left', 'left',
            'far_left']:
    sql_condition_dict[media_type] = """SELECT hostname FROM hosts_{med_type}_rev_stat
                                        WHERE perccum > 0.01""".format(med_type=media_type)
                                        
                                                                                
                                        
                                        
os.environ['SQLITE_TMPDIR'] = '/home/alex/'
with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

    
    c = conn.cursor()
    # count total number of tweets pointing outside of twitter
    c.execute("""SELECT COUNT(DISTINCT tweet_id) FROM urls
                         WHERE final_hostname IN
                         (""" + '\nUNION\n'.join(sql_condition_dict.values()) + ")")
    
    news_num_tweets = c.fetchall()
    print(news_num_tweets)
    
    c.execute("""SELECT COUNT(DISTINCT user_id) FROM urls
                         WHERE final_hostname IN
                         (""" + '\nUNION\n'.join(sql_condition_dict.values()) + ")")
    news_num_users = c.fetchall()
    print(news_num_users)


#%% see proportion of treated URLs

with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

    df_host_counts = pd.read_sql_query("""SELECT COUNT(*) , final_hostname FROM urls
                                        WHERE final_hostname  != 'amp.twimg.com' 
                         AND final_hostname != 'twitter.com'
    GROUP BY final_hostname
    ORDER BY COUNT(*) 	 DESC""", conn)
    
    
#%% make data for num users/num tweets table
        
UPDATE_RES = True

    
sql_count_tweet_ids =  """SELECT COUNT(DISTINCT tweet_id) FROM urls
                            WHERE final_hostname IN """

sql_user_ids =  """SELECT DISTINCT user_id FROM urls
                            WHERE final_hostname IN """
                                                        

sql_condition_dict = {}
                                     
for media_type in ['fake', 'far_right', 'right', 'lean_right', 'center', 'lean_left', 'left',
            'far_left']:
    sql_condition_dict[media_type] = """SELECT hostname FROM hosts_{med_type}_rev_stat
                                        WHERE perccum > 0.01""".format(med_type=media_type)
                                        
sql_condition_dict['fake_0p05perccum'] = """SELECT hostname FROM hosts_fake_all_rev_stat
                                        WHERE perccum > 0.005"""
                                        
sql_condition_dict['breitbart'] = """SELECT hostname FROM hosts_far_right_rev_stat
                                     WHERE hostname = 'breitbart.com'"""                                        
                                        
sql_condition_dict['not_breitbart'] = """SELECT hostname FROM hosts_far_right_rev_stat
                                         WHERE perccum > 0.01
                                         AND hostname IS NOT 'breitbart.com'"""
                                         
sql_condition_dict['shareblue'] = """SELECT hostname FROM hosts_far_left_rev_stat
                                     WHERE hostname = 'bluenationreview.com'
                                     OR hostname = 'shareblue.com'"""                                        
                                        
sql_condition_dict['not_shareblue'] = """SELECT hostname FROM hosts_far_left_rev_stat
                                         WHERE perccum > 0.01
                                         AND hostname IS NOT 'bluenationreview.com'
                                         AND hostname IS NOT 'shareblue.com'"""
                                         
sql_condition_dict['fake_far_right'] = """SELECT hostname FROM hosts_fake_rev_stat
                                         WHERE perccum > 0.01
                                         UNION
                                         SELECT hostname FROM hosts_far_right_rev_stat
                                         WHERE perccum > 0.01"""                    

sql_condition_dict['fake_far_right_far_left'] = """SELECT hostname FROM hosts_fake_rev_stat
                                         WHERE perccum > 0.01
                                         UNION
                                         SELECT hostname FROM hosts_far_right_rev_stat
                                         WHERE perccum > 0.01
                                         UNION
                                         SELECT hostname FROM hosts_far_left_rev_stat
                                         WHERE perccum > 0.01"""    
                                         
sql_condition_dict['right_lean_right'] = """SELECT hostname FROM hosts_lean_right_rev_stat
                                         WHERE perccum > 0.01
                                         UNION
                                         SELECT hostname FROM hosts_right_rev_stat
                                         WHERE perccum > 0.01"""                                         

sql_condition_dict['left_lean_left'] = """SELECT hostname FROM hosts_lean_left_rev_stat
                                         WHERE perccum > 0.01
                                         UNION
                                         SELECT hostname FROM hosts_left_rev_stat
                                         WHERE perccum > 0.01"""
                                         
media_list_to_read = sql_condition_dict.items()

#media_list_to_read = ['lean_left', 'center', 'lean_right']
#media_list_to_read = ['breitbart', 'not_breitbart', 'shareblue', 'not_shareblue']
#media_list_to_read = ['fake_far_right', 'right_lean_right', 'left_lean_left']
media_list_to_read = ['fake_far_right_far_left']
                                         
with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

    
    c = conn.cursor()
    
    if UPDATE_RES:
        
        tweet_counts = pd.read_pickle(os.path.join(save_dir, 'tweet_counts_per_cat.pickle'))

    else:
        tweet_counts = dict()
        
    user_lists = dict()
    
    t0 = time.time()
    for key in media_list_to_read:
    
        
        print(key)
        print(time.time() - t0)
        
        c.execute(sql_count_tweet_ids + "(" + sql_condition_dict[key] + ")")
        
        tweet_counts[key] = c.fetchall()[0][0]
        
        c.execute(sql_user_ids + "(" + sql_condition_dict[key] + ")")
        
        user_lists[key] = c.fetchall()
        

    
#%%
#for key in tweet_counts.keys():
#    tweet_counts[key] = tweet_counts[key]
    
#pd.to_pickle(tweet_counts, os.path.join(save_dir, 'tweet_counts_per_cat.pickle'))
    
if UPDATE_RES:
    user_counts = pd.read_pickle(os.path.join(save_dir, 'user_counts_per_cat.pickle'))
else:
    user_counts = dict()
    
for key in user_lists.keys():
    user_counts[key] = len(user_lists[key])

pd.to_pickle(user_counts, os.path.join(save_dir, 'user_counts_per_cat.pickle'))    
   

if UPDATE_RES:
    user_sets = pd.read_pickle(os.path.join(save_dir, 'user_sets_per_cat.pickle'))    
else:
    user_sets = dict()
    
for key, val in user_lists.items():
    user_sets[key] = set(u for u, in val)
    
pd.to_pickle(user_sets, os.path.join(save_dir, 'user_sets_per_cat.pickle'))    
    
#%%

UPDATE_RES = True
#media_list_to_read = ['breitbart', 'not_breitbart', 'shareblue', 'not_shareblue']
#media_list_to_read = ['fake_far_right', 'right_lean_right', 'left_lean_left']
media_list_to_read = ['fake_far_right_far_left']


os.environ['SQLITE_TMPDIR'] = '/home/alex'
    
#same for official client        
with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:

    
    c = conn.cursor()
    
    if UPDATE_RES:
        tweet_counts_off = pd.read_pickle(os.path.join(save_dir, 'tweet_counts_off_per_cat.pickle'))
    else:    
        tweet_counts_off = dict()
    user_lists_off = dict()
    
    
    for key in media_list_to_read:
        
        print(key)
        
        c.execute(sql_count_tweet_ids + "(" + sql_condition_dict[key] + """)
                    AND
                    source_content IN ('{source_seq}')"""\
                    .format(source_seq="','".join(official_twitter_clients)))
        
        tweet_counts_off[key] = c.fetchall()[0][0]
        
        c.execute(sql_user_ids + "(" + sql_condition_dict[key] + """)
                    AND
                    source_content IN ('{source_seq}')"""\
                    .format(source_seq="','".join(official_twitter_clients)))
        
        user_lists_off[key] = [u for u, in c.fetchall()]
        
#%%
#for key in tweet_counts_off.keys():
#    tweet_counts_off[key] = tweet_counts_off[key][0][0]
    
#pd.to_pickle(tweet_counts, os.path.join(save_dir, 'tweet_counts_per_cat.pickle'))
#pd.to_pickle(tweet_counts_off, os.path.join(save_dir, 'tweet_counts_off_per_cat.pickle'))    
    
if UPDATE_RES:
    user_counts_off = pd.read_pickle(os.path.join(save_dir, 'user_counts_off_per_cat.pickle'))
else:    
    user_counts_off = dict()
    
for key in user_lists_off.keys():
    user_counts_off[key] = len(user_lists_off[key])

pd.to_pickle(user_counts_off, os.path.join(save_dir, 'user_counts_off_per_cat.pickle'))    
   
if UPDATE_RES:
    user_sets_off = pd.read_pickle(os.path.join(save_dir, 'user_sets_off_per_cat.pickle'))
else:    
    user_sets_off = dict()
    
for key, val in user_lists_off.items():
    user_sets_off[key] = set(val)
    
pd.to_pickle(user_sets_off, os.path.join(save_dir, 'user_sets_off_per_cat.pickle'))            
        
#%% really make table
#load evrythings
tweet_counts = pd.read_pickle(os.path.join(save_dir, 'tweet_counts_per_cat.pickle'))
user_counts = pd.read_pickle(os.path.join(save_dir, 'user_counts_per_cat.pickle'))

tweet_counts_off = pd.read_pickle(os.path.join(save_dir, 'tweet_counts_off_per_cat.pickle'))
user_counts_off = pd.read_pickle(os.path.join(save_dir, 'user_counts_off_per_cat.pickle'))

user_sets = pd.read_pickle(os.path.join(save_dir, 'user_sets_per_cat.pickle')) 
user_sets_off = pd.read_pickle(os.path.join(save_dir, 'user_sets_off_per_cat.pickle'))
#%%
df_count_table = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake','far_right','right','lean_right',
                                            'center','lean_left','left', 'far_left'])

for key in df_count_table.index:
    df_count_table.loc[key,'N_t'] = tweet_counts[key]
    df_count_table.loc[key,'N_u'] = user_counts[key]
    df_count_table.loc[key,'N_t/N_u'] = tweet_counts[key]/user_counts[key]
    
df_count_table['r_t'] = df_count_table['N_t']/df_count_table['N_t'].sum()
df_count_table['r_u'] = df_count_table['N_u']/df_count_table['N_u'].sum()   

#%%
df_count_table_paired = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake_far_right_far_left','right_lean_right',
                                            'center','left_lean_left'])

for key in df_count_table_paired.index:
    df_count_table_paired.loc[key,'N_t'] = tweet_counts[key]
    df_count_table_paired.loc[key,'N_u'] = user_counts[key]
    df_count_table_paired.loc[key,'N_t/N_u'] = tweet_counts[key]/user_counts[key]
    
df_count_table_paired['r_t'] = df_count_table_paired['N_t']/df_count_table_paired['N_t'].sum()
df_count_table_paired['r_u'] = df_count_table_paired['N_u']/df_count_table_paired['N_u'].sum()   

#df_count_table_paired.to_pickle(os.path.join(save_dir,'df_count_table_paired.pickle'))

#%%
df_count_table_fake_0p05 = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake_0p05perccum','far_right','right','lean_right',
                                            'center','lean_left','left', 'far_left'])


for key in df_count_table_fake_0p05.index:
    df_count_table_fake_0p05.loc[key,'N_t'] = tweet_counts[key]
    df_count_table_fake_0p05.loc[key,'N_u'] = user_counts[key]
    df_count_table_fake_0p05.loc[key,'N_t/N_u'] = tweet_counts[key]/user_counts[key]
    
df_count_table_fake_0p05['r_t'] = df_count_table_fake_0p05['N_t']/df_count_table_fake_0p05['N_t'].sum()
df_count_table_fake_0p05['r_u'] = df_count_table_fake_0p05['N_u']/df_count_table_fake_0p05['N_u'].sum()   

 
#%%
df_count_table_breitblue = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake','breitbart', 'not_breitbart','right','lean_right',
                                            'center','lean_left','left', 'shareblue', 'not_shareblue'])


for key in df_count_table_breitblue.index:
    df_count_table_breitblue.loc[key,'N_t'] = tweet_counts[key]
    df_count_table_breitblue.loc[key,'N_u'] = user_counts[key]
    df_count_table_breitblue.loc[key,'N_t/N_u'] = tweet_counts[key]/user_counts[key]
    
df_count_table_breitblue['r_t'] = df_count_table_breitblue['N_t']/df_count_table_breitblue['N_t'].sum()
df_count_table_breitblue['r_u'] = df_count_table_breitblue['N_u']/df_count_table_breitblue['N_u'].sum()   

   
#%%   
df_count_table.to_pickle(os.path.join(save_dir,'df_count_table.pickle'))
df_count_table_fake_0p05.to_pickle(os.path.join(save_dir,'df_count_table_fake_0p05.pickle'))
df_count_table_breitblue.to_pickle(os.path.join(save_dir,'df_count_table_breitblue.pickle'))
#df_count_table = pd.read_pickle(os.path.join(save_dir,'df_count_table.pickle'))



    #%%
df_count_table_off = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake','far_right','right','lean_right',
                                            'center','lean_left','left', 'far_left'])
for key in df_count_table_off.index:
    df_count_table_off.loc[key,'N_t'] = tweet_counts_off[key]
    df_count_table_off.loc[key,'N_u'] = user_counts_off[key]
    df_count_table_off.loc[key,'N_t/N_u'] = tweet_counts_off[key]/user_counts_off[key]
    
df_count_table_off['r_t'] = df_count_table_off['N_t']/df_count_table_off['N_t'].sum()
df_count_table_off['r_u'] = df_count_table_off['N_u']/df_count_table_off['N_u'].sum()    

df_count_table_off.to_pickle(os.path.join(save_dir,'df_count_table_off.pickle'))
#df_count_table_off = pd.read_pickle('../data/urls/df_count_table_off.pickle')

    #%% breitblue off clients
df_count_table_breitblue_off = pd.DataFrame(columns=['N_t', 'r_t', 'N_u',
                                       'r_u', 'N_t/N_u'], 
                                     index=['fake','breitbart', 'not_breitbart','right','lean_right',
                                            'center','lean_left','left', 'shareblue', 'not_shareblue'])
for key in df_count_table_breitblue_off.index:
    df_count_table_breitblue_off.loc[key,'N_t'] = tweet_counts_off[key]
    df_count_table_breitblue_off.loc[key,'N_u'] = user_counts_off[key]
    df_count_table_breitblue_off.loc[key,'N_t/N_u'] = tweet_counts_off[key]/user_counts_off[key]
    
df_count_table_breitblue_off['r_t'] = df_count_table_breitblue_off['N_t']/df_count_table_breitblue_off['N_t'].sum()
df_count_table_breitblue_off['r_u'] = df_count_table_breitblue_off['N_u']/df_count_table_breitblue_off['N_u'].sum()    

df_count_table_breitblue_off.to_pickle(os.path.join(save_dir,'df_count_table_breitblue_off.pickle'))

#%% url stats
df_url_stats = df_count_table.copy()

# proportion of non-official clients
df_url_stats['p_{t,n/o}'] = (df_count_table['N_t'] - df_count_table_off['N_t'])/df_count_table['N_t']

df_url_stats['p_{u,n/o}'] = (df_count_table['N_u'] - df_count_table_off['N_u'])/df_count_table['N_u']

df_url_stats['N_{t,n/o}/N_{u,n/o}'] =  (df_count_table['N_t'] - df_count_table_off['N_t'])/(df_count_table['N_u'] - df_count_table_off['N_u'])




print(df_url_stats.to_latex(escape=False, float_format=lambda f: "{0:.2f}".format(f)))


#%% breitblue table
df_url_stats_breitblue = df_count_table_breitblue.copy()

# proportion of non-official clients
df_url_stats_breitblue['p_{t,n/o}'] = (df_count_table_breitblue['N_t'] - df_count_table_breitblue_off['N_t'])/df_count_table_breitblue['N_t']

df_url_stats_breitblue['p_{u,n/o}'] = (df_count_table_breitblue['N_u'] - df_count_table_breitblue_off['N_u'])/df_count_table_breitblue['N_u']

df_url_stats_breitblue['N_{t,n/o}/N_{u,n/o}'] =  (df_count_table_breitblue['N_t'] - df_count_table_breitblue_off['N_t'])/(df_count_table_breitblue['N_u'] - df_count_table_breitblue_off['N_u'])




print(df_url_stats_breitblue.to_latex(escape=False, float_format=lambda f: "{0:.3f}".format(f)))

#%%

n_t = [df_url_stats.N_t.loc['fake'],
       df_url_stats.N_t.loc['far_right'],
       df_url_stats.N_t.loc['right'],
       df_url_stats.N_t.loc['lean_right'],
       df_url_stats.N_t.loc['center'],
       df_url_stats.N_t.loc['lean_left'],
       df_url_stats.N_t.loc['left'],
       df_url_stats.N_t.loc['far_left']]

n_u = [df_url_stats.N_u.loc['fake'],
       df_url_stats.N_u.loc['far_right'],
       df_url_stats.N_u.loc['right'],
       df_url_stats.N_u.loc['lean_right'],
       df_url_stats.N_u.loc['center'],
       df_url_stats.N_u.loc['lean_left'],
       df_url_stats.N_u.loc['left'],
       df_url_stats.N_u.loc['far_left']]

x = range(1,len(n_t)+1)

labels = ['fake news',
          'extreme bias\n(right)',
          'right',
          'right\nleaning',
          'center',
          'left\nleaning',
          'left',
          'extreme bias\n(left)']

fig, (ax1,ax2) = plt.subplots(2,1, figsize=[6.4, 7.4])


from palettable.cartocolors.diverging import Geyser_7_r as colormap
from PlotUtils import darker


#colors = [col for col in cmap.colors]
colors = [darker(c, 0.1) for c in colormap.mpl_colors]
colors.insert(0, (0.5,0.5,0.5))
bars1 = ax1.bar(x, n_t, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels('')
ax1.set_ylabel('Number of tweets ($N_t$)')

bars2 = ax2.bar(x, n_u, color=colors)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=60)
ax2.set_ylabel('Number of users ($N_u$)')

fig.text(0.03,0.95,'a', weight='bold')
fig.text(0.03,0.57,'b', weight='bold')

#%% paired
    


n_t = [df_count_table_paired.N_t.loc['fake_far_right_far_left'],
       df_count_table_paired.N_t.loc['right_lean_right'],
       df_count_table_paired.N_t.loc['center'],
       df_count_table_paired.N_t.loc['left_lean_left']]

n_u = [df_count_table_paired.N_u.loc['fake_far_right_far_left'],
       df_count_table_paired.N_u.loc['right_lean_right'],
       df_count_table_paired.N_u.loc['center'],
       df_count_table_paired.N_u.loc['left_lean_left']]

x = range(1,len(n_t)+1)

labels = ['fake news &\nextreme bias',
          'right &\nright leaning',
          'center',
          'left &\n leftleaning']

fig, (ax1,ax2) = plt.subplots(2,1, figsize=[6.4, 7.4])


from palettable.cartocolors.diverging import Geyser_3_r as colormap
from PlotUtils import darker


#colors = [col for col in cmap.colors]
colors = [darker(c, 0.1) for c in colormap.mpl_colors]
colors.insert(0, (0.5,0.5,0.5))
bars1 = ax1.bar(x, n_t, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels('')
ax1.set_ylabel('Number of tweets ($N_t$)')

bars2 = ax2.bar(x, n_u, color=colors)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=60)
ax2.set_ylabel('Number of users ($N_u$)')

fig.text(0.03,0.95,'a', weight='bold')
fig.text(0.03,0.57,'b', weight='bold')
