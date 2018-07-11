# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import sqlite3
import pandas as pd

import gc
import time
import pickle




#plt.style.use('alex_paper')
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

#%% load user and tweet list
#user_fake_counts = pd.read_pickle('../data/user_fake_counts.pickle')

tweet_db_file1 = '../databases_ssd/complete_trump_vs_hillary_db.sqlite'
tweet_db_file2 = '../databases_ssd/complete_trump_vs_hillary_sep-nov_db.sqlite'
urls_db_file = '../databases_ssd/urls_db.sqlite'

save_dir_host_count = os.path.join(save_dir,'host_counts/')
os.makedirs(save_dir_host_count, exist_ok=True)
raise Exception

#%% select all urls comming from host groups

def get_df_counts(df, filename=None, resample_freq='D'):
    # count daily tweets
    df_tid_count = df.resample(resample_freq, on='datetime_EST').tweet_id.nunique()
    # discard original retweeted tweets
    df_tid_count_no_retweet = df.loc[df.retweeted_status == 0].resample(resample_freq, on='datetime_EST').tweet_id.nunique()
    # only official clients
    df_tid_count_off_cli = df.loc[df.source_content.isin(official_twitter_clients)].resample(resample_freq, on='datetime_EST').tweet_id.nunique()
    # count daily tweets
    df_uid_count = df.resample(resample_freq, on='datetime_EST').user_id.nunique()
    # discard original retweeted tweets
    df_uid_count_no_retweet = df.loc[df.retweeted_status == 0].resample(resample_freq, on='datetime_EST').user_id.nunique()
    # only official clients
    df_uid_count_off_cli = df.loc[df.source_content.isin(official_twitter_clients)].resample(resample_freq, on='datetime_EST').user_id.nunique()
    
    # global measures
    df_global = pd.DataFrame(data = {'num_tweet' : df.tweet_id.nunique(),
                                     'num_users' : df.user_id.nunique(),
                                     'num_tweet_no_retweet' : df.loc[df.retweeted_status == 0].tweet_id.nunique(),
                                     'num_users_no_retweet' : df.loc[df.retweeted_status == 0].user_id.nunique(),
                                     'num_tweet_off_cli' : df.loc[df.source_content.isin(official_twitter_clients)].tweet_id.nunique(),
                                     'num_users_off_cli' : df.loc[df.source_content.isin(official_twitter_clients)].user_id.nunique()
                                     },
                            index = [1])
    
    res = {'df_tid_count':df_tid_count,
            'df_uid_count' : df_uid_count,
            'df_tid_count_no_retweet': df_tid_count_no_retweet,
            'df_tid_count_off_cli': df_tid_count_off_cli,
            'df_uid_count_no_retweet': df_uid_count_no_retweet,
            'df_uid_count_off_cli': df_uid_count_off_cli,
            'df_global':df_global}
    
    if filename is not None:
        #save
        with open(filename, 'wb') as fopen:
            pickle.dump(res, fopen)
    else:
        return res
    

#%%
t0 = time.time()
t00 = time.time()
        
resample_freq='5min'        
with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
    
#    
#    print('not_breitbart')
#
#    df_not_breitbart_all = pd.read_sql_query("""SELECT * FROM urls
#                         WHERE final_hostname IN (
#                                 SELECT hostname FROM hosts_far_right_rev_stat
#                                 WHERE perccum > 0.01
#                                 AND hostname IS NOT 'breitbart.com')
#                         """, conn)
#                         
#    get_df_counts(df_not_breitbart_all, save_dir_host_count + 'df_not_breitbart_counts_' + resample_freq + '.pickle',
#                  resample_freq=resample_freq)                                     
#    
#    del df_not_breitbart_all
#    gc.collect()
#    print(time.time()-t0)
    ##################
    
    
    print('not_shareblue')

    df_not_shareblue_all = pd.read_sql_query("""SELECT * FROM urls
                         WHERE final_hostname IN (
                                 SELECT hostname FROM hosts_far_left_rev_stat
                                 WHERE perccum > 0.01
                                 AND hostname IS NOT 'bluenationreview.com'
                                 AND hostname IS NOT 'shareblue.com')
                         """, conn)
                         
    get_df_counts(df_not_shareblue_all, save_dir_host_count + 'df_not_shareblue_counts_' + resample_freq + '.pickle',
                  resample_freq=resample_freq)                                     
    
    del df_not_shareblue_all
    gc.collect()
    print(time.time()-t0)
    #################

    print('shareblue')
    
    df_shareblue_all = pd.read_sql_query("""SELECT * FROM urls
                         WHERE final_hostname = 'bluenationreview.com'
                         OR final_hostname = 'shareblue.com'
                         """, conn)
                         
    get_df_counts(df_shareblue_all, save_dir_host_count + 'df_shareblue_counts_' + resample_freq + '.pickle',
                  resample_freq=resample_freq)                                     
    
    del df_shareblue_all
    gc.collect()
    print(time.time()-t0)
    #################        
    #loop on the rest:
    
    sql_select = """SELECT * FROM urls
                         WHERE final_hostname IN (
                                 SELECT hostname FROM hosts_{hostype}_rev_stat
                                 WHERE perccum > 0.01)
                         """
     
                         
#    for host_type in ['fake', 'far_right', 'right', 'lean_right', 'center', 
#                      'lean_left', 'left',
#                      'far_left']:
#    for host_type in ['lean_right', 'center', 
#                      'lean_left']:
#    
#        print(host_type)
#        df = pd.read_sql_query(sql_select.format(hostype=host_type), conn)
#    
#        get_df_counts(df, save_dir_host_count + 'df_' + host_type + '_counts_' + resample_freq + '.pickle',
#                      resample_freq=resample_freq)
#        del df
#        gc.collect()
#        print(time.time()-t0)
        
    sql_select_not_aggr = """SELECT * FROM urls
                         WHERE final_hostname IN (
                                 SELECT hostname FROM hosts_{hostype}_rev_stat
                                 WHERE perccum > 0.01
                                 EXCEPT
                                 SELECT hostname FROM news_aggregators)
                         """
                         
    for host_type in ['fake', 'far_right', 
                      'lean_right']:
    
        print(host_type + '_not_aggr')
        df = pd.read_sql_query(sql_select_not_aggr.format(hostype=host_type), conn)
    
        get_df_counts(df, save_dir_host_count + 'df_' + host_type + '_not_aggr_counts_' + resample_freq + '.pickle',
                      resample_freq=resample_freq)
        del df
        gc.collect()
        print(time.time()-t0)
                    
        #################

    gc.collect()
    print(time.time()-t0)       
#raise Exception
    
#%% select tweets comming from specific users

t0 = time.time()
print('get influencers activity')
print(time.time()-t0)


def get_user_activity(user_id_list, resample_freq, db_file=urls_db_file,
                      save_filename=None):
    with sqlite3.connect(urls_db_file, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES) as conn:
        
        
        df = pd.read_sql_query("""SELECT * FROM urls WHERE 
                                        user_id IN ({user_ids})""".format(user_ids=','.join([str(uid) for uid in user_id_list])), conn)
        
        
    counts = get_df_counts(df, save_filename,
                  resample_freq=resample_freq)
    return counts
    
#%% load influencers 


from multiprocessing import Pool, Queue
ncpu = 4

def fct(args):
    infl_ranks, media_type, period, ranking, top_num, resample_freq, urls_db_file, save_dir_period = args
    print('****\n pid ' + str(os.getpid()) + ' processing ' + str(top_num))
    _ = get_user_activity(infl_ranks[media_type][period][ranking].user_id[:top_num].values,
                                  resample_freq,
                                  db_file=urls_db_file,
                                  save_filename=save_dir_period + '/infl_counts_' + ranking + '_' + \
                                     media_type + '_top' + str(top_num) + '_' + resample_freq + '.pickle') 

def process_queue(queue):
    while not queue.empty():
        args = queue.get()
        t0 = time.time()
        print('pid ' + str(os.getpid()) + ' processing ' + str(args[1:5]))
        fct(args)
        print('pid ' + str(os.getpid()) + ' done in ' + str(time.time() - t0))
    
    

save_dir_infl_counts = os.path.join(save_dir, 'influencers_counts/')
os.makedirs(save_dir_infl_counts, exist_ok=True)
resample_freq = '5min'    
ranking = 'CI_out'

top_nums = [100]

media_types =  ['fake_no_staff',  'far_right_no_staff', 
                'right_no_staff', 'left_no_staff', 'far_left_no_staff', 
                'breitbart_no_staff', 
                'not_breitbart_no_staff', 'lean_left_no_staff', 
                'center_no_staff', 'lean_right_no_staff',
                'not_shareblue_no_staff', 'shareblue_no_staff']
t0 = time.time()
queue = Queue()
for graph_type in ['simple']:
    
    print(graph_type)
    with open(os.path.join(save_dir, 'influencer_rankings_' + graph_type + \
              '.pickle'), 'rb') as fopen:
        
        infl_ranks = pickle.load(fopen)

    for period in ['june-nov']:
        print(period)
        
        save_dir_period = os.path.join(save_dir_infl_counts, graph_type, period)
        
        os.makedirs(save_dir_period, exist_ok=True)
        
        # start counts 

        for media_type in media_types:
            print(media_type)
            
            for top_num in top_nums:
                filename=save_dir_period + '/infl_counts_' + ranking + '_' + \
                                     media_type + '_top' + str(top_num) + '_' + resample_freq + '.pickle'
                
                if not os.path.exists(filename):
                    queue.put((infl_ranks, media_type, period, ranking, top_num, 
                               resample_freq, urls_db_file, save_dir_period))
                
                            
            print(time.time()-t0)

t1 = time.time()
with Pool(ncpu, process_queue, (queue,)) as pool:
    pool.close() # signal that we won't submit any more tasks to pool
    pool.join() # wait until all processes are done
    
print('Finished queue!')
print(time.time()-t1)
            
print('total time')
print(time.time() - t00)            
