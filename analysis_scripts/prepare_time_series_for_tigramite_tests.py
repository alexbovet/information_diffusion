# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import pandas as pd
import numpy as np

from datetime import timedelta, datetime

import pickle

#raise Exception

#%% functions
from pyloess import stl



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

multiplier = {'15min': 4,
                  'H' : 1}

raise Exception

#%% parameters

save_dir = '../data/urls/revisions/'

resample_freq = '15min'

ranking = 'CI_out'
top_nums = [100]

periods = ['june-nov']

graph_type = 'simple'


#which counts to use
count_type = 'df_tid_count'

media_types = ['fake', 'fake_not_aggr', 
               'breitbart',
            'not_breitbart', 'far_right', 'far_right_not_aggr', 'right',
            'right+lean_right', 'lean_right', 'lean_right_not_aggr',
            'center', 'lean_left', 'left',
            'far_left', 'not_shareblue', 'shareblue']


media_no_staff = ['fake_no_staff',  'far_right_no_staff', 
                'right_no_staff', 'left_no_staff', 'far_left_no_staff', 
                'breitbart_no_staff', 
                'not_breitbart_no_staff', 'lean_left_no_staff', 
                'center_no_staff', 'lean_right_no_staff',
                'not_shareblue_no_staff', 'shareblue_no_staff']


media_for_infl = media_no_staff

savefilesuffix = '_no_staff'

### STL params
n_p = multiplier[resample_freq]*24
n_s = n_p - 1


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

#%% read group counts

dfs_dict = dict()

for host_type in media_types:
    with open(os.path.join(save_dir, 'host_counts', 'df_' + host_type + '_counts_5min.pickle'), 'rb') as fopen:
        dfs_dict[host_type] = pickle.load(fopen)
        
#% dataframe with tid count without original retweeted statuses
        
df_tid_no_ret_og = pd.DataFrame()

for media_type in dfs_dict.keys():

    df_tid_no_ret_og[media_type] = dfs_dict[media_type]['df_tid_count_no_retweet']

#% resample
df_tid_no_ret_og = df_tid_no_ret_og.resample(resample_freq).sum()


#%% load influencers activity


for top_num in top_nums:
    print('****** top num : ' + str(top_num))

    for period in periods:
        print('   +++++ period : ' + str(period))
        
        save_dir_inf = os.path.join(save_dir, 'influencers_counts', graph_type,
                                period)
        dfs_infl_dict = dict()
        
        for media_type in media_for_infl:
            with open(save_dir_inf + '/infl_counts_' + ranking + '_' + \
                                 media_type + '_top' + str(top_num) + '_5min.pickle', 'rb') as fopen:
                
                dfs_infl_dict[media_type] = pickle.load(fopen)
        
                
        df_infl_with_ret = pd.DataFrame()
        
        for media_type in media_for_infl:
        
            df_infl_with_ret[media_type] = dfs_infl_dict[media_type]['df_tid_count']
            
        
        #% resample
        df_infl_with_ret = df_infl_with_ret.resample(resample_freq).sum()
    

        
       
        #%% 
        
        #prepare data
        
        df = pd.DataFrame(columns=['tot'], data=df_tot_tweets.copy())
        df['pro_h'] = df_num_tweets.n_pro_h
        df['pro_t'] = df_num_tweets.n_pro_t
        for key in df_tid_no_ret_og.columns:
            df[key] = df_tid_no_ret_og[key].copy()
            
        for key in df_infl_with_ret.columns:
            df['infl_' + key] = df_infl_with_ret[key].copy()
            
        # drop datetimes at both ends
        
        
        start_day = datetime(2016,6,1)
        stop_day = datetime(2016,11,9)
    
        
        df.drop(df.loc[np.logical_or(df.index < start_day, 
                                     df.index >= stop_day)].index,
                                    inplace=True)            
            
        # replace null values by nan
        for col in df.columns:
            df[col].loc[df['tot'].isna()] = np.nan
        
        
        # deal with missing values:
        all_time_index = df.index
        missing_ts = df.loc[df['tot'].isna()].index
       

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
        
        dfdrop.drop(dfdrop.loc[np.logical_or(dfdrop.index < start_day, 
                                     dfdrop.index >= stop_day)].index,
                                    inplace=True)        
        #%% LOESS smoothing
        
        
        # sremove seasonality and trend
        
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
        
        

        
        
        dftrend = pd.DataFrame(index=dfdrop.index)
        dfseasonal = pd.DataFrame(index=dfdrop.index)
        dfresiduals = pd.DataFrame(index=dfdrop.index)
        for col in stl_res_dict.keys():
            dftrend[col] = stl_res_dict[col].trend

            dfseasonal[col] = stl_res_dict[col].seasonal

            dfresiduals[col] = stl_res_dict[col].residuals

            
        
        
        save_dir_file = os.path.join(save_dir, 'time_series/', graph_type, period)
        
        os.makedirs(save_dir_file, exist_ok=True)
        
        filename_residuals = save_dir_file + '/timeseries_loess_residuals_freq' + \
                  resample_freq + '_ns' + str(stl_params['ns']) +\
                  '_topinflnum' + str(top_num) + '_ranking' + \
                  ranking + savefilesuffix + '.pickle'
                  
        filename_seasonal = save_dir_file + '/timeseries_loess_seasonal_freq' + \
                  resample_freq + '_ns' + str(stl_params['ns']) +\
                  '_topinflnum' + str(top_num) + '_ranking' + \
                  ranking + savefilesuffix + '.pickle'
                  
        filename_trend = save_dir_file + '/timeseries_loess_trend_freq' + \
                  resample_freq + '_ns' + str(stl_params['ns']) +\
                  '_topinflnum' + str(top_num) + '_ranking' + \
                  ranking + savefilesuffix + '.pickle'      
                  
        dfresiduals.to_pickle(filename_residuals)
        dfseasonal.to_pickle(filename_seasonal)
        dftrend.to_pickle(filename_trend)
         
        
