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

#import graph_tool.all as gt

plt.style.use('alex_paper')
save_dir = '../data/urls/revisions/'

raise Exception


#%%        
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite.independence_tests import RCOT2
from tigramite.models import LinearMediation

#% load results


res = pd.read_pickle(os.path.join(save_dir,'tigramite_res/pcmci_parallel_rcot2_june-nov_pc_alpha0.1_tau_max18_max_comb3.pickle'))

val_matrix = res['val_matrix']
p_matrix = res['p_matrix']
conf_matrix = res['conf_matrix']
parameters = res['parameters']

parameters['val_matrix'] = val_matrix
parameters['p_matrix'] = p_matrix
#%% load corresponding dataframe
suffix = ''
save_dir_file = os.path.join(save_dir, 'time_series/', parameters['graph_type'], 
                             parameters['period'])

filename_residuals = save_dir_file + '/timeseries_loess_residuals_freq' + \
          parameters['resample_freq'] + '_ns' + str(parameters['n_s']) +\
          '_topinflnum' + str(parameters['top_num']) + '_ranking' + \
          parameters['ranking'] + suffix + '.pickle'
          
          
dfresiduals = pd.read_pickle(filename_residuals)

data = dfresiduals[parameters['var_names']].values

T, N = data.shape

# Initialize dataframe object
dataframe = pp.DataFrame(data)

#%%

rcot = RCOT2(significance=parameters['cond_ind_test.significance'],
            num_f=parameters['cond_ind_test.num_f'])
pcmci = PCMCI(dataframe,
              cond_ind_test=rcot,
              selected_variables=parameters['selected_variables'],
              var_names=parameters['var_names'],
              verbosity=10)
              
q_matrix = pcmci.get_corrected_pvalues(p_matrix=p_matrix, fdr_method='fdr_bh')
q_matrix_tsbh = pcmci.get_corrected_pvalues(p_matrix=p_matrix, fdr_method='fdr_tsbh')

#%% print results
pcmci._print_significant_links(
        p_matrix = p_matrix, 
        q_matrix = q_matrix,
        val_matrix = val_matrix,
        alpha_level = 0.1)


#%% get selected parents and fit linear model
q_0 = 0.05

parameters['q_0'] = q_0
parameters['q_matrix'] = q_matrix

parent_dict = pcmci._return_significant_parents(q_matrix, val_matrix, q_0)
parameters['parent_dict'] = parent_dict

med = LinearMediation(dataframe=dataframe) #, data_transform=False)
med.fit_model(all_parents=parent_dict['parents'])

print(med.phi)
parameters['model.phi'] = med.phi
parameters['model.psi'] = med.psi

#%% find coefficients for each node

var_names = parameters['var_names']

tau_max = med.psi.shape[0]-1
for j in range(len(var_names)):
    print('CE --> ', var_names[j])
    for tau in range(1, tau_max+1):
        print('tau = ', tau)
        for i in range(len(var_names)):
            ICE_ijtau = med.get_ce(i, tau, j)
            if ICE_ijtau != 0.0:
                print('CE {i} --> {j} = {I}'.format(i=var_names[i],
                                              j=var_names[j],
                                              I=ICE_ijtau))
                
#max CE
for j in range(len(var_names)):
    print('CE --> ', var_names[j])
    for i in range(len(var_names)):
        ICE_ijmax = np.abs(med.psi[1:, j, i]).max()
        if ICE_ijmax != 0.0:
            print('CE max {i} --> {j} = {I}'.format(i=var_names[i],
                                          j=var_names[j],
                                          I=ICE_ijmax))


#%% confidence interval with bootstrap

#compute residuals

def bootstrapping_ar_model(model, num_bs=200, seed=52):
    

    T, N = model.data.shape
    
    std_data = np.zeros_like(model.data)
    #standardize
    for i in range(N):
        std_data[:,i] = (model.data[:,i] - model.data[:,i].mean())/model.data[:,i].std()
        
    # initial model coeffs
    phi = model.phi
    
    tau_max = phi.shape[0] - 1 
    
    residuals = np.zeros((T-tau_max, N))
    
    for i in range(T-tau_max):
        
        model_eval = np.zeros((1,N))
        for tau in range(1, tau_max+1):
            model_eval += np.dot(phi[tau],std_data[i+tau_max-tau])
            
        residuals[i,:] = std_data[i+tau_max,:] - model_eval
    
    # generate bootstrap data
    bs_models = []
    ts_indexes = np.arange(residuals.shape[0])
    np.random.seed(seed)
    for _ in range(num_bs):
        bs_residuals = residuals[np.random.choice(ts_indexes, size=T, replace=True),:]
        # bs model
        bs_x = np.zeros((T, N))
        for t in range(0,T):
            if t < tau_max:
                bs_x[t,:] = bs_residuals[t,:]
            else:
                model_eval = np.zeros((1,N))
                for tau in range(1, tau_max+1):
                    model_eval += np.dot(phi[tau],bs_x[t-tau])
                    
                bs_x[t,:] = model_eval + bs_residuals[t,:]
        #fit bs data
        bs_med = LinearMediation(dataframe=pp.DataFrame(data=bs_x)) #, data_transform=False)
        bs_med.fit_model(all_parents=parent_dict['parents'])
        bs_models.append(bs_med)
    
    return bs_models


#%% bootstap!
use_bootstrap = True
    

if use_bootstrap:    
    num_bs = 200
    bs_models = bootstrapping_ar_model(med, num_bs=num_bs)
    
    parameters['num_bs'] = 200
    
else:
    parameters['num_bs'] = None
#%% CE with uncertainties


df_CEmax = pd.DataFrame(columns=var_names, index=var_names)

df_CEmax_mean = pd.DataFrame(columns=var_names, index=var_names)
df_CEmax_std = pd.DataFrame(columns=var_names, index=var_names)
df_CEmax_tau = pd.DataFrame(columns=var_names, index=var_names)

for j in range(len(var_names)):
    print('CE --> ', var_names[j])
    for i in range(len(var_names)):
        ICE_ijmax = np.abs(med.psi[1:, j, i]).max()
        tau_at_max = np.abs(med.psi[1:, j, i]).argmax()+1
        if use_bootstrap:
            ICE_ijmax_dist = np.array([bs_m.psi[tau_at_max, j, i] for bs_m in bs_models])
        
        if ICE_ijmax != 0.0:
            if use_bootstrap:
                print('CE max {i} --> {j} = {I:.5f}, ({m:.5f} +- {s:.5f}), tau={tau}'.format(i=var_names[i],
                                          j=var_names[j],
                                          I=ICE_ijmax,
                                          m=np.abs(ICE_ijmax_dist.mean()),
                                          s=ICE_ijmax_dist.std(),
                                          tau=tau_at_max))
            else:
                print('CE max {i} --> {j} = {I:.5f}, tau={tau}'.format(i=var_names[i],
                                          j=var_names[j],
                                          I=ICE_ijmax,
                                          tau=tau_at_max))
            df_CEmax.loc[var_names[j], var_names[i]] = ICE_ijmax
            df_CEmax_tau.loc[var_names[j], var_names[i]] = tau_at_max
            if use_bootstrap:
                df_CEmax_mean.loc[var_names[j], var_names[i]] = np.abs(ICE_ijmax_dist.mean())
                df_CEmax_std.loc[var_names[j], var_names[i]] = ICE_ijmax_dist.std()
                
                

#%% save         
suffix = ''
file_name = os.path.join(save_dir,'tigramite_res/CE_max_' +\
                         '{cond_ind_test}_{period}_pc_alpha{pc_alpha}' +\
                         '_tau_max{tau_max}_q0{q0}_maxcomb{max_comb}_' +\
                         suffix + '.pickle')

file_name = file_name.format(cond_ind_test=parameters['cond_ind_test'],
                             period=parameters['period'],
                             pc_alpha=parameters['pc_alpha'],
                             tau_max=parameters['tau_max'],
                             q0=parameters['q_0'],
                             max_comb=parameters['max_combinations']
                             )

print(file_name)
#file_name += 'hercules'
if os.path.exists(file_name):
    raise Exception('file exists')
        
pd.to_pickle({'parameters' : parameters,
              'df_CEmax' : df_CEmax,
              'df_CEmax_mean' : df_CEmax_mean,
              'df_CEmax_std' : df_CEmax_std,
              'df_CEmax_tau' : df_CEmax_tau,
              }, 
              file_name)
