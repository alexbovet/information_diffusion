# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import CIcython

import time
import pickle
import graph_tool.all as gt

from multiprocessing import Pool, Queue
ncpu = 4

save_dir =  '../data/urls/revisions/'

#raise Exception
 
    
#%% add CI values to graph

    
def add_CI_to_graph(file):
    
    graph_file = os.path.join(save_dir, file)
    
    print('pid ' + str(os.getpid()) + ' loading graph ' + graph_file)
    
    graph = gt.load_graph(graph_file)
    
    
#    for direction in ['out', 'undir', 'in', 'both']:
    for direction in ['out', 'in']:
        
        print('pid ' + str(os.getpid()) + ' -- ' + direction)
        t0 = time.time()
        CIranks, CImap = CIcython.compute_graph_CI(graph, rad=2,
                                          direction=direction,
                                          verbose=True)
        t1 = time.time() - t0
        
        print('pid ' + str(os.getpid()) + ' -- ' + str(t1))
        
        print('pid ' + str(os.getpid()) + ' saving CIranks ' + direction +  '_' + file.strip('.gt'))
        with open('../data/urls/CI_l2_' + direction +  '_' + file.strip('.gt') + '_.pickle', 'wb') as fopen:
        
            pickle.dump({'CIranks': CIranks, 'CImap' : CImap, 'time' : t1}, fopen)
    
        graph.vp['CI_' + direction] = graph.new_vertex_property('int64_t', vals=0)
        graph.vp['CI_' + direction].a = CImap
        
    print('pid ' + str(os.getpid()) + ' -- adding katz centrality')
        
    
    graph.vp['katz'] = gt.katz(graph)
    
    graph.set_reversed(True)
    graph.vp['katz_rev'] = gt.katz(graph)
    graph.set_reversed(False)
        
    print('pid ' + str(os.getpid()) + ' saving graph ' + graph_file)
    graph.save(graph_file)


def process_queue(queue):
    while not queue.empty():
        file = queue.get()
        print('****\n pid ' + str(os.getpid()) + ' processing ' + file)
        t0 = time.time()
        add_CI_to_graph(file)
        print('pid ' + str(os.getpid()) + ' done in ' + str(time.time() - t0))
        
        
#%%        
files_in_dir = os.listdir(save_dir)

import datetime
date = datetime.datetime(2018,5,1,22)

files = [f for f in files_in_dir if f[:13] == 'retweet_graph' and\
             f[-3:] == '.gt' and\
             os.path.getmtime(save_dir+f) > date.timestamp()]        
#

#files = ['retweet_graph_lean_left_simple_june-nov.gt']


queue = Queue()
for file in files:
    queue.put(file)

t0 = time.time()
with Pool(ncpu, process_queue, (queue,)) as pool:
    pool.close() # signal that we won't submit any more tasks to pool
    pool.join() # wait until all processes are done
print('****************** Finished!')
print(time.time()-t0)
