# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0


import graph_tool.all as gt
import time
import heapq
from collections import deque
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.profile(True)
cpdef bfs_path(G, np.intp_t v_id, unsigned long rad,
                                  unsigned long num_vertices):
    """Breadth-First-Search
       returns list of edges of the BFS from node `v_id`
       up to a radius `rad` using Graph `G`. 
       `num_vertices` is the number of vertices of G.
    
    """

    cdef int[::1] color = np.zeros(num_vertices, dtype=np.int32) #zero=white
    cdef int[::1] dist = np.zeros(num_vertices, dtype=np.int32)
    cdef unsigned long long[::1] out_neighbours
    
    cdef unsigned long l = 0 # explored radius
    color[v_id] = 1 # 1=gray (discovered)
    
    cdef np.intp_t s_id
    cdef np.intp_t t_id
    cdef np.intp_t i # out-neighbours iterators
    cdef long imax # num of out-neighbours

    
    q = deque()
    
    edges = []
    
    q.append(v_id)
    while l <= rad and q:
        s_id = q.popleft()
        out_neighbours = G.get_out_neighbours(s_id)
        imax = out_neighbours.shape[0]
        for i in range(imax):
            t_id = out_neighbours[i]
            if color[t_id] == 0: #new node
                color[t_id] = 1
                l = dist[s_id] + 1
                dist[t_id] = l
                
                if l <= rad:
                    q.append(t_id)
                    edges.append((s_id, t_id))

    
    return edges

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned long long[::1] get_ball_boundary(G, np.intp_t v_id,
                                                unsigned long rad, 
                                                unsigned long num_vertices):
    """ 
        Returns an array of nodes ids of the graph `G`
        at the boundary of the ball of radius `rad` 
        centered on 'v_id'.
       `num_vertices` is the number of vertices of G.
                
    """
        
    cdef int[::1] color = np.zeros(num_vertices, dtype=np.int32) #zero=white
    cdef int[::1] dist = np.zeros(num_vertices, dtype=np.int32)
    cdef unsigned long long[::1] out_neighbours
    
    cdef unsigned long l = 0 # explored radius
    color[v_id] = 1 # 1=gray (discovered)
    
    cdef np.intp_t s_id
    cdef np.intp_t t_id
    cdef np.intp_t i # out-neighbours iterators
    cdef long imax # num of out-neighbours

    q = deque()
    
    boundary_nodes = []
    
    q.append(v_id)
    while l <= rad and q:
        s_id = q.popleft()
        out_neighbours = G.get_out_neighbours(s_id)
        imax = out_neighbours.shape[0]
        for i in range(imax):
            t_id = out_neighbours[i]
            if color[t_id] == 0: #new node
                color[t_id] = 1
                l = dist[s_id] + 1
                dist[t_id] = l
                
                if l < rad:
                    q.append(t_id)
                elif l == rad:
                    boundary_nodes.append(t_id)

    
    return np.array(boundary_nodes, np.uint64)



def get_ball_boundary_old(G, long v, unsigned long rad,
                          unsigned long num_vertices):
    """ 
        Returns a list of nodes ids at the boundary of the ball of radius `rad` 
        centered on 'v'.
        
        Cython version
        
    """
    cdef np.ndarray dist = np.zeros(num_vertices, 
                                     dtype=np.int64)
    
    bfs_iterator = gt.bfs_iterator(G, v)
    cdef unsigned long l
    cdef long s_id
    cdef long t_id
    
    l = 0
    boundary_nodes = []
    #we stop the search when we trespass the ball radius
    while l <= rad:
        try:
            e = next(bfs_iterator)
        except StopIteration:
            break
        s_id = G.vertex_index[e.source()]
        t_id = G.vertex_index[e.target()]
        l = dist[s_id] + 1
        dist[t_id] = l
        if l == rad:
            boundary_nodes.append(t_id)
    
    return boundary_nodes


def get_ball_old(G, long v, unsigned long rad, 
                            unsigned long num_vertices):
    """ 
        Returns a list of nodes in the ball of radius `rad` 
        centered on 'v', including the nodes at the boundary.
        
        Cython version
        
    """
        
    # Breadth-First-Search iterator
    bfs_iterator = gt.bfs_iterator(G, v)
    
#    dist = G.new_vertex_property("long", vals=0)
    cdef np.ndarray dist = np.zeros(num_vertices, 
                                     dtype=np.int64)
    
    boundary_nodes = []
    
    cdef unsigned long l
    cdef long s_id
    cdef long t_id
    
    l = 0

    t_id = v
    
    #we stop the search when we trespass the ball radius
    while l <= rad:
        boundary_nodes.append(t_id)
        try:
            e = next(bfs_iterator)
        except StopIteration:
            break
        s_id = G.vertex_index[e.source()]
        t_id = G.vertex_index[e.target()]
        l = dist[s_id] + 1
        dist[t_id] = l
            
            
    
    return boundary_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned long long[::1] get_ball(G, np.intp_t v_id, unsigned long rad,
                                       unsigned long num_vertices):
    """ 
        Returns an array of nodes ids of graph `G` in the ball of radius `rad` 
        centered on 'v_id', including the nodes at the boundary.
        `num_vertices` is the number of vertices of G.
                
    """
        
    cdef int[::1] color = np.zeros(num_vertices, dtype=np.int32) #zero=white
    cdef int[::1] dist = np.zeros(num_vertices, dtype=np.int32)
    cdef unsigned long[::1] out_neighbours
    
    cdef unsigned long l = 0 # explored radius
    color[v_id] = 1 # 1=gray (discovered)
    
    cdef np.intp_t s_id
    cdef np.intp_t t_id
    cdef np.intp_t i # out-neighbours iterators
    cdef long imax # num of out-neighbours

    q = deque()
    
    ball_nodes = []
    
    q.append(v_id)
    while l <= rad and q:
        s_id = q.popleft()
        out_neighbours = G.get_out_neighbours(s_id)
        imax = out_neighbours.shape[0]
        for i in range(imax):
            t_id = out_neighbours[i]
            if color[t_id] == 0: #new node
                color[t_id] = 1
                l = dist[s_id] + 1
                dist[t_id] = l
                
                if l <= rad:
                    q.append(t_id)
                    ball_nodes.append(t_id)

    return np.array(ball_nodes, dtype=np.uint64)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long long compute_node_CI(np.intp_t v_id, G, unsigned long rad,
                           long long[::1] k_map,
                           unsigned long num_vertices):
    """ 
        Computes and returns the CI value of vertex `v_id` 
        using a ball radius `rad` in the graph `G`.
        `k_map` is an array with the degree map to be used
        and `num_vertices` is the total number of vertices 
        of the graph.
        
    """
    
    if k_map[v_id] == 0 or k_map[v_id] == 1:
        return 0
    
    cdef unsigned long long[::1] boundary_nodes = get_ball_boundary(G, 
                                                                  v_id, 
                                                                  rad, 
                                                                  num_vertices)
                                                      
    cdef long long CI_balledge
    cdef long long kj
    cdef np.intp_t i
    cdef long imax = boundary_nodes.shape[0]
    CI_balledge = 0
    
    for i in range(imax):
        kj = k_map[boundary_nodes[i]]
        if kj > 0: # necessary check for CI directed
            CI_balledge += kj - 1
    
    return (k_map[v_id] - 1) * CI_balledge

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_node_CI_numpy(np.intp_t v_id, G, unsigned long rad,
                           unsigned long num_vertices):
    """ 
        Computes and returns the CI value of vertex `v_id` 
        using a ball radius `rad` in the graph `G`.
        `k_map` is an array with the degree map to be used
        and `num_vertices` is the total number of vertices 
        of the graph.
        
    """

    k_v = G.get_out_degrees([v_id])[0]
    
    if k_v == 0 or k_v == 1:
        return 0
    
    boundary_nodes = get_ball_boundary(G, 
                                      v_id, 
                                      rad, 
                                      num_vertices)

    boundary_degrees = G.get_out_degrees(boundary_nodes)
    
    return np.sum(boundary_degrees -1)*(k_v - 1)
    
 
    
@cython.boundscheck(False)
@cython.wraparound(False)    
def compute_graph_CI(G, unsigned long rad,
                       verbose=False,
                       direction='undir'):
    """
        Compute node ranks of graph `G` according to CI using
        a ball radius `rad`. 
        
        Returns a list of nodes sorted in descending order of their CI
        and a numpy array mapping node ids to their CI value.
    
        If `direction`='undir' (default), treat the graph as undirected
        even if it is directed.
        
        `direction`='out', compute only the CI_out based
        on the out-ball and k_out using the unreversed graph.
        
        `direction`='in' (default), compute only the CI_in based
        on the in-ball and k_in using the unreversed graph.
        
        `direction`='both' (default), compute CI as CI_out + CI_in,
        using the unreversed graph.
        
        Cython version.
    """
    
    G = gt.GraphView(G)
    
    if direction not in ['undir', 'in', 'out', 'both']:
        raise ValueError('unrecognised `direction`.')
        
    is_init_directed = G.is_directed()
    is_init_reversed = G.is_reversed()
    
    
    CI_type = 'CI_undirected'
    
    if is_init_directed:
        if direction == 'undir':
            G.set_directed(False)
        elif direction == 'in':
            # same as CI_out but on the reversed graph
            G.set_reversed(True)
            CI_type = 'CI_in'
        elif direction == 'out':
            G.set_reversed(False)
            CI_type = 'CI_out'
        elif direction == 'both':
            G.set_reversed(False)
            CI_type = 'CI_in + CI_out'
            
        
                
            
    # initial degree property map
    # for undirected network,  k = k_out and k_in = 0
    cdef np.ndarray[long long, ndim=1] k_out = np.array(G.degree_property_map('out').a,
                                               dtype=np.int64)
    cdef np.ndarray[long long, ndim=1] k_in = np.array(G.degree_property_map('in').a,
                                               dtype=np.int64)
    
    # initial number of vertices
    cdef unsigned long num_vertices = G.num_vertices()

    # vertex property map where the CI values will be stored
    cdef long long[::1] CImap = np.zeros(num_vertices, 
                                     dtype=np.int64)
    
    # CI map with values at removal
    cdef long long[::1] CImap_final = np.zeros(num_vertices, 
                                     dtype=np.int64)

    cdef long long top_CI_val 
    cdef unsigned long top_CI_vertex_id
    cdef unsigned long v_id
    
    # CI ranking
    CI_ranking = []
        
    # filter property map for removed nodes
    filt = G.new_vertex_property('bool', val=True)
    
    
    #############
    # initial computation of CI
    
    if verbose:
        t0 = time.time()
        print('initial computation of {CI_type} for entire graph'.format(CI_type=CI_type))
        
    # CI or CI_in or CI_out
    for v_id in range(num_vertices):
        CImap[v_id] = compute_node_CI(v_id, G, rad,
                                       k_map=k_out,
                                       num_vertices=num_vertices)
    #CI = CI_in + CI_out
    if direction == 'both':
        
        G.set_reversed(True)
        for v_id in range(num_vertices):            
            CImap[v_id] += compute_node_CI(v_id, G, rad,
                                           k_map=k_in,
                                           num_vertices=num_vertices)
        G.set_reversed(False)
        
        
    if verbose:
        print('*** done in ' + '{:.2f}'.format(time.time()-t0) + 's')
        
        
    # we use heapq with negative values of CI to make it a max-heap
    heap = list()
    for v_id in range(num_vertices):
        heap.append((-1*CImap[v_id], v_id))
    
    if verbose:
        t1 = time.time()
        print('heapification...')
    heapq.heapify(heap)
    if verbose:
        print('*** done in ' + '{:.2f}'.format(time.time()-t1) + 's')  
    
    # pop the top vertex
    top_CI_val, top_CI_vertex_id = heapq.heappop(heap)
    
    # reverse sign because we have a min-heap instead of a max-heap
    top_CI_val *= -1
    
    CImap_final[top_CI_vertex_id] = top_CI_val
    CI_ranking.append(top_CI_vertex_id)
    

    ################
    # Start removing node according to their CI rank and recompute CI vals
    # We just filter node out of the graph instead of removing them
    
    # first get the nodes that we will need to update
    ball_nodes = get_ball(G,top_CI_vertex_id,rad+1, num_vertices)
    vertices_to_update = set(ball_nodes) 
    
    if verbose:
        t2 = time.time()
        print('start removing nodes...')    
                    
    # filter top node
    filt[top_CI_vertex_id] = False
    G.set_vertex_filter(filt)
    
    # Update degree maps
    # we treat the undirected case by making k_in negative and 
    # updating k_tot = k_in + k_out
    
    k_out[top_CI_vertex_id] = 0
    k_in[top_CI_vertex_id] = 0
    # decrease degree of neighbors
    for neigh in G.get_out_neighbours(top_CI_vertex_id):
            k_in[neigh] -= 1
            
    # update k_tot
    if direction == 'undir':
        k_out += k_in
        k_in[:] = 0
        
    #####
    # Start looping to remove vertices
    #####
    cdef double cutoff
    cdef long step_check
    
    #stopping condition based on largest component size
    if num_vertices > 1000:
        cutoff = 0.01
        step_check = int(cutoff*num_vertices)
    else:
        cutoff = 1.0/num_vertices
        step_check = 10
    
    #largest weakly connected component
    
    cdef double gc_frac
    gc_fraq = float(gt.label_largest_component(G,
                    directed=False).a.sum())/num_vertices
                              
    cdef long lheap                                                   
    lheap = len(heap)
    
    while len(heap) > 0 and gc_fraq > cutoff:
        
        # pop the top vertex
        top_CI_val, top_CI_vertex_id = heapq.heappop(heap)

        # reverse sign because we have a min-heap instead of a max-heap
        top_CI_val *= -1
        
        # check if its CI value needs to be updated and that it's not the 
        # last vertex
        if top_CI_vertex_id in vertices_to_update \
            and len(heap) > 1:
            
            #update CI val
            top_CI_val = compute_node_CI(top_CI_vertex_id, G, rad,
                                             k_map=k_out,
                                             num_vertices=num_vertices)
            # CI = CI_out + CI_in
            if direction=='both':
                G.set_reversed(True)
                top_CI_val = compute_node_CI(top_CI_vertex_id, G, rad,
                                 k_map=k_in,
                                 num_vertices=num_vertices)
                G.set_reversed(False)
                
            vertices_to_update.remove(top_CI_vertex_id)

            # if the updated value is smaller than the 
            # current top in the heap, push back in the heap
            if top_CI_val < - heap[0][0]:
                heapq.heappush(heap, (-top_CI_val, top_CI_vertex_id))
                
                # go back to top of iteration
                continue

        # remove top_CI_vertex
        CImap_final[top_CI_vertex_id] = top_CI_val
        CI_ranking.append(top_CI_vertex_id)
        
        #update list of vertices to update
        ball_nodes = get_ball(G,top_CI_vertex_id,rad+1, num_vertices)
        vertices_to_update.update(ball_nodes)
        
        filt[top_CI_vertex_id] = False
        G.set_vertex_filter(filt)
        
        # Update degree maps            
        k_out[top_CI_vertex_id] = 0
        k_in[top_CI_vertex_id] = 0
        for neigh in G.get_out_neighbours(top_CI_vertex_id):
            k_in[neigh] -= 1
                
        if direction == 'undir':
            k_out = k_in + k_out
            k_in[:] = 0
        
        if not len(heap)%step_check and len(heap) != lheap:
            lheap = len(heap)
            # update gc_fraq every step_check
            if G.num_vertices() > 0:
                gc_fraq = float(gt.label_largest_component(G,
                                directed=False).a.sum())/num_vertices
            else:
                gc_fraq = 0
                                                               
            if verbose:
                print('  + number of removed nodes: ' + str(num_vertices - len(heap)))
                print('  + GC fraq: ' + str(gc_fraq))
                print('  + top CI val: ' + str(top_CI_val))
                print('  + in ' + '{:.2f}'.format(time.time()-t2) + 's')
        
        if top_CI_val < 1:
            break
        
    if verbose:
        print('*** done in ' + '{:.2f}'.format(time.time()-t2) + 's')   
        print('*** total time: ' + '{:.2f}'.format(time.time()-t0) + 's')   
        
    # put back in initial state    
    G.set_vertex_filter(None)
    
    G.set_directed(is_init_directed)
    G.set_reversed(is_init_reversed)
    
    return CI_ranking, np.asarray(CImap_final)
    
