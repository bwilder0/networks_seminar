def greedy(g, p, budget):
    import networkx as nx
    import heapq
    num_nodes = len(g)
    allowed_nodes = range(num_nodes)
    P = nx.to_numpy_matrix(g) * p
    rr_sets = make_rr_sets(P, 100, range(len(g)))
    S = set()
    upper_bounds = [(-eval_node_rr(u, S, num_nodes, rr_sets), u) for u in allowed_nodes]    
    heapq.heapify(upper_bounds)
    starting_objective = 0
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = eval_node_rr(u, S, num_nodes, rr_sets)
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective


def eval_node_rr(u, S, num_nodes, rr_sets):
    if hasattr(u, '__iter__'):
        S = S.union(u)
    else:
        S.add(u)
        
    val = 0.
    for r in rr_sets:
        if not r.isdisjoint(S):
            val += 1
    val /= len(rr_sets)
    val *= num_nodes
    
    if hasattr(u, '__iter__'):
        S.difference_update(u)
    else:
        S.remove(u)
    
    return val
    
def make_rr_sets(P, num_sets, counted_nodes):
    import numpy as np
    G = construct_reverse_adjacency(P)
    max_degree = max([len(x) for x in G])
    G_array = np.zeros((P.shape[0], max_degree), dtype=np.int)
    G_array[:] = -1
    for u in range(P.shape[0]):
        G_array[u, 0:len(G[u])] = G[u]
    results = simulate_rr(num_sets, P, G_array, np.array(counted_nodes, dtype=np.int))
    rr_sets = []
    for i in range(num_sets*len(counted_nodes)):
        new_set = set()
        new_set.update(np.where(results[i] == 1)[0])
        rr_sets.append(new_set)
    return rr_sets
    
def construct_reverse_adjacency(U):
    '''
    Converts an adjacency matrix into a reversed adjacency list
    '''
    G = []
    for u in range(U.shape[0]):
        G.append([])
        for v in range(U.shape[0]):
            if U[v,u] != 0:
                G[u].append(v)
    return G

def simulate_rr(num_iterations, P, G, sources):
    '''
    Generates reverse reachability sets under the ICM. Returns num_iterations sets for each node in sources.
    
    P: nxn array giving propagation probabilities -- this is read in reverse
    
    G: adjacency list representation of the graph with the edges reversed already. This is an array of dimension
    n x max degree, where G[i,j] gives the jth neighbor of i in the reversed graph. For j > degree(i), this is -1.
    
    '''    
    import numpy as np
    import random
    numnodes = P.shape[0]
    state = np.zeros((num_iterations*sources.shape[0], numnodes))
    curr_active = []    #stack which keeps nodes which were just activated
    for iter_num in range(num_iterations*sources.shape[0]): #generate num_iterations sets for each node in sources
        state[iter_num, sources[iter_num / num_iterations]] = 1  #set the current source node
        curr_active.append(sources[iter_num / num_iterations])
        while len(curr_active) > 0:   #keep triggering new activations until the cascade ends
            j = curr_active.pop()
            for neighbor in range(G.shape[1]):
                k = G[j, neighbor]
                if k == -1: #j has no more neighbors
                    break
                if state[iter_num, k] == 1:   #neighbor has already been activated
                    continue
                if random.random() <= P[k, j]:  #attempt activation
                    state[iter_num, k] = 1
                    curr_active.append(k)
    return state

