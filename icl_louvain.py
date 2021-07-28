#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import adjusted_rand_score as ARI

## Parser to give parameter values
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

## Set tolerance
parser.add_argument("-g", "--graph", dest="graph", default = 'icl2', type=str,
	help="Type of graph: icl1, icl2 or icl3.")

## Parse arguments
args = parser.parse_args()
graph = args.graph

## Bipartite graph
college_map = {}
college_inverse_map = {}
server_map = {}
server_inverse_map = {}
rows = []
cols = []

## Obtain the graph
k_college = 0
k_server = 0
G = Counter()
if graph == 'icl1':
    filter_ips = np.loadtxt('Data/hu_sp_ips.csv',delimiter=',',dtype=str)[:,0]
elif graph == 'icl2':
    filter_ips = np.loadtxt('Data/hu_ch_sp_cx_ips.csv',delimiter=',',dtype=str)[:,0]
elif graph == 'icl3':
    filter_ips = np.loadtxt('Data/hu_bl_sp_ee_cg_ips.csv',delimiter=',',dtype=str)[:,0]
else:
    raise ValueError('Invalid graph.')

## Obtain the full graph
with open('Data/college_edges_filter.csv') as f:
    for line in f:
        line = line.rstrip('\r\n').split(',')
        link = tuple(line[:2])
        src = link[0]
        dst = link[1]
        if src in filter_ips:
            G[src,dst] += 1
            if src not in college_map:
                college_map[src] = k_college
                college_inverse_map[k_college] = src
                k_college += 1
            rows += [college_map[src]]
            if dst not in server_map:
                server_map[dst] = k_server
                server_inverse_map[k_server] = dst
                k_server += 1
            cols += [server_map[dst]]

## Obtain labels
covs = np.loadtxt('Data/college_nodes_map_covs.csv',delimiter=',',dtype=str)
covs = covs[np.array([np.where(covs[:,0] == college_inverse_map[k])[0][0] for k in range(k_college)])][:,2]
d = dict(zip(set(covs), range(len(covs))))
true_labs = [d[x] for x in covs]
K = len(np.unique(true_labs))

## Obtain the adjacency matrix and the embeddings
A = csr_matrix(coo_matrix((np.repeat(1.0,len(rows)),(rows,cols)),shape=(k_college,k_server)))
## Set seed
np.random.seed(117)

## Alternative methods (community detection from adjacency matrix)
from sknetwork.hierarchy import BiLouvainHierarchy, BiParis, cut_straight
from sknetwork.clustering import BiLouvain
biparis = BiParis()
bilouvain = BiLouvain()
bihlouvain = BiLouvainHierarchy()
dendrogram_paris = biparis.fit_transform(A)
dendrogram_louvain = bihlouvain.fit_transform(A)
z_paris = cut_straight(dendrogram_paris, n_clusters=K)
z_louvain = bilouvain.fit_transform(A)
z_hlouvain = cut_straight(dendrogram_louvain, n_clusters=K)
print('Paris:\t', ARI(z_paris, true_labs))
print('Louvain:\t', ARI(z_louvain, true_labs))
print('HLouvain:\t', ARI(z_hlouvain, true_labs))