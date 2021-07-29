#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_rand_score as ARI
import dcsbm

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

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
filter_ips = np.loadtxt('Data/hu_ch_sp_cx_ips.csv',delimiter=',',dtype=str)[:,0]

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

## Obtain covariates
covs = np.loadtxt('Data/college_nodes_map_covs.csv',delimiter=',',dtype=str)
covs = covs[np.array([np.where(covs[:,0] == college_inverse_map[k])[0][0] for k in range(k_college)])][:,2]
d = dict(zip(set(covs), range(len(covs))))
labs = np.array([d[x] for x in covs])

## Obtain the adjacency matrix and the embeddings
A = coo_matrix((np.repeat(1.0,len(rows)),(rows,cols)),shape=(k_college,k_server))
dd = np.array(A.sum(axis=1))[:,0]

fig, ax = plt.subplots(figsize=(4.25,3.25))
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
lss = [':','-.','-','--']
group = ['Mathematics','Medicine','Chemistry','Civil Engineering']
for g in [2,3,0,1]:
    ix = np.where(labs == g)
    ax.hist(dd[labs == g], bins=10, color = cdict[g], alpha=0.25)

for g in [2,3,0,1]:
    ix = np.where(labs == g)  
    ax.hist(dd[labs == g], bins=10, histtype=u'step', linestyle = lss[g], edgecolor=cdict[g], linewidth=2, label = group[g])

ax.legend()
plt.xlabel('Out-degree')
plt.ylabel('Frequency')
plt.savefig("out_icl.pdf",bbox_inches='tight')
plt.show()

np.random.seed(117)
n = 1000
K = 4
z = np.array(list(range(4))*250)
B = np.ones((4,4))*.25 + np.array([.5,.25,.1,0])*np.diag(np.ones(4))
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i+1,n):
        A[i,j] = np.random.choice(2,size=1,p=[1-B[z[i],z[j]],B[z[i],z[j]]])
        A[j,i] = A[i,j]

dd = np.array(A.sum(axis=1))

fig, ax = plt.subplots(figsize=(4.25,3.25))
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
lss = [':','-.','-','--']
for g in [2,3,0,1]:
    ix = np.where(z == g)
    ax.hist(dd[z == g], bins=10, color = cdict[g], alpha=0.25)

for g in [2,3,0,1]:
    ix = np.where(z == g)  
    ax.hist(dd[z == g], bins=10, histtype=u'step', linestyle = lss[g], edgecolor=cdict[g], linewidth=2)

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.savefig("out_sbm.pdf",bbox_inches='tight')
plt.show()

np.random.seed(117)
rho = np.random.beta(size=n,a=2,b=4)
for i in range(n-1):
    for j in range(i+1,n):
        A[i,j] = np.random.choice(2,size=1,p=[1-rho[i]*rho[j]*B[z[i],z[j]],rho[i]*rho[j]*B[z[i],z[j]]])
        A[j,i] = A[i,j]

dd = np.array(A.sum(axis=1))

fig, ax = plt.subplots(figsize=(4.25,3.25))
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
lss = [':','-.','-','--']
for g in [2,3,0,1]:
    ix = np.where(z == g)
    ax.hist(dd[z == g], bins=10, color = cdict[g], alpha=0.25)

for g in [2,3,0,1]:
    ix = np.where(z == g)  
    ax.hist(dd[z == g], bins=10, histtype=u'step', linestyle = lss[g], edgecolor=cdict[g], linewidth=2)

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.savefig("out_dcsbm.pdf",bbox_inches='tight')
plt.show()