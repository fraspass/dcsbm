#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import scale
from scipy.stats import chi2, norm
from sklearn.metrics import adjusted_rand_score as ari
import dcsbm

#############################################
## Reproduces Table 1 for bipartite DCSBMs ##
#############################################

## Arguments
n = 1000
n_prime = 1500
M_sim = 250
K = 2
K_prime = 3
m = 10

## Set seed to repeat the simulation
np.random.seed(171171)
q = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])
q_prime = np.array([int(x) for x in np.linspace(0,n_prime,num=K_prime,endpoint=False)])
z = np.zeros(n,dtype=int)
z_prime = np.zeros(n_prime,dtype=int)
for k in range(K):
    z[q[k]:] = k

for k in range(K_prime):
    z_prime[q_prime[k]:] = k

bics = {}
aris = {}
bics_prime = {}
aris_prime = {}
for t in [None, 'normalised', 'theta']:
    bics[t] = np.zeros(shape=(M_sim,5,5))
    aris[t] = np.zeros(shape=(M_sim,5,5))
    bics_prime[t] = np.zeros(shape=(M_sim,5,5))
    aris_prime[t] = np.zeros(shape=(M_sim,5,5))

for s in range(M_sim):
    print('\rSimulation: ' + str(s), end='')
    B = np.tril(np.random.uniform(size=(K,K_prime)))
    rho = np.random.beta(a=2,b=1,size=n)
    rho_prime = np.random.beta(a=2,b=1,size=n_prime)
    ## Construct the adjacency matrix
    rows = []
    cols = []
    for i in range(n):
        for j in range(n_prime):
            if np.random.binomial(n=1,p=rho[i]*rho_prime[j]*B[z[i],z_prime[j]],size=1) == 1:
                rows += [i]
                cols += [j]
    ## Obtain the adjacency matrix and the embeddings
    A = coo_matrix((np.repeat(1.0,len(rows)),(rows,cols)),shape=(n,n_prime))
    U, S, V = svds(A, k=m)
    U, S, V = svds(A, k=m)
    X = np.dot(U[:,::-1], np.diag(np.sqrt(S[::-1])))
    Y = np.dot(V.T[:,::-1], np.diag(np.sqrt(S[::-1])))
    ## Remove empty rows
    zero1 = np.array(A.sum(axis=1),dtype=int)[:,0]
    zero2 = np.array(A.sum(axis=0),dtype=int)[0]
    X = X[zero1 > 0]
    Y = Y[zero2 > 0]
    zz1 = z[zero1 > 0]
    zz2 = z_prime[zero2 > 0]
    ## Model setup
    for d in range(1,6):
        for k in range(2,7):
            for t in [None, 'normalised', 'theta']:
                M = dcsbm.EGMM(K=k)
                ## Estimate
                z_est1 = M.fit_predict_approximate(X,d=d,transformation=t)
                bics[t][s,d-1,k-2] += [M.BIC()]
                aris[t][s,d-1,k-2] += [ari(z_est1,zz1)]
                z_est2 = M.fit_predict_approximate(Y,d=d,transformation=t)
                bics_prime[t][s,d-1,k-2] += [M.BIC()]
                aris_prime[t][s,d-1,k-2] += [ari(z_est2,zz2)]

def find_max(x):
    qq = np.where(x == np.max(x))
    return (qq[0][0], qq[1][0])

## Results
true_d = {}
true_K = {}
avg_ari = {}
true_d_prime = {}
true_K_prime = {}
avg_ari_prime = {}
for t in [None, 'normalised', 'theta']:
    true_d[t] = []
    true_K[t] = []
    avg_ari[t] = []
    true_d_prime[t] = []
    true_K_prime[t] = []
    avg_ari_prime[t] = []

for s in range(M_sim):
    true_d[None] += [find_max(bics[None][s])[0]]
    true_K[None] += [find_max(bics[None][s])[1]]
    true_d_prime[None] += [find_max(bics_prime[None][s])[0]]
    true_K_prime[None] += [find_max(bics_prime[None][s])[1]]
    avg_ari[None] += [aris[None][s][find_max(bics[None][s])]]
    avg_ari_prime[None] += [aris_prime[None][s][find_max(bics_prime[None][s])]]
    true_d['normalised'] += [find_max(bics['normalised'][s])[0]]
    true_K['normalised'] += [find_max(bics['normalised'][s])[1]]
    true_d_prime['normalised'] += [find_max(bics_prime['normalised'][s])[0]]
    true_K_prime['normalised'] += [find_max(bics_prime['normalised'][s])[1]]
    avg_ari['normalised'] += [aris['normalised'][s][find_max(bics['normalised'][s])]]
    avg_ari_prime['normalised'] += [aris_prime['normalised'][s][find_max(bics_prime['normalised'][s])]]
    true_d['theta'] += [find_max(bics['theta'][s])[0]]
    true_K['theta'] += [find_max(bics['theta'][s])[1]]
    true_d_prime['theta'] += [find_max(bics_prime['theta'][s])[0]]
    true_K_prime['theta'] += [find_max(bics_prime['theta'][s])[1]]
    avg_ari['theta'] += [aris['theta'][s][find_max(bics['theta'][s])]]
    avg_ari_prime['theta'] += [aris_prime['theta'][s][find_max(bics_prime['theta'][s])]]

for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results/out_bipartite_d_' + label + '.csv', true_d[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_bipartite_K_' + label + '.csv', true_K[t], fmt='%i', delimiter=',')
    np.savetxt('Results/avg_bipartite_ari_' + label + '.csv', avg_ari[t], fmt='%.6f', delimiter=',')
    np.savetxt('Results/out_bipartite_d_prime_' + label + '.csv', true_d_prime[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_bipartite_K_prime_' + label + '.csv', true_K_prime[t], fmt='%i', delimiter=',')
    np.savetxt('Results/avg_bipartite_ari_prime_' + label + '.csv', avg_ari_prime[t], fmt='%.6f', delimiter=',')