#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score as ari
import dcsbm

############################################################
## Reproduces the results for Table 1 (undirected DCSBMs) ##
############################################################

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, dest="n", default=1000, const=True, nargs="?",\
	help="Integer: number of nodes, default 1000")
parser.add_argument("-K", type=int, dest="K", default=2, const=True, nargs="?",\
	help="Integer: number of communities, default 2")

## Parse arguments
args = parser.parse_args()

## Arguments
n = args.n
K = args.K
M_sim = 250
m = 10

## Set seed to repeat the simulation
np.random.seed(171171)
q = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])
z = np.zeros(n,dtype=int)
for k in range(K):
    z[q[k]:] = k

## BICs and ARIs
bics = {}
aris = {}
for t in [None, 'normalised', 'theta']:
    bics[t] = np.zeros(shape=(M_sim,5,5))
    aris[t] = np.zeros(shape=(M_sim,5,5))

## Repeat M_sim times
for s in range(M_sim):
    print('\rSimulated graph: ' + str(s+1), end='')
    B = np.tril(np.random.uniform(size=(K,K)))
    B += np.tril(B,k=-1).T
    rho = np.random.beta(a=2,b=1,size=n)
    ## Construct the adjacency matrix
    A = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            A[i,j] = np.random.binomial(n=1,p=rho[i]*rho[j]*B[z[i],z[j]],size=1)
            A[j,i] = A[i,j]
    Lambda, Gamma = np.linalg.eigh(A)
    k = np.argsort(np.abs(Lambda))[::-1][:10]
    X = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
    ## Remove empty rows
    zero_index = np.array(A.sum(axis=0),dtype=int)
    X = X[zero_index > 0]
    zz = z[zero_index > 0]
    ## Model setup
    for d in range(1,6):
        for k in range(2,7):
            for t in [None, 'normalised', 'theta']:
                M = dcsbm.EGMM(K=k)
                z_est = M.fit_predict_approximate(X,d=d,transformation=t)
                bics[t][s,d-1,k-2] += [M.BIC()]
                aris[t][s,d-1,k-2] += [ari(z_est,zz)]

## Obtain maximum
def find_max(x):
    qq = np.where(x == np.max(x))
    return np.array([qq[0][0], qq[1][0]])

## Results
true_d = {}
true_K = {}
avg_ari = {}
for t in [None, 'normalised', 'theta']:
    true_d[t] = []
    true_K[t] = []
    avg_ari[t] = []

for s in range(M_sim):
    ## No transformation
    dK_none = find_max(bics[None][s])
    true_d[None] += [dK_none[0]]
    true_K[None] += [dK_none[1]]
    avg_ari[None] += [aris[None][s][dK_none[0],dK_none[1]]]
    ## Normalised
    dK_norm = find_max(bics['normalised'][s]) 
    true_d['normalised'] += [dK_norm[0]]
    true_K['normalised'] += [dK_norm[1]]
    avg_ari['normalised'] += [aris['normalised'][s][dK_norm[0],dK_norm[1]]]
    ## Theta
    dK_theta = find_max(bics['theta'][s]) 
    true_d['theta'] += [dK_theta[0]]
    true_K['theta'] += [dK_theta[1]]
    avg_ari['theta'] += [aris['theta'][s][dK_theta[0],dK_theta[1]]]

for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results/out_d_' + label + '_' + str(K) + '_' + str(n) + '.csv', true_d[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_K_' + label + '_' + str(K) + '_' + str(n) + '.csv', true_K[t], fmt='%i', delimiter=',')
    np.savetxt('Results/avg_ari_' + label + '_' + str(K) + '_' + str(n) + '.csv', avg_ari[t], fmt='%.6f', delimiter=',')