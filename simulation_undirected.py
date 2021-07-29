#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from sklearn.metrics import adjusted_rand_score as ari
import dcsbm
from bic_fit import GMM_bic

############################################################
## Reproduces the results for Table 1 (undirected DCSBMs) ##
############################################################

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, dest="n", default=1000, const=True, nargs="?",\
	help="Integer: number of nodes, default 1000")
parser.add_argument("-K", type=int, dest="K", default=2, const=True, nargs="?",\
	help="Integer: number of communities, default 2")
parser.add_argument("-s", type=int, dest="s", default=171171, const=True, nargs="?",\
	help="Integer: seed, default 171171")

## Obtain maximum
def find_max(x):
    qq = np.where(x == np.max(x))
    qq0 = qq[0][0] + 1
    qq1 = qq[1][0] + 2
    return np.array([qq0, qq1])

## Parse arguments
args = parser.parse_args()

## Arguments
n = args.n
K = args.K
M_sim = 250
m = 10
print('Number of nodes:', str(n))
print('Number of communities:', str(K))

## Set seed to repeat the simulation
np.random.seed(args.s)
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

## Results
est_d = {}
est_K = {}
bic_K = {}
est_ari = {}
bic_ari = {}
for t in [None, 'normalised', 'theta']:
    est_d[t] = []
    est_K[t] = []
    bic_K[t] = []
    est_ari[t] = []
    bic_ari[t] = []

## Repeat M_sim times
for s in range(M_sim):
    print('\rSimulated graph: ' + str(s+1), end='')
    try:
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
        ## Embeddings
        X = X[zero_index > 0]
        X_tilde = np.divide(X, np.linalg.norm(X,axis=1).reshape(-1,1))
        Theta = dcsbm.theta_transform(X)
        ## Filter indices
        zz = z[zero_index > 0]
        ## Model setup
        for d in range(1,6):
            for k in range(2,7):
                for t in [None, 'normalised', 'theta']:
                    M = dcsbm.EGMM(K=k)
                    z_est = M.fit_predict_approximate(X,d=d,transformation=t)
                    bics[t][s,d-1,k-2] += [M.BIC()]
                    aris[t][s,d-1,k-2] += [ari(z_est,zz)]
        ## Find estimates
        ## No transformation
        dK_none = find_max(bics[None][s])
        est_d[None] += [dK_none[0]]
        est_K[None] += [dK_none[1]]
        est_ari[None] += [aris[None][s][dK_none[0]-1,dK_none[1]-2]]
        U = GMM_bic(X[:,:dK_none[0]], K_star=10, n_init=5)
        bic_K[None] += [U[1]]
        bic_ari[None] += [ari(U[0],zz)]
        ## Normalised
        dK_norm = find_max(bics['normalised'][s]) 
        est_d['normalised'] += [dK_norm[0]]
        est_K['normalised'] += [dK_norm[1]]
        est_ari['normalised'] += [aris['normalised'][s][dK_norm[0]-1,dK_norm[1]-2]]
        U = GMM_bic(X_tilde[:,:dK_norm[0]], K_star=10, n_init=5)
        bic_K['normalised'] += [U[1]]
        bic_ari['normalised'] += [ari(U[0],zz)]
        ## Theta
        dK_theta = find_max(bics['theta'][s]) 
        est_d['theta'] += [dK_theta[0]]
        est_K['theta'] += [dK_theta[1]]
        est_ari['theta'] += [aris['theta'][s][dK_theta[0]-1,dK_theta[1]-2]]
        U = GMM_bic(X[:,:dK_theta[0]], K_star=10, n_init=5)
        bic_K['theta'] += [U[1]]
        bic_ari['theta'] += [ari(U[0],zz)]
    except:
        continue

for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results/out_d_' + label + '_' + str(K) + '_' + str(n) + '.csv', est_d[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_K_' + label + '_' + str(K) + '_' + str(n) + '.csv', est_K[t], fmt='%i', delimiter=',')
    np.savetxt('Results/est_ari_' + label + '_' + str(K) + '_' + str(n) + '.csv', est_ari[t], fmt='%.6f', delimiter=',')