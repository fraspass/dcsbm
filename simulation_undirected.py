#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from sklearn.metrics import adjusted_rand_score as ari
import dcsbm, bic_fit

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

## Parse arguments
args = parser.parse_args()

## Arguments
n = args.n
K = args.K
M_sim = 100
m = 10
print('Number of nodes:', str(n))
print('Number of communities:', str(K))

## Obtain maximum
def find_max(x):
    qq = np.where(x == np.max(x))
    qq0 = qq[0][0] + 1
    qq1 = qq[1][0] + 2
    return np.array([qq0, qq1])

## Set seed to repeat the simulation
np.random.seed(111)
## Degree corrections
rho = np.random.beta(a=1,b=1,size=n)
## Set seed
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
est_ari = {}
bic_K = {}
bic_ari = {}
embs = {}
for t in [None, 'normalised', 'theta']:
    est_d[t] = []
    est_K[t] = []
    est_ari[t] = []
    bic_K[t] = []
    bic_ari[t] = []

## Matrix of probabilities
## Bs = np.zeros((M_sim,K,K))
B = np.array([0.15,0.1,0.1,0.3]).reshape(2,2)

## Repeat M_sim times
for s in range(M_sim):
    try:
        ## B = np.tril(np.random.uniform(size=(K,K)))
        ## B += np.tril(B,k=-1).T
        ## Bs[s] = B
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
        ## Embeddings
        embs[None] = X
        embs['normalised'] = np.divide(X, np.linalg.norm(X, axis=1).reshape(-1,1))
        embs['theta'] = dcsbm.theta_transform(X)
        ## Model setup
        for d in range(1,6):
            for k in range(2,7):
                for t in [None, 'normalised', 'theta']:
                    print('\rSimulated graph: ' + str(s+1) + '\td: ' + str(d) + '\tK: ' + str(k), end='')
                    M = dcsbm.EGMM(K=k)
                    z_est = M.fit_predict_approximate(X,d=d,transformation=t)
                    bics[t][s,d-1,k-2] += [M.BIC()]
                    aris[t][s,d-1,k-2] += [ari(z_est,zz)]
        for t in [None, 'normalised', 'theta']:
            dK = find_max(bics[t][s])
            est_d[t] += [dK[0]]
            est_K[t] += [dK[1]]
            est_ari[t] += [aris[t][s][dK[0]-1,dK[1]-2]]
            if dK[0] == 1:
                U = embs[t][:,:dK[0]].reshape(-1,1)
            else:
                U = embs[t][:,:dK[0]]
            dK_bic = bic_fit.GMM_bic(X=U, K_star=6)
            bic_K[t] += [dK_bic[1]]
            bic_ari[t] += [ari(dK_bic[0],zz)]
    except:
        continue

## Save probability matrices
np.save('Results/B_' + str(args.s) + '.npy', Bs)
np.save('Results/rho_' + str(args.s) + '.npy', rho)

for s in range(M_sim):
    try:
        ## No transformation
        dK_none = find_max(bics[None][s])
        est_d[None] += [dK_none[0]]
        est_K[None] += [dK_none[1]]
        est_ari[None] += [aris[None][s][dK_none[0]-1,dK_none[1]-2]]
        ## Normalised
        dK_norm = find_max(bics['normalised'][s]) 
        est_d['normalised'] += [dK_norm[0]]
        est_K['normalised'] += [dK_norm[1]]
        est_ari['normalised'] += [aris['normalised'][s][dK_norm[0]-1,dK_norm[1]-2]]
        ## Theta
        dK_theta = find_max(bics['theta'][s]) 
        est_d['theta'] += [dK_theta[0]]
        est_K['theta'] += [dK_theta[1]]
        est_ari['theta'] += [aris['theta'][s][dK_theta[0]-1,dK_theta[1]-2]]
    except:
        continue

for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results/out_d_' + label + '_' + str(K) + '_' + str(n) + '_' + str(args.s) + '.csv', est_d[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_K_' + label + '_' + str(K) + '_' + str(n) + '_' + str(args.s) + '.csv', est_K[t], fmt='%i', delimiter=',')
    np.savetxt('Results/est_ari_' + label + '_' + str(K) + '_' + str(n) + '_' + str(args.s) + '.csv', est_ari[t], fmt='%.6f', delimiter=',')