#! /usr/bin/env python3
import numpy as np
import argparse
from sklearn.metrics import adjusted_rand_score as ari
import dcsbm

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument("-K", type=int, dest="K", default=2, const=True, nargs="?",\
	help="Integer: number of communities, default 2.")
parser.add_argument("-M", type=int, dest="M", default=25, const=True, nargs="?",\
	help="Integer: number of simulations, default M.")
parser.add_argument("-s", type=int, dest="s", default=171171, const=True, nargs="?",\
	help="Integer: seed, default 171171.")

## Parse arguments
args = parser.parse_args()

#############################################################
## Reproduces results in Section 6.1 for undirected DCSBMs ##
#############################################################

## Arguments
K = args.K
M_sim = args.M
m = 10

## Values of n
ns = [100, 200, 500, 1000, 2000]
n = int(np.max(ns))
n_max = int(np.max(ns))

## Summary
print('Number of nodes:', str(n_max))
print('Number of communities:', str(K))

## Obtain maximum
def find_max(x):
    qq = np.where(x == np.max(x))
    qq0 = qq[0][0] + 1
    qq1 = qq[1][0] + 2
    return np.array([qq0, qq1])

## Set seed to repeat the simulation
np.random.seed(111)
## Set seed
q = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])
z = np.zeros(n,dtype=int)
for k in range(K):
    z[q[k]:] = k

## Randomly shuffle
np.random.seed(171171)
np.random.shuffle(z)

## BICs and ARIs
bics = {}
aris = {}
for t in [None, 'normalised', 'theta']:
    for s in range(M_sim):
        for n in ns:
            bics[t,s,n] = np.zeros((5,5))
            aris[t,s,n] = np.zeros((5,5))

## Results
est_d = {}
est_K = {}
est_ari = {}
embs = {}
z_est = {}
z_est_temp = {}

## Matrix of probabilities
Bs = np.zeros((M_sim,K,K))

## Set seed
np.random.seed(args.s)
## Repeat M_sim times
for s in range(M_sim):
    B = np.tril(np.random.beta(a=1,b=1,size=(K,K)))
    B += np.tril(B,k=-1).T
    Bs[s] = B

## Set seed (again)
np.random.seed(args.s)

## Repeat M_sim times
for s in range(M_sim):
    A = np.zeros((n_max,n_max))
    ## Degree corrections
    rho = np.random.beta(a=2,b=1,size=n_max)
    B = Bs[s]
    for i in range(n_max-1):
        for j in range(i+1,n_max):
            A[i,j] = np.random.binomial(n=1,p=rho[i]*rho[j]*B[z[i],z[j]],size=1)
            A[j,i] = A[i,j]
    for n in ns: 
        Lambda, Gamma = np.linalg.eigh(A[:n,:n])
        k = np.argsort(np.abs(Lambda))[::-1][:m]
        X = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
        ## Remove empty rows
        zero_index = np.array(A[:n,:n].sum(axis=0),dtype=int)
        X = X[zero_index > 0]
        zz = z[:n][zero_index > 0]
        A_mat = A[:n,:n][zero_index > 0][:,zero_index > 0]
        ## Embeddings
        embs[None,s,n] = X
        embs['normalised',s,n] = np.divide(X, np.linalg.norm(X, axis=1).reshape(-1,1))
        embs['theta',s,n] = dcsbm.theta_transform(X)
        ## Model setup
        for d in range(1,6):
            for k in range(2,7):
                for t in [None, 'normalised', 'theta']:
                    if t is None:
                        method = 'Standard ASE'
                    elif t == 'normalised':
                        method = 'Row-normalised ASE'
                    else:
                        method = 'Spherical coordinates'
                    print('\rNumber of nodes: ' + str(n) + '\tSimulated graph: ' + str(s+1) + '\td: ' + str(d) + '\tK: ' + str(k) + '\tMethod: ' + method, end='\t\t\t')
                    M = dcsbm.EGMM(K=k)
                    z_est_temp[t,d-1,k-2] = M.fit_predict_approximate(X,d=d,transformation=t)
                    bics[t,s,n][d-1,k-2] = M.BIC()
                    aris[t,s,n][d-1,k-2] = ari(z_est_temp[t,d-1,k-2],zz)
        ## Obtain estimates
        for t in [None, 'normalised', 'theta']:
            dK = find_max(bics[t,s,n])
            est_d[t,s,n] = dK[0]
            est_K[t,s,n] = dK[1]
            z_est[t,s,n] = z_est_temp[t,dK[0]-1,dK[1]-2]
            est_ari[t,s,n] = aris[t,s,n][dK[0]-1,dK[1]-2]

## Calculate output
d_scores = {}
K_scores = {}
ari_scores = {}
for t in [None, 'normalised', 'theta']:
    d_scores[t] = np.zeros((M_sim, len(ns)))
    K_scores[t] = np.zeros((M_sim, len(ns)))
    ari_scores[t] = np.zeros((M_sim, len(ns)))
    for s in range(M_sim):
        for n in range(len(ns)):
            d_scores[t][s,n] = est_d[t,s,ns[n]]
            K_scores[t][s,n] = est_K[t,s,ns[n]]
            ari_scores[t][s,n] = est_ari[t,s,ns[n]]

## Save output
for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results/out_d_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', d_scores[t], fmt='%i', delimiter=',')
    np.savetxt('Results/out_K_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', K_scores[t], fmt='%i', delimiter=',')
    np.savetxt('Results/est_ari_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', ari_scores[t], fmt='%.6f', delimiter=',')