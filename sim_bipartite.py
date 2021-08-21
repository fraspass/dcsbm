#! /usr/bin/env python3
import numpy as np
import argparse
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_rand_score as ari
from scipy.sparse import coo_matrix
import dcsbm

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument("-M", type=int, dest="M", default=25, const=True, nargs="?",\
	help="Integer: number of simulations, default M.")
parser.add_argument("-s", type=int, dest="s", default=171171, const=True, nargs="?",\
	help="Integer: seed, default 171171.")

## Parse arguments
args = parser.parse_args()

#############################################################
## Reproduces results in Section 6.1 for bipartite DCScBMs ##
#############################################################

## Arguments
ns = [100, 200, 500, 1000, 2000]
ns_prime = [150, 300, 750, 1500, 3000]
M_sim = args.M
K = 2
K_prime = 3
m = 10

## Set maximum number of nodes
n = int(np.max(ns))
n_max = int(np.max(ns))
n_prime = int(np.max(ns_prime))
n_max_prime = int(np.max(ns_prime))

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
q_prime = np.array([int(x) for x in np.linspace(0,n_prime,num=K_prime,endpoint=False)])
z = np.zeros(n,dtype=int)
z_prime = np.zeros(n_prime,dtype=int)
for k in range(K):
    z[q[k]:] = k

for k in range(K_prime):
    z_prime[q_prime[k]:] = k

## Randomly shuffle
np.random.seed(171171)
np.random.shuffle(z)
np.random.shuffle(z_prime)

## BICs and ARIs
bics = {}
aris = {}
bics_prime = {}
aris_prime = {}
for t in [None, 'normalised', 'theta']:
    for s in range(M_sim):
        for n in ns:
            bics[t,s,n] = np.zeros(shape=(5,5))
            aris[t,s,n] = np.zeros(shape=(5,5))
        for n in ns_prime:
            bics_prime[t,s,n] = np.zeros(shape=(5,5))
            aris_prime[t,s,n] = np.zeros(shape=(5,5))

## Results
est_d = {}; est_d_prime = {}
est_K = {}; est_K_prime = {}
est_ari = {}; est_ari_prime = {}
embs = {}; embs_prime = {}
z_est = {}; z_est_prime = {}
z_est_temp = {}; z_est_temp_prime = {}

## Matrix of probabilities
Bs = np.zeros((M_sim,K,K_prime))

## Set seed
np.random.seed(args.s)
## Repeat M_sim times
for s in range(M_sim):
    Bs[s] = np.random.beta(a=1,b=1,size=(K,K_prime))

## Set seed (again)
np.random.seed(args.s)

## Repeat M_sim times
for s in range(M_sim):
    A = np.zeros((n_max,n_max_prime))
    B = Bs[s]
    ## Degree corrections
    rho = np.random.beta(a=2,b=1,size=n_max)
    rho_prime = np.random.beta(a=2,b=1,size=n_max_prime)
    ## Construct the adjacency matrix
    rows = []
    cols = []
    for i in range(n_max):
        for j in range(n_max_prime):
            if np.random.binomial(n=1,p=rho[i]*rho_prime[j]*B[z[i],z_prime[j]],size=1) == 1:
                rows += [i]
                cols += [j]
    ## Obtain the adjacency matrix and the embeddings
    A = coo_matrix((np.repeat(1.0,len(rows)),(rows,cols)),shape=(n,n_prime)).todense()
    for q in range(len(ns)): 
        U, S, V = svds(A[:ns[q],:ns_prime[q]], k=m)
        X = np.dot(U[:,::-1], np.diag(np.sqrt(S[::-1])))
        Y = np.dot(V.T[:,::-1], np.diag(np.sqrt(S[::-1])))
        ## Remove empty rows
        zero_index = np.array(A[:ns[q],:ns_prime[q]].sum(axis=1),dtype=int).reshape(-1)
        zero_index_prime = np.array(A[:ns[q],:ns_prime[q]].sum(axis=0),dtype=int).reshape(-1)
        X = X[zero_index > 0]
        zz = z[:ns[q]][zero_index > 0]
        Y = Y[zero_index_prime > 0]
        zz_prime = z_prime[:ns_prime[q]][zero_index_prime > 0]
        A_mat = A[:ns[q],:ns_prime[q]][zero_index > 0][:,zero_index_prime > 0]
        ## Embeddings
        embs[None,s,ns[q]] = X
        embs_prime[None,s,ns[q]] = Y
        embs['normalised',s,ns[q]] = np.divide(X, np.linalg.norm(X, axis=1).reshape(-1,1))
        embs_prime['normalised',s,ns_prime[q]] = np.divide(Y, np.linalg.norm(Y, axis=1).reshape(-1,1))
        embs['theta',s,ns_prime[q]] = dcsbm.theta_transform(X)
        embs_prime['theta',s,ns_prime[q]] = dcsbm.theta_transform(Y)
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
                    print('\rNumber of nodes: (' + str(ns[q]) + ', ' + str(ns_prime[q]) + ')\tSimulated graph: ' + str(s+1) + '\td: ' + str(d) + '\tK: ' + str(k) + '\tMethod: ' + method, end='\t\t\t', sep='')
                    ## Model
                    M = dcsbm.EGMM(K=k)
                    ## Source nodes
                    z_est_temp[t,d-1,k-2] = M.fit_predict_approximate(X,d=d,transformation=t)
                    bics[t,s,ns[q]][d-1,k-2] = M.BIC()
                    aris[t,s,ns[q]][d-1,k-2] = ari(z_est_temp[t,d-1,k-2],zz)
                    ## Destination nodes
                    z_est_temp_prime[t,d-1,k-2] = M.fit_predict_approximate(Y,d=d,transformation=t)
                    bics_prime[t,s,ns_prime[q]][d-1,k-2] = M.BIC()
                    aris_prime[t,s,ns_prime[q]][d-1,k-2] = ari(z_est_temp_prime[t,d-1,k-2],zz_prime)
        ## Obtain estimates
        for t in [None, 'normalised', 'theta']:
            ## Source nodes
            dK = find_max(bics[t,s,ns[q]])
            est_d[t,s,ns[q]] = dK[0]
            est_K[t,s,ns[q]] = dK[1]
            z_est[t,s,ns[q]] = z_est_temp[t,dK[0]-1,dK[1]-2]
            est_ari[t,s,ns[q]] = aris[t,s,ns[q]][dK[0]-1,dK[1]-2]
            ## Destination nodes nodes
            dK = find_max(bics_prime[t,s,ns_prime[q]])
            est_d_prime[t,s,ns_prime[q]] = dK[0]
            est_K_prime[t,s,ns_prime[q]] = dK[1]
            z_est_prime[t,s,ns_prime[q]] = z_est_temp_prime[t,dK[0]-1,dK[1]-2]
            est_ari_prime[t,s,ns_prime[q]] = aris_prime[t,s,ns_prime[q]][dK[0]-1,dK[1]-2]

## Calculate output
d_scores = {}; d_scores_prime = {}
K_scores = {}; K_scores_prime = {}
ari_scores = {}; ari_scores_prime = {}
for t in [None, 'normalised', 'theta']:
    d_scores[t] = np.zeros((M_sim, len(ns)))
    K_scores[t] = np.zeros((M_sim, len(ns)))
    ari_scores[t] = np.zeros((M_sim, len(ns)))
    d_scores_prime[t] = np.zeros((M_sim, len(ns)))
    K_scores_prime[t] = np.zeros((M_sim, len(ns)))
    ari_scores_prime[t] = np.zeros((M_sim, len(ns)))
    for s in range(M_sim):
        for n in range(len(ns)):
            d_scores[t][s,n] = est_d[t,s,ns[n]]
            K_scores[t][s,n] = est_K[t,s,ns[n]]
            ari_scores[t][s,n] = est_ari[t,s,ns[n]]
            d_scores_prime[t][s,n] = est_d_prime[t,s,ns_prime[n]]
            K_scores_prime[t][s,n] = est_K_prime[t,s,ns_prime[n]]
            ari_scores_prime[t][s,n] = est_ari_prime[t,s,ns_prime[n]]

## Save output
for t in [None, 'normalised', 'theta']:
    label = t if t != None else 'none'
    np.savetxt('Results_Bipartite/out_bip_d_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', d_scores[t], fmt='%i', delimiter=',')
    np.savetxt('Results_Bipartite/out_bip_K_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', K_scores[t], fmt='%i', delimiter=',')
    np.savetxt('Results_Bipartite/est_bip_ari_' + label + '_' + str(K) + '_' + str(args.s) + '.csv', ari_scores[t], fmt='%.6f', delimiter=',')
    np.savetxt('Results_Bipartite/out_bip_d_prime_' + label + '_' + str(K_prime) + '_' + str(args.s) + '.csv', d_scores_prime[t], fmt='%i', delimiter=',')
    np.savetxt('Results_Bipartite/out_bip_K_prime_' + label + '_' + str(K_prime) + '_' + str(args.s) + '.csv', K_scores_prime[t], fmt='%i', delimiter=',')
    np.savetxt('Results_Bipartite/est_bip_ari_prime_' + label + '_' + str(K_prime) + '_' + str(args.s) + '.csv', ari_scores_prime[t], fmt='%.6f', delimiter=',')