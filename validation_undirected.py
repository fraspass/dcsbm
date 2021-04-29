#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from sklearn.metrics import silhouette_score as ASW
from scipy.stats import kstest as KS
from sklearn.preprocessing import scale
from scipy.stats import chi2, norm
from scipy.linalg import orthogonal_procrustes as proc

###########################################################
## Reproduces results in Section 4.2 - Undirected graphs ##
###########################################################

## Takes a vector and returns its spherical coordinates
def cart_to_sphere(x):
    ## theta_1
    q = np.arccos(x[1] / np.linalg.norm(x[:2]))
    sphere_coord = [q] if x[0] >= 0 else [2*np.pi - q]
    ## Loop for theta_2, ..., theta_m-1
    for j in range(2,len(x)):
        sphere_coord += [2 * np.arccos(x[j] / np.linalg.norm(x[:(j+1)]))]
    ## Return the result in a numpy array
    return np.array(sphere_coord)

## Takes a matrix and returns the spherical coordinates obtained along the given axis
def theta_transform(X,axis=1):
    ## Apply the function theta_transform along the axis
    return np.apply_along_axis(func1d=cart_to_sphere, axis=axis, arr=X)

## Arguments
ns = [100, 200, 500, 1000, 2000]
M = 100
K = 3
m = 10

## Set seed to repeat the simulation
np.random.seed(171171)
mu = np.array([0.7,0.4,0.1,0.1,0.1,0.5,0.4,0.8,-0.1]).reshape(3,3)
B = np.dot(mu,mu.T)
q = {}
for n in ns:
    q[n] = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])

z = {}
for n in ns:
    z[n] = np.zeros(n,dtype=int)
    for k in range(K):
        z[n][q[n][k]:] = k

## Effective dimension in theta
p = K - 1

## Define the arrays
skew_pvals = []
kurt_pvals = []
skew_pvals_tilde = []
kurt_pvals_tilde = []
ks_redundant = np.zeros((len(ns),M,K,m-p-1))
ks_redundant_tot = np.zeros((len(ns),M,m-p-1))
mean_sim = np.zeros((len(ns),M,K,m-1))
cov_sim = np.zeros((len(ns),M,K,m-1,m-1))

X_true = {}
rhos = np.random.beta(a=1,b=1,size=ns[len(ns)-1])
for n in ns:
    X_true[n] = np.zeros((n,m))
    for i in range(n):
        X_true[n][i,:K] = rhos[i] * mu[z[n][i]]

## Repeat M times
ii = 0
Xs = {}
for n in ns:
    for s in range(M):
        print('\rNumber of nodes: ' + str(n) + '\tSimulation: ' + str(s+1), end='')
        ## Degree correction parameters
        ## Construct the adjacency matrix
        A = np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                A[i,j] = np.random.binomial(n=1,p=np.inner(X_true[n][i],X_true[n][j]),size=1)
                A[j,i] = A[i,j]
        ## Obtain the adjacency matrix and the embeddings
        S, U = eigsh(A, k=m)
        indices = np.argsort(np.abs(S))[::-1]
        XX = np.dot(U[:,indices], np.diag(np.abs(S[indices]) ** .5))
        Xs[s,n] = np.dot(XX,proc(XX,X_true[n])[0])
        ## Remove empty rows
        zero_index = np.array(A.sum(axis=0),dtype=int)
        Xs[s,n] = Xs[s,n][zero_index > 0]
        zz = z[n][zero_index > 0]
        ## Calculate the transformations of the embedding
        X_tilde = np.divide(Xs[s,n], np.linalg.norm(Xs[s,n],axis=1)[:,np.newaxis])
        theta = theta_transform(Xs[s,n])
        ## Loop over the groups
        for k in range(K):
            ## Number of units in cluster k
            nk = np.sum(zz==k)
            ## Mardia tests for X_tilde
            emb_k = X_tilde[zz==k]
            emb_k_var = np.linalg.inv(np.cov(emb_k[:,:K].T))
            Dk = np.dot(np.dot(scale(emb_k[:,:K],with_std=False),emb_k_var),scale(emb_k[:,:K],with_std=False).T)
            ## b1 (skewness)
            b1 = np.sum(Dk ** 3) / (6*nk)
            skew_pvals_tilde += [chi2.logsf(b1, df=K*(K+1)*(K+2)/6)]
            ## b2 (kurtosis)
            b2 = (np.mean(np.diag(Dk) ** 2) - K*(K+2)*(nk-1)/(nk+1)) / np.sqrt(8*p*(p+2) / nk)
            kurt_pvals_tilde += [norm.logsf(b2)]
            ## Repeat the calculations of the Mardia test for theta
            emb_k = theta[zz == k]
            emb_k_var = np.linalg.inv(np.cov(emb_k[:,:p].T))
            Dk = np.dot(np.dot(scale(emb_k[:,:p],with_std=False),emb_k_var),scale(emb_k[:,:p],with_std=False).T)
            ## b1 (skewness)
            b1 = np.sum(Dk ** 3) / (6*nk)
            skew_pvals += [chi2.logsf(b1, df=p*(p+1)*(p+2)/6)]
            ## b2 (kurtosis)
            b2 = (np.mean(np.diag(Dk) ** 2) - p*(p+2)*(nk-1)/(nk+1)) / np.sqrt(8*p*(p+2) / nk)
            kurt_pvals += [norm.logsf(b2)]
            ## KS test on the last dimensions
            clust_mean = np.pi
            clust_var = np.sqrt(np.sum(((emb_k[:,p:] - clust_mean) ** 2) / (nk-1), axis=0))
            ks_redundant[ii,s,k] = np.array([KS(emb_k[:,p:][:,d],'norm',args=(clust_mean,clust_var[d]))[0] for d in range(m-p-1)])
            ## Calculate mean and covariances
            mean_sim[ii,s,k] = np.mean(emb_k, axis=0)
            cov_sim[ii,s,k] = np.cov(emb_k.T)
        ## Calculate the KS score for the joint distribution
        tot_var = np.sqrt(np.sum(((theta[:,p:] - np.pi) ** 2) / (nk-1), axis=0)) 
        ks_redundant_tot[ii,s] = np.array([KS(theta[:,d+p],'norm',args=(np.pi,tot_var[d]))[0] for d in range(m-p-1)])
    ii += 1

## Save files
np.save('ks_redundant.npy',ks_redundant)
np.save('ks_redundant_tot.npy',ks_redundant_tot)
np.save('mean_sim.npy',mean_sim)
np.save('cov_sim.npy',cov_sim)
np.save('skew_pvals.npy',skew_pvals)
np.save('kurt_pvals.npy',kurt_pvals)
np.save('skew_pvals_tilde.npy',skew_pvals_tilde)
np.save('kurt_pvals_tilde.npy',kurt_pvals_tilde)