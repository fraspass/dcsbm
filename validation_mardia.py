#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics import silhouette_score as ASW
from scipy.stats import kstest as KS
from sklearn.preprocessing import scale
from scipy.stats import chi2, norm

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
n = 1500
M = 250
K = 3
m = 10

## Set seed to repeat the simulation
np.random.seed(171171)
mu = np.array([0.7,0.4,0.1,0.1,0.1,0.5,0.4,0.8,-0.1]).reshape(3,3)
B = np.dot(mu,mu.T)
rho = np.random.beta(a=2,b=1,size=n)
q = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])
z = np.zeros(n,dtype=int)
for k in range(K):
    z[q[k]:] = k

## Effective dimension in theta
p = K - 1

## Define the arrays
skew_pvals = []
kurt_pvals = []
skew_pvals_tilde = []
kurt_pvals_tilde = []

## Repeat M times
for s in range(M):
    print('\rIteration: '+str(s),end='')
    ## Construct the adjacency matrix
    rows = []
    cols = []
    for i in range(n-1):
        for j in range(i+1,n):
            if np.random.binomial(n=1,p=rho[i]*rho[j]*B[z[i],z[j]],size=1) == 1:
                rows += [i,j]
                cols += [j,i]
    ## Obtain the adjacency matrix and the embeddings
    A = coo_matrix((np.repeat(1.0,len(rows)),(rows,cols)),shape=(n,n))
    S, U = eigsh(A, k=m)
    indices = np.argsort(np.abs(S))[::-1]
    X = np.dot(U[:,indices], np.diag(np.abs(S[indices]) ** .5))
    ## Remove empty rows
    zero_index = np.array(A.sum(axis=0),dtype=int)[0]
    X = X[zero_index > 0]
    zz = z[zero_index > 0]
    ## Calculate the transformations of the embedding
    X_tilde = np.divide(X, np.linalg.norm(X,axis=1)[:,np.newaxis])
    theta = theta_transform(X)
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
        emb_k = theta[zz==k]
        emb_k_var = np.linalg.inv(np.cov(emb_k[:,:p].T))
        Dk = np.dot(np.dot(scale(emb_k[:,:p],with_std=False),emb_k_var),scale(emb_k[:,:p],with_std=False).T)
        ## b1 (skewness)
        b1 = np.sum(Dk ** 3) / (6*nk)
        skew_pvals += [chi2.logsf(b1, df=p*(p+1)*(p+2)/6)]
        ## b2 (kurtosis)
        b2 = (np.mean(np.diag(Dk) ** 2) - p*(p+2)*(nk-1)/(nk+1)) / np.sqrt(8*p*(p+2) / nk)
        kurt_pvals += [norm.logsf(b2)]

ks = np.sum(np.sign(np.array(kurt_pvals) - np.array(kurt_pvals_tilde)) > 0)
ss = np.sum(np.sign(np.array(skew_pvals) - np.array(skew_pvals_tilde)) > 0)

from scipy.stats import binom_test
binom_test([ks, M*K-ks], p=0.5, alternative='greater')
binom_test([ss, M*K-ss], p=0.5, alternative='greater')

np.save('skew_pvals.npy',skew_pvals)
np.save('kurt_pvals.npy',kurt_pvals)
np.save('skew_pvals_tilde.npy',skew_pvals_tilde)
np.save('kurt_pvals_tilde.npy',kurt_pvals_tilde)