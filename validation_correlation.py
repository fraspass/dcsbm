#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from scipy.linalg import orthogonal_procrustes as proc

###########################################################
## Reproduces results in Section 5.4 - Undirected graphs ##
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
n = 500
M = 100
K = 2
m = 10

## Set seed to repeat the simulation
np.random.seed(171171)
ps = np.linspace(0,0.3,num=7)[1:]
q = np.array([int(x) for x in np.linspace(0,n,num=K,endpoint=False)])
mu = {}; B = {}
for p in ps:
    mu[p] = np.array([0.5,p,p,0.3]).reshape(2,2)
    B[p] = np.dot(mu[p],mu[p].T)

## 
z = np.array([0] * (n//2) + [1] * (n//2))

## Define the arrays
mean_sim = np.zeros((len(ps),M,K,m-1))
cov_sim = np.zeros((len(ps),M,K,m-1,m-1))

## Truth
X_true = {}
rho = np.random.beta(a=1,b=1,size=n)
for p in ps:
    X_true[p] = np.zeros((n,m))
    for i in range(n):
        X_true[p][i,:K] = rho[i] * mu[p][z[i]]

## Repeat M times
ii = 0
Xs = {}
for p in ps:
    for s in range(M):
        print('\rCorrelation between blocks: ' + str(p) + '\tSimulation: ' + str(s+1), end='')
        ## Degree correction parameters
        ## Construct the adjacency matrix
        A = np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1,n):
                A[i,j] = np.random.binomial(n=1,p=np.inner(X_true[p][i],X_true[p][j]),size=1)
                A[j,i] = A[i,j]
        ## Obtain the adjacency matrix and the embeddings
        S, U = eigsh(A, k=m)
        indices = np.argsort(np.abs(S))[::-1]
        XX = np.dot(U[:,indices], np.diag(np.abs(S[indices]) ** .5))
        Xs[s,p] = np.dot(XX,proc(XX,X_true[p])[0])
        ## Remove empty rows
        zero_index = np.array(A.sum(axis=0),dtype=int)
        Xs[s,p] = Xs[s,p][zero_index > 0]
        zz = z[zero_index > 0]
        ## Calculate the transformations of the embedding
        X_tilde = np.divide(Xs[s,p], np.linalg.norm(Xs[s,p],axis=1)[:,np.newaxis])
        theta = theta_transform(Xs[s,p])
        ## Loop over the groups
        for k in range(K):
            ## Number of units in cluster k
            nk = np.sum(zz==k)
            ## Repeat the calculations of the Mardia test for theta
            emb_k = theta[zz == k]
            ## Calculate mean and covariances
            mean_sim[ii,s,k] = np.mean(emb_k, axis=0)
            cov_sim[ii,s,k] = np.cov(emb_k.T)
    ii += 1

## Latex plots
def pgfplots_boxplot(x,coordinate=0,color='black',fill='Blue4!50'):
    m = np.median(x)
    q = np.percentile(x,[25,75])
    iqr = q[1] - q[0]
    x_sort = np.sort(x) 
    tl = x_sort[np.where(x_sort > q[0] - 1.5 * iqr)[0][0]]
    tu = x_sort[np.where(x_sort < q[1] + 1.5 * iqr)[0][-1]]
    outliers_low = x_sort[x_sort < tl] 
    outliers_up = x_sort[x_sort > tu]
    print('%% Boxplot')
    print('\\addplot+[color='+color+',\nfill='+fill+',mark=*,mark size=1.5,mark options={black},solid,',sep='')
    print('boxplot prepared={draw position='+str(coordinate)+',',sep='')
    print('median=',str(m),',',sep='')
    print('upper quartile=',str(q[1]),',',sep='')
    print('lower quartile=',str(q[0]),',',sep='')
    print('upper whisker=',str(tu),',',sep='')
    print('lower whisker=',str(tl),',',sep='')
    print('},\n] ',end='')
    print('coordinates {')
    if len(outliers_low) > 0 or len(outliers_up) > 0:
        for o in outliers_low:
            print('('+str(coordinate)+','+str(o)+')')
        for o in outliers_up:
            print('('+str(coordinate)+','+str(o)+')')
    print('};\n')

## 
for u in range(6):
    pgfplots_boxplot(mean_sim[u,:,0,2].T,coordinate=u/5,fill='DeepSkyBlue4!70')

for u in range(6):
    pgfplots_boxplot(mean_sim[u,:,1,2].T,coordinate=u/5,fill='DeepSkyBlue4!70')

## 
for u in range(6):
    pgfplots_boxplot(cov_sim[u,:,0,0,2].T,coordinate=u/5,fill='DeepSkyBlue4!70')

for u in range(6):
    pgfplots_boxplot(cov_sim[u,:,1,0,2],coordinate=u/5,fill='DeepSkyBlue4!70')

## 
for u in range(6):
    pgfplots_boxplot(cov_sim[u,:,0,2,3].T,coordinate=u/5,fill='DeepSkyBlue4!70')

for u in range(6):
    pgfplots_boxplot(cov_sim[u,:,1,2,3],coordinate=u/5,fill='DeepSkyBlue4!70')
