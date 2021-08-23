#! /usr/bin/env python3
import numpy as np
import argparse
import dcsbm
from sklearn.preprocessing import scale
from scipy.stats import chi2, norm

## PARSER to give parameter values
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)

## Maximum dimension for the embedding
parser.add_argument("-m", "--dimension", dest="dimension", default=4, type=int,
	help="Initial dimension of the embedding.")
## Graph type
parser.add_argument("-g", "--graph", dest="graph", default = 'icl2', type=str,
	help="Type of graph: icl1, icl2 or icl3.")

## Parse arguments
args = parser.parse_args()
graph = 'icl3' ## args.graph
m = args.dimension

## Import data
if graph == 'icl1':
    labs = np.loadtxt('Data/labs1.csv', delimiter=',')
    X = np.load('Data/X_ICL1.npy')[:,:m]
elif graph == 'icl2':
    labs = np.loadtxt('Data/labs2.csv', delimiter=',')
    X = np.load('Data/X_ICL2.npy')[:,:m]
elif graph == 'icl3':
    labs = np.loadtxt('Data/labs3.csv', delimiter=',')
    X = np.load('Data/X_ICL3.npy')[:,:m]
else:
    raise ValueError('Invalid graph.')

## Number of communities
K = len(np.unique(labs))
X = X[:,:m]

## Obtain X_tilde
X_tilde = np.divide(X, np.linalg.norm(X,axis=1)[:,np.newaxis])
## Obtain Theta
Theta = dcsbm.theta_transform(X)

kurt_pvals = []; kurt_pvals_tilde = []; kurt_pvals_theta = []
skew_pvals = []; skew_pvals_tilde = []; skew_pvals_theta = []
for k in range(K):
    ## Number of units in cluster k
    nk = np.sum(labs==k)
    ## Mardia tests for X
    emb_k = X[labs==k]
    emb_k_var = np.linalg.inv(np.cov(emb_k.T))
    Dk = np.dot(np.dot(scale(emb_k,with_std=False),emb_k_var),scale(emb_k,with_std=False).T)
    ## b1 (skewness)
    b1 = np.sum(Dk ** 3) / (6*nk)
    skew_pvals += [chi2.logsf(b1, df=m*(m+1)*(m+2)/6)]
    ## b2 (kurtosis)
    b2 = (np.mean(np.diag(Dk) ** 2) - m*(m+2)*(nk-1)/(nk+1)) / np.sqrt(8*m*(m+2) / nk)
    kurt_pvals += [norm.logsf(b2)]
    ## Mardia tests for X_tilde
    emb_k = X_tilde[labs==k]
    emb_k_var = np.linalg.inv(np.cov(emb_k.T))
    Dk = np.dot(np.dot(scale(emb_k,with_std=False),emb_k_var),scale(emb_k,with_std=False).T)
    ## b1 (skewness)
    b1 = np.sum(Dk ** 3) / (6*nk)
    skew_pvals_tilde += [chi2.logsf(b1, df=m*(m+1)*(m+2)/6)]
    ## b2 (kurtosis)
    b2 = (np.mean(np.diag(Dk) ** 2) - m*(m+2)*(nk-1)/(nk+1)) / np.sqrt(8*m*(m+2) / nk)
    kurt_pvals_tilde += [norm.logsf(b2)]
    ## Mardia tests for Theta
    emb_k = Theta[labs==k]
    emb_k_var = np.linalg.inv(np.cov(emb_k.T))
    Dk = np.dot(np.dot(scale(emb_k,with_std=False),emb_k_var),scale(emb_k,with_std=False).T)
    ## b1 (skewness)
    p = m-1
    b1 = np.sum(Dk ** 3) / (6*nk)
    skew_pvals_theta += [chi2.logsf(b1, df=p*(p+1)*(p+2)/6)]
    ## b2 (kurtosis)
    b2 = (np.mean(np.diag(Dk) ** 2) - p*(p+2)*(nk-1)/(nk+1)) / np.sqrt(8*p*(p+2) / nk)
    kurt_pvals_theta += [norm.logsf(b2)]

kurt_pvals = np.array(kurt_pvals); kurt_pvals_tilde = np.array(kurt_pvals_tilde); kurt_pvals_theta = np.array(kurt_pvals_theta)
skew_pvals = np.array(skew_pvals); skew_pvals_tilde = np.array(skew_pvals_tilde); skew_pvals_theta = np.array(skew_pvals_theta)

np.mean(kurt_pvals_theta - kurt_pvals_tilde)
np.mean(kurt_pvals_theta - kurt_pvals)

np.mean(skew_pvals_theta - skew_pvals_tilde)
np.mean(skew_pvals_theta - skew_pvals)