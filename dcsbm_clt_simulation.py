#! /usr/bin/env python3
import numpy as np

## Parameters
n = 1000
K = 2
### Means
m = np.zeros((2,2))
m[0] = np.array([0.75,0.25])
m[1] = np.array([0.25,0.75])

### Allocations
z = np.array([0]*(n//2) + [1]*(n//2))

### Degree-corrected blockmodel
np.random.seed(1771)
X = np.zeros((n,2))
X1 = np.zeros((n,2))
rhos = np.random.uniform(size=n)
for i in range(n):
    X1[i] = m[z[i]]
    X[i] = rhos[i] * m[z[i]]

## Adjacency matrix
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i+1,n):
        A[i,j] = np.random.binomial(n=1, p=np.inner(X[i],X[j]), size=1)
        A[j,i] = A[i,j]

## Embedding
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k], np.diag(np.sqrt(np.abs(Lambda[k]))))

## Procrustes analysis
from scipy.linalg import orthogonal_procrustes as proc
X_tilde = np.dot(X_hat, proc(X_hat, X)[0])

## Save
np.savetxt('emb.csv', X_tilde, fmt='%.5f', delimiter=',')

## Indices
indices = [25, 88, 9, 941, 984, 900]
np.savetxt('emb_reduced.csv', X_tilde[indices], fmt='%.5f', delimiter=',')
np.savetxt('emb_reduced_true.csv', X[indices], fmt='%.5f', delimiter=',')

## Simulation to obtain the required expectations
np.random.seed(1771)
J = int(1e6)
z_sim = np.random.choice(2, size=J)
rhos_sim = np.random.uniform(size=J)
X_sim = rhos_sim.reshape(-1,1) * m[z_sim]
Delta_inv = np.linalg.inv(np.matmul(X_sim.T,X_sim) / J)
Delta_prods = np.zeros((J,2,2))
Delta_est = np.zeros((len(indices),2,2))
for i in range(len(indices)):
    x = X[indices[i]].reshape(-1,1)
    for j in range(J):
        X1 = X_sim[j].reshape(-1,1)
        xX1 = np.matmul(x.T,X1)
        Delta_prods[j] = (xX1 - (xX1 ** 2)) * np.matmul(X1, X1.T)
    Delta_est[i] = np.matmul(np.matmul(Delta_inv, np.mean(Delta_prods,axis=0)), Delta_inv) / n

## Function to extract the parameters required by pgfplots
def ellipse_parameters(Delta):
    from scipy.stats import chi2
    Lambda, Gamma = np.linalg.eig(Delta)
    quantiles = chi2.ppf([0.5, 0.75, 0.90], df=2)
    radii = np.sqrt(np.array([Lambda * q for q in quantiles])) * 10 ## Note the scaling factor for tikz plot: 1cm == 0.1
    angles = np.array([np.degrees(np.arctan2(Gamma[0,1], Gamma[0,0])), np.degrees(np.arctan2(Gamma[1,1], Gamma[1,0]))])
    angles2 = np.array([np.degrees(np.arctan2(Gamma[0,0], Gamma[0,1])), np.degrees(np.arctan2(Gamma[1,0], Gamma[1,1]))])
    return radii, angles, angles2

## Print output
for i in range(len(indices)):
    r, a, a2 = ellipse_parameters(Delta_est[i])
    for line in r:
        loc = f'{X[indices[i],0]:.5f}'+', '+f'{X[indices[i],1]:.5f}'
        string_radii = 'x radius='f'{line[0]:.10f}'+'cm, y radius='+f'{line[1]:.10f}'+'cm'
        # ax = a[1] if z[indices[i]] == 0 else a[1] 
        # print('\draw[rotate around={',f'{ax:.3f}',':(axis cs: ',loc,')}] (axis cs: ',loc,') circle [',string_radii,'];',sep='')
        ax = a2[1] if z[indices[i]] == 0 else a2[1] 
        print('\draw[rotate around={',f'{ax:.3f}',':(axis cs: ',loc,')}] (axis cs: ',loc,') circle [',string_radii,'];',sep='')
    print('')

simulate_points = False
if simulate_points:
    ## Simulation
    np.random.seed(117)
    JJ = 1000
    U = np.zeros((JJ,2))
    V = np.zeros((JJ,2))
    P = np.matmul(X,X.T)
    for j in range(JJ):
        print('\rSimulation: ', str(j+1), '/', str(JJ), sep='', end='')
        A2 = np.tril(np.random.binomial(n=1,p=P), k=-1)
        A_sim = A2 + A2.T
        ## Embedding
        Lambda_sim, Gamma_sim = np.linalg.eigh(A_sim)
        k = np.argsort(np.abs(Lambda_sim))[::-1][:2]
        X_hat_sim = np.dot(Gamma_sim[:,k],np.diag(np.sqrt(np.abs(Lambda_sim[k]))))
        X_tilde_sim = np.dot(X_hat_sim,proc(X_hat_sim,X)[0])
        U[j] = X_tilde_sim[indices[-1]]
        V[j] = X_tilde_sim[indices[0]]
    print('')
    np.savetxt('emb_900.csv',U,fmt='%.5f',delimiter=',')
    np.savetxt('emb_400.csv',V,fmt='%.5f',delimiter=',')

## Observed covariance
data = np.loadtxt('emb_900.csv', delimiter=',')
C = np.cov(data.T)
r, a, a2 = ellipse_parameters(C)
for line in r:
    loc = f'{X[indices[i],0]:.5f}'+', '+f'{X[indices[i],1]:.5f}'
    string_radii = 'x radius='f'{line[0]:.10f}'+'cm, y radius='+f'{line[1]:.10f}'+'cm'
    ax = a2[1] if z[indices[i]] == 0 else a2[1] 
    print('\draw[dashed,rotate around={',f'{ax:.3f}',':(axis cs: ',loc,')}] (axis cs: ',loc,') circle [',string_radii,'];',sep='')
