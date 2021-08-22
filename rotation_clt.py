#! /usr/bin/env python3
import numpy as np

def rotation_matrix(theta):
    R = np.zeros((2,2))
    ct = np.cos(theta)
    st = np.sin(theta)
    R[0,0] = ct
    R[0,1] = st
    R[1,0] = -st
    R[1,1] = ct
    return R

def reflection_matrix(theta):
    R = np.zeros((2,2))
    ct = np.cos(2*theta)
    st = np.sin(2*theta)
    R[0,0] = -ct
    R[0,1] = st
    R[1,0] = st
    R[1,1] = ct
    return R

def angle(x,axis=1):
    a = np.arctan2(x[1],x[0]) if axis==0 else np.arctan2(x[0],x[1])
    if a < 0:
        return 2*np.pi + a
    else:
        return a

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

## Estimate difference between the two curves
def estimate_difference(theta, t):
    if t+theta > 2*np.pi:
        return -theta
    else:
        return theta 

from scipy.stats import ortho_group
np.random.seed(111)
d = 2
n = 100
n_sim = 100
m = np.zeros((2,2))
m[0] = [0.25,0.75]; m[1] = [0.75,0.25]
aa1 = []; aa2 = []; aa3 = []
X = np.random.uniform(size=n).reshape(-1,1) * m[np.random.choice(d,size=n),:d]
A = np.random.binomial(n=1,p=np.matmul(X,X.T))
A = np.tril(A,k=-1) + np.tril(A,k=-1).T
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:d]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
for _ in range(n_sim):
    Q = ortho_group.rvs(d)
    X_hat2 = np.matmul(X_hat,Q.T)
    T1 = theta_transform(X_hat)
    T2 = theta_transform(X_hat2)
    a1 = (T1-T2) % (2*np.pi)
    aa1 += [np.all(np.round(a1,4) == np.round(a1[0],4))]
    a2 = (T1+T2) % (2*np.pi)
    aa2 += [np.all(np.round(a2,4) == np.round(a2[0],4))]
    a3 = (T2-T1) % (2*np.pi)
    aa3 += [np.all(np.round(a3,4) == np.round(a3[0],4))]