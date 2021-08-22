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

def rotation_matrix_3d(alpha,beta,gamma):
    R1 = rotation_matrix(alpha)
    R2 = rotation_matrix(beta)
    R3 = rotation_matrix(gamma)
    RR1 = np.zeros((3,3)); RR1[:2,:2] = R1; RR1[2,2] = 1
    RR2 = np.zeros((3,3)); RR2[0,0] = R2[0,0]; RR2[0,2] = R2[0,1]; RR2[2,0] = R2[1,0]; RR2[2,2] = R2[1,1]; RR2[1,1] = 1
    RR3 = np.zeros((3,3)); RR3[1:,1:] = R3; RR3[0,0] = 1
    return RR1, RR2, RR3, np.matmul(np.matmul(RR1,RR2),RR3)

def reflection_matrix(theta):
    R = np.zeros((2,2))
    ct = np.cos(2*theta)
    st = np.sin(2*theta)
    R[0,0] = ct
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
d = 3
n = 10
m = np.zeros((2,3))
m[0] = [0.25,0.75,0.1]; m[1] = [0.75,0.25,0.2]
## Q = np.zeros((d,d)); Q[0,0] = 1; Q[1,2] = -1; Q[2,1] = 1
Q = ortho_group.rvs(d)
## X = np.random.uniform(size=n).reshape(-1,1) * m[np.random.choice(2,size=n),:d]
m = np.random.uniform(size=(100,100))
X = np.random.uniform(size=n).reshape(-1,1) * m[np.random.choice(d+2,size=n),:d]
X = np.random.uniform(size=n).reshape(-1,1) * np.random.uniform(size=(n,d))
X2 = np.matmul(X,Q.T)
## Thetas
T1 = theta_transform(X)
T2 = theta_transform(X2)
## Affine transformation
A = np.matmul(np.linalg.inv(np.matmul(T1.T,T1)),np.matmul(T1.T,T2))
T12 = np.matmul(T1,A)
## 
TT1 = np.hstack((T1,np.ones(T1.shape[0]).reshape(-1,1)))
TT2 = np.hstack((T2,np.ones(T2.shape[0]).reshape(-1,1)))
AA = np.matmul(np.linalg.inv(np.matmul(TT1.T,TT1)),np.matmul(TT1.T,TT2))
TT12 = np.matmul(TT1,AA)

from scipy.linalg import orthogonal_procrustes as proc
T3 = np.matmul(T2,proc(T2,T1)[0])

plt.scatter(X[:,0],X[:,1],c='gray')
plt.scatter(X2[:,0],X2[:,1],c='gray')
plt.xlim([-0.5,1])
plt.ylim([-0.5,1])
for i in range(10):
    plt.text(X[i,0],X[i,1],s=str(i+1))
    plt.text(X2[i,0],X2[i,1],s=str(i+1),c='red')

plt.axhline(y=0)
plt.axvline(x=0)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.array([0.1, 0.5, 1.0])
#ax.scatter(X[:,0], X[:,1], X[:,2], c=np.array(['#009E73','#0072B2'])[z], s=1, alpha=0.75)
ax.scatter(X[0], X[1], X[2], s=25, c='#009E73')
X2 = np.matmul(rotation_matrix_3d(np.pi/2,np.pi/2,np.pi/2)[0], X)
ax.scatter(X2[0], X2[1], X2[2], s=25, c='#009E73')
U = np.zeros((23,3))
U[0] = X
X2 = np.matmul(rotation_matrix_3d(0,0,0)[2],rotation_matrix_3d(t,t,t)[0],X.reshape(-1,1))
V = np.zeros((23,3))
V[0] = X2
count = 0
for t in np.linspace(0,2*np.pi,25)[1:][:-1]:
    ## X2 = np.matmul(np.matmul(rotation_matrix_3d(t,t,t)[0],rotation_matrix_3d(t,t,t)[2]),X.reshape(-1,1))
    X2 = np.matmul(rotation_matrix_3d(np.pi/2*3/7,np.pi/2*3/7,np.pi/2*3/7)[2],np.matmul(rotation_matrix_3d(t,t,t)[0],X.reshape(-1,1)))
    U[count] = X2.reshape(-1)
    print(cart_to_sphere(X2))
    ax.scatter(X2[0], X2[1], X2[2], s=25, c='#0072B2')
    X2 = np.matmul(rotation_matrix_3d(t,t,t)[0],X.reshape(-1,1))
    V[count] = X2.reshape(-1)
    ax.scatter(X2[0], X2[1], X2[2], s=25, c='#0072B2')
    count += 1

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=25, azim=45)
plt.show()## ; plt.clf(); plt.cla(); plt.close()

T1 = theta_transform(U)
T2 = theta_transform(V)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(T1[:,0], T1[:,1], T2[:,0], s=25, c='#0072B2')
plt.show()

plt.scatter(T1[:,0],T1[:,1], c='#009E73')
plt.scatter(T2[:,0],T2[:,1], c='#0072B2')
plt.scatter(T3[:,0],T3[:,1], c='red')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(T1[:,0], T1[:,1], T2[:,0], s=25, c='#0072B2')

import numpy as np
TT1 = np.hstack((T1,np.ones(T1.shape[0]).reshape(-1,1)))
TT2 = np.hstack((T2,np.ones(T2.shape[0]).reshape(-1,1)))

## Simulation
x = [-0.5,-0.1] ## np.ones(2)
t = float(cart_to_sphere(x))
rs = np.linspace(0,2*np.pi,10000)
aa = []; aa2 = []; aa_est = []
for theta in rs:
    aa += [float(cart_to_sphere(np.matmul(reflection_matrix(theta),x))) - t]
    aa2 += [angle(np.matmul(reflection_matrix(theta),x)) - t]
    aa_est += [estimate_difference(theta,t)]
    #aa2 += [float(cart_to_sphere(np.matmul(reflection_matrix(theta),x))) - t]

plt.scatter(rs,aa,s=2.0,c='red')
plt.scatter(rs,aa2,s=2.0,c='green')
##plt.scatter(rs,aa_est,s=2.0,c='blue')
plt.axhline(2*np.pi,ls='dotted',c='blue')
plt.axhline(np.pi,ls='dotted',c='blue')
plt.axhline(0,ls='dotted',c='blue')
plt.show()





## Simulation
x = np.ones(2)
angle(x) ## pi/4
rs = np.linspace(0,2*np.pi,1000)
aa = []; aa2 = []; aa_conj = []
for theta in rs:
    uu = (angle(x) + theta) if angle(x) + theta <= np.pi else (angle(x) + theta - 2*np.pi) 
    aa_conj += [uu]
    aa += [angle(np.matmul(rotation_matrix(theta),x))]
    aa2 += [angle(np.matmul(reflection_matrix(theta),x))]

import matplotlib.pyplot as plt
plt.scatter(rs,aa,s=2.0,c='red')
plt.scatter(rs,aa_conj,s=0.1,c='green', ls='--')
plt.axhline(angle(x),ls='dotted',c='blue')
plt.show()
## Obtain the slope
slope(np.array([rs[800],aa[800]]),np.array([rs[900],aa[900]])) ## 1.0

plt.scatter(rs[:500],aa2[:500],s=0.5,c='red')
plt.axhline(angle(x),ls='dotted',c='blue')
plt.show()


## Three dimensions
from scipy.sparse import ortho_group
x = np.ones(3)
t = cart_to_sphere(x)
for _ in 
Q = ortho_group.rvs(3)
