#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture

#########################
## Auxiliary functions ##
#########################

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

##################
### EGMM class ###
##################

## EGMM stands for Extended Gaussian Mixture Model (in this case, it is a 'constrained' GMM)

class EGMM:

    ## Initialise the class with the number of components
    def __init__(self, K):
        self.K = K

    ## Initialise the model at random
    def initialise_random(self, X, X_red=None):
        if not hasattr(self,'n'):
            self.n = X.shape[0]
        self.psi = np.ones(self.K) / self.K
        self.vartheta = X[np.random.choice(self.n, size=self.K)]
        self.Sigma = np.array([np.cov(X.T) for _ in range(self.K)])
        if X_red is not None:
            self.sigma = np.array([np.var(X_red,axis=0) for _ in range(self.K)])
    
    ## Initialise the model using k-means
    def initialise_kmeans(self, X, X_red=None):
        initial_fit = kmeans(n_clusters=self.K,n_init=10).fit(X)
        self.vartheta = initial_fit.cluster_centers_
        self.psi = np.array([Counter(initial_fit.labels_)[k] for k in range(self.K)])
        self.Sigma = np.array([np.cov(X[initial_fit.labels_ == k,:].T) for k in range(self.K)])
        if X_red is not None:
            self.sigma = np.array([np.var(X_red[initial_fit.labels_ == k,:],axis=0) for k in range(self.K)])

    ## E-step: calculate the responsibilities using the parameters obtained at the previous iteration
    def E_step(self, X, X_red, mean_red=0):
        self.responsibilities = self.calculate_responsibilities(X=X, X_red=X_red, mean_red=mean_red)

    ## M-step: update the parameters using the responsibilities estimated at the E-step
    def M_step(self, X, X_red, mean_red=0):
        for k in range(self.K):
            respo = self.responsibilities[:,k]
            respo_sum = respo.sum()
            self.vartheta[k] = np.divide(np.multiply(X, respo[:, np.newaxis]).sum(axis=0), respo_sum)
            self.Sigma[k] = np.cov(X.T, aweights=(respo/respo_sum).flatten(), bias=True)
            if X_red is not None :
                self.sigma[k] = np.divide(np.multiply((X_red - mean_red) ** 2, respo[:, np.newaxis]).sum(axis=0), respo_sum)
        self.psi = self.responsibilities.mean(axis=0)
    
    ## Setup the model
    def model_setup(self, X, d, transformation):
        ## Check if the transformation has an admissible value
        if transformation not in [None,'normalised','theta','score']:
            raise ValueError("ValueError: 'transformation' is not in the list of admissible models.")
        else:
            self.transformation = transformation
        ## Obtain n, m and d
        self.n, self.m = X.shape
        self.d = d
        ## According to the model choice, transform X accordingly
        if self.transformation == 'normalised':
            XX = np.divide(X, np.linalg.norm(X,axis=1)[:,np.newaxis])
        elif self.transformation == 'theta':
            self.m -= 1
            XX = theta_transform(X)
        elif self.transformation == 'score':
            self.m -=1 
            XX = (X / X[:,0][:,np.newaxis])[:,1:]
        else:
            XX = X
        ## Separate the two matrices
        self.X = XX[:,:self.d]
        if self.d >= self.m:
            self.X_red = None
        else:
            self.X_red = XX[:,self.d:]

    ## Function for model fitting
    def fit(self, X, d, max_iter=250, tolerance=1e-5, random_init=False, transformation=None, verbose=False):
        ## Obtain parameters
        self.model_setup(X=X, d=d, transformation=transformation)
        ## Randomly initialise, or use kmeans
        if not random_init:
            self.initialise_kmeans(X=self.X, X_red=self.X_red)
        else:
            self.initialise_random(X=self.X, X_red=self.X_red)
        ## Initialise the number of iterations
        iteration = 0
        convergence = False
        old_lik = 0.0
        ## Loop until one of the two conditions is violated
        while iteration < max_iter and convergence == False:
            ## E-step
            self.E_step(X=self.X, X_red=self.X_red, mean_red=0 if self.transformation != 'theta' else np.pi)
            ## M-step
            self.M_step(X=self.X, X_red=self.X_red, mean_red=0 if self.transformation != 'theta' else np.pi)
            ## Increase the number of iterations and calculate the convergence criterion
            iteration += 1
            current_lik = self.complete_log_likelihood(self.X,self.X_red, mean_red=0 if self.transformation != 'theta' else np.pi)
            if iteration > 1:
                convergence = ((current_lik - old_lik) < tolerance)
            old_lik = current_lik
            ## Print summary if necessary
            if verbose:
                print("\r+++ Iteration number +++ ", str(iteration), sep='', end="")
        if verbose:
            print('')

    ## Calculate responsibilities
    def calculate_responsibilities(self, X, X_red, mean_red=0):
        ## Likelihoods for the main part of the embedding
        left_probs = np.zeros((self.n, self.K))
        ## Likelihoods for the redundant part of the embedding
        if X_red is not None:
            right_probs = np.zeros((self.n, self.K))
        ## Loop over the groups
        for k in range(self.K):
            left_distribution = mvn(mean=self.vartheta[k],cov=self.Sigma[k],allow_singular=True)
            left_probs[:,k] = left_distribution.pdf(X)
            if X_red is not None:
                right_distribution = mvn(mean=mean_red * np.ones(X_red.shape[1]),cov=np.diag(self.sigma[k]),allow_singular=True)
                right_probs[:,k] = right_distribution.pdf(X_red)
        ## Numerator and denominator for the responsibilities
        if X_red is not None:
            num = np.multiply(self.psi, np.multiply(left_probs, right_probs))
        else:    
            num = np.multiply(self.psi, left_probs)
        den = num.sum(axis=1)
        ## Calculate and return the result
        responsibilities = np.divide(num, den[:, np.newaxis])
        return responsibilities
    
    ## Prediction
    def predict(self, X, X_red):
        ## Calculate the responsibilities
        probs = self.calculate_responsibilities(X, X_red, mean_red=0 if self.transformation != 'theta' else np.pi)
        try:
            probs = probs.sum(axis=2)
        except:
            pass
        ## Return the argmax
        return np.argmax(probs, axis=1)

    ## Mix predict and fit
    def fit_predict(self, X, d, max_iter=250, tolerance=1e-5, random_init=False, transformation=None, verbose=False):
        self.fit(X=X, d=d, max_iter=max_iter, tolerance=tolerance, random_init=random_init, transformation=transformation, verbose=verbose)
        return self.predict(X=self.X, X_red=self.X_red)

    ## Function to calculate the complete log-likelihood
    def complete_log_likelihood(self,X,X_red=None, mean_red=0):
        loglik = 0.0
        for i in range(self.n):
            if X_red is None:
                loglik += np.log(np.sum([self.psi[k] * mvn.pdf(X[i], self.vartheta[k], self.Sigma[k], allow_singular=True) for k in range(self.K)]))
            else:
                loglik += np.log(np.sum([self.psi[k] * mvn.pdf(X[i], self.vartheta[k], self.Sigma[k], allow_singular=True) * 
                    mvn.pdf(X_red[i], mean_red * np.ones(X_red.shape[1]), np.diag(self.sigma[k]), allow_singular=True) for k in range(self.K)]))
        return loglik

    ## Calculate BIC
    def BIC(self):
        ## Complete log-likelihood
        loglik = self.complete_log_likelihood(X=self.X,X_red=self.X_red, mean_red=0 if self.transformation != 'theta' else np.pi)
        ## Number of observations
        n = self.X.shape[0]
        ## Dimension d
        d = self.X.shape[1]
        ## Dimension m-d
        try:
            m_minus_d = self.X_red.shape[1]
        except:
            m_minus_d = 0
        ## Number of parameters
        n_parameters = self.K * (d + d*(d+1)/2 + m_minus_d + 1)  
        return (2 * loglik - n_parameters * np.log(n))

    ## Approximate maximum likelihood procedure using sklearn (much faster)
    def fit_approximate(self, X, d, transformation):
        ## Setup for the model
        self.model_setup(X=X, d=d, transformation=transformation)
        ## Calculate the GMM only on the initial embeddings
        M = GaussianMixture(n_components=self.K,n_init=10).fit(self.X)
        ## Set the parameter values
        self.vartheta = M.means_
        self.Sigma = M.covariances_
        self.psi = M.weights_
        if self.d < self.m:
            self.sigma = np.zeros((self.K,self.m-self.d))
        ## Use the estimated parameters to calculate responsiblities using only the top-d components
        self.E_step(X=self.X, X_red=None)
        ## Use the responsibilites for the full model
        self.M_step(X=self.X, X_red=self.X_red, mean_red=0 if self.transformation != 'theta' else np.pi)

    ## Fit using the approximate MLE and predict the values
    def fit_predict_approximate(self, X, d, transformation=None):
        ## Calculate the approximate MLE
        self.fit_approximate(X=X, d=d, transformation=transformation)
        return self.predict(X=self.X, X_red=self.X_red)