#! /usr/bin/env python3
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

## Call mclust in R for fitting the model
def GMM_bic(X, K_star, n_init=5):
    bics = []
    for k in range(2,K_star+1):
        M = GMM(n_components=k, n_init=n_init)
        M.fit(X)
        bics += [M.bic(X)]
    K_star = np.argmin(bics) + 2
    return(GMM(n_components=K_star, n_init=n_init).fit_predict(X), K_star)