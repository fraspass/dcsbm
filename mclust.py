#! /usr/bin/env python3
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r.source('mclust.R')

## Call mclust in R for fitting the model
def mclust(X):
    XR = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    z = ro.r.mclust_fit(XR).astype(int)
    return(z, np.unique(z).shape)