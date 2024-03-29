#! /usr/bin/env python3
import numpy as np
from scipy.stats import norm

## Calculate elbow of the scree-plot using the criterion of Zhu and Ghodsi (2006)
def zhu(d):
    d = np.sort(d)[::-1]
    p = len(d)
    profile_likelihood = np.zeros(p)
    for q in range(1,p-1):
        mu1 = np.mean(d[:q])
        mu2 = np.mean(d[q:])
        sd = np.sqrt(((q-1) * (np.std(d[:q]) ** 2) + (p-q-1) * (np.std(d[q:]) ** 2)) / (p-2))
        profile_likelihood[q] = norm.logpdf(d[:q],loc=mu1,scale=sd).sum() + norm.logpdf(d[q:],loc=mu2,scale=sd).sum()
    return profile_likelihood[1:p-1], np.argmax(profile_likelihood[1:p-1])+1

## Find the first x elbows of the scree-plot, iterating the criterion of Zhu and Ghodsi (2006)
def iterate_zhu(d,x=4):
    results = np.zeros(x,dtype=int)
    results[0] = zhu(d)[1]
    for i in range(x-1):
        results[i+1] = results[i] + zhu(d[results[i]:])[1]
    return results