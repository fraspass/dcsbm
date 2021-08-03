# Spectral clustering on spherical coordinates under the degree-corrected stochastic blockmodel

The code in this repository can be used to reproduce the results and simulations in *Sanna Passino, F., Heard, N. A., and Rubin-Delanchy, P. (2020+) "Spectral clustering on spherical coordinates under the degree-corrected stochastic blockmodel"* ([link to the arXiv publication](https://arxiv.org/abs/2011.04558)). 

## Understanding the code

The main tool for inference on DCSBMs is the class `EGMM` contained in the file `dcsbm.py`. The class can be initialised using simply the number of communities `K`, a positive integer. The parameters for the EM algorithm are initialised at random (`initialise_random`) or with *k*-means (`initialise_kmeans`). An estimate of the communities <img src="svgs/95c0ae7a7506f6eb20e570a5a2df73ec.svg?invert_in_darkmode" align=middle width=217.2027264pt height=24.65753399999998pt/> for a <img src="svgs/767f121dc0c2dd47cf8bad2cffad01de.svg?invert_in_darkmode" align=middle width=57.17660189999999pt height=24.65753399999998pt/>-dimensional embedding `X` can be obtained with the function `fit_predict` applied on the class. The function has the following arguments: 
* `X`: a `numpy` array containing the <img src="svgs/767f121dc0c2dd47cf8bad2cffad01de.svg?invert_in_darkmode" align=middle width=57.17660189999999pt height=24.65753399999998pt/>-dimensional embedding;
* `d`: an integer representing the dimension of the latent positions, with <img src="svgs/219f060a62e2fc6614130a661551cac4.svg?invert_in_darkmode" align=middle width=75.04353449999998pt height=22.831056599999986pt/>;
*  `transformation`: a string, chosen between `normalised`, `theta`, and `score`, representing the required transformation of the embedding (row-normalisation, spherical coordintes or SCORE - if no value is provided for `transformation`, the default is `None`, corresponding clustering on the standard adjacency spectral embedding);
* `random_init`: a `True/False` boolean value used for initialisation (if `True`, the function `initialise_random` is used, otherwise `initialise_kmeans`);
* `verbose` (boolean), `max_iter` (integer), `tolerance` (float): other parameters for the EM algorithm. 
For example, the following snippet initialises the model with `K` communities and estimates the clusters using the spherical coordinates transformation (`theta`) with assumed dimension `d` of the latent positions: 
```python
M = dcsbm.EGMM(K=K)
z = M.fit_predict(X, d=d, transformation='theta', verbose=False, random_init=False, max_iter=500, tolerance=1e-6)
```

## Reproducing the results in the paper

The figures in the paper could be reproduced using the following files:
- Figure 1: `sim_pictures_intro.py`;
- Figure 3: `sim_pictures.py`;
- Figures 4 and 5: `sim.py`;
- Figure 6: `harry_potter.py`;
- Figure 7 and 8: `drosophila.py`.

For security reasons, the ICL network data have *not* been made available, but the code to run the quadratic model and truncated power basis spline model are available in `icl.py`. Similarly, `icl_gp.py` reproduces the same models under an explicit Gaussian process setting.