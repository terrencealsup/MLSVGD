# MLSVGD
This code implements a multilevel extension of Stein variation gradient descent (MLSVGD) for Bayesian inference with hierarchical surrogate models.

The main implementation is contained in the file `MLSVGD.py` and is follows closely the original SVGD implementation by Dilin Wang, which can be found at
https://github.com/dilinwang820/Stein-Variational-Gradient-Descent

An example notebook is also provided, `MLSVGD-SteadyStateDiffusion.ipynb`, which shows how to use the implementation as well as compares both SVGD and MLSVGD for Bayesian inference.  For this example, the forward model is implemented in the file `forward_model.py`.  This implementation should be compatible with most Python versions >= 3.6.


---
- T. Alsup, L. Venturi, and B. Peherstorfer. [Multilevel Stein variational gradient descent with applications to Bayesian inverse problems](https://msml21.github.io/papers/id52.pdf). In *Mathematical and Scientific Machine Learning (MSML) 2021*, 2021
