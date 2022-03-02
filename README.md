# MLSVGD
Implementation of multilevel Stein variation gradient descent (MLSVGD)

For more details of the algorithm see the paper:
[Multilevel Stein variational gradient descent with applications to Bayesian inverse problems](https://msml21.github.io/papers/id52.pdf)
by T. Alsup, L. Venturi, and B. Peherstorfer. In Mathematical and Scientific Machine Learning (MSML). 2021.

The main implementation is contained in the file `MLSVGD.py` and is follows closely the original SVGD implementation by Dilin Wang, which can be found at
https://github.com/dilinwang820/Stein-Variational-Gradient-Descent


---
An example notebook is provided, `MLSVGD-SteadyStateDiffusion.ipynb`, which shows how to use the implementation as well as compares both SVGD and MLSVGD for Bayesian inference.
