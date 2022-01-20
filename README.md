# ConvexSoS

## Overview

This repository complements the paper [Learning PSD-valued functions using kernel sums-of-squares](https://arxiv.org/abs/2111.11306) (Muzellec B., Bach F., Rudi, A.):

### PSD-valued regression

The `psd_models/` folder contains code for PSD-valued regression:

- `psd_optim.py` implements the PSDRegressor class;
- `psd_utils.py` contains utilities, such as functions to build kernel features.

### Convex regression

The `convex_models/` folder contains code for convex regression:

- `cvx_optim.py` implements the CVXRegressor class;
- `cvx_utils.py` contains utilities, such as functions to build kernel features or to sample convex constraints;
- `lin_cvx_reg.py` implements piecewise linear convex regression;
- `sobol_seq.py` and `sobol.cpp` implement Sobol quasi-random sampling.

Two example notebook are available: `PSD regression example.ipynb` and `Convex regression example.ipynb`

## Reference

Muzellec B., Bach F., Rudi, A.: [Learning PSD-valued functions using kernel sums-of-squares](https://arxiv.org/abs/2111.11306)

```
@article{muzellec2021learning,
  title={Learning PSD-valued functions using kernel sums-of-squares},
  author={Muzellec, Boris and Bach, Francis and Rudi, Alessandro},
  journal={arXiv preprint arXiv:2111.11306},
  year={2021}
}
```

## Dependencies
- Python 3+

To use the piecewise linear model from `lin_cvx_reg.py`, [cvxopt](https://cvxopt.org/) is also required.
