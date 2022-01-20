import numpy as np
import scipy.linalg

import time
import logging

from psd_utils import kernel_func
from cvx_utils import make_features, grid, sobol_sample


class CVXRegressor():
    """
    Convex regression with kernel sum-of-squares Hessian constraints.

    Parameters
    ----------

    bw: float, default=1e-1
        Kernel bandwith.

    rho: float, default=1e-3
        Squared norm regularization factor for the function f.

    lbda_1: float, default=1e-3
        Trace norm regularization factor for SoS constraints.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor for SoS constraints.

    max_iter: int, default=10000
        Maximum number of gradient descent iterations during training.

    precision: float, default=1e-6
        Gradient squared norm threshold to stop gradient descent.

    kernel_type: str, default='gaussian'
        Kernel function. Only the Gaussian kernel is currently supported.

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    nystrom: bool, default=False
        Whether to perform Nystrom approximation of kernel matrices.

    r: int, default=25
        Nystrom approximation rank (only applicable when nystrom is True).

    sampling: str, default='grid'
        SoS constraints subsampling scheme.
        If 'grid', will sample a regular grid.
        If 'sobol', will sample points from a Sobol quasi-random sequence.
        Else, takes support points X.

    n_fill: int, default=25
        The number of points on which SoS constraints are subsampled.
        If sampling is 'grid', n_fill will be rounded down to
        the closest power of dim.
        Else, n_fill is ignored.
    """
    def __init__(self,
                 bw=1e-1,
                 rho=1e-3,
                 lbda_1=1e-3,
                 lbda_2=1e-3,
                 max_iter=10000,
                 precision=1e-6,
                 kernel_type='gaussian',
                 eps=1e-12,
                 nystrom=False,
                 r=25,
                 sampling='grid',
                 n_fill=25):
        self.bw = bw
        self.rho = rho
        self.lbda_1 = lbda_1
        self.lbda_2 = lbda_2
        self.max_iter = max_iter
        self.precision = precision
        self.kernel_type = kernel_type
        self.eps = eps
        self.nystrom = nystrom
        self.r = r
        self.sampling = sampling
        self.n_fill = n_fill

    def fit(self, X, Y, verbose=False, report_interval=1000, G_init=None):
        """
        Fit the model.

        Parameters
        ----------
        X: (n x d) ndarray
            Features.

        Y: (n,) ndarray
            Target.

        verbose: bool, default=True
            If True, output loss to log during iterations.

        report_interval: int, default=1000
            If verbose is True, the interval between two loss reports.

        G_init: (n_fill x d x d) ndarray, default=None
            Warmstart dual variable.
            Useful e.g. for cross-validation.

        """

        self.X = X
        n = len(X)

        if self.sampling in ['grid', 'sobol']:
            if (not hasattr(self, 'lower')) or self.lower is None:
                self.lower = X.min()
            if (not hasattr(self, 'upper')) or self.upper is None:
                self.upper = X.max()
            if self.sampling == 'grid':
                X_fill = grid(self.n_fill, X.shape[1], lower=self.lower, upper=self.upper)
            elif self.sampling == 'sobol':
                X_fill = sobol_sample(self.n_fill, X.shape[1], lower=self.lower, upper=self.upper)
        else:
            X_fill = X

        K, ddK_xv, Phi, Psi = make_features(X, X_fill, self.bw, self.eps, self.nystrom, self.r)

        self.G_, self.losses_, self.grad_norms_ = accelerated_gd(Y, Psi, K, ddK_xv, rho=self.rho,
                                                                 lbda_1=self.lbda_1,
                                                                 lbda_2=self.lbda_2,
                                                                 precision=self.precision,
                                                                 max_iter=self.max_iter,
                                                                 eps=self.eps,
                                                                 verbose=verbose,
                                                                 report_interval=report_interval,
                                                                 G_init=G_init)
        # Build primal coefficients
        Z = 1. / n * K @ Y + .5 * (self.G_ * ddK_xv).sum((1, 2, 3))
        W = np.linalg.inv(K @ K / n + self.rho * K + self.eps * np.eye(n))
        self.coeffs = W @ Z

    def predict(self, X):
        """
        Return the value of the model at X.

        Parameters
        ----------
        X: (n x d) ndarray
            Features.

        Returns
        ----------
        (n,) ndarray
        """
        assert hasattr(self, 'coeffs'), "The model is not fitted."
        return (self.coeffs * kernel_func(X, self.X, self.bw, self.kernel_type)).sum(-1)


# Loss and grad functions
def neg_part(G, Psi, lbda_1=0, grad=True):
    """
    Returns the squared norm of [-Phi diag(G) Phi.T + lbda_1 * Id]_+ and its gradient

    Parameters
    ----------
    G: (n_fill x d x d) ndarray
        Dual variable.

    Psi: (n_fill x (r * d) x d) ndarray
        Kernel features.

    lbda_1: float, default=0
        Trace norm regularization factor.

    grad: bool, default=True
        If True, output the gradient of the negative part.

    """
    n, dim = G.shape[:2]
    r = Psi.shape[1]

    vals, vecs = np.linalg.eigh(((Psi @ G) @ Psi.transpose(0, 2, 1)).sum(0) + lbda_1 * np.eye(r))

    idxs = np.where(vals < 0)[0]

    if len(idxs) == 0:
        if grad:
            return 0, np.zeros_like(G)
        else:
            return 0

    else:
        max_idx = idxs[-1] + 1
        if grad:
            grad_ = 2 * (Psi.transpose(0, 2, 1) @ ((vecs[:, :max_idx] * vals[:max_idx]
                                                    @ vecs[:, :max_idx].T) @ Psi))
            return (vals[:max_idx] ** 2).sum(), grad_
        else:
            return (vals[:max_idx] ** 2).sum()


def loss_grad_func(G, Y, Psi, K, ddK_xv, rho=1e-3, lbda_1=1e-3, lbda_2=1e-3, eps=1e-12, W=None):
    """
    Compute the dual loss and its gradient.

    Parameters
    ----------

    G: (n_fill x d x d) ndarray
        Dual variable.

    Y: (n,) ndarray
        Target.

    Psi: (n_fill x (r * d) x d) ndarray
        Kernel features.

    K: (n_fill x n_fill) ndarray
        Kernel matrix.

    ddK_xv: (n_fill x n x d x d) ndarray
        Kernel second order derivatives.

    rho: float, default=1e-3
        Squared norm regularization factor for the function f.

    lbda_1: float, default=1e-3
        Trace norm regularization factor for SoS constraints.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor for SoS constraints.

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    W: (n x n) ndarray, default=None
        If not None, avoids recomputing (K @ K / n + rho * K + eps * Ib)^{-1}.

    Returns
    -------
    loss: float
        Loss value.

    grad: (n_fill x d x d) ndarray
        Gradient w.r.t. G.
    """
    n = len(Y)
    neg_pen, neg_grad = neg_part(G, Psi, lbda_1=lbda_1, grad=True)
    Z = 1 / n * K @ Y + .5 * (G * ddK_xv).sum((1, 2, 3))
    if W is None:
        W = np.linalg.inv(K @ K / n + rho * K + eps * np.eye(n))
    WZ = W @ Z
    loss = Z.T @ WZ + neg_pen / (2 * lbda_2)
    grad = (ddK_xv.T @ WZ).transpose(2, 1, 0) + neg_grad / (2 * lbda_2)
    return loss, grad


# Accelerated gradient descent
def accelerated_gd(Y, Psi, K, ddK_xv, rho=1e-3, lbda_1=1e-3, lbda_2=1e-3, precision=1e-3,
                   max_iter=10000, eps=1e-12, verbose=True, report_interval=1000, G_init=None):
    """
    Performs accelerated gradient descent in the dual.

    Parameters
    ----------

    Y: (n,) ndarray
        Target.

    Psi: (n_fill x (r * d) x d) ndarray
        Kernel features.

    K: (n_fill x n_fill) ndarray
        Kernel matrix.

    ddK_xv: (n_fill x (r * d) x d) ndarray
        Kernel second order derivatives.

    rho: float, default=1e-3
        Squared norm regularization factor for the function f.

    lbda_1: float, default=1e-3
        Trace norm regularization factor for SoS constraints.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor for SoS constraints.

    precision: float, default=1e-6
        Gradient squared norm threshold to stop gradient descent.

    max_iter: int, default=10000
        Maximum number of gradient descent iterations during training.

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    verbose: bool, default=True
        If True, output loss to log during iterations.

    report_interval: int, default=1000
        If verbose is True, the interval between two loss reports.

    G_init: (n_fill x d x d) ndarray, default=None
        Warmstart dual variable.
        Useful e.g. for cross-validation.

    Returns
    -------
    G: (n_fill x d x d) ndarray
        Fitted dual variable.

    losses: ndarray
        Loss values throughout GD iterations.

    norms: ndarray
        Squared gradient norms throughout GD iterations.
    """

    n, nfill, dim, dim = ddK_xv.shape

    # Compute constants
    W = np.linalg.inv(K @ K / n + rho * K + eps * np.eye(n))
    Q = ddK_xv.reshape(n, -1).T @ W @ ddK_xv.reshape(n, -1)
    L = scipy.linalg.eigh(K ** 2, eigvals_only=True).max() / lbda_2 \
        + scipy.linalg.eigh(Q, eigvals_only=True).max() / 2
    lr = 1. / L

    # Gradient descent
    if G_init is None:
        eta = np.zeros((nfill, dim, dim))
    else:
        eta = G_init.copy()

    G = eta.copy()

    norms = []
    losses = []

    start_time = time.time()

    for i in range(max_iter):
        # Loss and gradient:
        loss, grad = loss_grad_func(eta, Y, Psi, K, ddK_xv, rho, lbda_2, lbda_2, eps=eps, W=W)

        # Gradient update
        G_ = eta - lr * grad
        eta = G_ + i / (i + 3) * (G_ - G)

        norms.append((grad**2).sum() / nfill)
        losses.append(loss.item())

        # Check if we should stop
        if norms[-1] < precision:
            if verbose:
                logging.info(f"iter {i}:\tloss: {loss:.2e}\tgrad norm: {norms[i]:.2e}")
                logging.info(f"Precision {precision:.2e} reached in "
                             f"{time.time() - start_time:.2e} seconds\n")
            return G_, losses, norms

        G = G_

        if verbose:
            if i % report_interval == 0:
                logging.info(f"iter {i}:\tloss: {loss:.2e}\tgrad norm: {norms[i]:.2e}")

    if verbose:
        logging.info(f"iter {i}:\tloss: {loss:.2e}\tgrad norm: {norms[i]:.2e}")
        logging.info(f"Precision {precision:.2e} not reached in "
                     f"{time.time() - start_time:.2e} seconds\n")

    return G, np.array(losses), np.array(norms)
