import numpy as np
import scipy.linalg

import time
import logging

from psd_utils import kernel_func, make_features


class PSDRegressor():
    """
    PSD least-squares regression with kernel sum-of-squares model

    Parameters
    ----------

    bw: float, default=1e-1
        Kernel bandwith.

    lbda_1: float, default=1e-3
        Trace norm regularization factor.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor.

    max_iter: int, default=10000
        Maximum number of gradient descent iterations during training.

    precision: float, default=1e-6
        Gradient squared norm threshold to stop gradient descent.

    kernel_type: str, default='gaussian'
        Kernel function.
        Supported values : {'gaussian' or 'exponential'}

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    """
    def __init__(self,
                 bw=1e-1,
                 lbda_1=1e-3,
                 lbda_2=1e-3,
                 max_iter=10000,
                 precision=1e-6,
                 kernel_type='gaussian',
                 eps=1e-12):
        self.bw = bw
        self.lbda_1 = lbda_1
        self.lbda_2 = lbda_2
        self.max_iter = max_iter
        self.precision = precision
        self.kernel_type = kernel_type
        self.eps = eps

    def fit(self, X, Y, verbose=False, report_interval=1000, G_init=None):
        """
        Fits

        Parameters
        ----------
        X: (n x p) ndarray
            Features.

        Y: (n x d x d) ndarray
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
        n, dim = Y.shape[:2]
        self.dim = dim

        K, self.Phi, Psi = make_features(self.X, self.bw, self.dim, self.eps, self.kernel_type)

        self.G_, self.losses_, self.grad_norms_ = accelerated_gd(Y, Psi, K, lbda_1=self.lbda_1,
                                                                 lbda_2=self.lbda_2,
                                                                 precision=self.precision,
                                                                 max_iter=self.max_iter,
                                                                 verbose=verbose,
                                                                 report_interval=report_interval,
                                                                 G_init=G_init)
        # Build primal coefficients
        vals, vecs = np.linalg.eigh(((Psi @ self.G_) @ Psi.transpose(0, 2, 1)).sum(0)
                                    + self.lbda_1 * np.eye(n * self.dim))
        idxs = np.where(vals < 0)[0]

        if len(idxs) == 0:
            self.B = np.zeros((n * self.dim, n * self.dim))

        else:
            max_idx = idxs[-1] + 1
            H = -vecs[:, :max_idx] * vals[:max_idx] @ vecs[:, :max_idx].T
            Pi = np.kron(np.linalg.inv(self.Phi), np.eye(self.dim))
            B = Pi @ H @ Pi.T / self.lbda_2
            self.B = (B + B.T) / 2

    def predict(self, X):
        """
        Return the value of the model at X.

        Parameters
        ----------
        X: (n x p) ndarray
            Features.

        Returns
        -------
        (n x d x d) ndarray
        """
        assert hasattr(self, 'B'), "The model is not fitted."
        v = kernel_func(X, self.X, self.bw, self.kernel_type).T
        Ri = np.linalg.inv(self.Phi).T
        P = np.kron(Ri @ v, np.eye(self.dim)).reshape(len(X), self.dim, -1)
        return np.array([(P @ self.B)[i, :, :] @ P[i, :, :].T for i in range(len(P))]).squeeze()


# Optimization of a least squares objective
def loss_func(G, Y, Psi, lbda_1=0, lbda_2=1e-3, grad=True):
    """
    Compute the dual loss and its gradient.

    Parameters
    ----------

    G: (n_fill x d x d) ndarray
        Dual variable.

    Y: (n x d x d) ndarray
        Target.

    Psi: (n_fill x (r * d) x d) ndarray
        Kernel features.

    lbda_1: float, default=1e-3
        Trace norm regularization factor.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor.

    Returns
    -------
    loss: float
        Loss value.

    grad: (n_fill x d x d) ndarray
        Gradient w.r.t. G.
    """

    n = len(Y)

    neg_pen, neg_grad = neg_part(G, Psi, lbda_1=lbda_1, grad=True)
    loss = .5 * (n * (G ** 2).sum() + (G * Y).sum() + neg_pen / lbda_2)

    if grad:
        grad_ = n * G + Y / 2 + neg_grad / (2 * lbda_2)
        return loss, grad_
    else:
        return loss


def neg_part(G, Psi, lbda_1=0, grad=True):
    """
    Returns the squared norm of [-Phi diag(G) Phi.T]_+ and its gradient

    Parameters
    ----------
    X: (n x p) ndarray
        Features.

    Y: (n x d x d) ndarray
        Target.

    verbose: bool, default=True
        If True, output loss to log during iterations.

    report_interval: int, default=1000
        If verbose is True, the interval between two loss reports.

    G_init: (n_fill x d x d) ndarray, default=None
        Warmstart dual variable.
        Useful e.g. for cross-validation.
    """

    n, dim = G.shape[:2]
    r = Psi.shape[1]

    vals, vecs = np.linalg.eigh(((Psi @ G) @ Psi.transpose(0, 2, 1)).sum(0)
                                + lbda_1 * np.eye(r))

    idxs = np.where(vals < 0)[0]

    if len(idxs) == 0:
        if grad:
            return 0, np.zeros_like(G)
        else:
            return 0

    else:
        max_idx = idxs[-1] + 1
        if grad:
            grad_ = 2 * (Psi.transpose(0, 2, 1) @ ((vecs[:, :max_idx] *
                                                    vals[:max_idx] @ vecs[:, :max_idx].T) @ Psi))
            return (vals[:max_idx] ** 2).sum(), grad_
        else:
            return (vals[:max_idx] ** 2).sum()


def accelerated_gd(Y, Psi, K, lbda_1=0, lbda_2=1e-3, precision=1e-3, max_iter=10000,
                   verbose=True, report_interval=1000, G_init=None):
    """
    Performs accelerated gradient descent in the dual.

    Parameters
    ----------

    Y: (n x d x d) ndarray
        Target.

    Psi: (n x (n * d) x d) ndarray
        Kernel features.

    K: (n x n) ndarray
        Kernel matrix.

    lbda_1: float, default=1e-3
        Trace norm regularization factor.

    lbda_2: float, default=1e-3
        Squared Frobenius norm regularization factor.

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

    n, dim = Y.shape[:2]

    # Compute constants
    mu = n
    L = n + scipy.linalg.eigh(K ** 2, eigvals_only=True).max() / lbda_2
    lr = 1. / L
    acc = (1 - np.sqrt(mu / L)) / (1 + np.sqrt(mu / L))

    # Gradient descent
    if G_init is None:
        eta = np.zeros(Y.shape)
    else:
        eta = G_init.copy()

    G = eta.copy()

    losses = []
    norms = []

    start_time = time.time()

    for i in range(max_iter):

        # Compute loss and gradient:
        loss, grad = loss_func(eta, Y, Psi, lbda_1, lbda_2, grad=True)

        # Gradient update
        G_ = eta - lr * grad
        eta = G_ + acc * (G_ - G)

        losses.append(loss)
        norms.append((grad**2).sum())

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
