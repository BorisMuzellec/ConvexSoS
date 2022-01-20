import numpy as np
import logging

from sobol_seq import i4_sobol_generate


def kernel_func(x, X, bw=1., type='gaussian'):
    """
    Build kernel K(x, X)

    Parameters
    ----------
    x: (m x p) ndarray
        Support points.

    X: (n x p) ndarray
        Support points.

    bw: float, default=1.
        Kernel bandwidth.

    type: str, default="gaussian"
        Kernel type in {"gaussian", "exponential"}.

    Returns
    -------
    K: (m x n) ndarray
        Kernel matrix.
    """

    assert type in ['gaussian', 'exponential'], 'Kernel type not supported.'
    if type == 'gaussian':
        return np.exp(-((x[:, None] - X)**2).sum(-1) / (2 * bw))
    elif type == 'exponential':
        return np.exp(-np.sqrt(((x[:, None] - X)**2).sum(-1)) / (2 * bw))


def grid(n_fill, dim=2, lower=-1., upper=1.):
    """
    Sample n_fill points on a [lower, upper]^dim regular grid.

    Parameters
    ----------
    n_fill: int
        Number of points to sample.
        If n_fill is not of the form k ** dim, it will be rounded to the closest power of dim.

    dim: int, default=2
        Grid dimension.

    lower: float, default=-1.
        Lower bound on the grid.

    upper: float, default=1.
        Upper bound on the grid.

    Returns
    -------
    (n_fill x dim) ndarray
    """
    ngrid = int(n_fill ** (1 / dim))
    if ngrid ** dim != n_fill:
        logging.warning(f"n_fill = {n_fill} is not a power of dim = {dim}. "
                        f"n_fill will be set to {ngrid ** dim}")
    xy = np.meshgrid(*[np.linspace(lower, upper, ngrid) for i in range(dim)])
    return np.array(xy).reshape(dim, -1).T


def sobol_sample(n_fill, dim=2, lower=-1, upper=1):
    """
    Sample n_fill points from a Sobol quasi-random sequence.

    Parameters
    ----------
    n_fill: int
        Number of points to sample.

    dim: int, default=2
        Grid dimension.

    lower: float, default=-1.
        Lower bound on the grid.

    upper: float, default=1.
        Upper bound on the grid.

    Returns
    -------
    sob: (n_fill x dim) ndarray
    """
    sob = i4_sobol_generate(dim, n_fill, skip=3000)
    sob = (upper - lower) * sob + lower
    return sob


def kernel_hess(x, X, bw=1., type='gaussian', K=None):
    """
    Compute second order derivatives of the kernel w.r.t. X evaluated at x.

    Parameters
    ----------
    x: (n x d) ndarray
        Points at which to evaluate the hessian.

    X: (n_fill x d) ndarray
        Support points.

    bw: float, default=1.
        Kernel bandwidth.

    type: str, default="gaussian"
        Kernel type.
        Only the Gaussian kernel is currently supported.

    K: ndarray, default=None
        If provided, avoids recomputing k(x, X).

    Returns
    -------
    (n_fill x n x d x d) ndarray
        Kernel second order derivatives.
    """

    assert type == 'gaussian', "Only the Gaussian kernel is supported for now"

    _, dim = X.shape
    if K is None:
        K = kernel_func(x, X, bw, type)
    if type == 'gaussian':
        return ((x[:, None] - X)[:, :, :, None] * (x[:, None] - X)[:, :, None, :] / bw ** 2
                - np.eye(dim)[None, None, :, :] / bw) * K[:, :, None, None]


def make_features(X, V, bw, eps=1e-12, nystrom=False, r=20,
                  verbose=False, kernel_type='gaussian'):
    """
    Build kernel matrices and features required for convex regression.

    Parameters
    ----------
    X: (n x d) ndarray
        Support points of f.

    V: (n_fill x d) ndarray
        Points at which SoS constraints are enforced.

    bw: float, default=1.
        Kernel bandwidth.

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    nystrom: bool, default=False
        Whether to perform Nystrom approximation of kernel matrices.

    r: int, default=25
        Nystrom approximation rank (only applicable when nystrom is True).

    verbose: bool, default=False
        If true, print possible warnings to log.

    type: str, default="gaussian"
        Kernel type.
        Only the Gaussian kernel is currently supported.

    Returns
    -------

    K: (n_fill x n_fill) ndarray
        Kernel matrix.

    ddK_xv: (n_fill x n x d x d) ndarray
        Kernel second order derivatives.

    Psi: (n_fill x r) ndarray
        Scalar kernel features.

    Psi: (n_fill x (r * d) x d) ndarray
        Matrix kernel features.
    """

    n, dim = X.shape
    n_fill = len(V)

    K = kernel_func(X, X, bw, kernel_type) + eps * np.eye(n)
    K_xv = kernel_func(X, V, bw, kernel_type)
    ddK_xv = kernel_hess(X, V, bw, kernel_type, K=K_xv)

    if n_fill <= r:
        nystrom = False
        if verbose:
            logging.info("Nystrom rank bigger than full rank, performing no approximation.")

    # Features
    if nystrom:
        # Sample columns uniformly at random, without replacement
        idxs = np.random.choice(n_fill, r, replace=False)

        # Build indexes
        nidxs = np.array([i for i in range(n_fill) if i not in idxs])
        perm = np.concatenate([idxs, nidxs])
        inv_perm = np.argsort(perm)

        M = kernel_func(V[perm, None], V[idxs], bw).squeeze()
        M[:r, :r] += eps * np.eye(r)

        L = np.linalg.cholesky(M[:r, :r])
        R = (np.linalg.inv(L) @ M.T)[:, inv_perm]

        # Nyström Features
        Phi = R
        Psi = np.kron(Phi, np.eye(dim)).T.reshape(n_fill, dim, dim * r).transpose(0, 2, 1)
    else:
        K2 = kernel_func(V, V, bw, kernel_type) + eps * np.eye(n_fill)
        Phi = np.linalg.cholesky(K2).T
        Psi = np.kron(Phi, np.eye(dim)).T.reshape(n_fill, dim, dim * n_fill).transpose(0, 2, 1)

    return K, ddK_xv, Phi, Psi
