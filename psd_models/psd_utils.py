import numpy as np


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


def make_features(X, bw=1., dim=2, eps=1e-12, type='gaussian'):
    """
    Build kernel matrices and features required for PSD regression.

    Parameters
    ----------
    X: (n x p) ndarray
        Support points.

    bw: float, default=1.
        Kernel bandwidth.

    dim: int, default=2
        Desired matrix dimension.

    eps: float, default=1e-12
        Kernel regularization K + eps * Id (to avoid singular kernel matrices).

    type: str, default="gaussian"
        Kernel type in {"gaussian", "exponential"}.

    Returns
    -------
    K: (n x n) ndarray
        Kernel matrix.

    Psi: (n x n) ndarray
        Scalar kernel features.

    Psi: (n x (n * dim) x dim) ndarray
        Matrix kernel features.
    """

    n = len(X)
    K = kernel_func(X, X, bw, type) + eps * np.eye(n)
    Phi = np.linalg.cholesky(K).T
    Psi = np.kron(Phi, np.eye(dim)).T.reshape(n, dim, n * dim).transpose(0, 2, 1)

    return K, Phi, Psi


def PSD_func(x, B, Phi, X, bw=1., kernel_type='gaussian'):
    """
    Evaluate the PSD K-SoS at x

    Parameters
    ----------
    x: (p,) ndarray
        Point at which the SoS model is queried.

    B: (nd x nd) ndarray
        SoS parameter.

    Psi: (n x n) ndarray
        Scalar kernel features.

    X: (n x p) ndarray
        Support points.

    bw: float, default=1.
        Kernel bandwidth.

    kernel_type: str, default="gaussian"
        Kernel type in {"gaussian", "exponential"}.

    Returns
    -------
    (d x d) ndarray
    """
    n = len(Phi)
    dim = len(B) // n
    v = kernel_func(x, X, bw, type=kernel_type).T
    Ri = np.linalg.inv(Phi).T
    P = np.kron(Ri @ v, np.eye(dim))
    return P.T @ B @ P
