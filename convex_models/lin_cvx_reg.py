# Least-squares fit of a convex function.
import numpy as np

from cvxopt import solvers, matrix, spmatrix, mul
from pickle import load
solvers.options['show_progress'] = 0

def fit_affine_convex(X, Y):

    u, y = matrix(X), matrix(Y)
    m, dim = X.shape[:2]

    # minimize    (1/2) * || yhat - y ||_2^2
    # subject to  yhat[j] >= yhat[i] + g[i]' * (u[j] - u[i]), j, i = 0,...,m-1
    #
    # Variables  yhat (m), g (m).

    nvars = m + m * dim
    P = spmatrix(1.0, range(m), range(m), (nvars, nvars))
    q = matrix(0.0, (nvars,1))
    q[:m] = -y

    # m blocks (i = 0,...,m-1) of linear inequalities
    #
    #     yhat[i] + g[i]' * (u[j] - u[i]) <= yhat[j], j = 0,...,m-1.

    G = spmatrix([],[],[], (m**2, nvars))
    I = spmatrix(1.0, range(m), range(m))
    for i in range(m):
        # coefficients of yhat[i]
        G[list(range(i*m, (i+1)*m)), i] = 1.0

        # coefficients of g[i]
        for j in range(dim):
            G[list(range(i*m, (i+1)*m)), m + i * dim + j] = u[:, j] - u[i, j]

        # coefficients of yhat[j]
        G[list(range(i*m, (i+1)*m)), list(range(m))] -= I

    h = matrix(0.0, (m**2,1))

    sol = solvers.qp(P, q, G, h)
    yhat = sol['x'][:m]
    g = sol['x'][m:]

    return np.array(yhat).flatten(), np.array(g).reshape(m, dim)

def predict_affine_convex(x, yhat, g, X):
    return np.array([max(yhat + (g * (t - X)).sum(-1)) for t in x])
