import numpy as np

def MSBMF_F(M, D, R, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter):
    """
    MSBMF: Drug repositioning based on multi-similarity bilinear matrix factorization.
    Usage: X, Y, iter = MSBMF(M, D_list, R_list, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter)

    Inputs:
        M         - the target matrix with only known entries and the unobserved entries are 0.
        D_list    - list of disease similarity matrices.
        R_list    - list of drug similarity matrices.
        lambda1   - regularization parameter.
        lambda2   - regularization parameter.
        lambda3   - regularization parameter.
        k         - the latent dimension of matrix factorization.
        tol1, tol2- tolerance of termination conditions.
        maxiter   - maximum number of iterations.

    Outputs:
        X, Y      - two latent low-rank matrices of the completed matrix.
        iter      - the number of iterations.
    """
    np.random.seed(2019)  # Set random seed
    omega = (M != 0).astype(float)
    omega_ = np.ones_like(omega) - omega

    # Initialize U, V, P, Q matrices
    U = np.random.rand(M.shape[0], k)
    V = np.random.rand(M.shape[1], k)
    P = np.random.rand(D.shape[1], k)
    Q = np.random.rand(R.shape[1], k)
    
    X = U
    Y = V
    Z = M.copy()
    W1 = np.zeros_like(U)
    W2 = np.zeros_like(V)
    XY = M.copy()

    rho = 1.05
    mu = 1e-4
    max_mu = 1e20

    stop1 = 1
    stop2 = 1

    for i in range(maxiter):
        # Update U matrix
        U = np.dot(Z, V) + lambda2 * np.dot(D, P) - W1 + mu * X
        U = np.dot(U, np.linalg.inv(np.dot(V.T, V) + lambda2 * np.dot(P.T, P) + (lambda1 + mu) * np.eye(k)))

        # Update V matrix
        V = np.dot(Z.T, U) + lambda2 * np.dot(R, Q) - W2 + mu * Y
        V = np.dot(V, np.linalg.inv(np.dot(U.T, U) + lambda2 * np.dot(Q.T, Q) + (lambda1 + mu) * np.eye(k)))

        # Update P matrix
        P = np.dot(lambda2 * D.T, U)
        P = np.dot(P, np.linalg.inv(lambda2 * np.dot(U.T, U) + lambda3 * np.eye(k)))

        # Update Q matrix
        Q = np.dot(lambda2 * R.T, V)
        Q = np.dot(Q, np.linalg.inv(lambda2 * np.dot(V.T, V) + lambda3 * np.eye(k)))

        # Update X
        X = U + (1 / mu) * W1
        X[X < 0] = 0  # Enforce non-negative constraint

        # Update Y
        Y = V + (1 / mu) * W2
        Y[Y < 0] = 0  # Enforce non-negative constraint

        # Update Z
        Z = M * omega + np.dot(U, V.T) * omega_

        # Update W1 and W2
        W1 = W1 + mu * (U - X)
        W2 = W2 + mu * (V - Y)

        # Calculate stopping criteria
        stop1_0 = stop1
        stop1 = np.linalg.norm(np.dot(X, Y.T) - XY, 'fro') / np.linalg.norm(XY, 'fro')
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        # Update XY for the next iteration
        XY = np.dot(X, Y.T)

        # Check for convergence
        if stop1 < tol1 and stop2 < tol2:
            iter = i + 1
            break
        else:
            iter = i + 1
            mu = min(mu * rho, max_mu)

    return X, Y, iter
