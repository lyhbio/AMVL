import numpy as np

def A_VDA_GMSBMF(matDV, Wdd, Wvv, gm, w, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter):
    """
    A_VDA_GMSBMF: Matrix factorization for drug-disease association prediction.
    Args:
    matDV : target interaction matrix between diseases and drugs
    Wdd : disease similarity matrix
    Wvv : drug similarity matrix
    gm : gamma parameter for Gaussian kernel
    w : weighting parameter for similarity matrices
    lambda1, lambda2, lambda3 : regularization parameters
    k : latent dimension
    tol1, tol2 : tolerance for convergence
    maxiter : maximum number of iterations
    Returns:
    M_recovery : completed interaction matrix
    """
    w1 = w
    w2 = w
    Gdd, Gvv = getGIPSim_IN(matDV, gm, gm, False, False)
    Sdd = w1 * Gdd + (1 - w1) * Wdd
    Svv = w2 * Gvv + (1 - w2) * Wvv

    U, V, iter = A_MSBMF_IN(matDV, Sdd, Svv, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter)
    M_recovery = np.dot(U, V.T)

    return M_recovery

def A_MSBMF_IN(M, D, R, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter):
    """
    A_MSBMF_IN: Multi-Similarity Bilinear Matrix Factorization (MSBMF)
    Args:
    M : target matrix with only known entries and the unobserved entries are 0.
    D : disease similarity matrix
    R : drug similarity matrix
    lambda1, lambda2, lambda3 : regularization parameters
    k : latent dimension of matrix factorization
    tol1, tol2 : tolerance of termination conditions
    maxiter : maximum number of iterations
    Returns:
    X, Y : two latent low-rank matrices of the completed matrix
    iter : the number of iterations
    """
    np.random.seed(2019)  # fix random seed
    omega = (M != 0).astype(float)
    omega_ = np.ones(omega.shape) - omega
    U = np.random.rand(M.shape[0], k)
    V = np.random.rand(M.shape[1], k)
    P = np.random.rand(D.shape[1], k)
    Q = np.random.rand(R.shape[1], k)
    X = U.copy()
    Y = V.copy()
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
        U = np.dot(Z, V) + lambda2 * np.dot(D, P) - W1 + mu * X
        U = np.dot(U, np.linalg.inv(np.dot(V.T, V) + lambda2 * np.dot(P.T, P) + (lambda1 + mu) * np.eye(k)))

        V = np.dot(Z.T, U) + lambda2 * np.dot(R, Q) - W2 + mu * Y
        V = np.dot(V, np.linalg.inv(np.dot(U.T, U) + lambda2 * np.dot(Q.T, Q) + (lambda1 + mu) * np.eye(k)))

        P = np.dot(lambda2 * D.T, U)
        P = np.dot(P, np.linalg.inv(lambda2 * np.dot(U.T, U) + lambda3 * np.eye(k)))

        Q = np.dot(lambda2 * R.T, V)
        Q = np.dot(Q, np.linalg.inv(lambda2 * np.dot(V.T, V) + lambda3 * np.eye(k)))

        X = U + (1 / mu) * W1
        X[X < 0] = 0

        Y = V + (1 / mu) * W2
        Y[Y < 0] = 0

        Z = M * omega + np.dot(U, V.T) * omega_

        W1 = W1 + mu * (U - X)
        W2 = W2 + mu * (V - Y)

        stop1_0 = stop1
        stop1 = np.linalg.norm(np.dot(X, Y.T) - XY, 'fro') / np.linalg.norm(XY, 'fro')
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        XY = np.dot(X, Y.T)

        if stop1 < tol1 and stop2 < tol2:
            iter = i + 1
            break
        else:
            iter = i + 1
            mu = min(mu * rho, max_mu)

    return X, Y, iter

def getGIPSim_IN(Adm_interaction, gamma0_d, gamma0_m, AvoidIsolatedNodes=False, RemoveNonoverlapPairs=True):
    """
    getGIPSim_IN: Calculate Gaussian Interaction Profile kernel similarity
    Args:
    Adm_interaction : interaction matrix between disease and drug
    gamma0_d, gamma0_m : Gaussian kernel parameters
    AvoidIsolatedNodes : whether to avoid isolated nodes
    RemoveNonoverlapPairs : whether to remove non-overlapping pairs
    Returns:
    kd, km : Gaussian interaction profile similarity matrices for diseases and drugs
    """
    nd_all, nm_all = Adm_interaction.shape

    if AvoidIsolatedNodes:
        nodes_d = np.sum(Adm_interaction, axis=1) != 0
        nodes_m = np.sum(Adm_interaction, axis=0) != 0
        Adm = Adm_interaction[nodes_d, :][:, nodes_m]
    else:
        Adm = Adm_interaction

    nd, nm = Adm.shape
    SumOfSquares = np.sum(Adm**2)

    kd = None
    km = None

    if gamma0_d is not None:
        gamma_d = gamma0_d / (SumOfSquares / nd)
        D = np.dot(Adm, Adm.T)
        dd = np.diag(D)
        kd2 = np.exp(-gamma_d * (dd[:, None] + dd[None, :] - 2 * D))
        if RemoveNonoverlapPairs:
            kd2[D == 0] = 0
        if AvoidIsolatedNodes:
            kd = np.zeros((nd_all, nd_all))
            kd[np.ix_(nodes_d, nodes_d)] = kd2
        else:
            kd = kd2

    if gamma0_m is not None:
        gamma_m = gamma0_m / (SumOfSquares / nm)
        E = np.dot(Adm.T, Adm)
        mm = np.diag(E)
        km2 = np.exp(-gamma_m * (mm[:, None] + mm[None, :] - 2 * E))
        if RemoveNonoverlapPairs:
            km2[E == 0] = 0
        if AvoidIsolatedNodes:
            km = np.zeros((nm_all, nm_all))
            km[np.ix_(nodes_m, nodes_m)] = km2
        else:
            km = km2

    return kd, km
