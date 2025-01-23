###### ITRPCA ######
import numpy as np

def ITRPCA_F(Trr, Tdd, P_TMat, p, K, rat1, rat2):
    # WKNN Step
    dn, dr = P_TMat.shape
    P_TMat_new = WKNN(P_TMat, np.mean(Tdd, axis=2), np.mean(Trr, axis=2), K, 0.95)
    
    # Drug-tensor processing
    tr1, tr2, dr_num = Trr.shape
    Tdr = np.repeat(P_TMat_new[:, :, np.newaxis], dr_num, axis=2)
    R_ori = np.concatenate([Trr, Tdr], axis=0) * 255

    n1, n2, n3 = R_ori.shape
    n = min(n1, n2)
    
    # Compute the weights
    ind1 = [ffindw(R_ori[:, :, i], rat1) for i in range(dr_num)]
    a1 = round(np.mean(ind1))
    ind2 = [ffindw(R_ori[:, :, i], rat2) for i in range(dr_num)]
    a2 = -a1 + 2 + round(np.mean(ind2))
    
    w = np.concatenate([np.ones(a1), 2*np.ones(a2), 4*np.ones(n-a1-a2)])
    kao = 1 / (5 * np.sqrt(n1 * n2 * n3))
    
    # Tensor robust PCA with low-rank constraint
    R_ResultMat123, _, _, _, _ = itrpca_tnn_lp_stop(R_ori, kao, w, p, dr_num)
    R_ResultMat123 /= 255
    R_ResultMat = np.mean(R_ResultMat123[(n1-dn):n1, :dr, :dr_num], axis=2)
    
    # Disease-tensor processing
    td1, td2, dd_num = Tdd.shape
    Tdr = np.repeat(P_TMat_new[:, :, np.newaxis], dd_num, axis=2)
    D_ori = np.concatenate([Tdr, Tdd], axis=1) * 255

    nn1, nn2, nn3 = D_ori.shape
    nn = min(nn1, nn2)
    
    # Compute the weights
    indd1 = [ffindw(D_ori[:, :, i], rat1) for i in range(dd_num)]
    b1 = round(np.mean(indd1))
    indd2 = [ffindw(D_ori[:, :, i], rat2) for i in range(dd_num)]
    b2 = -b1 + 2 + round(np.mean(indd2))
    
    w = np.concatenate([np.ones(b1), 2*np.ones(b2), 4*np.ones(nn-b1-b2)])
    kao = 1 / (5 * np.sqrt(nn1 * nn2 * nn3))
    
    D_ResultMat123, _, _, _, _ = itrpca_tnn_lp_stop(D_ori, kao, w, p, dd_num)
    D_ResultMat123 /= 255
    D_ResultMat = np.mean(D_ResultMat123[:dn, :dr, :dd_num], axis=2)
    
    # Final Drug-Disease association matrix
    M_ResultMat = (R_ResultMat + D_ResultMat) / 2
    
    return M_ResultMat

def WKNN(DR_mat, D_mat, R_mat, K, r):
    rows, cols = DR_mat.shape
    y_d = np.zeros((rows, cols))
    y_r = np.zeros((rows, cols))

    knn_network_d = KNN(D_mat, K)
    for i in range(rows):
        w = np.zeros(K)
        sort_d, idx_d = np.sort(knn_network_d[i, :])[::-1], np.argsort(knn_network_d[i, :])[::-1]
        sum_d = np.sum(sort_d[:K])
        if sum_d == 0:
            sum_d = 1e-8

        for j in range(K):
            w[j] = r**(j) * sort_d[j]
            y_d[i, :] += w[j] * DR_mat[idx_d[j], :]
        y_d[i, :] /= sum_d
    
    knn_network_r = KNN(R_mat, K)
    for i in range(cols):
        w = np.zeros(K)
        sort_r, idx_r = np.sort(knn_network_r[i, :])[::-1], np.argsort(knn_network_r[i, :])[::-1]
        sum_r = np.sum(sort_r[:K])
        if sum_r == 0:
            sum_r = 1e-8

        for j in range(K):
            w[j] = r**(j) * sort_r[j]
            y_r[:, i] += w[j] * DR_mat[:, idx_r[j]]
        y_r[:, i] /= sum_r
    
    y_dr = (y_d + y_r) / 2
    DR_mat_new = np.maximum(DR_mat, y_dr)
    
    return DR_mat_new

def KNN(network, k):
    rows, cols = network.shape
    network = network - np.diag(np.diag(network))
    knn_network = np.zeros((rows, cols))
    sort_network = np.argsort(network, axis=1)[:, ::-1]
    for i in range(rows):
        knn_network[i, sort_network[i, :k]] = np.sort(network[i, :])[::-1][:k]
    return knn_network

def itrpca_tnn_lp_stop(X, lambda_, weight, p, dimdim):
    rho = 1.1
    mu = 1e-2
    max_mu = 1e10
    maxiter = 5
    tol1 = 1e-3
    tol2 = 1e-4

    dim = X.shape
    L = np.zeros(dim)
    S = L.copy()
    Y = L.copy()

    stop1 = 1
    X_0 = X.copy()
    for iter in range(maxiter):
        Lk = L.copy()
        Sk = S.copy()

        # Update L
        L, tnnL = prox_tnn(-S + X - Y / mu, weight / mu, p)
        L = np.clip(L, 0, 255)

        # Update S
        S = prox_l1(-L + X - Y / mu, lambda_ / mu)

        dY = L + S - X

        # Convergence condition
        X_1 = L
        sum_norm = np.sum([np.linalg.norm(X_1[:, :, j] - X_0[:, :, j], 'fro') / np.linalg.norm(X_0[:, :, j], 'fro') for j in range(dimdim)])
        stop1 = sum_norm
        stop2 = abs(stop1 - stop1) / max(1, abs(stop1))
        X_0 = X_1.copy()

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

        if stop1 < tol1 and stop2 < tol2:
            break

    return L, S, iter, stop1, stop2

def prox_tnn(Y, tau, p):
    # Placeholder for the proximal operator for tensor nuclear norm
    return Y, np.linalg.norm(Y)

def prox_l1(Y, lambda_):
    # Placeholder for the proximal operator for the L1 norm
    return np.sign(Y) * np.maximum(np.abs(Y) - lambda_, 0)

def ffindw(X, rat):
    # Placeholder for a function to determine the weighting index
    return int(X.shape[0] * rat)
