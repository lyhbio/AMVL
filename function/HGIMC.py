import numpy as np
from scipy.linalg import svd

def fBMC(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b):
    """
    fBMC: Bounded Matrix Completion
    Args:
    alpha, beta : parameters needed to give.
    T : the target matrix with only known entries and the unobserved entries are 0.
    trIndex : a matrix recording the observed positions in the target matrix.
    tol1, tol2 : tolerance of termination conditions.
    maxiter : maximum number of iterations.
    a, b : the left and right endpoints of the bounded interval.
    Returns:
    T_recovery : the completed matrix.
    iter : the number of iterations.
    """
    X = T.copy()
    W = X.copy()
    Y = X.copy()

    i = 1
    stop1 = 1
    stop2 = 1
    while stop1 > tol1 or stop2 > tol2:
        # the process of computing W
        tran = (1/beta) * (Y + alpha * (T * trIndex)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * trIndex)
        W[W < a] = a
        W[W > b] = b

        # the process of computing X
        X_1 = svt(W - (1/beta) * Y, 1/beta)

        # the process of computing Y
        Y = Y + beta * (X_1 - W)

        stop1_0 = stop1
        stop1 = np.linalg.norm(X_1 - X, 'fro') / np.linalg.norm(X, 'fro')
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        X = X_1
        i += 1

        if i < maxiter:
            iter = i - 1
        else:
            iter = maxiter
            print("Warning: reach maximum iteration~~do not converge!!!")
            break

    T_recovery = W
    return T_recovery, iter

def fGRB(A, sigma):
    """
    fGRB: Gaussian Radial Basis Function
    Args:
    A : the original similarity matrix.
    sigma : Gaussian bandwidth.
    Returns:
    B : the comprehensive similarity matrix.
    """
    m, n = A.shape
    B = np.zeros((m, n))

    for i in range(m):
        for j in range(i + 1):
            B[i, j] = np.exp(-np.linalg.norm(A[i, :] - A[j, :])**2 / (2 * sigma**2))
            B[j, i] = B[i, j]

    return B

def fHGI(alpha, A_DD, A_RR, A_DR):
    """
    fHGI: Heterogeneous Graph Inference
    Args:
    alpha : regularization parameter
    A_DD : disease similarity matrix
    A_RR : drug similarity matrix
    A_DR : drug-disease association matrix
    Returns:
    T_recovery : completed drug-disease association matrix
    """
    normWdd = fNorm(A_DD)
    normWrr = fNorm(A_RR)
    Wdr0 = A_DR.copy()

    Wdr_i = Wdr0.copy()
    Wdr_I = alpha * np.dot(np.dot(normWdd, Wdr_i), normWrr) + (1 - alpha) * Wdr0

    while np.max(np.abs(Wdr_I - Wdr_i)) > 1e-10:
        Wdr_i = Wdr_I
        Wdr_I = alpha * np.dot(np.dot(normWdd, Wdr_i), normWrr) + (1 - alpha) * Wdr0

    T_recovery = Wdr_I
    return T_recovery

def fNorm(A):
    """
    fNorm: The normalization function of similarity matrix
    Args:
    A : input matrix
    Returns:
    B : normalized matrix
    """
    num1, num2 = A.shape
    rnM = np.sum(A, axis=1)
    cnM = np.sum(A, axis=0)

    B = np.zeros((num1, num2))
    for i in range(num1):
        rsum = rnM[i]
        for j in range(num2):
            csum = cnM[j]
            if rsum == 0 or csum == 0:
                B[i, j] = 0
            else:
                B[i, j] = A[i, j] / np.sqrt(rsum * csum)

    return B

def svt(Y, x):
    """
    svt: Singular Value Thresholding operator for matrix Y by thresholding parameter x.
    Args:
    Y : input matrix
    x : thresholding parameter
    Returns:
    E : matrix after singular value thresholding
    """
    U, S, VT = svd(Y, full_matrices=False)
    v_new = np.maximum(S - x, 0)
    E = np.dot(U, np.dot(np.diag(v_new), VT))
    return E
