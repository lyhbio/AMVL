import numpy as np

def BMC_F(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b):
    """
    fBMC: Bounded Matrix Completion
    Usage: T_recovery, iter = fBMC(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b)

    Inputs:
        alpha, beta   - regularization parameters.
        T             - the target matrix with only known entries and the unobserved entries are 0.
        trIndex       - a matrix recording the observed positions in the target matrix.
        tol1, tol2    - tolerance of termination conditions.
        maxiter       - maximum number of iterations.
        a, b          - the left and right endpoints of the bounded interval.

    Outputs:
        T_recovery    - the completed matrix.
        iter          - the number of iterations.
    """
    # Initialization
    X = T.copy()
    W = X.copy()
    Y = X.copy()

    i = 1
    stop1 = 1
    stop2 = 1

    while stop1 > tol1 or stop2 > tol2:
        # Process of computing W
        tran = (1/beta) * (Y + alpha * (T * trIndex)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * trIndex)
        W[W < a] = a
        W[W > b] = b

        # Process of computing X using singular value thresholding (SVT)
        X_1 = svt(W - (1/beta) * Y, 1/beta)

        # Process of computing Y
        Y = Y + beta * (X_1 - W)

        # Calculate stopping criteria
        stop1_0 = stop1
        stop1 = np.linalg.norm(X_1 - X, 'fro') / np.linalg.norm(X, 'fro')
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        X = X_1
        i += 1

        if i < maxiter:
            iter = i - 1
        else:
            iter = maxiter
            print("Warning: reached maximum iteration, did not converge!!!")
            break

    T_recovery = W
    return T_recovery, iter

def svt(Y, x):
    """
    SVT: Singular Value Thresholding operator for matrix Y by thresholding parameter x.

    Inputs:
        Y - Input matrix.
        x - Threshold parameter.

    Outputs:
        E - Matrix after applying singular value thresholding.
    """
    # Perform singular value decomposition
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s = np.maximum(s - x, 0)  # Apply the thresholding on singular values
    E = np.dot(U, np.dot(np.diag(s), Vt))  # Reconstruct the matrix with thresholded singular values
    return E
