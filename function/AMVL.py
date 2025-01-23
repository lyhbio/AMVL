import cupy as cp
import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

def AMVL(Wrd, Wrr_list, Wdd_list, params):
    """
    Training framework that combines BMC (Bi-relational Matrix Completion) and multi-view learning.
    
    Args:
    - Wrd (np.ndarray): Drug-disease interaction matrix.
    - Wrr_list (list of np.ndarray): List of drug similarity matrices.
    - Wdd_list (list of np.ndarray): List of disease similarity matrices.
    - params (dict): Dictionary of parameters for the algorithm.
    
    Returns:
    - F_final (np.ndarray): Final prediction matrix.
    """
    # Unpack parameters with default values
    alpha = params.get('alpha', 10)
    beta = params.get('beta', 10)
    lamdaR = params.get('lamdaR', 0.1)
    lamdaD = params.get('lamdaD', 0.1)
    threshold = params.get('threshold', 0.8)
    max_iter = params.get('max_iter', 300)
    tol1 = params.get('tol1', 2e-3)
    tol2 = params.get('tol2', 1e-5)
    gip_w = params.get('gip_w', 0.2)

    # Initialize interaction matrix T as a copy of Wrd
    T = Wrd.copy()
    
    # Step 1: BMC matrix completion
    print(">>> Step 1: Starting BMC matrix completion...")
    T_mc, _ = BMC_F(alpha, beta, T, tol1, tol2, max_iter)

    # Calculate GIP similarity matrices for diseases and drugs using the thresholded matrix
    print(">>> Calculating GIP similarity matrices for disease-disease and drug-drug interactions...")
    Grr, Gdd = gip_similarity(T_mc * (T_mc > threshold))

    # Combine Gaussian Interaction Profile (GIP) similarity with original similarity matrices
    Grr_list = [gip_w * Grr + (1 - gip_w) * Wrr for Wrr in Wrr_list]
    Gdd_list = [gip_w * Gdd + (1 - gip_w) * Wdd for Wdd in Wdd_list]

    # Remove diagonal elements to eliminate self-interactions
    print(">>> Removing diagonal elements to eliminate self-interactions...")
    Wrr_ML = [w.copy() for w in Grr_list]
    Wdd_ML = [w.copy() for w in Gdd_list]

    for i in range(len(Wrr_ML)):
        np.fill_diagonal(Wrr_ML[i], 0)
    for i in range(len(Wdd_ML)):
        np.fill_diagonal(Wdd_ML[i], 0)

    # Step 2: Matrix Factorization with similarity regularization
    print(">>> Step 2: Performing Matrix Factorization with similarity regularization...")
    U, V, iter_count = MSBMF_F(T.T, Wdd_list, Wrr_list, lamdaD, lamdaR, min(T.shape), tol1, tol2, max_iter)
    print(f">>> Matrix factorization completed in {iter_count} iterations.")

    # Recover the matrix by combining factorized matrices
    T_msbmf = np.dot(U, V.T).T
    T_recovery = np.maximum(T_msbmf, T_mc)

    # Threshold the recovered matrix to eliminate weak associations
    print(">>> Thresholding the recovered matrix based on the threshold value...")
    T_update_mc = T_recovery * (T_recovery > threshold)

    # Step 3: Multi-view learning
    print(">>> Step 3: Starting multi-view learning...")
    _, _, F_mv = MVL_F(Wrr_ML, Wdd_ML, T_update_mc, lamdaR, lamdaD)

    # Step 4: Final combination of matrices and ensuring values are within the range [0, 1]
    print(">>> Step 4: Combining results and ensuring values are in [0, 1] range...")
    F_final = np.maximum(T_update_mc, F_mv)
    F_final = np.clip(F_final, 0, 1)

    print(">>> AdaMVL training complete. Returning final prediction matrix.")
    return F_final

###### GIP ######
def gip_similarity(mat_wrd, gamma_d=0.5, gamma_m=0.5):
    """
    Compute the Gaussian interaction profile similarity for diseases and drugs using the RBF kernel.

    Parameters
    ----------
    mat_dv : ndarray of shape (nd, nm)
        Disease-drug interaction matrix where rows represent diseases and columns represent drugs.
    gamma_d : float
        RBF kernel parameter controlling the width of the similarity distribution for diseases.
    gamma_m : float
        RBF kernel parameter controlling the width of the similarity distribution for drugs.

    Returns
    -------
    gdd : ndarray of shape (nd, nd)
        Disease-disease similarity matrix.
    gvv : ndarray of shape (nm, nm)
        Drug-drug similarity matrix.
    """
    # Compute drug-drug similarity matrix
    grr = rbf_kernel(mat_wrd, mat_wrd, gamma=gamma_d)

    # Compute disease-disease similarity matrix
    gdd = rbf_kernel(mat_wrd.T, mat_wrd.T, gamma=gamma_m)

    return grr, gdd

###### MSBMF #######
def MSBMF_F(M, D_list, R_list, lambda1, lambda2, k, tol1, tol2, maxiter, use_gpu=False, seed=42):
    """
    MSBMF: Drug repositioning based on multi-similarity bilinear matrix factorization.
    Usage: X, Y, iter = MSBMF(M, D_list, R_list, lambda1, lambda2, lambda3, k, tol1, tol2, maxiter)

    Inputs:
        M         - the target matrix with only known entries and the unobserved entries are 0.
        D_list    - list of disease similarity matrices.
        R_list    - list of drug similarity matrices.
        lambda1   - regularization parameter.
        lambda2   - regularization parameter.
        k         - the latent dimension of matrix factorization.
        tol1, tol2- tolerance of termination conditions.
        maxiter   - maximum number of iterations.

    Outputs:
        X, Y      - two latent low-rank matrices of the completed matrix.
        iter      - the number of iterations.
    """
    xp = cp if use_gpu else np

    # Convert inputs to CuPy arrays
    M = xp.array(M)
    D = xp.hstack([xp.array(d) for d in D_list])
    R = xp.hstack([xp.array(r) for r in R_list])
    
    xp.random.seed(seed)  # Set random seed
    omega = (M != 0).astype(float)
    omega_ = xp.ones_like(omega) - omega

    # Initialize U, V, P, Q matrices
    U = xp.random.rand(M.shape[0], k)
    V = xp.random.rand(M.shape[1], k)
    P = xp.random.rand(D.shape[1], k)
    Q = xp.random.rand(R.shape[1], k)
    
    X = U
    Y = V
    Z = M.copy()
    W1 = xp.zeros_like(U)
    W2 = xp.zeros_like(V)
    XY = M.copy()

    rho = 1.05
    mu = 1e-4
    max_mu = 1e20

    stop1 = 1
    stop2 = 1

    for i in range(maxiter):
        # Update U matrix
        U = xp.dot(Z, V) + lambda2 * xp.dot(D, P) - W1 + mu * X
        U = xp.dot(U, xp.linalg.inv(xp.dot(V.T, V) + lambda2 * xp.dot(P.T, P) + (lambda1 + mu) * xp.eye(k)))

        # Update V matrix
        V = xp.dot(Z.T, U) + lambda2 * xp.dot(R, Q) - W2 + mu * Y
        V = xp.dot(V, xp.linalg.inv(xp.dot(U.T, U) + lambda2 * xp.dot(Q.T, Q) + (lambda1 + mu) * xp.eye(k)))

        # Update P matrix
        P = xp.dot(lambda2 * D.T, U)
        P = xp.dot(P, xp.linalg.inv(lambda2 * xp.dot(U.T, U) + lambda2 * xp.eye(k)))

        # Update Q matrix
        Q = xp.dot(lambda2 * R.T, V)
        Q = xp.dot(Q, xp.linalg.inv(lambda2 * xp.dot(V.T, V) + lambda2 * xp.eye(k)))

        # Update X
        X = U + (1 / mu) * W1
        X[X < 0] = 0  # Enforce non-negative constraint

        # Update Y
        Y = V + (1 / mu) * W2
        Y[Y < 0] = 0  # Enforce non-negative constraint

        # Update Z
        Z = M * omega + xp.dot(U, V.T) * omega_

        # Update W1 and W2
        W1 = W1 + mu * (U - X)
        W2 = W2 + mu * (V - Y)

        # Calculate stopping criteria
        stop1_0 = stop1
        stop1 = xp.linalg.norm(xp.dot(X, Y.T) - XY, 'fro') / xp.linalg.norm(XY, 'fro')
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))

        # Update XY for the next iteration
        XY = xp.dot(X, Y.T)

        # Check for convergence
        if stop1 < tol1 and stop2 < tol2:
            iter = i + 1
            break
        else:
            iter = i + 1
            mu = min(mu * rho, max_mu)

    if use_gpu:
        # Convert results back to NumPy arrays
        X = xp.asnumpy(X)
        Y = xp.asnumpy(Y)

    return X, Y, iter

###### Multi-View Learning ######
def l2_distance(a, b):
    """
    Compute the squared Euclidean distance between column vectors.

    Args:
    - a (np.ndarray): First input matrix.
    - b (np.ndarray): Second input matrix.

    Returns:
    - d (np.ndarray): Matrix of pairwise squared distances.
    """
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    d = np.add.outer(aa, bb) - 2 * ab
    d = np.maximum(d, 0)  # Ensure non-negative distances
    return d

def e_proj_simplex(v, k=1):
    """
    Project the vector v onto the simplex defined by sum(x) = k and x >= 0.
    Optimized using a sorting algorithm with time complexity O(n log n).

    Args:
    - v (np.ndarray): Input vector.
    - k (float): Value of the sum constraint.

    Returns:
    - w (np.ndarray): The projected vector.
    """
    n = len(v)
    if n == 0:
        return np.array([])
    u = np.sort(v)[::-1]  # Sort v in descending order
    cssv = np.cumsum(u) - k
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    if np.any(cond):
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = np.maximum(v - theta, 0)
    else:
        w = np.zeros_like(v)
    return w

def mini_job(mv_matrix, F, lambda_):
    """
    Update similarity matrix for multi-view learning by minimizing distance with respect to lambda.

    Args:
    - mv_matrix (list of np.ndarray): List of similarity matrices from different views.
    - F (np.ndarray): Feature matrix.
    - lambda_ (float): Regularization parameter.

    Returns:
    - S (np.ndarray): Updated similarity matrix.
    """
    distd = l2_distance(F.T, F.T)
    # Pre-compute the average of mv_matrix to avoid redundant calculations
    a0 = np.mean(mv_matrix, axis=0)
    ad = a0 - 0.5 * lambda_ * distd
    S = np.apply_along_axis(e_proj_simplex, 1, ad)
    return S

def mini_job_gpu(AR, F, lamda):
    # GPU-accelerated version of mini_job with enhanced precision.
    distd_gpu = cp.asarray(l2_distance(cp.asnumpy(F.T), cp.asnumpy(F.T)))
    a0_gpu = sum([cp.asarray(ar) for ar in AR]) / len(AR)
    ad_gpu = a0_gpu - 0.5 * lamda * distd_gpu
    # Apply projection with more iterations for better precision
    S_gpu = cp.apply_along_axis(lambda v: cp.asarray(e_proj_simplex(cp.asnumpy(v), k=1)), 1, ad_gpu)
    return S_gpu

def solve_sylvester_admm(A, B, C, rho=1e-1, max_iter=100, tol=1e-6):
    """
    Solve the Sylvester equation using ADMM with increased precision.

    Args:
    - A (np.ndarray): Coefficient matrix A.
    - B (np.ndarray): Coefficient matrix B.
    - C (np.ndarray): Constant matrix C.
    - rho (float): Augmented Lagrangian parameter.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Tolerance for convergence.

    Returns:
    - X (np.ndarray): Solution matrix X.
    """
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    C_gpu = cp.asarray(C)

    m, n = C_gpu.shape
    X_gpu = cp.zeros((m, n))
    Z_gpu = cp.zeros((m, n))
    U_gpu = cp.zeros((m, n))

    I_m = cp.eye(m)
    I_n = cp.eye(n)

    A_inv_gpu = cp.linalg.inv(A_gpu + rho * I_m)
    B_inv_gpu = cp.linalg.inv(B_gpu + rho * I_n)

    for _ in range(max_iter):
        V_gpu = C_gpu - Z_gpu + U_gpu
        X_gpu = A_inv_gpu @ V_gpu @ B_inv_gpu

        Z_old_gpu = Z_gpu.copy()
        Z_gpu = (X_gpu + U_gpu)

        U_gpu = U_gpu + (X_gpu - Z_gpu)

        r_norm = cp.linalg.norm(X_gpu - Z_gpu, ord='fro')
        s_norm = cp.linalg.norm(-rho * (Z_gpu - Z_old_gpu), ord='fro')

        if r_norm < tol and s_norm < tol:
            break

    return cp.asnumpy(X_gpu)

def MVL_F(AR, AD, A, lamdaR, lamdaD):
    """
    Python implementation of Multi-View Prediction with ADMM to solve Sylvester equation.

    Args:
    - AR (list of np.ndarray): List of drug similarity matrices from multiple views.
    - AD (list of np.ndarray): List of disease similarity matrices from multiple views.
    - A (np.ndarray): Drug-disease association matrix.
    - lamdaR (float): Regularization parameter for drug similarity.
    - lamdaD (float): Regularization parameter for disease similarity.

    Returns:
    - SR (np.ndarray): Updated drug similarity matrix.
    - SD (np.ndarray): Updated disease similarity matrix.
    - F (np.ndarray): Multi-view prediction result.
    """
    print(f">>> Performing multi-view learning with {len(AR)} drug views and {len(AD)} disease views...")

    NITER = 100
    thresh = 1e-10
    epsilon = 1e-5

    F_old = A.copy()
    F = A.copy()
    num_drugs = A.shape[0]
    num_diseases = A.shape[1]
    SR = np.zeros_like(AR[0])
    SD = np.zeros_like(AD[0])

    idR = np.eye(num_drugs)
    idD = np.eye(num_diseases)

    for iter in range(NITER):
        # Update drug similarity matrix SR (using GPU acceleration with enhanced precision)
        F_gpu = cp.asarray(F)
        SR_gpu = mini_job_gpu([cp.asarray(ar) for ar in AR], F_gpu, lamdaR)
        SR = cp.asnumpy(SR_gpu)

        # Update disease similarity matrix SD (using GPU acceleration with enhanced precision)
        SD_gpu = mini_job_gpu([cp.asarray(ad) for ad in AD], F_gpu.T, lamdaD)
        SD = cp.asnumpy(SD_gpu)

        # Process drug similarity matrix
        SR0 = SR - np.diag(np.diag(SR))
        SR1 = (SR0 + SR0.T) / 2
        DR_diag = np.sum(SR1, axis=1) + epsilon
        DR_sqrt_inv = np.diag(1 / np.sqrt(DR_diag))
        LSR = idR - DR_sqrt_inv @ SR1 @ DR_sqrt_inv

        # Process disease similarity matrix
        SD0 = SD - np.diag(np.diag(SD))
        SD1 = (SD0 + SD0.T) / 2
        DD_diag = np.sum(SD1, axis=1) + epsilon
        DD_sqrt_inv = np.diag(1 / np.sqrt(DD_diag))
        LSD = idD - DD_sqrt_inv @ SD1 @ DD_sqrt_inv

        # Solve the Sylvester equation using ADMM with more iterations for better precision
        F = solve_sylvester_admm(2 * lamdaR * LSR + idR, 2 * lamdaD * LSD + idD, A)

        # Compute the change using Frobenius norm
        diff = np.linalg.norm(F - F_old, 'fro')
        if diff < thresh:
            print(f'>>> Iteration stops at {iter + 1} step.')
            break

        F_old = F.copy()

    return SR, SD, F

###### Bounded Matrix Completion ######
def BMC_F(alpha, beta, T, tol1, tol2, max_iter):
    """
    BMC: Bounded Matrix Completion (BMC) Algorithm using CuPy for GPU acceleration.
    This algorithm completes the input matrix T by optimizing it while ensuring that the values are within the specified bounds [0, 1].

    Args:
    - alpha, beta (float): Parameters for the algorithm controlling the influence of different terms.
    - T (np.ndarray or cp.ndarray): Target matrix with observed entries, while unobserved entries are set to 0.
    - tr_index (np.ndarray or cp.ndarray): Binary matrix indicating the observed entries in T (1 for observed, 0 for unobserved).
    - tol1, tol2 (float): Tolerance values for convergence criteria.
    - max_iter (int): Maximum number of iterations allowed.

    Returns:
    - T_recovery (cp.ndarray): Completed matrix after BMC.
    - iter_count (int): Number of iterations performed by the algorithm.
    """
    # Initialization
    tr_index = (T != 0).astype(float)

    # Convert numpy arrays to cupy arrays if necessary
    if isinstance(T, np.ndarray):
        T = cp.array(T)
    if isinstance(tr_index, np.ndarray):
        tr_index = cp.array(tr_index)

    # Initialize variables
    X = T.copy()  # X will store the current estimate of the completed matrix
    W = X.copy()  # W will store the matrix after boundary constraints
    Y = cp.zeros_like(X)  # Dual variable initialized to zero

    i = 1  # Iteration counter
    stop1 = 1  # Convergence criterion 1 (change in X)
    stop2 = 1  # Convergence criterion 2 (stability of stop1)

    # Start the BMC iteration process
    while (stop1 > tol1 or stop2 > tol2) and i <= max_iter:
        # Step 1: Update W using the current X and Y matrices
        tran = (1 / beta) * (Y + alpha * (T * tr_index)) + X
        W = tran - (alpha / (alpha + beta)) * (tran * tr_index)
        # Apply boundary constraints on W to ensure values are within [0, 1]
        W = cp.clip(W, 0, 1)

        # Step 2: Update X using Singular Value Thresholding (SVT) on W - (1 / beta) * Y
        X_new = svt(W - (1 / beta) * Y, 1 / beta)

        # Step 3: Update the dual variable Y
        Y = Y + beta * (X_new - W)

        # Step 4: Calculate the stopping criteria
        stop1_0 = stop1  # Store the previous stop1 value
        stop1 = cp.linalg.norm(X_new - X, 'fro') / (cp.linalg.norm(X, 'fro') + 1e-8)  # Frobenius norm difference
        stop2 = abs(stop1 - stop1_0) / max(1, abs(stop1_0))  # Check stability of stop1

        # Update X for the next iteration
        X = X_new.copy()
        i += 1  # Increment iteration counter

    # Check if the algorithm reached the maximum number of iterations
    if i > max_iter:
        print('>>> Warning: Reached maximum iteration without convergence!')
    else:
        print(f'>>> BMC converged in {i - 1} iterations.')

    # Return the recovered matrix and the actual number of iterations
    T_recovery = cp.asnumpy(W)
    iter_count = i - 1
    return T_recovery, iter_count

def svt(M, tau):
    """
    Singular Value Thresholding (SVT) function using CuPy.
    Performs Singular Value Decomposition (SVD) on the input matrix M and applies soft-thresholding to singular values.

    Args:
    - M (cp.ndarray): Input matrix (on GPU).
    - tau (float): Threshold parameter.

    Returns:
    - X (cp.ndarray): Matrix after thresholding.
    """
    # Singular Value Decomposition using CuPy
    U, S, VT = cp.linalg.svd(M, full_matrices=False)
    # Apply soft-thresholding to the singular values
    S_threshold = cp.maximum(S - tau, 0)
    # Reconstruct the matrix
    X = cp.dot(U, cp.dot(cp.diag(S_threshold), VT))
    return X
