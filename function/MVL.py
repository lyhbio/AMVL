import numpy as np
from scipy.linalg import solve_sylvester

def MVL_F(AR, AD, A, lamdaR, lamdaD):
    """
    Python implementation of Multi-View Prediction.

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
    NITER = 50
    thresh = 1e-10
    epsilon = 1e-5  # Regularization term to avoid division by zero

    F_old = A.copy()
    F = A.copy()
    num_drugs = A.shape[0]
    num_diseases = A.shape[1]
    SR = np.zeros_like(AR[0])
    SD = np.zeros_like(AD[0])

    idR = np.eye(num_drugs)
    idD = np.eye(num_diseases)

    for iter in range(NITER):
        # Update drug similarity matrix SR
        SR = mini_job(AR, F, lamdaR)
        # Update disease similarity matrix SD
        SD = mini_job(AD, F.T, lamdaD)

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

        # Solve the Sylvester equation
        F = solve_sylvester(2 * lamdaR * LSR + idR, 2 * lamdaD * LSD + idD, A)

        # Compute the change using Frobenius norm
        diff = np.linalg.norm(F - F_old, 'fro')
        if diff < thresh:
            break

        F_old = F.copy()

    return SR, SD, F

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
