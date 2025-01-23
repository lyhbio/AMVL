import numpy as np

def WeightImputeLogFactorization(R, drug, disease, W, rank, lR, lM, lN, num_iter, learn_rate, rnseed):
    """
    WeightImputeLogFactorization function for matrix factorization using gradient descent.
    
    Args:
        R (np.array): Input relationship matrix (m, n).
        drug (np.array): Drug similarity matrix (m, m).
        disease (np.array): Disease similarity matrix (n, n).
        W (np.array): Weight matrix.
        rank (int): Rank size for matrix factorization.
        lR, lM, lN (float): Regularization parameters.
        num_iter (int): Maximum number of iterations.
        learn_rate (float): Learning rate for gradient descent.
        rnseed (int): Random seed for initialization.
    
    Returns:
        tF, tG: Factorized latent matrices.
    """
    # Set random seed
    np.random.seed(rnseed)

    # Get the shape of R
    m, n = R.shape

    # Initialize tF and tG matrices with a normal distribution
    tF = np.random.normal(0, 1/np.sqrt(rank), (m, rank))
    tG = np.random.normal(0, 1/np.sqrt(rank), (n, rank))
    tGt = tG.T

    # Identity matrices
    Im = np.eye(m)
    In = np.eye(n)

    # Initialize cumulative gradients
    df_sum = np.zeros((m, rank))
    dg_sum = np.zeros((n, rank))

    # Additional parameter initialization
    itr = 0
    Rt = R.T
    Wt = W.T

    # Get diagonal processed matrices
    DM, nM = GetDiag(drug, 10)
    DMM = DM - nM

    DN, nN = GetDiag(disease, 10)
    DNN = DN - nN

    # Set initial error values
    frobF_new = np.finfo(np.float64).max
    frobG_new = np.finfo(np.float64).max
    MINDIFF = 0.00001
    diff = np.finfo(np.float64).max

    # Iterative loop until convergence or max iterations
    while diff > MINDIFF and itr < num_iter:
        itr += 1  # Increment iteration counter
        frobF = frobF_new
        frobG = frobG_new

        # Calculate the prediction matrix P
        P = GetProbability(np.dot(tF, tGt))

        # Compute gradient of tF and update
        df = np.dot((W * (P - R)), tG) + 2 * lR * tF + 2 * lM * np.dot(DMM, tF)
        df_sum += df ** 2
        tF -= df * (learn_rate / np.sqrt(df_sum))

        # Recalculate P
        P = GetProbability(np.dot(tF, tGt))
        Pt = P.T

        # Compute gradient of tG and update
        dg = np.dot((Wt * (Pt - Rt)), tF) + 2 * lR * tG + 2 * lN * np.dot(DNN, tG)
        dg_sum += dg ** 2
        tG -= dg * (learn_rate / np.sqrt(dg_sum))
        tGt = tG.T

        # Calculate Frobenius norm
        frobF_new = np.linalg.norm(tF, 'fro')
        frobG_new = np.linalg.norm(tG, 'fro')

        # Calculate convergence difference
        diff = max(abs(frobF - frobF_new), abs(frobG - frobG_new))

    return tF, tG

def GetDiag(A, J):
    """
    Returns a diagonal matrix `DA` and a symmetric matrix `newA` based on the input matrix `A` and parameter `J`.
    
    Args:
        A (np.array): Input matrix.
        J (int): Number of top elements to retain in each row after removing diagonal elements.
    
    Returns:
        DA (np.array): Diagonal matrix with elements derived from the sum of top J elements.
        newA (np.array): Symmetric matrix with only top J elements retained.
    """
    m, n = A.shape

    # Set diagonal elements to zero
    np.fill_diagonal(A, 0)

    # Sort A in descending order along each row, get sorted values (SH) and their indices (ind)
    SH = -np.sort(-A, axis=1)  # Sort in descending order
    ind = np.argsort(-A, axis=1)  # Get indices of sorted elements

    # Initialize matrix to store top J elements in each row
    TOP_HORIZ = np.zeros((m, m))

    # Retain only top J elements in each row, with their original positions
    for i in range(m):
        for j in range(J):
            # The next largest element in row i is at position col
            col = ind[i, j]
            TOP_HORIZ[i, col] = A[i, col]  # Keep the top J elements, others remain zero

    # Calculate diagonal matrix DA
    DA = np.diag((np.sum(TOP_HORIZ, axis=0) + np.sum(TOP_HORIZ, axis=1)) / 2)

    # Create symmetric matrix newA
    newA = (TOP_HORIZ + TOP_HORIZ.T) / 2

    return DA, newA

def GetProbability(M):
    """
    Applies the sigmoid function element-wise to the input matrix M.

    Args:
        M (np.array): Input matrix.

    Returns:
        np.array: Matrix after applying the sigmoid function.
    """
    return np.exp(M) / (1 + np.exp(M))
