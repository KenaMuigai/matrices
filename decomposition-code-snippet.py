# Name 1: Agrawal Akruti
# Enrolment number 1: 3851641
# Name 2: Muigai Joy Makena
# Enrolment number 2: 3830606

def matrix_addition(A, B):
    """
    Perform matrix addition.

    Args:
    A (list of list of floats): The first matrix.
    B (list of list of floats): The second matrix.

    Returns:
    list of list of floats: The resulting matrix after addition.
    """
    # Implement exercise 1.1
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    return result

def matrix_multiplication(A, B):
    """
    Perform matrix multiplication.

    Args:
    A (list of list of floats): The first matrix.
    B (list of list of floats): The second matrix.

    Returns:
    list of list of floats: The resulting matrix after multiplication.
    """
    # Implement exercise 1.2
    result = []
    num_rows_A = len(A)
    num_cols_B = len(B[0])
    num_cols_A = len(A[0])

    for i in range(num_rows_A):
        row = []
        for j in range(num_cols_B):
            val = 0
            for k in range(num_cols_A):
                val += A[i][k] * B[k][j]
            row.append(val)
        result.append(row)

    return result

def is_lower_triangular(matrix):
    """
    Check if a given matrix is lower triangular.

    Args:
    matrix (list of list of floats): The matrix to check.

    Returns:
    bool: True if the matrix is lower triangular, False otherwise.
    """
    # Implement exercise 1.3
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != 0:
                return False
    return True

def row_swap_maybe(L, P, k):
    """
    Perform row swap on a given matrix.

    Args:
    L (list of list of floats): The matrix L_{(k)}
    P (list of list of floats): The matrix P_{(k)}
    k (int): The current row index.

    Returns:
    tuple: A tuple containing the updated matrices L and P.
    """
    # Implement exercise 1.4
    n = len(L)

    # If pivot is non-zero, no need to swap
    if L[k][k] != 0:
        return L, P

    # Find a row q > k such that L[q][k] != 0
    q = -1
    for i in range(k + 1, n):
        if L[i][k] != 0:
            q = i
            break

    # If no such row exists, return as-is (or raise error)
    if q == -1:
        return L, P

    # Swap rows k and q in L
    L[k], L[q] = L[q], L[k]

    # Swap rows k and q in P
    P[k], P[q] = P[q], P[k]

    return L, P

def update_matrices(L, R, k):
    """
    Define R' and update matrices R and L.

    Args:
    L (list of list of floats): The k-triangular matrix  L_{(k)}.
    R (list of list of floats): The upper triangular matrix  R_{(k)}.
    k (int): The current row index.

    Returns:
    tuple: A tuple containing the updated matrices L and R as defined in step (c) of the exercise sheet.
    """
    # Implement exercise 1.5
    n = len(L)
    R_prime = [[0.0 for _ in range(n)] for _ in range(n)]

    # Build R'
    for j in range(k + 1, n):
        R_prime[k][j] = L[k][j] / L[k][k]

    # Update R = R + R'
    R_updated = matrix_addition(R, R_prime)

    # Build (E - R')
    E_minus_Rp = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            E_minus_Rp[i][j] -= R_prime[i][j]

    # Update L = L * (E - R')
    L_updated = matrix_multiplication(L, E_minus_Rp)

    return L_updated, R_updated



def triangular_decomposition(A):
    """
    Perform triangular decomposition on a given matrix.

    Args:
    A (list of list of floats): The matrix to decompose.

    Returns:
    tuple: A tuple containing the permutation matrix P, the lower triangular matrix L, and the upper triangular matrix R as defined via the algorithm of the exercise sheet.
    """
    # Implement exercise 1.6
    import copy
    n = len(A)
    E = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    P = copy.deepcopy(E)
    R = copy.deepcopy(E)
    L = copy.deepcopy(A)

    k = 0
    while True:
        # Step (a): if L is already lower triangular, stop
        if is_lower_triangular(L):
            return P, L, R

        # Step (b): check for zero pivot and perform row swap
        if L[k][k] == 0:
            L, P = row_swap_maybe(L, P, k)

        # Step (c): update L and R using the algorithm
        L, R = update_matrices(L, R, k)

        k += 1
        if k >= n:
            return P, L, R

if __name__ == "__main__":
    A = [[6.0, 6.0, 0.0], [6.0, 13.0, 0.0], [3.0, -46.0, 70.0]]
    P, L, R = triangular_decomposition(A)

    print("P =", P)
    print("L =", L)
    print("R =", R)

