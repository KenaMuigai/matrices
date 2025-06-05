# Name 1: Agrawal Akruti
# Enrolment number 1: 3851641
# Name 2: Muigai Joy
# Enrolment number 2: 3830606

import csv
import random
import math
import numpy as np

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def euclidean_distance(v1, v2):
    """
    Compute Euclidean distance between two vectors.

    Args:
        v1, v2 (list of floats): Vectors.

    Returns:
        float: Euclidean distance.
    """
    return math.sqrt(sum((a - b)**2 for a, b in zip(v1, v2)))

def dot_product(v1, v2):
    """
    Compute dot product of two vectors.

    Args:
       v1, v2 (list of floats): Vectors.

    Returns:
        float: dot product.
    """
    return sum(a * b for a, b in zip(v1, v2))


def mat_transpose(matrix):
    """
    Transpose a matrix.

    Args:
      matrix (list of list of floats): A matrix.

    Returns:
        list of list of float: the transposed matrix.
    """
    return [list(col) for col in zip(*matrix)]


def matmul(A, B):
    """
    Multiply two matrices A and B.

    Args:
      A (list of list of floats): matrix A.
      B (list of list of floats): matrix B.

    Returns:
        list of list of float: the product of A and B.
    """
    result = []
    for row in A:
        new_row = []
        for col in zip(*B):
            new_row.append(sum(a * b for a, b in zip(row, col)))
        result.append(new_row)
    return result

def split_matrix(B, num_test=5000):
    """
    Randomly split matrix B into test matrix T and training matrix A.

    Args:
        B (list of list of floats): Full data matrix.
        num_test (int): Number of rows for test matrix.

    Returns:
        tuple: (T, A) where
            T (list of list of floats): Test matrix.
            A (list of list of floats): Training matrix.
    """
    indices = list(range(len(B)))
    test_indices = set(random.sample(indices, num_test))
    T = [B[i] for i in test_indices]
    A = [B[i] for i in indices if i not in test_indices]
    return T, A

# -------------------------------------------------
# Exercise 1.1: Data Loading and Preprocessing
# -------------------------------------------------

def load_and_preprocess_data(file_path):
    """
    Load Spotify data from CSV, remove first three columns, and convert decade to numeric year.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list of list of floats: Preprocessed data matrix.
    """
    # Implement Exercise 1.1
    B = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            row = row[3:]  # Remove first 3 columns
            decade_str = row[-1]
            if decade_str == '60s':
                decade = 1960
            elif decade_str == '70s':
                decade = 1970
            elif decade_str == '80s':
                decade = 1980
            elif decade_str == '90s':
                decade = 1990
            elif decade_str == '00s':
                decade = 2000
            elif decade_str == '10s':
                decade = 2010
            elif decade_str == '20s':
                decade = 2020
            else:
                continue  # skip invalid rows
            features = list(map(float, row[:-1])) + [float(decade)]
            B.append(features)
    return B

# -------------------------------------------------
# Exercise 2: Normalisation
# -------------------------------------------------

def normalise_matrix(matrix):
    """
    Normalise each column (except the last one) by subtracting mean and dividing by std dev.

    Args:
        matrix (list of list of floats): Data matrix.

    Returns:
        list of list of floats: Normalised matrix.
    """
    # Implement Exercise 1.2
    cols = list(zip(*matrix))
    normalized_cols = []
    for i, col in enumerate(cols):
        if i == len(cols) - 1:  # Last column: decade
            normalized_cols.append(col)
        else:
            mean = sum(col) / len(col)
            std = math.sqrt(sum((x - mean) ** 2 for x in col) / len(col))
            normalized_col = [(x - mean) / std if std != 0 else 0 for x in col]
            normalized_cols.append(normalized_col)
    normalized_matrix = list(map(list, zip(*normalized_cols)))
    return normalized_matrix


# -------------------------------------------------
# Exercise 3: Decade Extraction
# -------------------------------------------------

def extract_unique_decades(matrix):
    """
    Create a sorted list of unique decades from the last column.

    Args:
        matrix (list of list floats): Data matrix.

    Returns:
        list of floats: Sorted unique decades.
    """
    # Implement Exercise 1.3
    decades = sorted(set(row[-1] for row in matrix))
    return decades


# -------------------------------------------------
# Exercise 4: Create Decade Matrices
# -------------------------------------------------

def create_decade_matrices(matrix, decades):
    """
    For each decade, create a matrix containing rows of that decade.

    Args:
        matrix (list of list of floats): Normalised matrix.
        decades (list of floats): List of unique decades.

    Returns:
        dict: Mapping decade to corresponding matrix.
              Keys are floats.
              Values are lists of lists of floats.
    """
    # Implement Exercise 1.4
    decade_matrices = {}
    for dec in decades:
        decade_matrices[dec] = [row for row in matrix if row[-1] == dec]
    return decade_matrices



# -------------------------------------------------
# Exercise 5: Mean Vectors
# -------------------------------------------------

def compute_mean_vectors(decade_matrices):
    """
    Compute mean vector (excluding last column) for each decade matrix.

    Args:
        decade_matrices (dict): Mapping of decade to matrix.
                                Keys are floats.
                                Values are lists of lists of floats.

    Returns:
        dict: Mapping of decade to mean vector.
              Keys are floats.
              Values are lists of floats.

    """
    # Implement Exercise 1.5
    mean_vectors = {}
    for dec, mat in decade_matrices.items():
        mat_no_decade = [row[:-1] for row in mat]
        cols = list(zip(*mat_no_decade))
        mean_vector = [sum(col) / len(col) for col in cols]
        mean_vectors[dec] = mean_vector
    return mean_vectors



# -------------------------------------------------
# Exercise 6: Nearest Mean Classification
# -------------------------------------------------

def compute_success_rate_means(matrix, mean_vectors):
    """
    Compute success rate of nearest mean classification.

    Args:
        matrix (list of list of floats): (Test) data matrix.
        mean_vectors (dict): Decade to mean vector mapping.
                             Keys are floats.
                             Values are lists of floats.

    Returns:
        float: Fraction of correctly classified rows.
    """
    # Implement Exercise 1.6
    correct = 0
    for row in matrix:
        features = row[:-1]
        actual_decade = row[-1]
        predicted_decade = min(mean_vectors, key=lambda dec: euclidean_distance(features, mean_vectors[dec]))
        if predicted_decade == actual_decade:
            correct += 1
    return correct / len(matrix)


# -------------------------------------------------
# Exercise 7: SVD-based Classification
# -------------------------------------------------

def compute_success_rate_svd(matrix, decade_matrices, k):
    """
    Compute the classification success rate for SVD-based method.

    Args::
        matrix (list of list): Normalized data matrix (test or train set).
        decade_matrices (dict): A with decades (floats) as keys and
                                corresponding matrices (lists of lists of floats)
                                as values; such as the output of
                                'create_decade_matrices'
                                Values are lists of lists of floats.
         k(int): the number of factors to be considered

    Returns:
        float: Fraction of correctly classified rows.
    """
    # Implement Exercise 1.7
    svd_decades = {}
    for dec, mat in decade_matrices.items():
        A_dec = np.array([row[:-1] for row in mat])
        U, S, Vt = np.linalg.svd(A_dec, full_matrices=False)
        svd_decades[dec] = (U[:, :k], S[:k], Vt[:k, :])

    correct = 0
    for row in matrix:
        v = np.array(row[:-1])
        actual_decade = row[-1]
        min_residual = float('inf')
        predicted_decade = None
        for dec, (U, S, Vt) in svd_decades.items():
            Vk = Vt.T
            projection = Vk @ (Vk.T @ v)
            residual = np.linalg.norm(v - projection)
            if residual < min_residual:
                min_residual = residual
                predicted_decade = dec
        if predicted_decade == actual_decade:
            correct += 1
    return correct / len(matrix)


# -------------------------------------------------
# Main script for demonstration
# -------------------------------------------------


# Load data and preprocess
B = load_and_preprocess_data("spotify_dataset.csv")

# TODO:
# Split B into two matrices T and A,
T, A = split_matrix(B, num_test=5000)

# normalise appropriately
T_norm = normalise_matrix(T)
A_norm = normalise_matrix(A)

# determine the decades
decades = extract_unique_decades(A_norm)

# Compute for each decade i the matrix A_i using the function decade_matrices
decade_matrices = create_decade_matrices(A_norm, decades)

# Compute the mean vectors
mean_vectors = compute_mean_vectors(decade_matrices)

print("Nearest-mean success rate (test set):", compute_success_rate_means(T_norm, mean_vectors))
print("Nearest-mean success rate (train set):", compute_success_rate_means(A_norm, mean_vectors))

# SVD-based classification example (with k=4)
k = 4

# Compute SVD-based success rate
train_success = compute_success_rate_svd(A_norm,decade_matrices, k)
test_success = compute_success_rate_svd(T_norm,decade_matrices, k)

print(f"SVD classification success rate (train set, k={k}):", train_success)
print(f"SVD classification success rate (test set, k={k}):", test_success)
