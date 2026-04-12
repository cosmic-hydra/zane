# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Optimized molecular fingerprint operations in Cython.

High-performance implementation of fingerprinting and similarity calculations
that would be slow in pure Python.
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport exp, sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ctypedef np.int32_t INT32_t
ctypedef np.uint8_t UINT8_t


@cython.boundscheck(False)
@cython.wraparound(False)
def tanimoto_similarity_batch(
    np.ndarray[DTYPE_t, ndim=2] fingerprints1,
    np.ndarray[DTYPE_t, ndim=2] fingerprints2
):
    """
    Compute Tanimoto similarity between two sets of fingerprints.

    Parameters
    ----------
    fingerprints1 : ndarray, shape (n_mols1, n_features)
        First set of molecular fingerprints
    fingerprints2 : ndarray, shape (n_mols2, n_features)
        Second set of molecular fingerprints

    Returns
    -------
    similarities : ndarray, shape (n_mols1, n_mols2)
        Pairwise Tanimoto similarity scores
    """
    cdef int n_mols1 = fingerprints1.shape[0]
    cdef int n_mols2 = fingerprints2.shape[0]
    cdef int n_features = fingerprints1.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] similarities = np.zeros(
        (n_mols1, n_mols2), dtype=DTYPE
    )

    cdef int i, j, k
    cdef DTYPE_t intersection, union, val1, val2

    for i in prange(n_mols1, nogil=True, schedule='static'):
        for j in range(n_mols2):
            intersection = 0.0
            union = 0.0
            for k in range(n_features):
                val1 = fingerprints1[i, k]
                val2 = fingerprints2[j, k]
                intersection += min(val1, val2)
                union += max(val1, val2)

            if union > 0:
                similarities[i, j] = intersection / union
            else:
                similarities[i, j] = 0.0

    return similarities


@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_distance_batch(
    np.ndarray[DTYPE_t, ndim=2] points1,
    np.ndarray[DTYPE_t, ndim=2] points2
):
    """
    Compute pairwise Euclidean distances between two sets of points.

    Parameters
    ----------
    points1 : ndarray, shape (n_points1, n_dims)
        First set of points
    points2 : ndarray, shape (n_points2, n_dims)
        Second set of points

    Returns
    -------
    distances : ndarray, shape (n_points1, n_points2)
        Pairwise Euclidean distances
    """
    cdef int n1 = points1.shape[0]
    cdef int n2 = points2.shape[0]
    cdef int n_dims = points1.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.zeros(
        (n1, n2), dtype=DTYPE
    )

    cdef int i, j, k
    cdef DTYPE_t diff, dist_sq

    for i in prange(n1, nogil=True, schedule='static'):
        for j in range(n2):
            dist_sq = 0.0
            for k in range(n_dims):
                diff = points1[i, k] - points2[j, k]
                dist_sq += diff * diff
            distances[i, j] = sqrt(dist_sq)

    return distances


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_transform(np.ndarray[DTYPE_t, ndim=1] x):
    """
    Apply sigmoid transformation to array elements.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Input array

    Returns
    -------
    y : ndarray, shape (n,)
        Sigmoid transformed output
    """
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(n, dtype=DTYPE)

    cdef int i
    cdef DTYPE_t val

    for i in prange(n, nogil=True, schedule='static'):
        val = x[i]
        y[i] = 1.0 / (1.0 + exp(-val))

    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity(
    np.ndarray[DTYPE_t, ndim=1] vec1,
    np.ndarray[DTYPE_t, ndim=1] vec2
):
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : ndarray, shape (n,)
        First vector
    vec2 : ndarray, shape (n,)
        Second vector

    Returns
    -------
    similarity : float
        Cosine similarity score (0-1)
    """
    cdef int n = vec1.shape[0]
    cdef DTYPE_t dot_product = 0.0
    cdef DTYPE_t norm1 = 0.0
    cdef DTYPE_t norm2 = 0.0

    cdef int i
    cdef DTYPE_t v1, v2

    for i in range(n):
        v1 = vec1[i]
        v2 = vec2[i]
        dot_product += v1 * v2
        norm1 += v1 * v1
        norm2 += v2 * v2

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (sqrt(norm1) * sqrt(norm2))


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_softmax(np.ndarray[DTYPE_t, ndim=2] X):
    """
    Compute softmax along rows of a matrix.

    Parameters
    ----------
    X : ndarray, shape (n_rows, n_cols)
        Input matrix

    Returns
    -------
    Y : ndarray, shape (n_rows, n_cols)
        Softmax-transformed matrix
    """
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] Y = np.zeros(
        (n_rows, n_cols), dtype=DTYPE
    )

    cdef int i, j
    cdef DTYPE_t max_val, exp_sum, exp_val

    for i in range(n_rows):
        # Find max for numerical stability
        max_val = X[i, 0]
        for j in range(1, n_cols):
            if X[i, j] > max_val:
                max_val = X[i, j]

        # Compute exponentials and sum
        exp_sum = 0.0
        for j in range(n_cols):
            exp_val = exp(X[i, j] - max_val)
            Y[i, j] = exp_val
            exp_sum += exp_val

        # Normalize
        for j in range(n_cols):
            Y[i, j] /= exp_sum

    return Y
