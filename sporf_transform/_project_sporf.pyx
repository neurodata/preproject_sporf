# cython: language_level=3, boundscheck=False

# Author: Jesse Patsolic <jpatsol1@jhu.edu>
#
# License: Apache 2.0

import scipy.sparse
import numpy as np

from libc.math cimport floor 
from libc.stdio cimport printf 
cimport numpy as np
np.import_array()



cdef int ternary(int M, int d, int [:] fv, int [:] fi, int [:] fj) nogil except -1:
    """
    Create an augmented dataset of X: Each output feature is a [1:k]-sparse linear
    combination of the input features.

    Output: $\tilde{X}_{n \times p} \cdot A_{p \times d}$
    """

    cdef int a,b

    for i in range(M):
        a = <int> floor(fv[i] / d)
        b = fv[i] % d
        fi[i] = a
        fj[i] = b

    return 0


def sporf_ternary(int p, int d, double k):

    M = <int> floor(p*d*k)
    N = p * d

    f = np.array(np.random.choice(N, M, replace = False), dtype = np.dtype("i"))
    w = np.array(np.random.choice((-1,1), M), dtype = np.dtype("i"))
    
    fi = np.zeros(M, dtype = np.dtype("i"))
    fj = np.zeros(M, dtype = np.dtype("i"))

    ternary(M, d, f, fi, fj)
    
    A = scipy.sparse.csr_matrix((w, (fi,fj)), shape = (p,d))
    
    return(A)


cdef int ternaryColumns(int p, int d, int nnz, int [:] pois, int [:] fi, int [:] fj) except -1:
    """
    Create a sparse matrix A_{p \times d} that is filled by column:
    For j in 1:d
    1) Sample nnz ~ Pois(\lambda) + 1
    2) Sample fi ~ Unif(nnz from {0,...,p-1})
    3) A[fi, j] <- Sample(nnz from {-1,1})
    """

    cdef int i
    cdef int k = 0
    cdef int c = 0 ## index into fi and fj
    cdef np.ndarray F
    cdef int[:] Fv

    for i in range(d):
        k = pois[i]
        F = np.array(np.random.choice(p, k, replace = False), dtype = np.dtype("i"))
        Fv = F
        for j in range(k):
            fi[c] = Fv[j]
            fj[c] = i
            c += 1
            
    return 0

def sporf_ternaryColumns(int p, int d, double lam):

    pois = np.array(np.random.poisson(lam = lam, size=d) + 1, dtype = np.dtype("i"))
    nnz = np.sum(pois)

    w = np.array(np.random.choice((-1,1), nnz), dtype = np.dtype("i"))

    fi = np.zeros(nnz, dtype = np.dtype("i"))
    fj = np.zeros(nnz, dtype = np.dtype("i"))

    ternaryColumns(p, d, nnz, pois, fi, fj)
    
    A = scipy.sparse.csr_matrix((w, (fi,fj)), shape = (p,d))
    
    return(A)
