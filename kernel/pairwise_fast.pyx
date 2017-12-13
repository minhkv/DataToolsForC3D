#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Lars Buitinck
#
# License: BSD 3 clause

from libc.string cimport memset
import numpy as np
cimport numpy as np

cdef extern from "cblas.h":
    double cblas_dasum(int, const double *, int) nogil

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t

cdef fused floating1d:
    float[::1]
    double[::1]

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t


np.import_array()


def _chi_square_kernel_fast(floating_array_2d_t X,
                      floating_array_2d_t Y,
                      floating_array_2d_t result):
    cdef np.npy_intp i, j, k
    cdef np.npy_intp n_samples_X = X.shape[0]
    cdef np.npy_intp n_samples_Y = Y.shape[0]
    cdef np.npy_intp n_features = X.shape[1]
    cdef double res, nom, denom

    with nogil:
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = 2 * X[i, k] * Y[j, k]
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res  += denom / nom
                result[i, j] = res
