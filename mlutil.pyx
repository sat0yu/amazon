#coding: utf-8;
import numpy as np
cimport numpy as np

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

def randomSwapOverSampling(np.ndarray[DTYPE_int_t, ndim=2] X, int nCreateData = 0):
    nCreateData = len(X) if nCreateData == 0 else nCreateData
    cdef int dim = len(X[0])
    cdef np.ndarray[DTYPE_int_t, ndim=2] created = X.copy()

    # shuffle ndarray given as argument
    np.random.shuffle(created)
    created = created[:nCreateData]

    # create new data
    cdef int i
    for i in range(nCreateData):
        idx = np.random.randint(dim)
        created[i][idx] = X[np.random.randint(nCreateData)][idx]

    return created
