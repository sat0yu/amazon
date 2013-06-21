#coding: utf-8;
import numpy as np
cimport numpy as np

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

def randomSwapOverSampling(np.ndarray[DTYPE_int_t, ndim=2] X, int gain_ratio=1):
    cdef int N = len(X)
    cdef int dim = len(X[0])
    cdef int i, j, idx
    cdef np.ndarray[DTYPE_int_t, ndim=2] gained, created
    cdef bint isFirst = True
    for i in range(gain_ratio):
        # copy original data
        created = X.copy()

        # shuffle ndarray given as argument
        np.random.shuffle(created)

        # create new data
        for j in range(N):
            idx = np.random.randint(dim)
            created[j][idx] = X[np.random.randint(N)][idx]

        # add created data
        if isFirst:
            gained = created
            isFirst = False
        else:
            gained = np.vstack((gained, created))

    return gained

def dividingUnderSampling(np.ndarray[DTYPE_int_t, ndim=2] major, np.ndarray[DTYPE_int_t, ndim=2] minor):
    cdef list trainset = []
    cdef int i, idx
    cdef int nMajor = major.shape[0]
    cdef int nMinor = minor.shape[0]
    cdef int ratio = nMajor / nMinor
    
    # validation arguments
    if not ratio > 0:
        raise ValueError('Requied two arguments, the former\'s length is larger than later\'s')
    if major.shape[1] is not minor.shape[1]:
        raise ValueError('Requied two arguments, those size is the same')

    # divide and concatenate, and create train
    np.random.shuffle(major)
    for i in range(ratio):
        idx = i * nMinor
        if i < ratio - 1:
            trainset.append( np.vstack( (minor, major[idx:idx+nMinor,:]) ) )
        else:
            trainset.append( np.vstack( (minor, major[idx:,:]) ) )

    return trainset
