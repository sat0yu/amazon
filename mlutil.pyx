#coding: utf-8;
import numpy as np
cimport numpy as np

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

def createLabeledDataset(np.ndarray[DTYPE_int_t, ndim=2] labeled, np.ndarray[DTYPE_int_t, ndim=2] unlabeled, int label_idx=0):
    cdef float nLabeled = float(labeled.shape[0])
    cdef float nUnlabeled = float(unlabeled.shape[0])
    cdef float ratio = nUnlabeled / nLabeled

    np.random.shuffle(labeled)
    cdef np.ndarray[DTYPE_int_t, ndim=2] posdata = labeled[labeled[:,label_idx]==1,:]
    cdef np.ndarray[DTYPE_int_t, ndim=2] negdata = labeled[labeled[:,label_idx]==0,:]
    cdef float nPosdata = float(posdata.shape[0])
    cdef float nNegdata = float(negdata.shape[0])

    # HOW TO CALC. nMetaLabeled
    #------------------------------
    # nLabeled      | nUnlabeled (= nLabeled * ratio)
    # nMetaLabeled  | nMetaUnlabeled (= nMetalabeled * ratio)
    #------------------------------
    # nMetalabeled + nMetaUnlabeled = nLabeled
    # nMetalabeled + (nMetaLabeled * ratio) = nLabeled
    # nMetalabeled = nLabeled / (1 + ratio)

    cdef int nMetaLabeled = int( nLabeled / (1. + ratio) )
    cdef float scale = float(nMetaLabeled) / float(nLabeled)
    cdef int nMetaPos = int( nPosdata * scale )
    cdef int nMetaNeg = int( nNegdata * scale )

    cdef np.ndarray[DTYPE_int_t, ndim=2] metaLabeled = np.vstack( (posdata[:nMetaPos,:], negdata[:nMetaNeg,:]) )
    cdef np.ndarray[DTYPE_int_t, ndim=2] metaUnlabeled = np.vstack( (posdata[nMetaPos:,:], negdata[nMetaNeg:,:]) )

    return (metaLabeled, metaUnlabeled)

def randomSwapOverSampling(np.ndarray[DTYPE_int_t, ndim=2] X, int gain_ratio=1, int nSwap=1):
    cdef int N = len(X)
    cdef int dim = len(X[0])
    cdef int i, j, idx
    cdef np.ndarray[DTYPE_int_t, ndim=2] gained, created
    cdef np.ndarray[DTYPE_int_t, ndim=1] indices
    cdef bint isFirst = True

    for i in range(gain_ratio):
        # copy original data
        created = X.copy()

        # shuffle ndarray given as argument
        np.random.shuffle(created)

        # create new data
        for j in range(N):
            indices = np.random.randint(0, dim, nSwap)
            for idx in indices:
                created[j][idx] = X[np.random.randint(N)][idx]

        # add created data
        if isFirst:
            gained = created
            isFirst = False
        else:
            gained = np.vstack((gained, created))

    return gained

def dividingUnderSampling(np.ndarray[DTYPE_int_t, ndim=2] major, np.ndarray[DTYPE_int_t, ndim=2] minor, int ratio=1):
    cdef list trainset = []
    cdef int i, idx, width
    cdef int nMajor = major.shape[0]
    cdef int nMinor = minor.shape[0]
    cdef int nDivide = (nMajor / nMinor) / ratio

    # validation arguments
    if not nDivide > 0:
        raise ValueError('Requied two arguments, the former\'s length is larger than later\'s')
    if major.shape[1] is not minor.shape[1]:
        raise ValueError('Requied two arguments, those size is the same')

    # divide and concatenate, and create train
    np.random.shuffle(major)
    width = nMinor * ratio
    for i in range(nDivide):
        idx = i * width
        if i < nDivide - 1:
            trainset.append( np.vstack( (minor, major[idx:idx+width,:]) ) )
        else:
            trainset.append( np.vstack( (minor, major[idx:,:]) ) )

    return trainset
