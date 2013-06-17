#coding: utf-8
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import sys

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

class IntKernel():
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2): pass

    def gram(self, np.ndarray[DTYPE_int_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_int_t, ndim=2] gm = np.identity(N, dtype=DTYPE_int)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, np.ndarray[DTYPE_int_t, ndim=2] X1, np.ndarray[DTYPE_int_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_int_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_int)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class FloatKernel():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def val(self, np.ndarray[DTYPE_float_t, ndim=2] X1, np.ndarray[DTYPE_float_t, ndim=2] X2): pass

    def gram(self, np.ndarray[DTYPE_float_t, ndim=2] X):
        cdef int N = len(X)
        cdef np.ndarray[DTYPE_float_t, ndim=2] gm = np.identity(N, dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, np.ndarray[DTYPE_float_t, ndim=2] X1, np.ndarray[DTYPE_float_t, ndim=2] X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        cdef np.ndarray[DTYPE_float_t, ndim=2] mat = np.zeros((N,M), dtype=DTYPE_float)
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class HammingKernel(IntKernel):
    def __init__(self, int d=1):
        self.__d = d

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        cdef int i
        cdef int N = len(x)
        cdef int k = 0
        for i in range(N):
            k = k + (1 if x[i] == y[i] else 0)
        return k**self.__d

class GaussKernel(FloatKernel):
    def __init__(self, double beta):
        self.__beta = beta

    def val(self, np.ndarray[DTYPE_float_t, ndim=1] vec1, np.ndarray[DTYPE_float_t, ndim=1] vec2):
        cdef double dist = np.linalg.norm(vec1-vec2)
        return np.exp(-self.__beta*(dist**2))

def PCA(data, kernel, d):
    # create gram matric
    gm = kernel.gram(data)

    # calculate [J][GM]
    N = len(data)
    j = np.identity(N) - (1.0/N)* np.ones((N,N))
    jgm = np.dot(j, gm)

    # calculate eigen value and vector
    lmd, um = np.linalg.eig(jgm)

    # extract d basis vector(s) as basis matrix(N*d)
    bm = um[:,:d]

    # mapping
    mapped = np.dot(bm.T, gm)
    
    # return data mapped to lower dimention space
    return mapped.T
