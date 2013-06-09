#coding: utf-8
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import sys

class Kernel():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def val(self, X1, X2): pass

    def gram(self, X):
        cdef N = len(X)
        gm = np.identity(N)
        cdef i,j
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, X1, X2):
        cdef int N = len(X1)
        cdef int M = len(X2)
        mat = np.zeros((N,M))
        cdef int i,j
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class HammingKernel(Kernel):
    def __init__(self, int d=1):
        self.__d = d

    def val(self, x, y):
        cdef int i
        cdef int N = len(x)
        cdef int k = 0
        for i in range(N):
            k = k + (1 if x[i] == y[i] else 0)
        return k**self.__d

class GaussKernel(Kernel):
    def __init__(self, double beta):
        self.__beta = beta

    def val(self, vec1, vec2):
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
