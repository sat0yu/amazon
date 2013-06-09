#coding: utf-8
import numpy as np
from abc import ABCMeta, abstractmethod
import sys

class Kernel():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def val(self, X1, X2): pass

    def gram(self, X):
        N = len(X)
        gm = np.identity(N)
        for i in range(N):
            for j in range(i, N):
                gm[j][i] = gm[i][j] = self.val(X[i], X[j])
        return gm

    def matrix(self, X1, X2):
        N, M = len(X1), len(X2)
        mat = np.zeros((N,M))
        for i in range(N):
            for j in range(M):
                mat[i][j] = self.val(X1[i], X2[j])
        return mat

class HammingKernel(Kernel):
    def __init__(self, d=1):
        self.__d = d

    def val(self, x, y):
        return sum([ 1 if xi == yi else 0 for xi,yi in zip(x,y) ])**self.__d

class GaussKernel(Kernel):
    def __init__(self, beta):
        self.__beta = beta

    def val(self, vec1, vec2):
        dist = np.linalg.norm(vec1-vec2)
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
