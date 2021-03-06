#coding: utf-8;
import numpy as np
cimport numpy as np
from sklearn import svm
from crossValidation import *
from os import path
from multiprocessing import Pool

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel

DTYPE = np.int
ctypedef np.int_t DTYPE_t

class CustomHammingKernel(kernel.IntKernel):
    def __init__(self, _hash, int _idx, double _var=1.0, int _d=1):
        self.__hash = _hash
        self.__idx = _idx
        self.__var = _var
        self.__d = _d

    def val(self, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
        cdef int i
        cdef int N = len(x)
        cdef int hash_x = self.__hash.get(int(x[self.__idx]), 0)
        cdef int hash_y = self.__hash.get(int(y[self.__idx]), 0)
        cdef double k = np.exp( ( -(hash_x - hash_y)**2 )/ self.__var)
        for i in range(N):
            k = k + (1.0 if x[i] == y[i] else 0.0)
        return k**self.__d

def train_and_test(args):
    kernel, traindata, testdata = args
    labels = traindata[:,0]
    train = traindata[:,1:]
    #print 'traindata shape: ', train.shape
    #print 'testdata shape: ', testdata.shape

    #precomputing
    gram = kernel.gram(train)
    print 'gram matrix: ', gram.shape
    mat = kernel.matrix(testdata, train)
    print 'test matrix: ', mat.shape

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    return clf.predict(mat).astype(np.int)

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=DTYPE, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=DTYPE, delimiter=',', skiprows=1)
    
    #read hash
    cdef int i, N
    cdef char* line
    hashfile = open("hash.dat", "rb").readlines()
    N = len(hashfile)
    hashdata = {}
    for i in range(N):
        line = hashfile[i]
        try:
            key, val = map(int, line.strip().split(' '))
            hashdata[val] = key
        except ValueError:
            print line.strip().split(' ')
    #print 'hashdata: ', hashdata

    #separate id column from other features
    #ids = testdata[:500,0]
    #test = testdata[:500,1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    #instantiate kernel
    chk = CustomHammingKernel(hashdata, 0, 1.0, 2)

    #initialize predictions
    predictions = np.zeros_like(ids, dtype=np.int)

    #initialize pool and args
    pool = Pool(2)
    argset = []

    #imbalanced data processing
    np.random.shuffle(traindata)
    pos = traindata[traindata[:,0]==1,:]
    neg = traindata[traindata[:,0]==0,:]
    cdef int nPos = pos.shape[0]
    cdef int nNeg = neg.shape[0]
    cdef int j, rate
    if nPos > nNeg:
        rate = nPos / nNeg
        print 'pos:%d, neg:%d, rate(pos/neg):%d' % (nPos, nNeg, rate)
        for j in range(rate):
            if j < rate - 1:
                argset.append((
                    chk,
                    np.vstack( (neg, pos[j*nNeg:(j+1)*nNeg,:]) ),
                    test,
                ))
            else:
                argset.append( (chk, np.vstack( (neg, pos[j*nNeg:,:]) ), test) )
            
    else:
        rate = nNeg / nPos
        print 'pos:%d, neg:%d, rate(neg/pos):%d' % (nPos, nNeg, rate)
        for j in range(rate):
            if j < rate - 1:
                argset.append((
                    chk,
                    np.vstack( (neg, pos[j*nPos:(j+1)*nPos,:]) ),
                    test,
                ))
            else:
                argset.append( (chk,np.vstack( (neg, pos[j*nPos:,:]) ),test) )
            
    #multiprocessing
    predictions = sum( pool.map(train_and_test, argset) )

    #average
    predictions = np.round( predictions.astype(np.float) / rate )
    
    #output
    output = np.vstack((ids,predictions)).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
