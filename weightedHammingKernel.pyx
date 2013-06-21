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
import mlutil

DTYPE_int = np.int
DTYPE_float = np.float
ctypedef np.int_t DTYPE_int_t
ctypedef np.float_t DTYPE_float_t

class WeightendHammingKernel(kernel.IntKernel):
    def __init__(self, np.ndarray[DTYPE_float_t, ndim=1] _w, int _d=1):
        self.__w = _w
        self.__d = _d

    def val(self, np.ndarray[DTYPE_int_t, ndim=1] x, np.ndarray[DTYPE_int_t, ndim=1] y):
        # in numpy, == operator of ndarrays means
        # correspondings for each feature
        correspondings = (x == y)

        # in numpy, True equals 1 and False equals 0
        # so, numpy.dot() can calculate expectedly
        return (np.dot(self.__w, correspondings))**self.__d

def train_and_test(args):
    kernel, traindata, testdata = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]

    #precomputing
    print 'gram matrix processing start'
    gram = kernel.gram(train)
    print 'gram matrix: ', gram.shape, gram.dtype, 'processing end'
    
    print 'test matrix processing start'
    mat = kernel.matrix(testdata, train)
    print 'test matrix: ', mat.shape, mat.dtype, 'processing end'

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    prediction = clf.predict(mat).astype(np.int)
    print 'predictoins : ', prediction.shape, prediction.dtype
    return prediction

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=DTYPE_int, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=DTYPE_int, delimiter=',', skiprows=1)

    #separate id column from other features
    #ids = testdata[:500,0]
    #test = testdata[:500,1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=DTYPE_float)
    whk = WeightendHammingKernel(uniq/max(uniq), 1)

    #initialize predictions
    predictions = np.zeros_like(ids, dtype=np.int)

    #initialize pool and args
    pool = Pool(2)
    argset = []

    #imbalanced data processing
    pos = traindata[traindata[:,0]==1,:]
    neg = traindata[traindata[:,0]==0,:]
    rate = len(pos) / len(neg)
    ## RSOS
    gain = mlutil.randomSwapOverSampling(neg)
    neg = np.vstack( (neg, gain) )
    print 'given %d minority data' % len(gain)
    ## DUS
    trainset = mlutil.dividingUnderSampling(pos, neg)
    print 'given %d trainset' % len(trainset)

    #multiprocessing
    argset = [ (whk,t,test) for t in trainset ]
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
