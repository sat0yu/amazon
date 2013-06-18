#coding: utf-8;
import numpy as np
cimport numpy as np
from sklearn import svm
from crossValidation import *
import kernel
from os import path
from multiprocessing import Pool

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
    gram = kernel.gram(train)
    print 'gram matrix: ', gram.shape, gram.dtype
    mat = kernel.matrix(testdata, train)
    print 'test matrix: ', mat.shape, mat.dtype

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
    whk = WeightendHammingKernel(uniq/max(uniq), 2.0)

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
                    whk,
                    np.vstack( (neg, pos[j*nNeg:(j+1)*nNeg,:]) ),
                    test,
                ))
            else:
                argset.append( (whk, np.vstack( (neg, pos[j*nNeg:,:]) ), test) )

    else:
        rate = nNeg / nPos
        print 'pos:%d, neg:%d, rate(neg/pos):%d' % (nPos, nNeg, rate)
        for j in range(rate):
            if j < rate - 1:
                argset.append((
                    whk,
                    np.vstack( (neg, pos[j*nPos:(j+1)*nPos,:]) ),
                    test,
                ))
            else:
                argset.append( (whk,np.vstack( (neg, pos[j*nPos:,:]) ),test) )

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
