#coding: utf-8;
import numpy as np
from sklearn import svm
from crossValidation import *
import kernel
from os import path
import sys

class CustomHammingKernel(kernel.Kernel):
    def __init__(self, _hash, int _idx, double _var=1.0, int _d=1):
        self.__hash = _hash
        self.__idx = _idx
        self.__var = _var
        self.__d = _d

    def val(self, x, y):
        cdef int i
        cdef int N = len(x)
        cdef int hash_x = self.__hash.get(int(x[self.__idx]), 0)
        cdef int hash_y = self.__hash.get(int(y[self.__idx]), 0)
        cdef double k = np.exp( ( -(hash_x - hash_y)**2 )/ self.__var)
        # print hash_x, hash_y
        # sys.stdout.flush()
        for i in range(N):
            k = k + (1.0 if x[i] == y[i] else 0.0)
        return k**self.__d

class SVM(svm.SVC, Classifier):
    def train(self, train, label):
        self.fit(train, label)

    def validate(self, param, train, label):
        return self.score(train, label)

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), delimiter=',', skiprows=1)
    
    #read hash
    hashfile = open("hash.dat", "rb").readlines()
    hashdata = {}
    for line in hashfile:
        try:
            key, val = map(int, line.strip().split(' '))
            hashdata[val] = key
        except ValueError:
            print val, key
    print 'hashdata: ', hashdata

    # labels = traindata[:500,0]
    # train = traindata[:500,1:]
    # ids = testdata[:100,0]
    # test = testdata[:100,1:]
    labels = traindata[:,0]
    train = traindata[:,1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    print 'traindata shape: ', train.shape
    print 'testdata shape: ', test.shape

    #instantiate kernel
    chk = CustomHammingKernel(hashdata, 0, 1.0, 2)

    #precomputing
    gram = chk.gram(train)
    print 'gram matrix: ', gram.shape
    mat = chk.matrix(test, train)
    print 'test matrix: ', mat.shape

    #train and classify
    clf = SVM(kernel='precomputed')
    clf.train(gram, labels)
    predictions = clf.predict(mat)
    
    #output
    output = np.vstack((ids,predictions)).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
