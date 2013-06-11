#coding: utf-8;
import numpy as np
from sklearn import svm
from crossValidation import *
import kernel
from os import path

class CustomHammingKernel(kernel.Kernel):
    def __init__(self, _hash, int _idx, double _c=1.0):
        self.__hash = _hash
        self.__idx = _idx
        self.__c = _c

    def val(self, x, y):
        cdef int i
        cdef int N = len(x)
        cdef int hash_x = self.__hash.get(int(x[self.__idx]), 0)
        cdef int hash_y = self.__hash.get(int(y[self.__idx]), 0)
        cdef int diff = abs(hash_x - hash_y)
        cdef int k = 0
        for i in range(N):
            k = k + (1 if x[i] == y[i] and i != self.__idx else 0)
        return k / (diff + self.__c)

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
        key, val = map(int, line.strip().split(' '))
        hashdata[val] = key
    print 'hashdata: ', hashdata

    labels = traindata[:500,0]
    train = traindata[:500,1:]
    ids = testdata[:100,0]
    test = testdata[:100,1:]

    print 'traindata shape: ', train.shape
    print 'testdata shape: ', test.shape

    #instantiate kernel
    chk = CustomHammingKernel(hashdata, 0, 1.0)

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
