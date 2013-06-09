#coding: utf-8;
import numpy as np
from sklearn import svm
from crossValidation import *
import kernel

class SVM(svm.SVC, Classifier):
    def train(self, train, label):
        self.fit(train, label)

    def validate(self, param, train, label):
        return self.score(train, label)

if __name__ == '__main__':
    #read data
    traindata = np.loadtxt(open("samplingdata/train.csv", "rb"), delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("samplingdata/test.csv", "rb"), delimiter=',', skiprows=1)

    labels = traindata[:,0]
    train = traindata[:,1:]
    test = testdata[:,1:]

    print 'traindata shape: ', train.shape
    print 'testdata shape: ', test.shape

    #instantiate kernel
    hk = kernel.HammingKernel(2)

    #precomputing
    gram = hk.gram(train)
    print 'gram matrix: ', gram.shape
    mat = hk.matrix(test, train)
    print 'test matrix: ', mat.shape

    #train and classify
    clf = SVM(kernel='precomputed')
    clf.train(gram, labels)
    predictions = clf.predict(mat)
    np.savetxt("hammingKernelSVM.csv", predictions.astype(int), fmt="%d", delimiter=',')
