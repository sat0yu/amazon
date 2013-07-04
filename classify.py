#coding: utf-8;
import numpy as np
from sklearn import svm
from os import path
from multiprocessing import Pool

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

def train_and_test(args):
    i, kernel, traindata, testdata = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]

    #precomputing
    print '[%d] gram matrix processing start (%d,%d)' % (i,len(train),len(train))
    gram = kernel.gram(train)
    print '[%d] test matrix processing start (%d,%d)' % (i,len(testdata),len(train))
    mat = kernel.matrix(testdata, train)

    #train and classify
    #class_weight set as auto mode
    clf = svm.SVC(kernel='precomputed', class_weight='auto')
    clf.fit(gram, labels)
    prediction = clf.predict(mat).astype(np.int)
    return prediction

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    print 'train data: %d' % len(traindata)
    print 'test data: %d' % len(testdata)

    #separate id column from other features
    #ids = testdata[:len(traindata),0]
    #test = testdata[:len(traindata),1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    #instantiate kernel
    hashfile = open("hash.dat", "rb").readlines()
    hashdata = {}
    for line in hashfile:
        try:
            key, val = map(int, line.strip().split(' '))
            hashdata[val] = key
        except ValueError:
            print line.strip().split(' ')
    chk = kernel.CustomHammingKernel(hashdata, 0, 1.0, 2)

    #imbalanced data processing
    pos = traindata[traindata[:,0]==1,:]
    neg = traindata[traindata[:,0]==0,:]
    print 'positive data: %d' % len(pos)
    print 'negative data: %d' % len(neg)
    
    ## RSOS
    # gain = mlutil.randomSwapOverSampling(neg)
    # neg = np.vstack( (neg, gain) )
    # print 'given %d minority data' % len(gain)

    ## DUS
    trainset = mlutil.dividingUnderSampling(pos, neg, ratio=2)
    print 'given %d trainset' % len(trainset)
    print 'each traindata of trainset: ', trainset[0].shape

    #multiprocessing
    pool = Pool(4)
    argset = [ (i,chk,t,test) for i,t in enumerate(trainset) ]
    predictions = pool.map(train_and_test, argset)

    #average
    prediction = np.round( sum(predictions) / float( len(predictions) ) )

    #output
    output = np.vstack( (ids, prediction) ).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
