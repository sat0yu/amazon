#coding: utf-8;
import numpy as np
from sklearn import svm
from os import path
from multiprocessing import Pool

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    print 'train data: %d' % len(traindata)
    print 'test data: %d' % len(testdata)

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    #imbalanced data processing
    posdata = traindata[traindata[:,0]==1,:]
    negdata = traindata[traindata[:,0]==0,:]
    print 'positive data: %d' % len(posdata)
    print 'negative data: %d' % len(negdata)

    #RSOS
    stripped_negdata = negdata[:,1:]    
    gained = np.vstack( (stripped_negdata, stripped_negdata) )
    gained = np.vstack( (gained, mlutil.randomSwapOverSampling(stripped_negdata, 7, 1)) )
    gained = np.vstack( (gained, mlutil.randomSwapOverSampling(stripped_negdata, 7, 2)) )
    print 'gained negative data: %d' % len(gained)
    labels = np.zeros( (len(gained), 1), dtype=np.int)
    labeled = np.hstack( (labels, gained) )
    merged = np.vstack( (negdata, labeled) )
    traindata = np.vstack( (posdata, merged) )

    #separate action/id column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    #train and classify
    print 'gram matrix: (%d,%d)' % (train.shape[0],train.shape[0])
    gram = whk.gram(train)
    print 'test matrix: (%d,%d)' % (test.shape[0],train.shape[0])
    mat = whk.matrix(test, train)
    clf = svm.SVC(kernel='precomputed', class_weight={0:1.0, 1:0.1})
    clf.fit(gram, labels)
    predict = clf.predict(mat).astype(np.int)

    #output
    output = np.vstack( (ids, predict) ).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
