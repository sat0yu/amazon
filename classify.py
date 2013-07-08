#coding: utf-8;
import numpy as np
from sklearn import svm
from os import path
from multiprocessing import Pool
import sys

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

def weak_classifier(args):
    i, kernel, traindata, evaldata, target = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]
    answers = evaldata[:,0]
    test = evaldata[:,1:]

    #precomputing
    print '[%d] gram matrix processing start (%d,%d)' % (i,len(train),len(train))
    gram = kernel.gram(train)
    print '[%d] test matrix processing start (%d,%d)' % (i,len(test),len(train))
    test_mat = kernel.matrix(test, train)
    print '[%d] target matrix processing start (%d,%d)' % (i,len(target),len(train))
    target_mat = kernel.matrix(target, train)

    #train, class_weight set to auto mode
    clf = svm.SVC(kernel='precomputed', class_weight='auto')
    clf.fit(gram, labels)

    #evaluate classifier
    predict = clf.predict(test_mat).astype(np.int)

    ## since presict.shape is equal answers.shape,
    ## answers indices can be used as predict indices
    posPredict = predict[answers[:]==1]
    negPredict = predict[answers[:]==0]

    ## posCorrect(negPredict) is represented on {0, 1}
    nPosCorrect = float( sum(posPredict) )
    nNegCorrect = float( len(negPredict) - sum(negPredict) )
    accP = nPosCorrect / len( answers[answers[:]==1] )
    accN = nNegCorrect / len( answers[answers[:]==0] )

    ## calc. err as (1 - g)
    ## g is the geometric mean of accP and accN
    err = 1 - np.sqrt(accP * accN) if accP*accN < 1. else 0.000001
    beta = np.sqrt( err / (1 - err) )
    alpha = np.log( 1 / beta )
    print "[%d] acc+: %f, acc-: %f, err: %f, beta: %f, alpha: %f" % (i,accP,accN,err,beta,alpha)
    sys.stdout.flush()

    #classify
    predict = clf.predict(target_mat).astype(np.int)

    #replace labels from 0 to -1 and weighten them
    weighted_predict = err * ( (2 * predict) - np.ones_like(predict) )
     
    return weighted_predict

def execute():
    #read data
    traindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    print 'train data: %d' % len(traindata)
    print 'test data: %d' % len(testdata)

    #separate id column from other features
    ids = testdata[:,0]
    test = testdata[:,1:]

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 3)

    #imbalanced data processing
    pos = traindata[traindata[:,0]==1,:]
    neg = traindata[traindata[:,0]==0,:]
    print 'positive data: %d' % len(pos)
    print 'negative data: %d' % len(neg)
    
    ## DUS
    trainset = mlutil.dividingUnderSampling(pos, neg, ratio=2)
    print 'given %d trainset' % len(trainset)
    print 'each traindata of trainset: ', trainset[0].shape

    #multiprocessing
    pool = Pool(4)
    argset = [ (i, whk, t, traindata, test) for i,t in enumerate(trainset) ]
    predictions = pool.map(weak_classifier, argset)

    #calc. predict,
    #since each prediction is represented based on {-1, 1}
    #i.e. (alpha * -1) or (alpha * 1)
    predict = np.sign( sum(predictions) )
    predict = ( predict + np.ones_like(predict) ) / 2

    #output
    output = np.vstack( (ids, predict) ).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
