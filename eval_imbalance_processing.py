#coding: utf-8;
import numpy as np
from os import path
from sklearn import svm
import sys
from multiprocessing import Pool

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

def evaluate_RSOS(args):
    rate, kernel, traindata, testdata = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]
    answers = testdata[:,0]
    test = testdata[:,1:]
    nPosData = len( answers[answers[:]==1] )
    nNegData = len( answers[answers[:]==0] )

    #precomputing
    print '[%d] gram matrix processing start' % rate
    gram = kernel.gram(train)
    sys.stdout.flush()

    print '[%d] test matrix processing start' % rate
    mat = kernel.matrix(test, train)
    sys.stdout.flush()

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    prediction = clf.predict(mat).astype(np.int)

    #output
    posPredict = prediction[answers[:]==1]
    negPredict = prediction[answers[:]==0]
    nPosCorrect = float( sum(posPredict) )
    nNegCorrect = float( len(negPredict) - sum(negPredict) )
    accP = nPosCorrect / nPosData
    accN = nNegCorrect / nNegData
    g = np.sqrt( accP * accN )
    print 'rate: %d\t acc+: %f(%d/%d), acc-: %f(%d/%d), g: %f' % (rate,accP,int(nPosCorrect),nPosData,accN,int(nNegCorrect),nNegData,g)
    sys.stdout.flush()

def execute():
    #read data
    rawtraindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    rawtestdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    
    #create metadata
    metatraindata, metatestdata = mlutil.createLabeledDataset(rawtraindata, rawtestdata)
    posdata = metatraindata[metatraindata[:,0]==1,:]
    negdata = metatraindata[metatraindata[:,0]==0,:]
    print 'meta traindata: ', metatraindata.shape
    print 'meta train pos data: ', posdata.shape
    print 'meta train neg data: ', negdata.shape
    print 'meta testdata: ', metatestdata.shape

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    #RSOS
    eval_ratio = range( len(posdata) / len(negdata) )
    args = []
    for i in eval_ratio:
        if i == 0:
            gained = negdata
        else:
            gained = np.vstack( (negdata, mlutil.randomSwapOverSampling(negdata, i)) )
        metatraindata = np.vstack( (posdata, gained) )
        args.append( (i, whk, metatraindata, metatestdata) )
    
    #multiprocessing
    pool = Pool(2)
    pool.map(evaluate_RSOS, args)


if __name__ == '__main__':
    execute()
