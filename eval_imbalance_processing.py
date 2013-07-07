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

def train_and_test(args):
    kernel, traindata, testdata = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]

    #precomputing
    gram = kernel.gram(train)
    mat = kernel.matrix(testdata, train)

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    return clf.predict(mat).astype(np.int)

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
    answers = metatestdata[:,0]
    test = metatestdata[:,1:]
    nPosData = len( answers[answers[:]==1] )
    nNegData = len( answers[answers[:]==0] )

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    for r in range(1,len(posdata)/len(negdata)):
        #DUS
        trainset = mlutil.dividingUnderSampling(posdata, negdata, ratio=r)
        print 'ratio: %d, given %d trainset' % (r,len(trainset))

        #multiprocessing
        pool = Pool(4)
        argset = [ (whk,train,test) for train in trainset ]
        predictions = pool.map(train_and_test, argset)

        #average
        sumup = sum(predictions)
        prediction = np.round( sumup.astype(np.float) / len(predictions) )

        #output
        posPredict = prediction[answers[:]==1]
        negPredict = prediction[answers[:]==0]
        nPosCorrect = sum( posPredict )
        nNegCorrect = len( negPredict ) - sum( negPredict )
        accP = nPosCorrect / nPosData
        accN = nNegCorrect / nNegData
        g = np.sqrt( accP * accN )
        print 'ratio: %d\t acc+: %f(%d/%d), acc-: %f(%d/%d), g: %f' % (r,accP,int(nPosCorrect),nPosData,accN,int(nNegCorrect),nNegData,g)
        sys.stdout.flush()

if __name__ == '__main__':
    execute()
