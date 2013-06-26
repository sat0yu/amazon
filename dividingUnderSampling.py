#coding: utf-8;
import numpy as np
from sklearn import svm
from crossValidation import *
from os import path
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
    print 'gram matrix processing start'
    gram = kernel.gram(train)
    print 'gram matrix: ', gram.shape, gram.dtype, 'processing end'
    
    print 'test matrix processing start'
    mat = kernel.matrix(testdata, train)
    print 'test matrix: ', mat.shape, mat.dtype, 'processing end'

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    return clf.predict(mat).astype(np.int)

def execute():
    #read data
    rawtraindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    rawtestdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)

    #ready rawdata
    np.random.shuffle(rawtraindata)
    posdata = rawtraindata[rawtraindata[:,0]==1,:]
    negdata = rawtraindata[rawtraindata[:,0]==0,:]
    nRawtrain = float(rawtraindata.shape[0])
    nRawtest = float(rawtestdata.shape[0])
    nPosdata = float(posdata.shape[0])
    nNegdata = float(negdata.shape[0])

    #create metadata
    ratio = nRawtest / nRawtrain
    nMetatrain = int( nRawtrain / (1. + ratio) )
    scale = float(nMetatrain) / float(nRawtrain)
    nMetaPos = int( nPosdata * scale )
    nMetaNeg = int( nNegdata * scale )
    metatraindata = np.vstack( (posdata[:nMetaPos,:], negdata[:nMetaNeg,:]) )
    metatestdata = np.vstack( (posdata[nMetaPos:,:], negdata[nMetaNeg:,:]) )
    print 'meta traindata: ', metatraindata.shape
    print 'meta testdata: ', metatestdata.shape
    metapos = metatraindata[metatraindata[:,0]==1,:]
    metaneg = metatraindata[metatraindata[:,0]==0,:]
    print 'meta train pos data: ', metapos.shape
    print 'meta train neg data: ', metaneg.shape

    #separate label column from other features
    metalabels = metatestdata[:,0]
    metatest = metatestdata[:,1:]

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    ratio_list = [1,2,4,8,16]
    for i,r in enumerate(ratio_list):
        #DUS
        trainset = mlutil.dividingUnderSampling(metapos, metaneg, ratio=r)
        print 'given %d trainset' % len(trainset)

        #multiprocessing
        pool = Pool(4)
        argset = [ (whk,train,metatest) for train in trainset ]
        predictions = pool.map(train_and_test, argset)

        #average
        sumup = sum(predictions)
        prediction = np.round( sumup.astype(np.float) / len(predictions) )

        #print result
        output = np.vstack((metalabels,prediction)).T
        ap = output[output[:,0]==1,:]
        an = output[output[:,0]==0,:]
        tp = sum( (ap[:,0] == ap[:,1]) ) / float( len(ap) )
        tn = sum( (an[:,0] == an[:,1]) ) / float( len(an) )
        g = ( float(tp) * float(tn) )**0.5
        print "rate:%d,%lf,%lf,%lf" % (r,tp,tn,g)
        filename = path.splitext(__file__)[0] + "_%d.csv" % r
        np.savetxt(filename, output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
