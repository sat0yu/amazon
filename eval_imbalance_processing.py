#coding: utf-8;
import numpy as np
from os import path
from sklearn import svm
import sys

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

def execute():
    #read data
    rawtraindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    rawtestdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    
    #create metadata
    metatraindata, metatestdata = mlutil.createLabeledDataset(rawtraindata, rawtestdata)
    print 'meta traindata: ', metatraindata.shape
    print 'meta train pos data: ', (metatraindata[metatraindata[:,0]==1,:]).shape
    print 'meta train neg data: ', (metatraindata[metatraindata[:,0]==0,:]).shape
    print 'meta testdata: ', metatestdata.shape

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    #precomputing
    #separate ACTION column from other features
    labels = metatraindata[:,0]
    metatrain = metatraindata[:,1:]
    print 'gram matrix processing start'
    sys.stdout.flush()
    gram = whk.gram(metatrain)

    #precomputing
    #separate id column from other features
    metalabels = metatestdata[:,0]
    nPosData = len( metalabels[metalabels[:]==1] )
    nNegData = len( metalabels[metalabels[:]==0] )
    metatest = metatestdata[:,1:]
    print 'test matrix processing start'
    sys.stdout.flush()
    mat = whk.matrix(metatest, metatrain)

    #classify    
    weight_list = np.arange(0.01, 0.1, 0.01)
    weight_list = np.hstack((weight_list, np.arange(0.1, 1, 0.1) ))
    weight_list = np.hstack((weight_list, np.arange(1, 10, 1) ))
    for w0 in weight_list:
        for w1 in weight_list:
            clf = svm.SVC(kernel='precomputed', class_weight={0:w0, 1:w1})
            clf.fit(gram, labels)
            prediction = clf.predict(mat).astype(np.int)

            #output
            posPredict = prediction[metalabels[:]==1]
            negPredict = prediction[metalabels[:]==0]
            nPosCorrect = float( sum(posPredict) )
            nNegCorrect = float( len(negPredict) - sum(negPredict) )
            accP = nPosCorrect / nPosData
            accN = nNegCorrect / nNegData
            g = np.sqrt( accP * accN )
            print '0:%4.2f, 1:%4.2f, acc+: %f(%d/%d), acc-: %f(%d/%d), g: %f' % (w0,w1,accP,int(nPosCorrect),nPosData,accN,int(nNegCorrect),nNegData,g)
            sys.stdout.flush()

    #print result, using 'auto' mode
    clf = svm.SVC(kernel='precomputed', class_weight='auto')
    clf.fit(gram, labels)
    prediction = clf.predict(mat).astype(np.int)
    posPredict = prediction[metalabels[:]==1]
    negPredict = prediction[metalabels[:]==0]
    accP = float( sum(posPredict) ) / len( metalabels[metalabels[:]==1] )
    accN = (float( len(negPredict) - sum(negPredict) )) / len( metalabels[metalabels[:]==0] )
    print '"auto", acc+: %f, acc-: %f, g: %f' % (accP,accN,np.sqrt(accP * accN))
    sys.stdout.flush()

if __name__ == '__main__':
    execute()
