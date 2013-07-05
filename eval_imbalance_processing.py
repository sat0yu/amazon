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
    gram = whk.gram(metatrain)
    
    #initialize classifiers
    clfs = []
    weight_list = [0.01, 0.05, 0.1, 0.5, 1., 5., 10.]
    for i in weight_list:
        for j in weight_list:
            clfs.append( svm.SVC(kernel='precomputed', class_weight={0:i, 1:j}) )
    ## add 'auto' mode
    clfs.append( svm.SVC(kernel='precomputed', class_weight='auto') )

    #train each classifier
    map(lambda clf_i: clf_i.fit(gram, labels), clfs)

    #precomputing
    #separate id column from other features
    metalabels = metatestdata[:,0]
    nPosData = len( metalabels[metalabels[:]==1] )
    nNegData = len( metalabels[metalabels[:]==0] )
    metatest = metatestdata[:,1:]
    print 'test matrix processing start'
    mat = whk.matrix(metatest, metatrain)

    #classify    
    for i,w0 in enumerate(weight_list):
        for j,w1 in enumerate(weight_list):
            prediction = clfs[i*len(weight_list)+j].predict(mat).astype(np.int)

            #output
            posPredict = prediction[metalabels[:]==1]
            negPredict = prediction[metalabels[:]==0]
            nPosCorrect = float( sum(posPredict) )
            nNegCorrect = float( len(negPredict) - sum(negPredict) )
            accP = nPosCorrect / nPosData
            accN = nNegCorrect / nNegData
            g = np.sqrt( accP * accN )
            print '{0:%.2f, 1:%.2f}\t acc+: %f(%d/%d), acc-: %f(%d/%d), g: %f' % (w0,w1,accP,int(nPosCorrect),nPosData,accN,int(nNegCorrect),nNegData,g)
            sys.stdout.flush()

    #print result, using 'auto' mode
    prediction = clfs[len(clfs)-1].predict(mat).astype(np.int)
    posPredict = prediction[metalabels[:]==1]
    negPredict = prediction[metalabels[:]==0]
    accP = float( sum(posPredict) ) / len( metalabels[metalabels[:]==1] )
    accN = (float( len(negPredict) - sum(negPredict) )) / len( metalabels[metalabels[:]==0] )
    print '"auto"\t\t acc+: %f, acc-: %f, g: %f' % (accP,accN,np.sqrt(accP * accN))
    sys.stdout.flush()

if __name__ == '__main__':
    execute()
