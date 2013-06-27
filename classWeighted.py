#coding: utf-8;
import numpy as np
from os import path
from sklearn import svm

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

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
    print 'gram matrix: ', gram.shape, gram.dtype, 'processing end'
    
    #initialize classifiers
    clfs = []
    weight_list = [0.1, 0.2, 0.5, 1, 2, 5]
    for i in weight_list:
        for j in weight_list:
            clfs.append( svm.SVC(kernel='precomputed', class_weight={0:i, 1:j}) )

    #train each classifier
    map(lambda clf_i: clf_i.fit(gram, labels), clfs)

    #precomputing
    #separate id column from other features
    metalabels = metatestdata[:,0]
    metatest = metatestdata[:,1:]
    print 'test matrix processing start'
    mat = whk.matrix(metatest, metatrain)
    print 'test matrix: ', mat.shape, mat.dtype, 'processing end'

    #classify    
    for i,w0 in enumerate(weight_list):
        for j,w1 in enumerate(weight_list):
            prediction = clfs[i*len(weight_list)+j].predict(mat).astype(np.int)

            #output
            output = np.vstack((metalabels, prediction)).T
            ap = output[output[:,0]==1,:]
            an = output[output[:,0]==0,:]
            tp = sum( (ap[:,0] == ap[:,1]) ) / float( len(ap) )
            tn = sum( (an[:,0] == an[:,1]) ) / float( len(an) )
            g = ( float(tp) * float(tn) )**0.5
            print 'w0:%.1f,w1:%.1f,%lf,%lf,%lf' % (w0,w1,tp,tn,g)
            filename = path.splitext(__file__)[0]
            filename += "_" + str(w0) + "_" + str(w1) 
            filename += ".csv"
            np.savetxt(filename, output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
