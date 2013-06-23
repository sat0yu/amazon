#coding: utf-8;
import numpy as np
from os import path
import sys
from sklearn import svm
from multiprocessing import Pool

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
    print 'meta testdata: ', metatestdata.shape
    metapos = metatraindata[metatraindata[:,0]==1,:]
    metaneg = metatraindata[metatraindata[:,0]==0,:]
    print 'meta train pos data: ', metapos.shape
    print 'meta train neg data: ', metaneg.shape

    #separate id column from other features
    metalabels = metatestdata[:,0]
    metatest = metatestdata[:,1:]

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 1)

    #RSOS
    eval_ratio = range( int( nPosdata / nNegdata ) )
    evaluates = []
    args = []
    for i in eval_ratio:
        if i == 0:
            gained = metaneg
        else:
            gained = np.vstack( (metaneg, mlutil.randomSwapOverSampling(metaneg, i)) )
        metatrain = np.vstack( (metapos, gained) )
        print 'meta train [%d]: ' % i, metatrain.shape
        args.append( (whk, metatrain, metatest) )
    
    pool = Pool(4)
    predicts = pool.map(train_and_predictoin, args)

    for i, predict in enumerate(predicts):
        #print result
        table = np.vstack((metalabels, predict)).T
        ap = table[table[:,0]==1,:]
        an = table[table[:,0]==0,:]
        tp = sum( (ap[:,0] == ap[:,1]) ) / float( len(ap) )
        tn = sum( (an[:,0] == an[:,1]) ) / float( len(an) )
        g = ( float(tp) * float(tn) )**0.5
        print 'g [%d]: ' % i, g
        sys.stdout.flush()

        #output
        output = np.vstack((metalabels, predict)).T
        filename = path.splitext(__file__)[0] + '_' + str(i)
        np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

def train_and_predictoin(args):
    kernel, traindata, test = args

    #separate ACTION column from other features
    labels = traindata[:,0]
    train = traindata[:,1:]

    #precomputing
    print 'gram matrix processing start'
    gram = kernel.gram(train)
    print 'gram matrix: ', gram.shape, gram.dtype, 'processing end'

    print 'test matrix processing start'
    mat = kernel.matrix(test, train)
    print 'test matrix: ', mat.shape, mat.dtype, 'processing end'

    #train and classify
    clf = svm.SVC(kernel='precomputed')
    clf.fit(gram, labels)
    prediction = clf.predict(mat).astype(np.int)
    print 'predictoins : ', prediction.shape, prediction.dtype

    return prediction

if __name__ == '__main__':
    execute()
