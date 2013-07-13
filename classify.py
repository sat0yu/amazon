#coding: utf-8;
import numpy as np
from sklearn import svm
import ctypes
from os import path
from multiprocessing import Pool, Array
import sys

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import kernel
import mlutil

#global statement
shared_gram_base = None
shared_evalmat_base = None

def train_and_test(args):
    i, w0, w1, data = args
    gramshape, labels, evalmatshape, answers = data
    
    global shared_gram_base
    global shared_evalmat_base

    print shared_gram_base
    print shared_evalmat_base

    gram = np.frombuffer( shared_gram_base )
    gram = gram.reshape(gramshape)
    evalmat = np.frombuffer( shared_evalmat_base )
    evalmat = evalmat.reshape(evalmatshape)
    
    print "[%d] %d, %d" % (i,ctypes.addressof(gram.base.base), ctypes.addressof(evalmat.base.base))

    clf = svm.SVC(kernel='precomputed', class_weight={0:w0, 1:w1})
    clf.fit(gram, labels)
    predict = clf.predict(evalmat).astype(np.int)

    posPredict = predict[answers[:]==1]
    negPredict = predict[answers[:]==0]
    nPosCorrect = float( sum(posPredict) )
    nNegCorrect = float( len(negPredict) - sum(negPredict) )
    accP = nPosCorrect / len( answers[answers[:]==1] )
    accN = nNegCorrect / len( answers[answers[:]==0] )
    g = np.sqrt( accP * accN )

    return (g, np.sqrt(w0*w1), w0, w1)

def execute():
    #read data
    rawtraindata = np.loadtxt(open("rawdata/train.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    testdata = np.loadtxt(open("rawdata/test.csv", "rb"), dtype=np.int, delimiter=',', skiprows=1)
    print 'train data: %d' % len(rawtraindata)
    print 'test data: %d' % len(testdata)

    #instantiate kernel
    uniq = np.array([7519, 4244, 129, 178, 450, 344, 2359, 68, 344], dtype=np.float)
    whk = kernel.WeightendHammingKernel(uniq/max(uniq), 2)

    #imbalanced data processing
    posdata = rawtraindata[rawtraindata[:,0]==1,:]
    negdata = rawtraindata[rawtraindata[:,0]==0,:]
    print 'positive data: %d' % len(posdata)
    print 'negative data: %d' % len(negdata)

    #RSOS
    stripped_negdata = negdata[:,1:]    
    gained = np.vstack( (stripped_negdata, stripped_negdata) )
    gained = np.vstack( (gained, mlutil.randomSwapOverSampling(stripped_negdata, 3, 1)) )
    gained = np.vstack( (gained, mlutil.randomSwapOverSampling(stripped_negdata, 6, 2)) )
    gained = np.vstack( (gained, mlutil.randomSwapOverSampling(stripped_negdata, 3, 3)) )
    print 'gained negative data: %d' % len(gained)
    labels = np.zeros( (len(gained), 1), dtype=np.int)
    labeled = np.hstack( (labels, gained) )
    merged = np.vstack( (negdata, labeled) )
    traindata = np.vstack( (posdata, merged) )

    #separate action/id column from other features
    answers = rawtraindata[:,0]
    rawtrain = rawtraindata[:,1:]
    labels = traindata[:,0]
    train = traindata[:,1:]
    ids = testdata[:,0]
    test = testdata[:,1:]

    #precomputing
    gram = whk.matrix_multiprocessing(train, train, 4)
    global shared_gram_base
    shared_gram_base = Array(ctypes.c_double, gram.shape[0]*gram.shape[1], lock=False)
    shared_gram = np.frombuffer( shared_gram_base )
    shared_gram = shared_gram.reshape(gram.shape)
    shared_gram[:] = gram
    print 'gram matrix: (%d,%d)' % (train.shape[0],train.shape[0])
    print shared_gram.base.base

    evalmat = whk.matrix_multiprocessing(rawtrain, train, 4)
    global shared_evalmat_base
    shared_evalmat_base = Array(ctypes.c_double, evalmat.shape[0]*evalmat.shape[1], lock=False)
    shared_evalmat = np.frombuffer( shared_evalmat_base )
    shared_evalmat = shared_evalmat.reshape(evalmat.shape)
    shared_evalmat[:] = evalmat
    print 'evaluation matrix: (%d,%d)' % (rawtrain.shape[0],train.shape[0])
    print shared_evalmat.base.base

    #evaluate class_weight
    print "class_weight evaluating..."
    cw_list = np.arange(0.01, 0.1, 0.01)
    cw_list = np.hstack(( cw_list, np.arange(0.1, 1, 0.1) ))
    cw_list = np.hstack(( cw_list, np.arange(1, 10, 1.) ))
    data = (gram.shape, labels, evalmat.shape, answers)
    args = [(i*len(cw_list)+j,w0,w1,data) for j,w0 in enumerate(cw_list) for i,w1 in enumerate(cw_list)]
    pool = Pool(4)
    scored_cw = pool.map( train_and_test, args )
    print "class_weight evaluation has been done."

    #precomputing
    testmat = whk.matrix_multiprocessing(test, train, 4)
    print 'test mat', testmat.shape

    #weighred decision by majority
    nClassifier = 5
    predictions = []
    scored_cw.sort(reverse=True)
    for g,m,w0,w1 in scored_cw[:nClassifier]:
        print 'g:%f, m:%f, w0:%f, w1:%f' % (g,m,w0,w1)
        clf = svm.SVC(kernel='precomputed', class_weight={0:w0, 1:w1})
        clf.fit(gram, labels)
        predict = clf.predict(testmat).astype(np.int)

        ###### weight = g?? ######
        weight = g
        predict = weight * ( (2 * predict) - np.ones_like(predict) )
        predictions.append( predict )

    #calc. predict,
    #since each prediction is represented based on {-1, 1}
    #i.e. (alpha * -1) or (alpha * 1)
    prediction = np.sign( sum(predictions) )
    prediction = ( prediction + np.ones_like(prediction) ) / 2

    #output
    output = np.vstack( (ids, prediction) ).T
    print 'output: ', output.shape
    filename = path.splitext(__file__)[0]
    np.savetxt(filename+".csv", output.astype(int), fmt="%d", delimiter=',')

if __name__ == '__main__':
    execute()
