#coding: utf-8;
import numpy as np
from os import path

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

if __name__ == '__main__':
    execute()
