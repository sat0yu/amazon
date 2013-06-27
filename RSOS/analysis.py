#coding: utf-8;
import numpy as np
from os import path
import sys
import glob

def execute():
    #ready rawdata
    csv_paths = glob.glob('*.csv')
    csv_files = []
    for path in csv_paths:
        csv_files.append( np.loadtxt(open(path, "rb"), dtype=np.int, delimiter=',') )

    for i, csv in enumerate(csv_files):
        #print result
        ap = csv[csv[:,0]==1,:]
        an = csv[csv[:,0]==0,:]
        tp = sum( (ap[:,0] == ap[:,1]) ) / float( len(ap) )
        tn = sum( (an[:,0] == an[:,1]) ) / float( len(an) )
        print 'sensitivity [%d]:' % i, tp
        print 'specificity [%d]:' % i, tn
        g = ( float(tp) * float(tn) )**0.5
        print 'g [%d]: ' % i, g
        sys.stdout.flush()

if __name__ == '__main__':
    execute()
