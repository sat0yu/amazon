from abc import ABCMeta, abstractmethod
import numpy as np

def CV(dataset, labelset, classifier, k=1):
    u'''
    k-fold cross validation on given classifier
    '''

    # validation
    m = len(dataset) / k
    print "given %d samples and k=%d, each training with %d samples" % (len(dataset), k, m)
    accuracies = []
    for l in range(k):
        sIdx = m*l
        preDataset, postDataset = dataset[:sIdx], dataset[sIdx+m:]
        preLabelset, postLabelset = labelset[:sIdx], labelset[sIdx+m:]
        trainDataset = np.vstack((preDataset, postDataset))
        trainLabelset = np.hstack((preLabelset, postLabelset))

        classifier.train(trainDataset, trainLabelset)
        acc = classifier.validate(None, dataset[sIdx:sIdx+m], labelset[sIdx:sIdx+m])
       
        print "Acc:%f (tested at dataset[%d:%d])" % (acc, sIdx, sIdx+m)
        accuracies.append(acc)

    # return the mean value of k-fold accuracies
    return sum(accuracies)/k
    

class Classifier():
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, dataset, labelset):
        u'''
        train classifier by given dataset and labelset.
        this methos must return some trained parameters.
        '''
        pass

    @abstractmethod
    def validate(self, params, dataset, labelset):
        u'''
        validate classifier on accuracy
        with given dataset and label set
        using given parameters.
        this method must return accuracy.
        '''
        pass
