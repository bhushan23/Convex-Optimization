######
######  This file includes different functions used in HW3
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        temp = features * w
        mFilter = np.maximum(0, 1 - np.multiply(labels, temp))
        value = np.sum(mFilter) / n
        return value

    elif order==1:
        # value = ( TODO: value )
        temp = features * w
        mFilter = np.maximum(0, 1 - np.multiply(labels, temp))
        #value = np.sum(mFilter) / n
        
        # subgradient = ( TODO: sungradient )
        tempSG = -1 * np.multiply(labels, features)
        mFilter = np.asarray(mFilter)
        mFilter2 = mFilter.reshape(mFilter.shape[0],)
    
        tempSG = tempSG[mFilter2 > 0]
        subgradient = np.sum(tempSG, axis=0) / n
        subgradient = subgradient.T
        
        #print('ORDER1:', value) # , subgradient)
        return (np.inf, subgradient)

    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
    
def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    
    if order==0:
        # value = ( TODO: value )
        temp = features * w
        mFilter = np.maximum(0, 1 - np.multiply(labels, temp))
        value = np.sum(mFilter) / n
        
        return value
    elif order==1:
        # value = ( TODO: value )
        #temp = features.dot(w)
        #mFilter = np.maximum(0, 1 - np.multiply(labels, temp))
        #value = np.sum(mFilter) / n
        
        # subgradient = ( TODO: sungradient )
        #print('BATCH:', minibatch_size)
        
        rSample = np.random.choice(features.shape[0], size = minibatch_size, replace = False)
        nFeatures = np.asarray(features)
        features = nFeatures[rSample]
        labels = labels[rSample]

        temp = features.dot(w)
        mFilter = np.maximum(0, 1 - np.multiply(labels, temp))

        tempSG = -1 * np.multiply(labels, features)
        mFilter = np.asarray(mFilter)
        mFilter2 = mFilter.reshape(mFilter.shape[0],)
    
        tempSG = tempSG[mFilter2 > 0]
        subgradient = np.sum(tempSG, axis=0) / minibatch_size
        subgradient = subgradient.T
        
        #print('ORDER1:', value) # , subgradient)
        return (np.inf, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
