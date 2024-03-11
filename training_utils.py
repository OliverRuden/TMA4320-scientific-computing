from layers import *
from neural_network import NeuralNetwork
from utils import onehot
import numpy as np
import matplotlib.pyplot as plt
from data_generators import *

def training(neuralNetwork, objectFunction, dataSet, nIter, m, alpha = 0.01, beta_1 = 0.9, beta_2 = 0.999):

    """
    Kommentar om training function
    """
    L = []
    for i in range(nIter):
        for batchNumber, x in enumerate(dataSet[0]):

            X = onehot(x,m)
            Z = neuralNetwork.forward(X)
            L.append(objectFunction.forward(Z,dataSet[1][batchNumber]))

            grad_Z = objectFunction.backward()
            neuralNetwork.backward(grad_Z)
            neuralNetwork.adamStep(i, batchNumber, len(dataSet[0]), alpha = alpha, beta_1 = beta_1, beta_2 = beta_2)

    return neuralNetwork, L

def generateSortData(numberOfBatches, batchSize, n_max, m):

    """
    Our implementation for generating sort data. 
    """
    x = np.random.randint(0,m,(numberOfBatches,batchSize, (n_max+1)//2))
    y = np.sort(np.copy(x), axis = 2)
    x = np.append(x,y, axis = 2)[:,:,:-1]

    return [x,y]