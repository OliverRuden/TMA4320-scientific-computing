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
        L_temp = 0
        for batchNumber in range(np.shape(dataSet[0])[0]):
            x = dataSet[0][batchNumber]
            X = onehot(x,m)
            Z = neuralNetwork.forward(X)
            L_temp += objectFunction.forward(Z,dataSet[1][batchNumber])
            grad_Z = objectFunction.backward()
            neuralNetwork.backward(grad_Z)
            neuralNetwork.adamStep(alpha = alpha, beta_1 = beta_1, beta_2 = beta_2)
        L.append(L_temp/np.shape(dataSet[0])[0])
    return L

def generateAllAddition():
    a=np.arange(0,10000,1)

    dict = {}
    x = [1000,100,10,1]

    for i in range(len(x)):

        if i==0:
            dict[x[i]] = a//x[i]
        else:
            dict[x[i]] = (a-(a//x[i-1])*x[i-1])//x[i]

    y = np.transpose(np.append(np.array([dict[1000]]),np.array([dict[100], dict[10], dict[1]]), axis = 0))
    y = y.astype(int)
    z = np.zeros(10**4)
    for i in range(len(a)):
        temp1 = y[i,0]*10 + y[i,1]
        temp2 = y[i,2]*10 + y[i,3]
        z[i] = temp1+temp2
    z = z.astype(int)
    z2 = np.zeros((np.shape(z) + (3,)))
    for i in range(np.shape(z)[0]):
        temp = str(z[i])
        while len(temp) < 3:
            temp = "0" + temp
        for j in range(3):
            z2[i,j] =  temp[j]
    z2 = z2.astype(int)
    return y, z2
