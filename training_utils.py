from layers import *
from neural_network import NeuralNetwork
from utils import onehot
import numpy as np
import matplotlib.pyplot as plt
from data_generators import *

def training(neuralNetwork, objectFunction, dataSet, nIter, m, alpha = 0.01, beta_1 = 0.9, beta_2 = 0.999):

    """
    This is our training function
    """
    L = []
    x_data, y_data = dataSet[0], dataSet[1]
    #Loop over all iterations

    for i in range(nIter):
        L_temp = 0
        #And all batches

        for batchNumber in range(np.shape(x_data)[0]):
            #Take a batch through the neural network
            x = x_data[batchNumber]
            X = onehot(x,m)
            Z = neuralNetwork.forward(X)

            #Change the parameters to reduce the loss
            L_temp += objectFunction.forward(Z,y_data[batchNumber])
            grad_Z = objectFunction.backward()
            neuralNetwork.backward(grad_Z)
            neuralNetwork.adamStep(alpha = alpha)
        
        #Append the average loss over all batches
        L.append(L_temp/np.shape(x_data[0])[0])

        #If the loss is sufficiently small, break
        if L[-1] < 0.0001:
            return L
    return L

def generateAllAddition():
    #Create a x-test array for input for addition
    a=np.arange(0,10000,1)
    add_dict = {}
    x = [1000,100,10,1]
    for i in range(len(x)):
        if i==0:
            add_dict[x[i]] = a//x[i]
        else:
            add_dict[x[i]] = (a-(a//x[i-1])*x[i-1])//x[i]
    y = np.transpose(np.append(np.array([add_dict[1000]]),np.array([add_dict[100], add_dict[10], add_dict[1]]), axis = 0))
    y = y.astype(int)

    #Create a y-test array for solution to input 
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

    #Return the two arrays
    return y, z2
