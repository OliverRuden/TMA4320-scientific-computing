import numpy as np
from utils import onehot

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']

    def adamStep(self, beta_1 = 0.9, beta_2 = 0.999, alpha = 0.01, epsilon = 10**(-8)):
        self.j += 1
        """
        The Adam algorithm as presented in algorithm 3 in the project description
        """

        # for param in self.params:
        #     G_j = self.params[param]["d"]
        #     """
        #     Initialize the matrices V and M for each matrix, j is a counter on which iteration it is on. 
        #     """
        #     if "V" not in self.params[param]:
        #         self.params[param]["V"] = np.zeros((totalBaseCase,) + np.shape(G_j))

        #     if "M" not in self.params[param]:
        #         self.params[param]["M"] = np.zeros((totalBaseCase,) + np.shape(G_j))

        #     self.params[param]["M"][k] = beta_1 * self.params[param]["M"][k] + (1 - beta_1) * G_j
        #     self.params[param]["V"][k] = beta_2 * self.params[param]["V"][k] + (1 - beta_2) * (np.multiply(G_j, G_j))
        #     j += 1
        #     Mhat = (1 / (1 - beta_1**j)) * self.params[param]["M"][k]
        #     Vhat = (1 / (1 - beta_2**j)) * self.params[param]["V"][k]
        #     self.params[param]["w"] -= alpha * (np.divide(Mhat, np.sqrt(Vhat) + epsilon))
        for param in self.params:
            G_j = self.params[param]["d"]

            if "m" not in self.params[param]:
                self.params[param]["m"] = np.zeros_like(G_j)
                self.params[param]["v"] = np.zeros_like(G_j)

            M = self.params[param]["m"]
            V = self.params[param]["v"]

            self.params[param]["m"] = beta_1 * M + (1 - beta_1) * G_j
            self.params[param]["v"] = beta_2 * V + (1 - beta_2) * (np.multiply(G_j, G_j))

            M_hat = M / (1-beta_1**(self.j))
            V_hat = V / (1-beta_2**(self.j))

            self.params[param]["w"] -= alpha * (np.divide(M_hat, np.sqrt(V_hat) + epsilon))

                
                



class Attention(Layer):

    def __init__(self, d, k):
        self.j = 0
        """
        Initializing the parameter matrices and adding these to the params-dictionary. These could've been implemented as LinearLayers, 
        but we decided not to do so. 
        """
        
        W_Q = np.random.randn(k,d)
        W_K = np.random.randn(k,d)
        W_O = np.random.randn(k,d)
        W_V = np.random.randn(k,d)

        "Creating a local softmax, since Attention requires a softmax function"

        self.localSoftmax = Softmax()

        self.params = {'W_Q': {'w':W_Q,'d':np.zeros((k,d))},
                       'W_K': {'w':W_K,'d':np.zeros((k,d))},
                       'W_O': {'w':W_O,'d':np.zeros((k,d))},
                       'W_V': {'w':W_V,'d':np.zeros((k,d))}}

        return 

    def forward(self,z):
        self.z = z
        """
        Initializing the D-matrix with the right size. The dimensions we put into softmax are n*n, where z is b*(d*n),
        so to get the right dimension, we take the length of z[0,0] to get n.
        """
        n = len(z[0,0])          
        D = np.zeros((n,n))

        "Making every lower triangular element of D negative infitiy."

        i1, i2 = np.tril_indices(n, -1)
        D[i1, i2] -= np.inf

        """
        Calculating the A and z_l matrices as in equation (20) in the project description.
        """

        self.A = self.localSoftmax.forward((np.einsum('beo,ke,kd,bdn->bon', z, self.params['W_Q']['w'], self.params['W_K']['w'], z, optimize = True) + D))

        self.zl = z + np.einsum('kd,kp,bpq,bqn->bdn', self.params['W_O']['w'], self.params['W_V']['w'], z, self.A, optimize = True)

        return self.zl

    def backward(self,grad):

        """
        Calculating g_OV and g_S, and returning the gradient as seen in equation (22) in the project description.

        Updating the gradients in the dictionary. 

        """

        g_OV = np.einsum('kd,ke,ben->bdn', self.params['W_V']['w'], self.params['W_O']['w'], grad, optimize=True)
        g_S = self.localSoftmax.backward(np.einsum('bdn,bdo->bno',self.z,g_OV, optimize=True))

        self.params['W_O']['d'] = np.mean(np.einsum('kd,bdn,bno,bfo->bkf',self.params['W_V']['w'], self.z, self.A, grad, optimize=True),axis=0)
        self.params['W_V']['d'] = np.mean(np.einsum('kd,bdn,bon,bfo->bkf',self.params['W_O']['w'], grad, self.A, self.z, optimize=True),axis=0)
        self.params['W_K']['d'] = np.mean(np.einsum('kd,bdn,bno,bfo->bkf',self.params['W_Q']['w'], self.z, g_S, self.z, optimize=True),axis=0)
        self.params['W_Q']['d'] = np.mean(np.einsum('kd,bdn,bon,bfo->bkf',self.params['W_K']['w'], self.z, g_S, self.z, optimize=True),axis=0)

        return grad + np.einsum('bdo,bno->bdn', g_OV, self.A, optimize = True) + np.einsum('ke,kd,bdn,bno->beo', self.params['W_K']['w'], self.params['W_Q']['w'], self.z, g_S, optimize=True) + np.einsum('ke,kd,bdn,bon->beo', self.params['W_Q']['w'], self.params['W_K']['w'], self.z, g_S, optimize = True)


class Softmax(Layer):

    def __init__(self):
        self.j = 0
        return
    
    def forward(self, z):

        """
        Initializing P and Q,

        returning the probability distribution P / Q.
        """

        self.P = np.exp(z - z.max(axis = 1, keepdims = True))  
        self.Q = np.sum(self.P, axis = 1, keepdims = True)
        self.z_l = np.divide(self.P, self.Q + 10 ** (-8))

        return self.z_l

    def backward(self, grad):

        """
        Returning the gradient as in equation (19) from the project description.
        """

        S = np.divide(self.P, np.multiply(self.Q, self.Q) + 10 ** (-8))

        return np.multiply(grad, self.z_l) - np.multiply(np.sum(np.multiply(grad, S), axis = 1, keepdims = True), self.P)


class CrossEntropy(Layer):

    def __init__(self):
        self.j = 0
        self.epsilon = 10**(-8)

    def forward(self, Z, y):
        self.Z = Z
        self.n = np.shape(y)[-1]
        self.y = y

        self.Y_hat = Z[:,:,-self.n:]
        self.m = np.shape(self.Y_hat)[1]
        self.Y = onehot(y,self.m)

        """
        Initialize the guesses, the one-vector and the solution
        """

        """
        Calculate the loss value and take the mean over all the testcases
        """
        Y_prod = np.multiply(self.Y_hat, self.Y)
        p = np.sum(Y_prod, axis = 1)
        q = -np.log(p + self.epsilon)
        value = np.mean(q)

        return value

    def backward(self):
        Y_mod = np.zeros_like(self.Z)
        Y_mod[:,:,-self.n:] = onehot(self.y, self.m)
        grad = -(1/self.n)*np.divide(Y_mod,self.Z+self.epsilon)
        return grad
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """
        self.j = 0

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x, optimize = True)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x, optimize = True)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad, optimize=True)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        self.j = 0
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   
        self.j = 0
        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'][:,:np.shape(grad)[2]] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)

    def adamStep(self, j, k, totalBaseCase, beta_1 = 0.9, beta_2 = 0.999, alpha = 0.01, epsilon = 10**(-8)):
        self.embed.adamStep(j, k, totalBaseCase, beta_1, beta_2, alpha, epsilon)

        super().adamStep(j, k, totalBaseCase, beta_1, beta_2, alpha, epsilon)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        self.j = 0
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)
    
    def adamStep(self, j, k, totalBaseCase, beta_1 = 0.9, beta_2 = 0.999, alpha = 0.01, epsilon = 10**(-8)):
        self.l1.adamStep(j, k, totalBaseCase, beta_1, beta_2, alpha, epsilon)

        self.l2.adamStep(j, k, totalBaseCase, beta_1, beta_2, alpha, epsilon)