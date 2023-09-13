import torch.nn as nn
import numpy as np
import torch

batch_size = 60

def batch_dig(z):
    dig = np.apply_along_axis(np.diag, -1, z.reshape(batch_size,len(z[0,0,:,]))).reshape(batch_size,1,len(z[0,0,:,]),len(z[0,0,:,]))
    return dig

class Sigmoid():
    
    def activation(self, z):
        return 1.0/(1.0+np.exp(-z))

    def gradient(self, z):
        z = self.activation(z)*(1-self.activation(z))
        return batch_dig(z)
    
    def fast(self):
        return 0
    
class Identity():
    
    def activation(self,z):
        return z
    
    def gradient(self, z):
        z = np.ones(z.shape)
        return batch_dig(z)
        
    def fast(self):
        return 0
    
class Tanh():
    def activation(self, x):
        return np.tanh(x)

    def gradient(self, x):
        z = 1-np.square(tanh(x))
        return batch_dig(z)
    
    def fast(self):
        return 0
        
class Softmax():

    def activation(self, x):
        m = nn.Softmax(dim=2)
        return m(torch.tensor(x)).numpy()

    def gradient(self, z):
        z = self.activation(z)
        dig = np.apply_along_axis(np.diag, -2, z).reshape(batch_size,1,len(z[0,0,:,]),len(z[0,0,:,]))
        return dig - z @ np.transpose(z,(0, 1, 3, 2))
        
    def fast(self):
        return 1

class Relu():

    def activation(self, x):
        return(np.maximum(0, x))
            
    def gradient(self, x):
        
        def v_d(a):
            if a > 0:
                return 1
            else:
                return 0
        
        vfunc = np.vectorize(v_d, otypes=[float])
        return batch_dig(vfunc(x))
        
    def fast(self):
        return 0


class leaky_relu():

    def __init__(self, slope):
        self.slope = slope

    def activation(self, x):
        activation_mask = 1.0 * (x >0) + self.slope * (x<0)
        activations= np.multiply(x, activation_mask)
        return activations


    def gradient(self, x):

        def v_d(a):
            if a > 0:
                return 1
            else:
                return self.slope
            
        vfunc = np.vectorize(v_d, otypes=[float])
        return batch_dig(vfunc(x))
    
    def fast(self):
        return 0
   
class XE():

    def Loss(self, predicted, label):
        loss= -np.sum(label*np.log(predicted))
        return loss/float(len(label[0,0,:,:]))
        
    def gradient(self, output_activations, y):
        grad = -(y / output_activations)
        return grad/float(len(y[0,0,:,:]))
        
    def fast(self):
        return 1

def softmax_XE_grad(output_activations, y):
    return (output_activations - y) / float(len(y[0,0,:,:]))

class MSE():
    def Loss(self , a,b):
        return np.sqrt(np.sum(np.square(a-b)))

    def gradient(self, output_activations, y):
        return (output_activations - y)
        
    def fast(self):
        return 0
