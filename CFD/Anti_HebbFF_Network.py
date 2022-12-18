import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torchviz
import sys
import copy
import gc
from memory_profiler import profile

# from torch.utils.tensorboard import SummaryWriter
    
    
class HebbFF(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim, batch_size, hid_nonl):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.training_loss = []
        self.Hidden_Layer_Activity = []
        # store the 'norm' of the plastic matrix
        self.A_norm =[]
        self.output_seq = 0
        self.predicted = 0
        self.accuracy = 0
        self.hid_nol = hid_nonl
        self.batch_size = batch_size
        self.loss = 0

        # different nonlinearity
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # initializing parameters
        self.w1 = torch.nn.Parameter(torch.empty(hid_dim,input_dim))
        nn.init.xavier_normal_(self.w1, gain=1.0)

        self.b1 = torch.ones(hid_dim,1)
        self.b1_parameter = torch.nn.Parameter(torch.tensor(0.3))
        
        self.w2 = torch.ones(out_dim,hid_dim)
        self.w2_parameter = torch.nn.Parameter(torch.tensor(-0.5))

        
        self.b2 = torch.nn.Parameter(torch.empty(out_dim,1))
        nn.init.xavier_normal_(self.b2, gain=1.0)

        # plastic matrix
        self.A = torch.zeros(self.batch_size, hid_dim,input_dim)
        # decay rate
        self.lmbda = torch.nn.Parameter(torch.tensor(0.3))
        self.eta = torch.nn.Parameter(torch.tensor(-0.5))

        '''
        self.w1: N * d
        self.b1: d * 1
        self.w2: 1 * N
        self.b2: 1 * 1
        '''

    def forward(self, x_t, Store, Batch = False):
        if Store == True and Batch == True:
            Norm_A = np.linalg.norm(torch.squeeze(self.A.detach()))
            self.A_norm.append(Norm_A)
            x_t = torch.unsqueeze(x_t,-1)
            #print('x_t',x_t.shape)

            h_t = self.hid_nol((self.w1 + self.A) @ x_t  + self.b1_parameter * self.b1)

            self.Hidden_Layer_Activity.append(h_t)
            self.A = self.lmbda * self.A + self.eta * (h_t @ torch.transpose(x_t,1,2))
            #print('A',self.A.shape)
            y_t = self.sigmoid(self.w2_parameter * self.w2 @ (h_t) + self.b2)
            y_t = torch.squeeze(y_t,-1)
            #print('y_t',y_t.shape)
        elif Store == False and Batch == True:
            '''
            x_t: B * d
            unsqueez x_t: B * d * 1
            self.A: B * N * d
            self.w1 + self.A: B * N * d
            h_t: B * N * 1
            y_t: B * 1 * 1
            '''
            x_t = torch.unsqueeze(x_t, -1)
            h_t = self.hid_nol((self.w1 + self.A) @ x_t + self.b1_parameter * self.b1)
            self.A = self.sigmoid(self.lmbda) * self.A + self.eta * (h_t @ torch.transpose(x_t, 1, 2))
            y_t = self.sigmoid(self.w2_parameter * self.w2 @ (h_t) + self.b2)
            y_t = torch.squeeze(y_t, -1)





        elif Store == True and Batch == False:

            x_t = torch.unsqueeze(x_t, -1)
            h_t = self.sigmoid((self.w1 * self.A) @ x_t + self.w1 @ x_t + self.b1_parameter * self.b1)
            #h_t = self.sigmoid((self.w1 + self.A) @ x_t + self.b1_parameter * self.b1)
            self.Hidden_Layer_Activity.append(h_t)
            self.A = self.sigmoid(self.lmbda) * self.A + self.eta * (h_t @ torch.transpose(x_t, 0, 1))
            y_t = self.sigmoid(self.w2_parameter * self.w2 @ (h_t) + self.b2)
            y_t = torch.squeeze(y_t, -1)

        elif Store == False and Batch == False:
            '''
            x_t: d
            unsqueez x_t: d * 1
            self.A: N * d
            self.w1 + self.A: N * d
            h_t: N * 1
            y_t: 1 * 1
            '''
            '''
            x_t = torch.unsqueeze(x_t, -1)
            h_t = self.sigmoid((self.w1 + self.A) @ x_t + self.b1_parameter * self.b1)
            self.A = self.lmbda * self.A + self.eta * (h_t @ torch.transpose(x_t, 0, 1))
            y_t = self.sigmoid(self.w2_parameter * self.w2 @ (h_t) + self.b2)
            y_t = torch.squeeze(y_t, -1)    
            '''

            x_t = torch.unsqueeze(x_t, -1)
            h_t = self.sigmoid((self.w1 * self.A) @ x_t + self.w1 @ x_t + self.b1_parameter * self.b1)
            #h_t = self.sigmoid((self.w1 + self.A) @ x_t + self.b1_parameter * self.b1)

            self.A = self.sigmoid(self.lmbda) * self.A + self.eta * (h_t @ torch.transpose(x_t, 0, 1))
            y_t = self.sigmoid(self.w2_parameter * self.w2 @ (h_t) + self.b2)
            y_t = torch.squeeze(y_t, -1)


        return y_t

def train(model,criterion,optimizer,input_seq,target_seq, Batch = True):
    model.accuracy = 0
    model.loss = 0
    model.output_seq = torch.zeros(target_seq.shape)
    optimizer.zero_grad()
    model.Hidden_Layer_Activity = []

    if Batch ==  True:
        T = len(input_seq[0])
        for i in range(T):
            model.output_seq[:,i,:] = model(input_seq[:,i,:], Store = False)

    else:
        T = len(input_seq)
        model.A = torch.zeros(model.hid_dim, model.input_dim)
        for i in range(T):
            model.output_seq[i, :] = model(input_seq[i, :], Store = False, Batch = False)


    #model.output_seq =torch.squeeze(model.output_seq)
    #target_seq =  torch.squeeze(target_seq)
    # B * T * d
    model.loss = criterion(model.output_seq, target_seq)
    model.predicted = (model.output_seq.detach()>=0.5).float()
    model.accuracy  = (model.predicted == target_seq).sum() / T

    model.loss.backward()
    optimizer.step()

    #model.A = model.A.detach()
    if Batch ==  True:
        model.A = torch.zeros(model.batch_size, model.hid_dim, model.input_dim)
    else:
        model.A = torch.zeros(model.hid_dim, model.input_dim)
    return model.accuracy, model.loss

    
def test(model,criterion,input_seq,target_seq,T, Batch  =  False):
    model.loss = 0
    model.accuracy = 0
    model.output_seq = torch.zeros(target_seq.shape)

    for i in range(T):
        model.output_seq[i, :] = model(input_seq[i, :], Store = False, Batch = False)
    model.output_seq =torch.squeeze(model.output_seq)
    target_seq =  torch.squeeze(target_seq)
    model.loss = criterion(model.output_seq, target_seq)
    # model.A = model.A.detach()
    model.A = torch.zeros(model.batch_size, model.hid_dim, model.input_dim)

    model.predicted = (model.output_seq>0.5).float()
    model.accuracy  = (model.predicted == target_seq).sum()
    
    return model.predicted,model.accuracy/(T * model.batch_size), model.loss


def get_hid(model, criterion, input_seq, target_seq, T):
    model.loss = 0
    model.accuracy = 0
    model.Hidden_Layer_Activity = []
    model.A_norm = []
    model.output_seq = torch.zeros(target_seq.shape)

    model.A = torch.zeros(model.hid_dim, model.input_dim)
    for i in range(T):
        model.output_seq[i, :] = model(input_seq[i, :], Store=True, Batch=False)

    model.loss = criterion(model.output_seq[::4], target_seq[::4])
    # model.A = model.A.detach()
    model.A = torch.zeros(model.batch_size, model.hid_dim, model.input_dim)
    model.predicted = (model.output_seq > 0.5).float()
    model.accuracy = (model.predicted == target_seq).sum()

    return model.predicted, model.accuracy / (T * model.batch_size), model.loss, model.Hidden_Layer_Activity

def A_converge_test(model,criterion,optimizer,input_seq,target_seq,T):
    model.loss = 0
    output_seq = torch.zeros(target_seq.shape)
    optimizer.zero_grad()
    model.Hidden_Layer_Activity = []
    model.A_norm = []


    for i in range(T):
        output_seq[:,i,:] = model(input_seq[:,i,:], Store=True)

    model.loss = criterion(output_seq, target_seq)
    model.training_loss.append(model.loss.item())

    model.loss.backward()
    optimizer.step()

    model.A = torch.zeros(model.batch_size, model.hid_dim, model.input_dim)
    return model.A_norm
