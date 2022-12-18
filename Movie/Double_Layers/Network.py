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
        self.d = input_dim
        self.N = hid_dim
        self.training_loss = []
        self.HL_add = []
        self.HL_mult = []

        # store the 'norm' of the plastic matrix
        self.A_norm =[]
        self.output_seq = 0
        self.predicted = 0
        self.accuracy = 0
        self.hid_nol = hid_nonl
        self.batch_size = batch_size
        self.loss = 0

        # different nonlinearity
        self.tanh = nn.Tanh()

        self.sigmoid = nn.Sigmoid()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)


        '''
        x_t: d
        unsqueez x_t: d * 1
        self.A_mult: N * d
        self.w_mult + self.A_mult: N * d
        h_t_mult: N * 1
        self.w_add : N * N
        self.A_add : N * N
        h_t_add: N * 1
        self.b_final : 1 * 1
        self.w_final : 1 * N
        y_t: 1 * 1
        '''

        self.A_mult = torch.zeros(self.N,self.d)
        self.A_add = torch.zeros(self.N,self.N)


        # initializing parameters
        self.w_mult = torch.nn.Parameter(torch.empty(self.N,self.d))
        self.b_mult = torch.nn.Parameter(torch.empty(self.N,1))
        self.w_add = torch.nn.Parameter(torch.empty(self.N,self.N))
        self.b_add = torch.nn.Parameter(torch.empty(self.N,1))
        self.w_final = torch.nn.Parameter(torch.empty(out_dim,self.N))
        self.b_final = torch.nn.Parameter(torch.empty(out_dim,1))

        nn.init.xavier_normal_(self.w_mult, gain=1.0)
        nn.init.xavier_normal_(self.b_mult, gain=1.0)
        nn.init.xavier_normal_(self.w_add, gain=1.0)
        nn.init.xavier_normal_(self.b_add, gain=1.0)
        nn.init.xavier_normal_(self.w_final, gain=1.0)
        nn.init.xavier_normal_(self.b_final, gain=1.0)

        # plastic matrix
        self.A_mult = torch.zeros(self.N,self.d)
        self.A_add = torch.zeros(self.N,self.N)

        # decay rate
        self.lmbda_mult = torch.nn.Parameter(torch.tensor(1.0))
        self.lmbda_add = torch.nn.Parameter(torch.tensor(1.0))
        self.eta_mult = torch.nn.Parameter(torch.tensor(-0.5))
        self.eta_add = torch.nn.Parameter(torch.tensor(-0.5))

        '''
        self.eta_mult = torch.nn.Parameter(torch.empty(self.N, self.d))
        self.eta_add = torch.nn.Parameter(torch.empty(self.N, self.N))
        nn.init.xavier_normal_(self.eta_mult, gain=1.0)
        nn.init.xavier_normal_(self.eta_add, gain=1.0)
        '''

    def forward(self, x_t, Store, Batch = True):
        if Store == True and Batch == True:
            Norm_A = np.linalg.norm(torch.squeeze(self.A.detach()))
            self.A_norm.append(Norm_A)
            x_t = torch.unsqueeze(x_t,-1)
            #print('x_t',x_t.shape)

            h_t = self.hid_nol((self.w1 + self.A) @ x_t  + self.b1_parameter * self.b1)

            self.Hidden_Layer_Activity.append(h_t)
            self.A = self.sigmoid(self.lmbda) * self.A +  self.self.eta * (h_t @ torch.transpose(x_t,1,2))
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

            h_t_mult = self.tanh((self.w_mult * self.A_mult) @ x_t + self.w_mult @ x_t + self.b_mult)
            self.A_mult = self.sigmoid(self.lmbda_mult) * self.A_mult + self.eta_mult * (h_t_mult @ torch.transpose(x_t, 0, 1))

            h_t_add = self.sigmoid((self.w_add + self.A_add) @ h_t_mult + self.b_add)
            self.A_add = self.sigmoid(self.lmbda_add) * self.A_add + self.eta_add * (h_t_add @ torch.transpose(h_t_mult, 0, 1))

            y_t = self.sigmoid(self.w_final @ (h_t_add) + self.b_final)
            y_t = torch.squeeze(y_t, -1)

            self.HL_mult.append(h_t_mult)
            self.HL_add.append(h_t_add)


        elif Store == False and Batch == False:
            '''
            x_t: d
            unsqueez x_t: d * 1
            self.A_mult: N * d
            self.w_mult + self.A_mult: N * d
            h_t_mult: N * 1
            self.w_add : N * N
            self.A_add : N * N
            h_t_add: N * 1
            self.b_final : 1 * 1
            self.w_uniform : 1 * N
            y_t: 1 * 1
            '''
            x_t = torch.unsqueeze(x_t, -1)
            h_t_mult = self.tanh((self.w_mult * self.A_mult) @ x_t + self.w_mult @ x_t + self.b_mult)
            self.A_mult = self.sigmoid(self.lmbda_mult) * self.A_mult + self.eta_mult * (h_t_mult @ torch.transpose(x_t, 0, 1))

            h_t_add = self.sigmoid((self.w_add + self.A_add) @ h_t_mult + self.b_add)
            self.A_add = self.sigmoid(self.lmbda_add) * self.A_add + self.eta_add * (h_t_add @ torch.transpose(h_t_mult, 0, 1))
            y_t = self.sigmoid(self.w_final @ (h_t_add) + self.b_final)
            y_t = torch.squeeze(y_t, -1)

        return y_t


def train(model, criterion, optimizer, input_seq, target_seq, scene_t, Batch=True):
    model.accuracy = 0
    model.loss = 0
    model.output_seq = []
    optimizer.zero_grad()
    model.Hidden_Layer_Activity = []
    model.output_seq = torch.zeros(len(target_seq) * scene_t)

    if Batch == True:
        T = len(input_seq[0])
        for i in range(T):
            model.output_seq[:, i, :] = model(input_seq[:, i, :], Store=False)

    else:
        T = len(input_seq)
        for t in range(T):
            scence = input_seq[t]
            for i in range(len(scence)):
                out = model(scence[i], Store=False, Batch=False)
                model.output_seq[scene_t * t + i] = out

    # model.output_seq =torch.squeeze(model.output_seq)
    # target_seq =  torch.squeeze(target_seq)
    # B * T * d
    # move_out_seq = torch.FloatTensor(move_out_seq)
    # model.loss = criterion(model.output_seq, target_seq)
    # model.predicted = (model.output_seq.detach()>=0.5).float()
    model.loss = criterion(model.output_seq[scene_t - 1::scene_t], target_seq)
    model.predicted = (model.output_seq[scene_t - 1::scene_t].detach() >= 0.5).float()
    model.accuracy = (model.predicted == target_seq).sum() / T

    model.loss.backward()
    optimizer.step()

    # model.A = model.A.detach()
    if Batch == True:
        model.A_mult = torch.zeros(model.A_mult.shape)
        model.A_add = torch.zeros(model.A_add.shape)
    else:
        model.A_mult = torch.zeros(model.A_mult.shape)
        model.A_add = torch.zeros(model.A_add.shape)


    return model.accuracy, model.loss


def test(model, criterion, input_seq, target_seq, T, scene_t, Batch=False):
    model.accuracy = 0
    model.loss = 0
    # model.output_seq = torch.zeros(len(target_seq)*5)\
    model.output_seq = torch.zeros(len(target_seq) * scene_t)

    if Batch == True:
        T = len(input_seq[0])
        for i in range(T):
            model.output_seq[:, i, :] = model(input_seq[:, i, :], Store=False)

    else:
        T = len(input_seq)
        # model.A = torch.zeros(model.hid_dim, model.input_dim)
        for t in range(T):
            scence = input_seq[t]
            for i in range(len(scence)):
                out = model(scence[i], Store=False, Batch=False)
                model.output_seq[scene_t * t + i] = out

    # model.output_seq =torch.squeeze(model.output_seq)
    # target_seq =  torch.squeeze(target_seq)
    # B * T * d
    # model.loss = criterion(model.output_seq[4::5], target_seq)
    # model.predicted = (model.output_seq[4::5].detach()>=0.5).float()
    model.loss = criterion(model.output_seq[scene_t - 1::scene_t], target_seq)
    model.predicted = (model.output_seq[scene_t - 1::scene_t].detach() >= 0.5).float()
    model.accuracy = (model.predicted == target_seq).sum() / T

    # model.A = model.A.detach()
    if Batch == True:
        model.A_mult = torch.zeros(model.A_mult.shape)
        model.A_add = torch.zeros(model.A_add.shape)
    else:
        model.A_mult = torch.zeros(model.A_mult.shape)
        model.A_add = torch.zeros(model.A_add.shape)

    return model.predicted, model.accuracy, model.loss


def get_hid(model, criterion, input_seq, target_seq, T, scene_t):
    model.loss = 0
    model.accuracy = 0
    model.HL_add = []
    model.HL_mult = []
    model.A_norm = []
    model.output_seq = torch.zeros(len(target_seq) * scene_t)
    move_out_seq = torch.zeros(target_seq.shape)

    T = len(input_seq)
    for t in range(T):
        scence = input_seq[t]
        for i in range(len(scence)):
            out = model(scence[i], Store=True, Batch=False)
            model.output_seq[scene_t * t + i] = out


    return model.HL_mult, model.HL_add


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
