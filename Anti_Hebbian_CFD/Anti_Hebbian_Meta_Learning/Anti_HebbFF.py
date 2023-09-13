import Anti_HebbFF_Network as AHF
import Data_Loader as DL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torchviz
import sys
import time
import timeit
import copy
import os
import seaborn as sns
from time import sleep
from rich.console import Console
from memory_profiler import profile

# from torch.utils.tensorboard import SummaryWriter

# meta parameters
vec_len = 25
lr = 0.001
batch_size = 1
acc = 0.99
hid_dim = 25
out_dim = 1
Sigmoid = nn.Sigmoid()
LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)

def load_model(PATH):

    model = AHF.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model


def test(model,r1, r2,criterion,PATH):
    ACC_R = []
    for i in range(r1,r2):
        T = max(1000,200 * i)
        input_seq,target_seq = DL.generate_seq(batch_size = model.batch_size, vec_len = vec_len, R = i, T = T, Batch = False)
        output_seq,accuracy,total_loss = AHF.test(model,criterion,input_seq,target_seq,T)
        ACC_R.append(accuracy.item())
        
        # false_positive =

    r_axis = np.arange(r1,r2)
    acc_plt = plt.figure()
    plt.plot(r_axis,ACC_R)
    plt.title("Accuracy across different intervals after training with R in [{},{}]".format(r1,r2))
    acc_plt.savefig(PATH + '/Acc',  dpi = 600, facecolor='w', transparent=True)
    acc_plt.clf()

def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def hidden_activity(model,r,criterion,PATH,T_step):

    model.batch_size = 1

    T = max(100, 20 * r)
    input_seq,target_seq = DL.generate_seq(batch_size = model.batch_size, vec_len = vec_len, R = r, T = T, Batch = False)
    output_seq,accuracy,total_loss,hidden_activity = AHF.get_hid(model,criterion,input_seq,target_seq,T)

    for i in range(len(hidden_activity)):
        hidden_activity[i] = torch.squeeze(hidden_activity[i]).detach().numpy()
    #print(hidden_activity.shape)
    hidden_activity = np.transpose(np.array(hidden_activity))
    hidden_activity = hidden_activity[:,r:r+T_step]
    x_axis_labels =  merge(output_seq[r:r+T_step].detach().numpy().astype(int).reshape(T_step).tolist(), target_seq[r:r+T_step].detach().numpy().astype(int).reshape(T_step).tolist())
    ha_plot= sns.heatmap(hidden_activity, xticklabels=x_axis_labels,linewidth=0.3, cmap= 'rocket_r')
    ha_figure = ha_plot.get_figure()
    plt.title('Hidden Activity')
    ha_figure.savefig(PATH + '/Hidden_Activity', dpi=600, facecolor='w', transparent=True)
    ha_figure.clf()

def test_data(R , PATH):
    input_seq,target_seq = DL.generate_seq(vec_len = vec_len, R = R, T = 30,Batch = False)
    data_plot= sns.heatmap(torch.transpose(torch.squeeze(input_seq),0,1), xticklabels=torch.squeeze(target_seq.detach()).numpy().astype(int),linewidth=0.3, cmap= 'rocket_r')
    data_figure = data_plot.get_figure()
    plt.title('data check {}'.format(R))
    data_figure.savefig(PATH + '/Data_Check_R{}'.format(R), dpi=600, facecolor='w', transparent=True)
    data_figure.clf()


def static_matrix(model, PATH):
    # static matrix
    sm_plot = sns.heatmap(model.w1.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.w1.detach().numpy()==0))
    sm_figure = sm_plot.get_figure()
    plt.title('Static Weight Matrix')
    sm_figure.savefig(PATH + '/W_1', dpi=600, facecolor='w', transparent=True)
    sm_figure.clf()

def acc_train(model, PATH ,acc, vec_len, R, T, criterion, optimizer,acc_num):
    writer = SummaryWriter(PATH + '/Training_Log')
    iterations = 0
    acc_flag = 0

    while acc_flag < acc_num:
        input_seq, target_seq = DL.generate_seq(batch_size=model.batch_size, vec_len=vec_len, R=R, T = T, Batch = False)
        model.A = torch.zeros(model.hid_dim, model.input_dim)
        accuracy, loss = AHF.train(model, criterion, optimizer, input_seq, target_seq, Batch = False)
        iterations += 1

        if accuracy.item() >= acc:
            acc_flag += 1
        else:
            acc_flag = 0

        # tensorboard

        writer.add_scalar('Performance / Training Loss', loss.item(), iterations)
        writer.add_scalar('Performance / Accuracy', accuracy.item(), iterations)

        Norm_w1 = np.linalg.norm(torch.squeeze(model.w1.detach()))
        Var_w1 = np.var(torch.squeeze(model.w1.detach()).numpy())

        Norm_b2 = np.linalg.norm(torch.squeeze(model.b2.detach()))

        writer.add_scalar('w1b1 / Norm of w1', Norm_w1, iterations)
        writer.add_scalar('w1b1 / Var of w1', Var_w1, iterations)


        writer.add_scalar('Parameters / Norm of b2', Norm_b2, iterations)
        writer.add_scalar('Parameters / Lambda', model.lmbda.item(), iterations)
        writer.add_scalar('Parameters / Eta', model.eta.item(), iterations)
        writer.add_scalar('Parameters / w2', model.w2_parameter.item(), iterations)
        writer.add_scalar('Parameters / b1', model.b1_parameter.item(), iterations)

        # gradient norm1
        '''
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Grad_norm', total_norm, iterations)
        '''

        # false positive and false negative

        dif_seq = (model.predicted - target_seq).detach().numpy()
        actual_positive = np.count_nonzero(target_seq == 1)
        actual_negative = np.count_nonzero(target_seq == 0)
        false_positive = np.count_nonzero(dif_seq == 1)
        false_negative = np.count_nonzero(dif_seq == -1)
        writer.add_scalar('FPFN / false positive rate', false_positive / actual_negative, iterations)
        writer.add_scalar('FPFN / false negative rate', false_negative / actual_positive, iterations)

        torch.save(model.state_dict(), PATH + '/Model_tmp')

    print('Training Summary: Total Number of iterations: {}'.format(iterations))

def Find_max_R(R):

    # Starting New Models
    HebbFF_CFD = AHF.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(HebbFF_CFD.parameters())
    
    while True:
        R += 1
        print('Current Training Interval Value: ',R)
        T = max(100, 20 * R)

        PATH = os.getcwd() + '/Result/Mult_v_{}_h_{}/R_{}'.format(vec_len, hid_dim, R)
        isExist = os.path.exists(PATH)
        
        if not isExist:
            os.makedirs(PATH)

        # training
        acc_train(HebbFF_CFD, PATH, acc, vec_len, R, T, criterion, optimizer,acc_num = 5)

        # saving the model
        torch.save(HebbFF_CFD.state_dict(), PATH + '/Model')

        # test accuracy
        test(HebbFF_CFD, 1, max (20, 5 * R), criterion, PATH)
        
        # hidden_layer heatmap

        hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=20)

        # static matrix
        static_matrix(HebbFF_CFD, PATH)

def reload(R,PATH):
    # loading previous model
    HebbFF_CFD = load_model(PATH)
    HebbFF_CFD.eval()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(HebbFF_CFD.parameters(),lr = 3e-3)

    while True:
        R += 1
        print('Current Training Interval Value: ', R)
        T = max(100, 20 * R)

        PATH = os.getcwd() + '/Result/BCE_v_{}_h_{}/R_{}'.format(vec_len, hid_dim, R)
        isExist = os.path.exists(PATH)

        if not isExist:
            os.makedirs(PATH)

        # training
        acc_train(HebbFF_CFD, PATH, acc, vec_len, R, T, criterion, optimizer, acc_num=5)

        # saving the model
        torch.save(HebbFF_CFD.state_dict(), PATH + '/Model')

        # test accuracy
        test(HebbFF_CFD, 1, max(20, 5 * R), criterion, PATH)

        # hidden_layer heatmap

        hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=20)

        # static matrix
        static_matrix(HebbFF_CFD, PATH)

def generate_graphs(R,PATH):
    # loading previous model
    HebbFF_CFD = load_model(PATH)
    criterion = nn.BCE()
    # test accuracy
    PATH = os.getcwd() + '/Result/Sigmoid_v_{}_h_{}/R_{}'.format(vec_len, hid_dim, R)
    test(HebbFF_CFD, 1, 20, criterion, PATH)
    # hidden_layer heatmap
    hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=30)
    # static matrix
    static_matrix(HebbFF_CFD, PATH)

def generate_norm_graph(model,R,criterion, optimizer):
    PATH = os.getcwd() + '/Result/Sigmoid_v_{}_h_{}/R_{}/Norm_Graph'.format(vec_len, hid_dim, R)
    T = 1000 * R

    input_seq, target_seq = DL.generate_seq(batch_size=model.batch_size, vec_len=vec_len, R=R, T=T)
    A_norm = AHF.A_converge_test(model, criterion, optimizer,input_seq, target_seq, T)
    model.A_norm = []

    r_axis = np.arange(T)
    cvg_plt = plt.figure()
    plt.plot(r_axis, A_norm)
    plt.title("New Model A Norm R{}".format(R))
    cvg_plt.savefig(PATH , dpi=600, facecolor='w', transparent=True)


if __name__ == "__main__":


    #PATH = os.getcwd() + '/Result/BCE_v_{}_h_{}/R_{}'.format(vec_len, hid_dim, 34)
    #reload(33,PATH + '/Model_tmp')

    '''
    PATH = os.getcwd() + '/Data_Check'
    for R in range(1,10):
        test_data(R,PATH)
    '''

    #generate_graphs(R = 2, PATH = PATH + '/Model')

    Find_max_R(R = 0)
    # Writer will output to ./runs/ directory by default

    '''
    test_converge = AHF.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(test_converge.parameters())
    generate_norm_graph(test_converge,1,criterion, optimizer)
    '''
