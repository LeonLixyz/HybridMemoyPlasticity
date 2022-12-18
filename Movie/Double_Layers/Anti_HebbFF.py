import Network as NN
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
VAR = 0.1
scene_t = 4
vec_len = 25
lr = 0.001
batch_size = 1
acc = 0.99
hid_dim = 25
out_dim = 1
Sigmoid = nn.Sigmoid()
LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)

def load_model(PATH):

    model = NN.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model


def test(model,r1, r2,criterion,PATH):
    ACC_R = []
    for i in range(r1,r2):
        T = max(1000,200 * i)
        input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=i, T=T, scene_t = scene_t, variation=VAR)
        output_seq,accuracy,total_loss = NN.test(model,criterion,input_seq,target_seq,T,scene_t)
        ACC_R.append(accuracy.item())
        

    r_axis = np.arange(r1,r2)
    acc_plt = plt.figure()
    plt.plot(r_axis,ACC_R)
    plt.title("Accuracy across different intervals after training with R in [{},{}]".format(r1,r2))
    acc_plt.savefig(PATH + '/Acc',  dpi = 600, facecolor='w', transparent=True)
    acc_plt.clf()
    plt.close()


def entire_output(model, criterion, PATH, R):
    T = max(100, 20 * R)
    input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=R, T=T, scene_t = scene_t,variation = VAR)
    output_seq, accuracy, total_loss = NN.test(model, criterion, input_seq, target_seq, T,scene_t)

    out_plt = plt.figure()
    plt.plot(torch.FloatTensor(model.output_seq).detach().numpy(), label = 'output')
    plt.scatter(np.arange(scene_t * T)[scene_t-1::scene_t], torch.FloatTensor(total_target).detach().numpy()[scene_t-1::scene_t], label = 'target',marker='.', color='r')
    plt.title("Full View of Output")
    plt.legend()
    out_plt.savefig(PATH + '/Total Output {}'.format(R), dpi=600, facecolor='w', transparent=True)
    out_plt.clf()
    plt.close()

def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def hidden_activity(model,r,criterion,PATH,T_step,scene_t):

    model.batch_size = 1

    T = max(100, 20 * r)
    input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=r, T=T, scene_t = scene_t,variation = VAR)
    HL_mult, HL_add = NN.get_hid(model,criterion,input_seq,target_seq,T,scene_t)

    for i in range(len(HL_mult)):
        HL_mult[i] = torch.squeeze(HL_mult[i]).detach().numpy()
        HL_add[i] = torch.squeeze(HL_add[i]).detach().numpy()

    #print(hidden_activity.shape)
    HL_mult = np.transpose(np.array(HL_mult))
    HL_add = np.transpose(np.array(HL_add))
    output_seq = (model.output_seq.detach()>=0.5).float()

    HL_mult = HL_mult[:,scene_t*r:scene_t*r+T_step]
    HL_add = HL_add[:,scene_t*r:scene_t*r+T_step]

    total_target_seq = total_target[scene_t*r:scene_t*r+T_step].detach().numpy().astype(int).reshape(T_step).tolist()
    print_target_seq = [(i, '')[i == 2] for i in total_target_seq]
    x_axis_labels =  merge(print_target_seq, output_seq[scene_t*r:scene_t*r+T_step].detach().numpy().astype(int).reshape(T_step).tolist())

    #x_axis_labels = np.around(total_target[scene_t*r:scene_t*r+T_step].detach().numpy(), decimals=1)

    HL_mult_plot= sns.heatmap(HL_mult, xticklabels=x_axis_labels,linewidth=0.3, cmap= 'vlag', mask=(HL_mult==0))
    HL_mult_figure = HL_mult_plot.get_figure()
    plt.title('Hidden Multiplicative Activity')
    HL_mult_figure.savefig(PATH + '/HL_mult', dpi=600, facecolor='w', transparent=True)
    HL_mult_figure.clf()
    plt.close()

    HL_add_plot= sns.heatmap(HL_add, xticklabels=x_axis_labels, linewidth=0.3, cmap= 'Reds' )
    HL_add_figure = HL_add_plot.get_figure()
    plt.title('Hidden Additive Activity')
    HL_add_figure.savefig(PATH + '/HL_add', dpi=600, facecolor='w', transparent=True)
    HL_add_figure.clf()
    plt.close()

def test_data(R , PATH):
    input_seq,target_seq = DL.generate_seq(vec_len = vec_len, R = R, T = 30,Batch = False)
    data_plot= sns.heatmap(torch.transpose(torch.squeeze(input_seq),0,1), xticklabels=torch.squeeze(target_seq.detach()).numpy().astype(int),linewidth=0.3, cmap= 'rocket_r')
    data_figure = data_plot.get_figure()
    plt.title('data check {}'.format(R))
    data_figure.savefig(PATH + '/Data_Check_R{}'.format(R), dpi=600, facecolor='w', transparent=True)
    data_figure.clf()
    plt.close()


def static_matrix(model, PATH):
    # static matrix
    sm1_plot = sns.heatmap(model.w_mult.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.w_mult.detach().numpy()==0))
    sm1_figure = sm1_plot.get_figure()
    plt.title('Multiplicative Weight Matrix')
    sm1_figure.savefig(PATH + '/W_multiplicative', dpi=600, facecolor='w', transparent=True)
    sm1_figure.clf()
    plt.close()

    sm2_plot = sns.heatmap(model.w_add.detach().numpy(), linewidth=0.3, center=0.00, cmap='vlag',mask=(model.w_add.detach().numpy() == 0))
    sm2_figure = sm2_plot.get_figure()
    plt.title('Additive Weight Matrix')
    sm2_figure.savefig(PATH + '/W_additive', dpi=600, facecolor='w', transparent=True)
    sm2_figure.clf()
    plt.close()

def lmbda_matrix(model, PATH):
    # static matrix
    lmbda1_plot = sns.heatmap(model.lmbda_mult.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.lmbda_mult.detach().numpy()==0))
    lmbda1_figure = lmbda1_plot.get_figure()
    plt.title('lmbda multiplicative Matrix')
    lmbda1_figure.savefig(PATH + '/lmbda_mult', dpi=600, facecolor='w', transparent=True)
    lmbda1_figure.clf()
    plt.close()

    lmbda2_plot = sns.heatmap(model.lmbda_add.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.lmbda_add.detach().numpy()==0))
    lmbda2_figure = lmbda2_plot.get_figure()
    plt.title('lmbda multiplicative Matrix')
    lmbda2_figure.savefig(PATH + '/lmbda_add', dpi=600, facecolor='w', transparent=True)
    lmbda2_figure.clf()
    plt.close()

def eta_matrix(model, PATH):
    # static matrix
    eta_plot = sns.heatmap(model.eta_mult.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.eta_mult.detach().numpy()==0))
    eta_figure = eta_plot.get_figure()
    plt.title('eta mutliplicative Matrix')
    eta_figure.savefig(PATH + '/eta_mult', dpi=600, facecolor='w', transparent=True)
    eta_figure.clf()
    plt.close()


    eta2_plot = sns.heatmap(model.eta_add.detach().numpy(), linewidth=0.3, center=0.00, cmap= 'vlag', mask=(model.eta_add.detach().numpy()==0))
    eta2_figure = eta2_plot.get_figure()
    plt.title('eta additive Matrix')
    eta2_figure.savefig(PATH + '/eta_add', dpi=600, facecolor='w', transparent=True)
    eta2_figure.clf()
    plt.close()

def acc_train(model, PATH ,acc, vec_len, R, T, criterion, optimizer,acc_num):
    writer = SummaryWriter(PATH + '/Training_Log')
    iterations = 0
    acc_flag = 0

    while acc_flag < acc_num:
        input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=R, T=T, scene_t = scene_t, variation=VAR)
        #model.A = torch.zeros(model.hid_dim, model.input_dim)
        accuracy, loss = NN.train(model, criterion, optimizer, input_seq, target_seq, scene_t,Batch = False)
        iterations += 1

        if accuracy.item() >= acc:
            acc_flag += 1
        else:
            acc_flag = 0

        # tensorboard

        writer.add_scalar('Performance / Training Loss', loss.item(), iterations)
        writer.add_scalar('Performance / Accuracy', accuracy.item(), iterations)


        Norm_w_mult = np.linalg.norm(torch.squeeze(model.w_mult.detach()))
        Var_w_mult = np.var(torch.squeeze(model.w_mult.detach()).numpy())
        Norm_w_add = np.linalg.norm(torch.squeeze(model.w_add.detach()))
        Var_w_add = np.var(torch.squeeze(model.w_add.detach()).numpy())
        Norm_w_final = np.linalg.norm(torch.squeeze(model.w_final.detach()))
        Var_w_final = np.var(torch.squeeze(model.w_final.detach()).numpy())

        Norm_b_mult = np.linalg.norm(torch.squeeze(model.b_mult.detach()))
        Var_b_mult = np.var(torch.squeeze(model.b_mult.detach()).numpy())
        Norm_b_add = np.linalg.norm(torch.squeeze(model.b_add.detach()))
        Var_b_add = np.var(torch.squeeze(model.b_add.detach()).numpy())
        Norm_b_final = np.linalg.norm(torch.squeeze(model.b_final.detach()))
        Var_b_final = np.var(torch.squeeze(model.b_final.detach()).numpy())


        '''
        Norm_lmbda_mult = np.linalg.norm(torch.squeeze(model.lmbda_mult.detach()))
        Var_lmbda_mult = np.var(torch.squeeze(model.lmbda_mult.detach()).numpy())
        Norm_lmbda_add = np.linalg.norm(torch.squeeze(model.lmbda_add.detach()))
        Var_lmbda_add = np.var(torch.squeeze(model.lmbda_add.detach()).numpy())
        '''
        '''
        
        Norm_eta_mult = np.linalg.norm(torch.squeeze(model.eta_mult.detach()))
        Var_eta_mult = np.var(torch.squeeze(model.eta_mult.detach()).numpy())
        Norm_eta_add = np.linalg.norm(torch.squeeze(model.eta_add.detach()))
        Var_eta_add = np.var(torch.squeeze(model.eta_add.detach()).numpy())
        '''

        #Norm_eta_mult = model.eta_mult.item()
        #Norm_eta_add = model.eta_add.item()




        writer.add_scalar('Weights/ Norm_w_mult', Norm_w_mult, iterations)
        writer.add_scalar('Weights/ Var_w_mult', Var_w_mult, iterations)
        writer.add_scalar('Weights/ Norm_b_mult', Norm_b_mult, iterations)
        writer.add_scalar('Weights/ Var_b_mult', Var_b_mult, iterations)

        writer.add_scalar('Weights/ Norm_w_add', Norm_w_add, iterations)
        writer.add_scalar('Weights/ Var_w_add', Var_w_add, iterations)
        writer.add_scalar('Weights/ Norm_b_add', Norm_b_add, iterations)
        writer.add_scalar('Weights/ Var_b_add', Var_b_add, iterations)

        writer.add_scalar('Weights/ Norm_w_final', Norm_w_final, iterations)
        writer.add_scalar('Weights/ Var_w_final', Var_w_final, iterations)
        writer.add_scalar('Weights/ Norm_b_final', Norm_b_final, iterations)
        writer.add_scalar('Weights/ Var_b_final', Var_b_final, iterations)

        #writer.add_scalar('Parameters / Norm_lmbda_mult', Norm_lmbda_mult, iterations)
        #writer.add_scalar('Parameters / Var_lmbda_mult', Var_lmbda_mult, iterations)
        #writer.add_scalar('Parameters / Norm_lmbda_add', Norm_lmbda_add, iterations)
        #writer.add_scalar('Parameters / Var_lmbda_add', Var_lmbda_add, iterations)

        #writer.add_scalar('Parameters / Norm_eta_mult', Norm_eta_mult, iterations)
        #writer.add_scalar('Parameters / Var_eta_mult', Var_eta_mult, iterations)
        #writer.add_scalar('Parameters / Norm_eta_add', Norm_eta_add, iterations)
        #writer.add_scalar('Parameters / Var_eta_add', Var_eta_add, iterations)

        writer.add_scalar('Synapse / Lambda_add', model.lmbda_add.item(), iterations)
        writer.add_scalar('Synapse / Eta_add', model.eta_add.item(), iterations)
        writer.add_scalar('Synapse / Lambda_mult', model.lmbda_mult.item(), iterations)
        writer.add_scalar('Synapse / Eta_mult', model.eta_mult.item(), iterations)

        # gradient norm1

        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Grad_norm', total_norm, iterations)

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
    HebbFF_CFD = NN.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(HebbFF_CFD.parameters())
    
    while True:
        R += 1
        print('Current Training Interval Value: ',R)
        T = max(100, 20 * R)

        PATH = os.getcwd() + '/Result/uniform_eta_d_{}_N_{}_Var_{}_st_{}/R_{}'.format(vec_len, hid_dim, VAR, scene_t, R)
        isExist = os.path.exists(PATH)
        
        if not isExist:
            os.makedirs(PATH)

        # training
        acc_train(HebbFF_CFD, PATH, acc, vec_len, R, T, criterion, optimizer,acc_num = 5)

        # saving the model
        hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=30, scene_t = scene_t)

        torch.save(HebbFF_CFD.state_dict(), PATH + '/Model')

        # test accuracy
        test(HebbFF_CFD, 1, max (20, 5 * R), criterion, PATH)
        entire_output(HebbFF_CFD, criterion, PATH, R)
        
        # static matrix

        static_matrix(HebbFF_CFD, PATH)
        #eta_matrix(HebbFF_CFD, PATH)
        #lmbda_matrix(HebbFF_CFD, PATH)

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

        #hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=20)

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
    A_norm = NN.A_converge_test(model, criterion, optimizer,input_seq, target_seq, T)
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
    test_converge = NN.HebbFF(input_dim = vec_len, hid_dim=hid_dim, out_dim=out_dim, batch_size = batch_size, hid_nonl = Sigmoid)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(test_converge.parameters())
    generate_norm_graph(test_converge,1,criterion, optimizer)
    '''
