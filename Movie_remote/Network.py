import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

from Networks.ML import ML
from Networks.Stack import Stack
from Networks.Dyn_RO import Dyn_RO
from Networks.Nonl_RO import Nonl_RO


# from torch.utils.tensorboard import SummaryWriter

    
def network_selector(input_dim, hid_dim_list, out_dim, hetero_list, plastic_list, network_name):
    if network_name == 'ML':
        return ML(input_dim, hid_dim_list, out_dim, hetero_list, plastic_list)
    elif network_name == 'Stack':
        return Stack(input_dim, hid_dim_list, out_dim, hetero_list, plastic_list)
    elif network_name == 'Dyn_RO':
        return Dyn_RO(input_dim, hid_dim_list, out_dim, hetero_list, plastic_list)
    elif network_name == 'Nonl_RO':
        return Nonl_RO(input_dim, hid_dim_list, out_dim, hetero_list, plastic_list)
    else:
        raise ValueError(f"Invalid network name '{network_name}'.")

        
def clear_plastic_matrix(model):
    for i in range(len(model.plastic_matrix_list)):
        model.plastic_matrix_list[i] = torch.zeros_like(model.plastic_matrix_list[i])


def train(model, criterion, optimizer, input_seq, target_seq, scene_t, device='cpu'):
    model.accuracy = 0
    model.loss = 0
    model.output_seq = []
    optimizer.zero_grad()
    model.HL_list = [[] for _ in range(len(model.hid_dim_list))]
    model.output_seq = torch.zeros(len(target_seq) * scene_t)

    T = len(input_seq)
    for t in range(T):
        scene = input_seq[t]
        for i in range(len(scene)):
            frame = scene[i]
            out = model(frame, Store=False)
            model.output_seq[scene_t * t + i] = out

    target_seq = target_seq

    model.loss = criterion(model.output_seq[scene_t - 1::scene_t], target_seq)
    model.predicted = (model.output_seq[scene_t - 1::scene_t].detach() >= 0.5).float()
    model.accuracy = (model.predicted == target_seq).sum() / T

    model.loss.backward()

    optimizer.step()

    clear_plastic_matrix(model)

    return model.accuracy, model.loss


    
def test(model,criterion,input_seq,target_seq,T,scene_t, device = 'cpu'):
    model.accuracy = 0
    model.loss = 0
    model.output_seq = torch.zeros(len(target_seq)*scene_t)


    T = len(input_seq)
    for t in range(T):
        scene = input_seq[t]
        for i in range(len(scene)):
            #frame = scene[i].to(device)
            frame = scene[i]
            out =  model(frame, Store = False)
            model.output_seq[scene_t*t+i] = out

    target_seq = target_seq
    model.loss = criterion(model.output_seq[scene_t-1::scene_t], target_seq)
    model.predicted = (model.output_seq[scene_t-1::scene_t].detach()>=0.5).float()
    model.accuracy  = (model.predicted == target_seq).sum() / T

    # clear the plastic matrix
    clear_plastic_matrix(model)  

    return model.predicted, model.accuracy, model.loss


def get_hid(model, input_seq, target_seq, T, scene_t, device = 'cpu'):
    model.loss = 0
    model.accuracy = 0
    # we want to store the plastic matrix as well
    plstic_plot_list = [[] for _ in range(len(model.plastic_matrix_list))]
    model.output_seq = torch.zeros(len(target_seq)*scene_t)

    if model.network_name == 'Stack':
        model.HL_list = [[]]
    else:
        model.HL_list = [[] for _ in range(len(model.hid_dim_list))]
    
    if hasattr(model, 'W_list'):
        model.W_list = [[] for _ in range(len(model.hid_dim_list))]

    T = len(input_seq)
    for t in range(T):
        scene = input_seq[t]
        for i in range(len(scene)):
            #frame = scene[i].to(device)
            frame = scene[i]
            out =  model(frame, Store = True)
            model.output_seq[scene_t * t + i] = out
            for j in range(len(model.plastic_matrix_list)):
                plstic_plot_list[j].append(model.plastic_matrix_list[j].detach().cpu().numpy())
        

    clear_plastic_matrix(model)
    
    if hasattr(model, 'W_list'):
        return model.HL_list, plstic_plot_list, model.W_list
    else:
        return model.HL_list, plstic_plot_list

