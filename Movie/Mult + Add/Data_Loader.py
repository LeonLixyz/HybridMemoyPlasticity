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
import os
import seaborn as sns


def generate_seq(vec_len, R, T, Batch =  True, batch_size = 1):
    batch_input = []
    batch_target = []

    if Batch ==  True:
        for batch in range (batch_size):
            input = []
            target = []
            for i in range(R):
                input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
                target.append(torch.zeros(1))

            for i in range(R, 2*R):
                if np.random.rand() > 0.5:
                    input.append(input[i-R])
                    target.append(torch.ones(1))
                else:
                    input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
                    target.append(torch.zeros(1))

            for i in range(2*R,T):
                if np.random.rand() > 0.5 and (not torch.equal(input[i-R],input[i-2*R])):
                    input.append(input[i-R])
                    target.append(torch.ones(1))
                else:
                    input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
                    target.append(torch.zeros(1))


            batch_input.append(torch.stack(input, dim=0))
            batch_target.append(torch.stack(target, dim=0))

        input_seq = torch.stack(batch_input, dim=0)
        target_seq = torch.stack(batch_target, dim=0)

    else:
        input = []
        target = []
        for i in range(R):
            input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
            target.append(torch.zeros(1))

        for i in range(R, 2 * R):
            if np.random.rand() >= 0.5:
                input.append(input[i - R])
                target.append(torch.ones(1))
            else:
                input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
                target.append(torch.zeros(1))

        for i in range(2 * R, T):
            if np.random.rand() >= 0.5 and (not torch.equal(input[i - R], input[i - 2 * R])):
                input.append(input[i - R])
                target.append(torch.ones(1))
            else:
                input.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
                target.append(torch.zeros(1))

        input_seq = torch.stack(input, dim=0)
        target_seq = torch.stack(target, dim=0)

    return(input_seq,target_seq)
    

def generate_mix_seq(batch_size, vec_len, R, T):
    mix_input_seq = []
    mix_target_seq = []
    for r in R:
        input_seq,target_seq = generate_seq(int(batch_size/len(R)), vec_len, r, T)
        mix_input_seq.append(input_seq)
        mix_target_seq.append(target_seq)
    
    mix_input_seq = torch.cat(mix_input_seq, dim=0)
    mix_target_seq = torch.cat(mix_target_seq, dim=0)
    shuf = np.random.permutation(len(mix_input_seq))

    return mix_input_seq[shuf],mix_target_seq[shuf]

def generate_new_scence(movie, target, total_target, scene_t,vec_len,variation):
    Scence = []
    Variable = int(vec_len * variation)
    Base = 2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1
    Scence.append(copy.deepcopy(Base))
    for i in range(1, scene_t):
        random_index = random.sample(range(vec_len), Variable)
        #next_frame = copy.deepcopy(Scence[-i])
        next_frame = copy.deepcopy(Base)
        next_frame[random_index] = next_frame[random_index] * -1
        Scence.append(copy.deepcopy(next_frame))

        #Scence.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
        total_target.append(torch.ones(1) * 2)
    total_target.append(torch.zeros(1))
    movie.append(Scence)
    target.append(torch.zeros(1))

def generate_movie(R,T,vec_len,scene_t,variation):
    movie = []
    target = []
    total_target =[]
    #scene_t = np.random.randint(low = 5, high = 10)
    for i in range(R):
        generate_new_scence(movie, target, total_target, scene_t, vec_len ,variation)

    for i in range(R, T):
        if np.random.rand() >= 0.5 and (not (target[i - R] == torch.ones(1))):
            movie.append(random.sample(movie[i - R], len(movie[i - R])))
            target.append(torch.ones(1))
            for j in range(len(movie[i - R]) - 1):
                total_target.append(torch.ones(1)*2)
            total_target.append(torch.ones(1))

        else:
            generate_new_scence(movie, target, total_target, scene_t, vec_len ,variation)

    target = torch.FloatTensor(target)
    total_target = torch.FloatTensor(total_target)


    return movie, target, total_target





