import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import random


import torch
import random
import numpy as np

def generate_new_scene_batch(movie, target, total_target, scene_t, vec_len, variation):
    scene = []
    variable = int(vec_len * variation)
    base = 2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1
    scene.append(base)

    for i in range(1, scene_t):
        random_index = random.sample(range(vec_len), variable)
        next_frame = base.clone()
        next_frame[random_index] *= -1
        scene.append(next_frame)
        total_target.append(torch.tensor([2]))

    total_target.append(torch.tensor([0]))
    movie.append(scene)
    target.append(torch.tensor([0]))

def generate_movie_batch(batch_size, R, T, vec_len, scene_t, variation):
    movies = []
    targets = []
    total_targets = []

    for b in range(batch_size):
        movie = []
        target = []
        total_target = []

        for i in range(R):
            generate_new_scene_batch(movie, target, total_target, scene_t, vec_len, variation)

        for i in range(R, T):
            if np.random.rand() >= 0.5 and target[i - R].item() != 1:
                movie.append(random.sample(movie[i - R], len(movie[i - R])))
                target.append(torch.tensor([1]))
                total_target.extend([torch.tensor([2])] * (len(movie[i - R]) - 1))
                total_target.append(torch.tensor([1]))

            else:
                generate_new_scene_batch(movie, target, total_target, scene_t, vec_len, variation)

        target = torch.FloatTensor(target)
        total_target = torch.FloatTensor(total_target)
        
        movies.append(movie)
        targets.append(target)
        total_targets.append(total_target)

    return movies, targets, total_targets

def generate_new_scene(movie, target, total_target, scene_t, vec_len, variation):
    scene = []
    Variable = int(vec_len * variation)
    Base = 2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1
    scene.append(Base.clone())
    for i in range(1, scene_t):
        random_index = random.sample(range(vec_len), Variable)
        next_frame = Base.clone()
        next_frame[random_index] = next_frame[random_index] * -1
        scene.append(next_frame.clone())

        #Scence.append(2 * torch.bernoulli(torch.empty(vec_len).uniform_(0, 1)) - 1)
        total_target.append(torch.ones(1) * 2)
    total_target.append(torch.zeros(1))
    movie.append(scene)
    target.append(torch.zeros(1))

def generate_movie(R,T,vec_len,scene_t,variation):
    movie = []
    target = []
    total_target =[]
    #scene_t = np.random.randint(low = 5, high = 10)
    for i in range(R):
        generate_new_scene(movie, target, total_target, scene_t, vec_len ,variation)

    for i in range(R, T):
        if np.random.rand() >= 0.5 and (not (target[i - R] == torch.ones(1))):
            movie.append(random.sample(movie[i - R], len(movie[i - R])))
            target.append(torch.ones(1))
            for j in range(len(movie[i - R]) - 1):
                total_target.append(torch.ones(1)*2)
            total_target.append(torch.ones(1))

        else:
            generate_new_scene(movie, target, total_target, scene_t, vec_len ,variation)

    target = torch.FloatTensor(target)
    total_target = torch.FloatTensor(total_target)

    return movie, target, total_target
