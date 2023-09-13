import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
    
    
class ML(nn.Module):
    def __init__(self, input_dim, hid_dim_list, out_dim, hetero_list, plastic_list):
        super().__init__()
        self.input_dim = input_dim
        hid_dim_list = hid_dim_list
        self.hid_dim_list = hid_dim_list
        self.training_loss = []
        self.HL_list = [[] for i in range(len(hid_dim_list))]
        self.plastic_list = plastic_list
        self.hetero_list = hetero_list
        self.network_name = 'ML'
        self.W_list = [[] for i in range(len(hid_dim_list))]


        #make it parameter list
        self.lambda_list = nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for i in range(len(plastic_list))])

        self.plastic_matrix_list = [torch.zeros(self.hid_dim_list[i], self.hid_dim_list[i-1]) for i in range(1, len(self.plastic_list))]
        self.plastic_matrix_list.insert(0, torch.zeros(self.hid_dim_list[0], self.input_dim))

        self.weight_matrix_list = nn.ParameterList()
        self.weight_matrix_list.append(nn.Parameter(torch.empty(self.hid_dim_list[0], self.input_dim)))
        for i in range(1, len(self.hid_dim_list)):
            self.weight_matrix_list.append(nn.Parameter(torch.empty(self.hid_dim_list[i], self.hid_dim_list[i-1])))


        for i in range(len(self.weight_matrix_list)):
            nn.init.xavier_normal_(self.weight_matrix_list[i], gain=1.0)

        self.w_final = torch.nn.Parameter(torch.empty(out_dim, self.hid_dim_list[-1]))
        nn.init.xavier_normal_(self.w_final, gain=1.0)
        self.bias_matrix_list = nn.ParameterList([nn.Parameter(torch.empty(self.hid_dim_list[i], 1)) for i in range(len(self.hid_dim_list))])
        for i in range(len(self.bias_matrix_list)):
            nn.init.xavier_normal_(self.bias_matrix_list[i], gain=1.0)

        self.b_final = torch.nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.b_final, gain=1.0)
    

        self.eta_matrix_list = nn.ParameterList([])
        for i in range(len(self.hetero_list)):
            if self.hetero_list[i] == 1:
                self.eta_matrix_list.append(torch.nn.Parameter(torch.empty(self.plastic_matrix_list[i].shape)))
                nn.init.xavier_normal_(self.eta_matrix_list[i], gain=1.0)
            elif self.hetero_list[i] == 0.5:
                self.eta_matrix_list.append(torch.nn.Parameter(torch.empty((self.hid_dim_list[i],1))))
                nn.init.xavier_normal_(self.eta_matrix_list[i], gain=1.0)
            else:
                self.eta_matrix_list.append(torch.nn.Parameter(torch.tensor(-0.5)))

        self.output_seq = 0
        self.predicted = 0
        self.accuracy = 0
        self.loss = 0
    

    def forward(self, x_t, Store):

        x_t = torch.unsqueeze(x_t, -1)
        prev_h_t = x_t
        for i in range(len(self.weight_matrix_list)):


            bias = self.bias_matrix_list[i]

            if self.plastic_list[i] == 'A':
                if i == len(self.weight_matrix_list) - 1:
                    h_t = F.sigmoid((self.weight_matrix_list[i] + self.plastic_matrix_list[i]) @ prev_h_t + bias)
                else:
                    h_t = F.tanh((self.weight_matrix_list[i] + self.plastic_matrix_list[i]) @ prev_h_t + bias)       
                if Store == True:
                    self.W_list[i].append(self.weight_matrix_list[i] + self.plastic_matrix_list[i])          

            elif self.plastic_list[i] == 'M':
                if i == len(self.weight_matrix_list) - 1:
                    h_t = F.sigmoid((self.weight_matrix_list[i] * self.plastic_matrix_list[i]) @ prev_h_t + self.weight_matrix_list[i] @ prev_h_t + bias)
                else:
                    h_t = F.tanh((self.weight_matrix_list[i] * self.plastic_matrix_list[i]) @ prev_h_t + self.weight_matrix_list[i] @ prev_h_t + bias)
                if Store == True:
                    self.W_list[i].append(self.weight_matrix_list[i] * self.plastic_matrix_list[i])

            self.plastic_matrix_list[i] = F.sigmoid(self.lambda_list[i]) * self.plastic_matrix_list[i] + self.eta_matrix_list[i] * (h_t @ torch.transpose(prev_h_t, 0, 1))

            if Store == True:
                self.HL_list[i].append(h_t)

            prev_h_t = h_t.clone()

        y_t = F.sigmoid(self.w_final @ h_t + self.b_final)
        y_t = torch.squeeze(y_t, -1)

        return y_t

