import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
    
    
class Stack(nn.Module):
    def __init__(self, input_dim, hid_dim_list, out_dim, hetero_list, plastic_list):
        super().__init__()
        self.input_dim = input_dim
        hid_dim_list = hid_dim_list
        self.hid_dim_list = hid_dim_list
        self.training_loss = []
        self.HL_list = [[]]
        self.plastic_list = plastic_list
        self.hetero_list = hetero_list
        self.network_name = 'Stack'

        #make it parameter list
        self.lambda_list = nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for i in range(len(plastic_list))])

        self.plastic_matrix_list = [torch.zeros(self.hid_dim_list[i], self.input_dim) for i in range(1, len(self.hid_dim_list))]
        self.plastic_matrix_list.insert(0, torch.zeros(self.hid_dim_list[0], self.input_dim))

        self.weight_matrix_list = nn.ParameterList()
        self.weight_matrix_list.append(nn.Parameter(torch.empty(self.hid_dim_list[0], self.input_dim)))
        for i in range(1, len(self.hid_dim_list)):
            self.weight_matrix_list.append(nn.Parameter(torch.empty(self.hid_dim_list[i], self.input_dim)))


        for i in range(len(self.weight_matrix_list)):
            nn.init.xavier_normal_(self.weight_matrix_list[i], gain=1.0)


        self.w_final = torch.nn.Parameter(torch.empty(out_dim, self.hid_dim_list.sum()))
        nn.init.xavier_normal_(self.w_final, gain=1.0)
        self.bias_matrix_list = nn.ParameterList([nn.Parameter(torch.empty(self.hid_dim_list[i], 1)) for i in range(len(self.hid_dim_list))])
        for i in range(len(self.bias_matrix_list)):
            nn.init.xavier_normal_(self.bias_matrix_list[i], gain=1.0)

        self.b_final = torch.nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.b_final, gain=1.0)
    

        self.eta_matrix_list = nn.ParameterList([])
        for i in range(len(self.hetero_list)):
            if self.hetero_list[i] == True:
                self.eta_matrix_list.append(torch.nn.Parameter(torch.empty(self.plastic_matrix_list[i].shape)))
                nn.init.xavier_normal_(self.eta_matrix_list[i], gain=1.0)
            else:
                self.eta_matrix_list.append(torch.nn.Parameter(torch.tensor(-0.5)))

        self.output_seq = 0
        self.predicted = 0
        self.accuracy = 0
        self.loss = 0


    def forward(self, x_t, Store):

        stack_ht = []
        x_t = torch.unsqueeze(x_t, -1)
        for i in range(len(self.plastic_list)):
            
            bias = self.bias_matrix_list[i]

            if self.plastic_list[i] == 'A':
                    h_t = F.sigmoid((self.weight_matrix_list[i] + self.plastic_matrix_list[i]) @ x_t + bias)
    
            elif self.plastic_list[i] == 'M':
                    h_t = F.sigmoid((self.weight_matrix_list[i] * self.plastic_matrix_list[i]) @ x_t + self.weight_matrix_list[i] @ x_t + bias)

            self.plastic_matrix_list[i] = F.sigmoid(self.lambda_list[i]) * self.plastic_matrix_list[i] + self.eta_matrix_list[i] * (h_t @ torch.transpose(x_t, 0, 1))

            stack_ht.append(h_t)
        
        # stack all hidden layers in stack_ht
        for i in range(len(self.plastic_list)):
            if i == 0:
                h_t = stack_ht[i]
            else:
                h_t = torch.cat((h_t, stack_ht[i]), 0)

        if Store == True:
            self.HL_list[0].append(h_t)

        y_t = F.sigmoid(self.w_final @ h_t + self.b_final)
        y_t = torch.squeeze(y_t, -1)

        return y_t

    def print_gradients(self):
        print("Gradients for weight_matrix_list:")
        for i, weight_matrix in enumerate(self.weight_matrix_list):
            print(f"Layer {i}:", weight_matrix.grad)
        
        print("Gradients for bias_matrix_list:")
        for i, bias_matrix in enumerate(self.bias_matrix_list):
            print(f"Layer {i}:", bias_matrix.grad)
        
        print("Gradients for lambda_list:")
        for i, lambda_param in enumerate(self.lambda_list):
            print(f"Layer {i}:", lambda_param.grad)
        
        print("Gradients for eta_matrix_list:")
        for i, eta_matrix in enumerate(self.eta_matrix_list):
            print(f"Layer {i}:", eta_matrix.grad)

        

