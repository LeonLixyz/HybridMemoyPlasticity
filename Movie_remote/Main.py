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
import sys
import os
import seaborn as sns
import ast
import glob


# from torch.utils.tensorboard import SummaryWriter

# meta parameters
VAR = 0.1
vec_len = 25
lr_rate = 0.001
acc = 0.99
out_dim = 1
Sigmoid = nn.Sigmoid()

LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)

network_name = sys.argv[1]
hid_dim_list =  np.array(ast.literal_eval(sys.argv[2]))
hetero_list = np.array(ast.literal_eval(sys.argv[3]))

input_string = sys.argv[4]
stripped_string = input_string.strip('[]').strip()
list_of_strings = stripped_string.split(',')
plastic_list = [s.strip() for s in list_of_strings]
scene_t = int(sys.argv[5])

metadata = {
    'network type': network_name,
    'Sequence of plastic matrix': plastic_list,
    'Hidden dimensions': hid_dim_list,
    'variation': VAR,
    'frames per scene': scene_t,
    'input': vec_len,
    'Hetero eta': hetero_list,
}

def load_model(PATH):

    model = NN.network_selector(input_dim = vec_len, hid_dim_list = hid_dim_list, out_dim = out_dim, hetero_list = hetero_list, plastic_list = plastic_list, network_name = network_name)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model


def test(model,r1, r2,criterion,PATH, device):
    ACC_R = []
    for i in range(r1,r2):
        T = max(1000,200 * i)
        input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=i, T=T, scene_t = scene_t,variation=VAR)
        output_seq,accuracy,total_loss = NN.test(model,criterion,input_seq,target_seq,T,scene_t, device)
        ACC_R.append(accuracy.item())
        

    r_axis = np.arange(r1,r2)
    acc_plt = plt.figure()
    plt.plot(r_axis,ACC_R)
    plt.title("Accuracy with different interval R in [{},{}]".format(r1,r2))
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

def graph_clip(matrix):
    # Calculate mean and standard deviation
    mean = np.mean(matrix)
    std_dev = np.std(matrix)

    # Calculate bounds
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev

    # Clip matrix
    clipped_matrix = np.clip(matrix, lower_bound, upper_bound)

    return clipped_matrix


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def hidden_activity(model, r, criterion, PATH, T_step, scene_t, device):

    T = max(100, 20 * r)
    input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=r, T=T, scene_t=scene_t, variation=VAR)

    if hasattr(model, 'W_list'):
        HL_list, plastic_list, W_list = NN.get_hid(model, input_seq, target_seq, T, scene_t, device)
    else:
        HL_list, plastic_list = NN.get_hid(model, input_seq, target_seq, T, scene_t, device)

    if hasattr(model, 'W_list'):
        # Generate heatmaps for all weight matrices in W_list side by side
        n_w_layers = len(W_list)
        for t in range(scene_t * r, scene_t * r + T_step):
            fig_w, axs_w = plt.subplots(1, n_w_layers, figsize=(8 * n_w_layers, 8), dpi=600)
            for w_idx, w in enumerate(W_list):
                w = [i.detach().numpy() for i in w]
                w = graph_clip(w)
                if n_w_layers == 1:
                    w_plot = sns.heatmap(w[t], linewidth=0.3, center=0.00, cmap='vlag', ax=axs_w, cbar_kws={'shrink': 0.5}, square=True)
                    axs_w.set_title(f'Actual Weight Matrix at time {t}')
                else:
                    w_plot = sns.heatmap(w[t], linewidth=0.3, center=0.00, cmap='vlag', ax=axs_w[w_idx], cbar_kws={'shrink': 0.5}, square=True)
                    axs_w[w_idx].set_title(f'Actual Weight Matrix {w_idx + 1} at time {t}')

            fig_w.tight_layout()
            # save in the plastic folder:
            if not os.path.exists(PATH + '/Actual_Weight'):
                os.makedirs(PATH + '/Actual_Weight')
            fig_w.savefig(PATH + f'/Actual_Weight/Actual_Weight_combined_{t}', dpi=600, facecolor='w', transparent=True)
            plt.close(fig_w)

    # hidden activity
    output_seq = (model.output_seq.detach() >= 0.5).float()
    total_target_seq = total_target[scene_t * r:scene_t * r + T_step].detach().numpy().astype(int).reshape(T_step).tolist()
    x_axis_labels = merge(output_seq[scene_t * r:scene_t * r + T_step].detach().numpy().astype(int).reshape(T_step).tolist(), total_target_seq)

    # Generate heatmaps for all hidden layers in HL_list side by side

    n_hl_layers = len(HL_list)

    fig_hl, axs_hl = plt.subplots(1, n_hl_layers, figsize=(8 * n_hl_layers, 8), dpi=600)

    for hl_idx, HL in enumerate(HL_list):
        for i in range(len(HL)):
            HL[i] = torch.squeeze(HL[i]).detach().numpy()
        HL = np.transpose(np.array(HL))
        HL = HL[:, scene_t * r:scene_t * r + T_step]
        if n_hl_layers == 1:
            hl_plot = sns.heatmap(HL, xticklabels=x_axis_labels, linewidth=0.3, center=0.00, cmap='vlag', ax=axs_hl, cbar_kws={'shrink': 0.5}, square=True)
            axs_hl.set_title(f'Hidden Activity {hl_idx + 1}')
        else:
            hl_plot = sns.heatmap(HL, xticklabels=x_axis_labels, linewidth=0.3, center=0.00, cmap='vlag', ax=axs_hl[hl_idx], cbar_kws={'shrink': 0.5}, square=True)
            axs_hl[hl_idx].set_title(f'Hidden Activity {hl_idx + 1}')

    fig_hl.tight_layout()
    fig_hl.savefig(PATH + f'/HL_Combined', dpi=600, facecolor='w', transparent=True)
    plt.close(fig_hl)

    # Generate heatmaps for all plastic matrices in plastic_list side by side
    n_plastic_layers = len(plastic_list)
    for t in range(scene_t * r, scene_t * r + T_step):
        fig_plastic, axs_plastic = plt.subplots(1, n_plastic_layers, figsize=(8 * n_plastic_layers, 8), dpi=600)
        for plastic_idx, plastic in enumerate(plastic_list):
            plastic = graph_clip(plastic)
            if n_plastic_layers == 1:
                plastic_plot = sns.heatmap(plastic[t], linewidth=0.3, center=0.00, cmap='vlag', ax=axs_plastic, cbar_kws={'shrink': 0.5}, square=True)
                axs_plastic.set_title(f'Plastic Matrix at time {t}')
            else:
                plastic_plot = sns.heatmap(plastic[t], linewidth=0.3, center=0.00, cmap='vlag', ax=axs_plastic[plastic_idx], cbar_kws={'shrink': 0.5}, square=True)
                axs_plastic[plastic_idx].set_title(f'Plastic Matrix {plastic_idx + 1} at time {t}')

        fig_plastic.tight_layout()
        # save in the plastic folder:
        if not os.path.exists(PATH + '/Plastic'):
            os.makedirs(PATH + '/Plastic')
        fig_plastic.savefig(PATH + f'/Plastic/Plastic_combined_{t}', dpi=600, facecolor='w', transparent=True)
        plt.close(fig_plastic)
    


def weight_matrix(model, PATH):
    n_layers = len(model.weight_matrix_list)

    fig, axs = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8), dpi=600)

    for idx, weight_matrix in enumerate(model.weight_matrix_list):
        weight_matrix_np = weight_matrix.detach().numpy()
        weight_matrix_np = graph_clip(weight_matrix_np)  # Clip the values between -1 and 1
        if n_layers == 1:
            weight_matrix_plot = sns.heatmap(weight_matrix_np, linewidth=0.3, center=0.00, cmap='vlag', ax=axs, cbar_kws={'shrink': 0.5}, square=True)
            axs.set_title(f'Layer Weight Matrix')
        else:    
            weight_matrix_plot = sns.heatmap(weight_matrix_np, linewidth=0.3, center=0.00, cmap='vlag', ax=axs[idx], cbar_kws={'shrink': 0.5}, square=True)
            axs[idx].set_title(f'Layer {idx + 1} Weight Matrix')

    fig.tight_layout()
    fig.savefig(PATH + f'/Weight_Matrices', dpi=600, facecolor='w', transparent=True)
    plt.close(fig)

    # plot the w_final
    w_final = model.w_final.detach().numpy()
    fig_w_final, axs_w_final = plt.subplots(1, 1, figsize=(8, 8), dpi=600)
    w_final_plot = sns.heatmap(w_final, linewidth=0.3, center=0.00, cmap='vlag', ax=axs_w_final, cbar_kws={'shrink': 0.5}, square=True)
    axs_w_final.set_title(f'Final Weight Matrix')
    fig_w_final.tight_layout()
    fig_w_final.savefig(PATH + f'/Final_Weight_Matrix', dpi=600, facecolor='w', transparent=True)
    plt.close(fig_w_final)



def eta_matrix(model, PATH):
    n_layers = len(model.eta_matrix_list)

    fig, axs = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8), dpi=600)

    for idx, eta_matrix in enumerate(model.eta_matrix_list):
        if eta_matrix.numel() > 1:
            eta_matrix_np = eta_matrix.detach().numpy()
            eta_matrix_np = graph_clip(eta_matrix_np)  # Clip the values between -1 and 1
            if n_layers == 1:
                eta_matrix_plot = sns.heatmap(eta_matrix_np, linewidth=0.3, center=0.00, cmap='vlag', ax=axs, cbar_kws={'shrink': 0.5}, square=True)
                axs.set_title(f'Eta Matrix')
            else:
                eta_matrix_plot = sns.heatmap(eta_matrix_np, linewidth=0.3, center=0.00, cmap='vlag', ax=axs[idx], cbar_kws={'shrink': 0.5}, square=True)
                axs[idx].set_title(f'Layer {idx + 1} eta Matrix')

    fig.tight_layout()
    fig.savefig(PATH + f'/Eta_Matrices', dpi=600, facecolor='w', transparent=True)
    plt.close(fig)



    
def acc_train(model, PATH ,acc, vec_len, R, T, criterion, optimizer,acc_num, device):
    writer = SummaryWriter(PATH + '/Training_Log')
    iterations = 0
    acc_flag = 0
    # iterations less than a million step:
    while acc_flag < acc_num and iterations < 1000000:

        input_seq, target_seq, total_target = DL.generate_movie(vec_len=vec_len, R=R, T=T,scene_t = scene_t, variation=VAR)
        #model.to(device)
        accuracy, loss = NN.train(model, criterion, optimizer, input_seq, target_seq, scene_t, device = device)
        iterations += 1

        if accuracy.item() >= acc:
            acc_flag += 1
        else:
            acc_flag = 0

        # tensorboard
        if iterations % 100 == 0:

            writer.add_scalar('Performance / Training Loss', loss.item(), iterations)
            writer.add_scalar('Performance / Accuracy', accuracy.item(), iterations)

            for idx, weight_matrix in enumerate(model.weight_matrix_list):
                norm_w = np.linalg.norm(torch.squeeze(weight_matrix.detach()))
                var_w = np.var(torch.squeeze(weight_matrix.detach()).numpy())
                writer.add_scalar(f'wb / Norm of w_{idx + 1}', norm_w, iterations)
                writer.add_scalar(f'wb / Var of w_{idx + 1}', var_w, iterations)

            for idx, eta_matrix in enumerate(model.eta_matrix_list):
                if eta_matrix.numel() == 1:
                    writer.add_scalar(f'Synapse / Eta_{idx + 1}', eta_matrix.item(), iterations)
                else:
                    norm_eta = np.linalg.norm(torch.squeeze(eta_matrix.detach()))
                    var_eta = np.var(torch.squeeze(eta_matrix.detach()).numpy())
                    writer.add_scalar(f'Synapse / Norm of Eta_{idx + 1}', norm_eta, iterations)
                    writer.add_scalar(f'Synapse / Var of Eta_{idx + 1}', var_eta, iterations)
            
            for idx, lambda_list in enumerate(model.lambda_list):
                writer.add_scalar (f'Synapse / Lambda_{idx + 1}', lambda_list.item(), iterations)

            for idx, bias_matrix in enumerate(model.bias_matrix_list):
                norm_b = np.linalg.norm(torch.squeeze(bias_matrix.detach()))
                var_b = np.var(torch.squeeze(bias_matrix.detach()).numpy())
                writer.add_scalar(f'wb / Norm of b_{idx + 1}', norm_b, iterations)
                writer.add_scalar(f'wb / Var of b_{idx + 1}', var_b, iterations)

            # w_final and b_final
            norm_w_final = np.linalg.norm(torch.squeeze(model.w_final.detach()))
            var_w_final = np.var(torch.squeeze(model.w_final.detach()).numpy())
            norm_b_final = np.linalg.norm(torch.squeeze(model.b_final.detach()))
            var_b_final = np.var(torch.squeeze(model.b_final.detach()).numpy())
            writer.add_scalar(f'wb / Norm of w_final', norm_w_final, iterations)
            writer.add_scalar(f'wb / Var of w_final', var_w_final, iterations)
            writer.add_scalar(f'wb / Norm of b_final', norm_b_final, iterations)
            writer.add_scalar(f'wb / Var of b_final', var_b_final, iterations)

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
  
        
def continual_train():
    #same as Find_max_R(R), however, it will load the previous trained model "Model_tmp" and continue training on that R
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    plastic_string = '_'.join(metadata['Sequence of plastic matrix'])

    hidden_dims_string = '_'.join(str(x) for x in metadata['Hidden dimensions'])
    hetero_eta_string = '_'.join(str(x) for x in metadata['Hetero eta'])
    folder_name = '{0}-{1}-in{2}-hid{3}-eta-{4}-fr-{5}-var{6}'.format(
        metadata['network type'],
        plastic_string,
        metadata['input'],
        hidden_dims_string,
        hetero_eta_string,
        metadata['frames per scene'],
        metadata['variation'],
    )

    cwd = os.getcwd() + '/Result/' + network_name +'/' + folder_name
    # now we have a lot of 'R_n' folders, we need to find the max n
    if not os.path.exists(cwd):
        os.makedirs(cwd)
        filename = 'metadata.txt'
        filepath = os.path.join(cwd, filename)
        # Open the file in write mode and write the metadata
        with open(filepath, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

        R = 1
        HebbFF_CFD = NN.network_selector(input_dim = vec_len, hid_dim_list = hid_dim_list, out_dim = out_dim, hetero_list = hetero_list, plastic_list = plastic_list, network_name = network_name)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(HebbFF_CFD.parameters(),lr=lr_rate)
    
    else:
        R = 0
        for folder in os.listdir(cwd):
            if folder.startswith('R_'):
                R = max(R, int(folder[2:]))

        PATH = cwd + '/R_{}'.format(R) + '/Model_tmp'
        HebbFF_CFD = NN.network_selector(input_dim = vec_len, hid_dim_list = hid_dim_list, out_dim = out_dim, hetero_list = hetero_list, plastic_list = plastic_list, network_name = network_name)
        HebbFF_CFD.load_state_dict(torch.load(PATH))
        HebbFF_CFD.eval()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(HebbFF_CFD.parameters(),lr=lr_rate)

    while True:
        print('Current Training Interval Value: ',R)

        T = max(100, 20 * R)

        PATH = cwd + '/R_{}'.format(R)

        isExist = os.path.exists(PATH)
        
        if not isExist:
            os.makedirs(PATH)

        # training
        acc_train(HebbFF_CFD, PATH, acc, vec_len, R, T, criterion, optimizer,acc_num = 5, device = device)

        # saving the model
        torch.save(HebbFF_CFD.state_dict(), PATH + '/Model')
        
        # test accuracy
        
        
        test(HebbFF_CFD, 1, max (20, 5 * R), criterion, PATH, device = device)
        entire_output(HebbFF_CFD, criterion, PATH, R)
        
        # static matrix
        
        hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=48,scene_t = scene_t, device = device)
        weight_matrix(HebbFF_CFD, PATH)
        eta_matrix(HebbFF_CFD, PATH)

        R += 1


def get_R_values(logs_dir, network_class, network):
    R_values = [os.path.basename(x) for x in glob.glob(os.path.join(logs_dir, network_class, network, '*')) if os.path.isdir(x)]
    R_values.sort(key=lambda x: int(x.split('_')[-1]))
    return R_values
        
def generate_graphs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    plastic_string = '_'.join(metadata['Sequence of plastic matrix'])

    hidden_dims_string = '_'.join(str(x) for x in metadata['Hidden dimensions'])
    hetero_eta_string = '_'.join(str(x) for x in metadata['Hetero eta'])
    folder_name = '{0}-{1}-in{2}-hid{3}-eta-{4}-fr-{5}-var{6}'.format(
        metadata['network type'],
        plastic_string,
        metadata['input'],
        hidden_dims_string,
        hetero_eta_string,
        metadata['frames per scene'],
        metadata['variation'],
    )

    cwd = os.getcwd() + '/Result/' + network_name +'/' + folder_name
    # iterate through all R_n folders, and load model and generate graphs
    R = 0
    while True:
        R += 1
        print('Current generating graph Interval Value: ',R)
        # check if path exists:
        PATH = cwd + '/R_{}'.format(R) + '/Model'
        print('model Path: ', PATH)
        isExist = os.path.exists(PATH)
        if not isExist:
            print('Model not found, break')
            break
        HebbFF_CFD = NN.network_selector(input_dim = vec_len, hid_dim_list = hid_dim_list, out_dim = out_dim, hetero_list = hetero_list, plastic_list = plastic_list, network_name = network_name)
        HebbFF_CFD.load_state_dict(torch.load(PATH))
        criterion = nn.BCELoss()

        PATH = cwd + '/R_{}'.format(R)
        print('generated at Path: ', PATH)
        
        hidden_activity(HebbFF_CFD, R, criterion, PATH, T_step=96,scene_t = scene_t, device = device)
        weight_matrix(HebbFF_CFD, PATH)
        eta_matrix(HebbFF_CFD, PATH)
        

            

if __name__ == "__main__":
    #continual_train()
    generate_graphs()
