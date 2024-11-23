import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

class PlasticityVisualizer:
    def __init__(self, save_path):
        self.save_path = save_path
        
    @staticmethod
    def graph_clip(matrix):
        """Clip matrix values to within 2 standard deviations of the mean."""
        mean = np.mean(matrix)
        std_dev = np.std(matrix)
        lower_bound = mean - 2 * std_dev
        upper_bound = mean + 2 * std_dev
        return np.clip(matrix, lower_bound, upper_bound)

    def plot_weight_matrices(self, model):
        """Plot static weight matrices of the model."""
        n_layers = len(model.weight_matrix_list)
        fig, axs = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8), dpi=600)

        for idx, weight_matrix in enumerate(model.weight_matrix_list):
            weight_matrix_np = weight_matrix.detach().numpy()
            weight_matrix_np = self.graph_clip(weight_matrix_np)
            
            if n_layers == 1:
                sns.heatmap(weight_matrix_np, linewidth=0.3, center=0.00, 
                           cmap='vlag', ax=axs, cbar_kws={'shrink': 0.5}, square=True)
                axs.set_title('Layer Weight Matrix')
            else:    
                sns.heatmap(weight_matrix_np, linewidth=0.3, center=0.00, 
                           cmap='vlag', ax=axs[idx], cbar_kws={'shrink': 0.5}, square=True)
                axs[idx].set_title(f'Layer {idx + 1} Weight Matrix')

        fig.tight_layout()
        fig.savefig(f'{self.save_path}/Weight_Matrices', dpi=600, facecolor='w', transparent=True)
        plt.close(fig)

        # Plot final weight matrix
        w_final = model.w_final.detach().numpy()
        fig_w_final, axs_w_final = plt.subplots(1, 1, figsize=(8, 8), dpi=600)
        sns.heatmap(w_final, linewidth=0.3, center=0.00, cmap='vlag', 
                   ax=axs_w_final, cbar_kws={'shrink': 0.5}, square=True)
        axs_w_final.set_title('Final Weight Matrix')
        fig_w_final.tight_layout()
        fig_w_final.savefig(f'{self.save_path}/Final_Weight_Matrix', dpi=600, facecolor='w', transparent=True)
        plt.close(fig_w_final)

    def plot_eta_matrices(self, model):
        """Plot learning rate matrices of the model."""
        n_layers = len(model.eta_matrix_list)
        fig, axs = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8), dpi=600)

        for idx, eta_matrix in enumerate(model.eta_matrix_list):
            if eta_matrix.numel() > 1:
                eta_matrix_np = eta_matrix.detach().numpy()
                eta_matrix_np = self.graph_clip(eta_matrix_np)
                
                if n_layers == 1:
                    sns.heatmap(eta_matrix_np, linewidth=0.3, center=0.00, 
                               cmap='vlag', ax=axs, cbar_kws={'shrink': 0.5}, square=True)
                    axs.set_title('Eta Matrix')
                else:
                    sns.heatmap(eta_matrix_np, linewidth=0.3, center=0.00, 
                               cmap='vlag', ax=axs[idx], cbar_kws={'shrink': 0.5}, square=True)
                    axs[idx].set_title(f'Layer {idx + 1} eta Matrix')

        fig.tight_layout()
        fig.savefig(f'{self.save_path}/Eta_Matrices', dpi=600, facecolor='w', transparent=True)
        plt.close(fig)

    def plot_hidden_activity(self, model, r, T_step, scene_t, HL_list, plastic_list, W_list=None):
        """Plot hidden layer activities and plastic matrices."""
        # Plot weight matrices if available
        if W_list is not None:
            self._plot_weight_time_series(W_list, r, T_step, scene_t)
            
        # Plot hidden layer activities
        self._plot_hidden_layers(model, HL_list, r, T_step, scene_t)
        
        # Plot plastic matrices
        self._plot_plastic_matrices(plastic_list, r, T_step, scene_t)

    def _plot_weight_time_series(self, W_list, r, T_step, scene_t):
        """Helper method to plot weight matrices over time."""
        n_w_layers = len(W_list)
        for t in range(scene_t * r, scene_t * r + T_step):
            fig_w, axs_w = plt.subplots(1, n_w_layers, figsize=(8 * n_w_layers, 8), dpi=600)
            for w_idx, w in enumerate(W_list):
                w = [i.detach().numpy() for i in w]
                w = self.graph_clip(w)
                if n_w_layers == 1:
                    sns.heatmap(w[t], linewidth=0.3, center=0.00, cmap='vlag', 
                              ax=axs_w, cbar_kws={'shrink': 0.5}, square=True)
                    axs_w.set_title(f'Actual Weight Matrix at time {t}')
                else:
                    sns.heatmap(w[t], linewidth=0.3, center=0.00, cmap='vlag', 
                              ax=axs_w[w_idx], cbar_kws={'shrink': 0.5}, square=True)
                    axs_w[w_idx].set_title(f'Actual Weight Matrix {w_idx + 1} at time {t}')

            fig_w.tight_layout()
            weight_path = f'{self.save_path}/Actual_Weight'
            os.makedirs(weight_path, exist_ok=True)
            fig_w.savefig(f'{weight_path}/Actual_Weight_combined_{t}', dpi=600, facecolor='w', transparent=True)
            plt.close(fig_w)