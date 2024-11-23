import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

def clear_plastic_matrix(model):
    """Clear plastic matrices in the model."""
    for i in range(len(model.plastic_matrix_list)):
        model.plastic_matrix_list[i] = torch.zeros_like(model.plastic_matrix_list[i])

def train_step(model, input_seq, target_seq, criterion, optimizer, device, store=False):
    """Matches the original training behavior."""
    model.train()
    optimizer.zero_grad()
    
    # Initialize model attributes for tracking
    model.accuracy = 0
    model.loss = 0
    model.output_seq = torch.zeros(len(target_seq) * model.scene_t)
    model.HL_list = [[] for _ in range(len(model.hid_dim_list))]
    
    # Process each scene and its frames
    T = len(input_seq)
    for t in range(T):
        scene = input_seq[t]
        for i in range(len(scene)):
            frame = scene[i].to(device)
            out = model(frame, Store=False)
            model.output_seq[model.scene_t * t + i] = out
    
    # Compute loss and accuracy
    model.loss = criterion(model.output_seq[model.scene_t-1::model.scene_t], target_seq)
    model.predicted = (model.output_seq[model.scene_t-1::model.scene_t].detach() >= 0.5).float()
    model.accuracy = (model.predicted == target_seq).sum() / T
    
    # Backward pass
    model.loss.backward()
    optimizer.step()
    
    # Clear plastic matrices
    clear_plastic_matrix(model)
    
    return model.accuracy, model.loss

class CFDTrainer:
    def __init__(self, model, optimizer, criterion, data_generator, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_generator = data_generator
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterations = 0
        
        # Add scene_t to model for easier access
        self.model.scene_t = config.scene_time
    
    def train_epoch(self, R, T, writer=None):
        """Train for one epoch with current R value."""
        input_seq, target_seq, total_target = self.data_generator.generate_movie(
            vec_len=self.config.vec_len,
            R=R,
            T=T,
            scene_t=self.config.scene_time,
            variation=self.config.VAR
        )
        
        # Convert sequences to tensors
        target_seq = torch.tensor(target_seq, dtype=torch.float32)
        
        accuracy, loss = train_step(
            model=self.model,
            input_seq=input_seq,
            target_seq=target_seq,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            store=False
        )
        
        if writer and self.iterations % 100 == 0:
            self._log_metrics(writer, accuracy, loss, target_seq)
            
        return accuracy, loss

    def curriculum_train(self, save_dir):
        """Implement curriculum training with increasing R."""
        R = 1
        while True:
            print(f'Current Training Interval Value: {R}')
            T = max(100, 20 * R)
            
            # Create directory for current R
            curr_path = os.path.join(save_dir, f'R_{R}')
            os.makedirs(curr_path, exist_ok=True)
            
            # Initialize tensorboard writer
            writer = SummaryWriter(os.path.join(curr_path, 'Training_Log'))
            
            # Train until accuracy threshold is met
            self.train_until_convergence(R, T, curr_path, writer)
            
            # Save model and generate visualizations
            self.save_and_visualize(curr_path, R)
            
            R += 1

    def train_until_convergence(self, R, T, save_path, writer):
        """Train until accuracy threshold is met or max iterations reached."""
        iterations = 0
        acc_flag = 0
        
        while acc_flag < self.config.acc_num and iterations < 1000000:
            accuracy, loss = self.train_epoch(R, T, writer)
            iterations += 1
            self.iterations += 1
            
            # Print training progress
            print(f'\rIteration {iterations}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}, Acc Flag = {acc_flag}', end='')
            
            if accuracy >= self.config.acc_threshold:
                acc_flag += 1
                print(f'\nReached threshold accuracy! Flag count: {acc_flag}/{self.config.acc_num}')
            else:
                acc_flag = 0
                
            if iterations % 100 == 0:
                print(f'\nSaving model checkpoint at iteration {iterations}')
                torch.save(self.model.state_dict(), os.path.join(save_path, 'Model_tmp'))
                
        print(f'\nTraining Summary: Total Number of iterations: {iterations}')
        print(f'Final Accuracy: {accuracy:.4f}, Final Loss: {loss:.4f}')

    def _log_metrics(self, writer, accuracy, loss, target_seq):
        """Log metrics to tensorboard."""
        writer.add_scalar('Performance/Training Loss', loss.item(), self.iterations)
        writer.add_scalar('Performance/Accuracy', accuracy.item(), self.iterations)

        # Log weight matrix norms and variances
        for idx, weight_matrix in enumerate(self.model.weight_matrix_list):
            norm_w = np.linalg.norm(torch.squeeze(weight_matrix.detach()))
            var_w = np.var(torch.squeeze(weight_matrix.detach()).numpy())
            writer.add_scalar(f'wb/Norm of w_{idx + 1}', norm_w, self.iterations)
            writer.add_scalar(f'wb/Var of w_{idx + 1}', var_w, self.iterations)

        # Log eta matrix metrics
        for idx, eta_matrix in enumerate(self.model.eta_matrix_list):
            if eta_matrix.numel() == 1:
                writer.add_scalar(f'Synapse/Eta_{idx + 1}', eta_matrix.item(), self.iterations)
            else:
                norm_eta = np.linalg.norm(torch.squeeze(eta_matrix.detach()))
                var_eta = np.var(torch.squeeze(eta_matrix.detach()).numpy())
                writer.add_scalar(f'Synapse/Norm of Eta_{idx + 1}', norm_eta, self.iterations)
                writer.add_scalar(f'Synapse/Var of Eta_{idx + 1}', var_eta, self.iterations)
        
        # Log lambda values
        for idx, lambda_val in enumerate(self.model.lambda_list):
            writer.add_scalar(f'Synapse/Lambda_{idx + 1}', lambda_val.item(), self.iterations)

        # Log bias matrix metrics
        for idx, bias_matrix in enumerate(self.model.bias_matrix_list):
            norm_b = np.linalg.norm(torch.squeeze(bias_matrix.detach()))
            var_b = np.var(torch.squeeze(bias_matrix.detach()).numpy())
            writer.add_scalar(f'wb/Norm of b_{idx + 1}', norm_b, self.iterations)
            writer.add_scalar(f'wb/Var of b_{idx + 1}', var_b, self.iterations)

        # Log final weight and bias metrics
        norm_w_final = np.linalg.norm(torch.squeeze(self.model.w_final.detach()))
        var_w_final = np.var(torch.squeeze(self.model.w_final.detach()).numpy())
        norm_b_final = np.linalg.norm(torch.squeeze(self.model.b_final.detach()))
        var_b_final = np.var(torch.squeeze(self.model.b_final.detach()).numpy())
        writer.add_scalar('wb/Norm of w_final', norm_w_final, self.iterations)
        writer.add_scalar('wb/Var of w_final', var_w_final, self.iterations)
        writer.add_scalar('wb/Norm of b_final', norm_b_final, self.iterations)
        writer.add_scalar('wb/Var of b_final', var_b_final, self.iterations)

        # Log false positive and false negative rates
        dif_seq = (self.model.predicted - target_seq).detach().numpy()
        actual_positive = np.count_nonzero(target_seq == 1)
        actual_negative = np.count_nonzero(target_seq == 0)
        false_positive = np.count_nonzero(dif_seq == 1)
        false_negative = np.count_nonzero(dif_seq == -1)
        
        if actual_negative > 0:
            writer.add_scalar('FPFN/false positive rate', false_positive / actual_negative, self.iterations)
        if actual_positive > 0:
            writer.add_scalar('FPFN/false negative rate', false_negative / actual_positive, self.iterations)

    def save_and_visualize(self, save_path, R):
        """Save model and generate visualization plots."""
        torch.save(self.model.state_dict(), os.path.join(save_path, 'Model'))