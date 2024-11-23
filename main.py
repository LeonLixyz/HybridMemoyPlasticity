import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from trainer import CFDTrainer
from plasticity import PlasticityVisualizer
import create_network as NN
import dataloader as DL
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="MetaLearn CFD: Continual Familiarity Detection")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'visualize'],
                        help='Mode to run the script: train or visualize')
    parser.add_argument('--network_type', type=str, required=True,
                        help='Type of network architecture (e.g., Nonl_RO, ML, Dyn_RO, Stack)')
    parser.add_argument('--hidden_dims', type=str, required=True,
                        help='Hidden layer dimensions as a string list (e.g., "[100,100]")')
    parser.add_argument('--hetero_rates', type=str, required=True,
                        help='Learning rates for each layer as a string list (e.g., "[1]")')
    parser.add_argument('--plastic_types', type=str, required=True,
                        help='Type of plasticity for each layer as a string list (e.g., "[M]")')
    parser.add_argument('--scene_time', type=int, default=4,
                        help='Number of frames per scene')
    parser.add_argument('--vec_len', type=int, default=25,
                        help='Input vector length')
    parser.add_argument('--lr_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--VAR', type=float, default=0.1,
                        help='Variation rate for data generation')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='Output dimension of the network')
    parser.add_argument('--acc_threshold', type=float, default=0.99,
                        help='Accuracy threshold for considering convergence')
    parser.add_argument('--acc_num', type=int, default=5,
                        help='Number of consecutive successful accuracy checks to consider convergence')
    return parser.parse_args()

def setup_experiment_folder(args):
    """Create experiment folder and save metadata."""
    plastic_list = ast.literal_eval(args.plastic_types)
    hidden_dims_list = ast.literal_eval(args.hidden_dims)
    hetero_list = ast.literal_eval(args.hetero_rates)
    
    plastic_string = '_'.join(plastic_list)
    hidden_dims_string = '_'.join(str(x) for x in hidden_dims_list)
    hetero_eta_string = '_'.join(str(x) for x in hetero_list)
    
    folder_name = (f'{args.network_type}-{plastic_string}-'
                  f'in{args.vec_len}-hid{hidden_dims_string}-'
                  f'eta-{hetero_eta_string}-fr-{args.scene_time}-'
                  f'var{args.VAR}')
    
    save_dir = os.path.join(os.getcwd(), 'Result', args.network_type, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
            
    return save_dir

def initialize_model(args):
    """Initialize model and training components."""
    model = NN.network_selector(
        input_dim=args.vec_len,
        hid_dim_list=eval(args.hidden_dims),
        out_dim=args.out_dim,
        hetero_list=eval(args.hetero_rates),
        plastic_list=eval(args.plastic_types),
        network_name=args.network_type
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)
    
    return model, criterion, optimizer

def train_model(args):
    """Main training function."""
    save_dir = setup_experiment_folder(args)
    
    # Initialize model and training components
    model, criterion, optimizer = initialize_model(args)
    
    # Initialize trainer
    trainer = CFDTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        data_generator=DL,
        config=args
    )
    
    # Start curriculum training
    trainer.curriculum_train(save_dir)

def generate_visualizations(args):
    """Generate visualizations for trained models."""
    save_dir = setup_experiment_folder(args)
    
    R = 0
    while True:
        R += 1
        model_path = os.path.join(save_dir, f'R_{R}', 'Model')
        
        if not os.path.exists(model_path):
            print(f'No more models found after R={R-1}')
            break
            
        print(f'Generating visualizations for R={R}')
        
        # Load model
        model = initialize_model(args)[0]
        model.load_state_dict(torch.load(model_path))
        
        # Create visualizations
        vis_path = os.path.join(save_dir, f'R_{R}')
        visualizer = PlasticityVisualizer(vis_path)
        visualizer.plot_weight_matrices(model)
        visualizer.plot_eta_matrices(model)
        
        # Generate hidden activity plots
        T = max(100, 20 * R)
        input_seq, target_seq, _ = DL.generate_movie(
            vec_len=args.vec_len,
            R=R,
            T=T,
            scene_t=args.scene_time,
            variation=args.VAR
        )
        
        if hasattr(model, 'W_list'):
            HL_list, plastic_list, W_list = model.get_hidden_states(input_seq, target_seq, T)
        else:
            HL_list, plastic_list = model.get_hidden_states(input_seq, target_seq, T)
            W_list = None
            
        visualizer.plot_hidden_activity(
            model=model,
            r=R,
            T_step=96,
            scene_t=args.scene_time,
            HL_list=HL_list,
            plastic_list=plastic_list,
            W_list=W_list
        )

if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'visualize':
        generate_visualizations(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
