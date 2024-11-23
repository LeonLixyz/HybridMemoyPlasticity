# Hybrid Memory Plasticity and Advanced Readout Architectures for Continual Scene Detection

This repository implements novel memory plasticity mechanisms that combine multiplicative and additive plasticities and
advanced readout architectures for continual scene detection tasks. Our approach achieves state-of-the-art results compared to
other biological plausible memory models. We also provide scripts for analyzing the plasticity and hidden activities of the network.

![Network Architecture Overview](./images/Neuro.png)

## Overview

The project introduces:
- Hybrid memory plasticity combining multiplicative and additive mechanisms
- Advanced readout architectures for effective memory retrieval
- Curriculum-based meta-learning strategy
- Comprehensive network interpretability analysis

## Network Types

- **Nonl_RO**: Nonlinear Readout architecture
- **Memo**: Memory Readout architecture
- **Dyn_RO**: Dynamic Readout architecture
- **Stack**: Stacked Plasticity architecture

## Plasticity Types

- **M**: Multiplicative plasticity
- **A**: Additive plasticity
- **M,A**: Multiplayer plasticity, with M first layer and A second layer

The scripts takes in any combination of plasticity types. For example, `[M,M,A]` will use multiplicative plasticity for the first two layers and additive plasticity for the last layer.

## Project Structure

```
.
├── Memory_Networks/          # Directory for network architectures
│   ├── Dyn_RO.py             # Dynamic Readout architecture
│   ├── Memo.py               # Memory Readout architecture
│   ├── Nonl_RO.py            # Nonlinear Readout architecture
│   └── Stack.py              # Stacked Plasticity architecture
├── data/                     # Directory for data-related files
│   ├── dataloader.py         # Data loading utilities
│   └── (other data files)    # Any additional data files or scripts
├── scripts/                  # Directory for scripts
│   ├── run_exp.sh            # Script to run a single experiment
│   ├── run_exp_all.sh        # Script to run all experiments
│   └── (other scripts)       # Any additional scripts
├── utils/                    # Directory for utility modules
│   ├── create_network.py     # Network creation utilities
│   ├── plasticity.py         # Plasticity visualization tools
│   └── trainer.py            # Training utilities
├── results/                  # Directory for storing results
│   └── (experiment results)  # Subdirectories for each experiment's results
├── README.md                 # Project documentation
└── main.py                   # Main execution script
```

## Parameters

- `--mode`: Mode to run the script. Options are `train` or `visualize`.
- `--network_type`: Type of network architecture (e.g., Nonl_RO, Memo, Dyn_RO, Stack).
- `--hidden_dims`: Hidden layer dimensions as a string list (e.g., "[100,100]").
- `--hetero_rates`: Learning rates for each layer as a string list (e.g., "[1]"). Note that this needs to be within the
  same shape as `--plastic_types`, as we will have different learning rates for each plasticity matrix. These rates determine the configuration of the plasticity matrices:
  - `1`: Fully heterogenous learning rates.
  - `0.5`: Column-wise learning rates.
  - `0`: Homogenous learning rates.
- `--plastic_types`: Type of plasticity for each layer as a string list (e.g., "['M']" for Multiplicative, "['A']" for Additive).
- `--scene_time`: Number of frames per scene (default: 4).
- `--vec_len`: Input vector length (default: 25).
- `--lr_rate`: Learning rate for the optimizer (default: 0.001).
- `--VAR`: Variation rate for data generation (default: 0.1).
- `--out_dim`: Output dimension of the network (default: 1).
- `--acc_threshold`: Accuracy threshold for considering convergence (default: 0.99).
- `--acc_num`: Number of consecutive successful accuracy checks to consider convergence (default: 10).

## Running Experiments

### Creating the Environment

```bash
conda create -n neuro python=3.9
conda activate neuro
pip install -r requirements.txt
```

### Training

To run a single experiment, use the following command:

```
python main.py --mode train --network_type <network_type> --hidden_dims <hidden_dims> --hetero_rates <hetero_rates> --plastic_types <plastic_types> --scene_time <scene_time>
```

or use the bash script:

```
./scripts/run_exp.sh --network_type <network_type> --hidden_dims <hidden_dims> --hetero_rates <hetero_rates> --plastic_types <plastic_types> --scene_time <scene_time>
```

**Example:**

```
python main.py --mode train --network_type Memo --hidden_dims "[100,100]" --hetero_rates "[1,1]" --plastic_types "['M','A']" --scene_time 4
```

```
./scripts/run_exp.sh --network_type Memo --hidden_dims "[100,100]" --hetero_rates "[1,1]" --plastic_types "['M','A']" --scene_time 4
```

#### Visualization

To generate visualizations for a trained model, use the following command:

```
python main.py --mode visualize --network_type <network_type> --hidden_dims <hidden_dims> --hetero_rates <hetero_rates> --plastic_types <plastic_types> --scene_time <scene_time>
```

**Example:**

```
python main.py --mode visualize --network_type Memo --hidden_dims "[100,100]" --hetero_rates "[1,1]" --plastic_types "['M','A']" --scene_time 4
```


### Batch Experiments
To run all experiments in our report, execute:

```
./scripts/run_exp_all.sh
```

This script will run a series of experiments with various configurations, such as different network types, hidden layer dimensions, and plasticity types.


## Results Directory Structure

After running experiments, results are stored in the `results/` directory with the following structure:

```
results/
└── <network_type>/
    └── <experiment_config>/
        ├── metadata.txt
        └── R_<interval>/
            ├── Model
            ├── Training_Log/
            ├── Weight_Matrices.png
            ├── Eta_Matrices.png
            └── Hidden_Activity.png
```

Each experiment directory contains training logs, model checkpoints, and visualization plots for analysis.

