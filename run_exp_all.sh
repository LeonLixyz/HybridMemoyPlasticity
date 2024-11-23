#!/bin/bash

# Define an array of commands
commands=(
    "--network_type 'Nonl_RO' --hidden_dims '[100,100]' --hetero_rates '[1]' --plastic_types \"['M']\" --scene_time 4"
    "--network_type 'Nonl_RO' --hidden_dims '[100,100]' --hetero_rates '[1]' --plastic_types \"['A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[100]' --hetero_rates '[1]' --plastic_types \"['M']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[100]' --hetero_rates '[1]' --plastic_types \"['A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[50,50]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[50,50]' --hetero_rates '[1,1]' --plastic_types \"['M','M']\" --scene_time 4"
    "--network_type 'Dyn_RO' --hidden_dims '[50,50]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
    "--network_type 'Dyn_RO' --hidden_dims '[50,50]' --hetero_rates '[1,1]' --plastic_types \"['M','M']\" --scene_time 4"
    "--network_type 'Stack' --hidden_dims '[50,50]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
    "--network_type 'Nonl_RO' --hidden_dims '[50,50]' --hetero_rates '[1]' --plastic_types \"['M']\" --scene_time 4"
    "--network_type 'Nonl_RO' --hidden_dims '[50,50]' --hetero_rates '[1]' --plastic_types \"['A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[50]' --hetero_rates '[1]' --plastic_types \"['M']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[50]' --hetero_rates '[1]' --plastic_types \"['A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[25,25]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
    "--network_type 'ML' --hidden_dims '[25,25]' --hetero_rates '[1,1]' --plastic_types \"['M','M']\" --scene_time 4"
    "--network_type 'Dyn_RO' --hidden_dims '[25,25]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
    "--network_type 'Dyn_RO' --hidden_dims '[25,25]' --hetero_rates '[1,1]' --plastic_types \"['M','M']\" --scene_time 4"
    "--network_type 'Stack' --hidden_dims '[25,25]' --hetero_rates '[1,1]' --plastic_types \"['M','A']\" --scene_time 4"
)

# Iterate over each command and execute it
for cmd in "${commands[@]}"; do
    echo "Running experiment with parameters: $cmd"
    python main.py --mode train $cmd
done