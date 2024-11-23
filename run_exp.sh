#!/bin/bash

# Default parameters
NETWORK_TYPE="Memo"
HIDDEN_DIMS="[100,100]"
HETERO_RATES="[1, 1]"
PLASTIC_TYPES="['M','A']"
SCENE_TIME=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --network_type) NETWORK_TYPE="$2"; shift ;;
        --hidden_dims) HIDDEN_DIMS="$2"; shift ;;
        --hetero_rates) HETERO_RATES="$2"; shift ;;
        --plastic_types) PLASTIC_TYPES="$2"; shift ;;
        --scene_time) SCENE_TIME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the experiment
echo "Running experiment with the following parameters:"
echo "Network Type: $NETWORK_TYPE"
echo "Hidden Dimensions: $HIDDEN_DIMS"
echo "Hetero Rates: $HETERO_RATES"
echo "Plastic Types: $PLASTIC_TYPES"
echo "Scene Time: $SCENE_TIME"

python main.py --mode train \
    --network_type "$NETWORK_TYPE" \
    --hidden_dims "$HIDDEN_DIMS" \
    --hetero_rates "$HETERO_RATES" \
    --plastic_types "$PLASTIC_TYPES" \
    --scene_time "$SCENE_TIME"