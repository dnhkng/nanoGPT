#!/bin/bash

# List of activation functions to test
FUNCTIONS=(
    "dynamic_hybrid"
    "gated_hybrid"
    "residual_hybrid"
    "adaptive_gating_hybrid"
    "scale_invariant_hybrid"
    "enhanced_gradient_stable_relu"
    "adaptive_gradient_stable_relu"
    "hybrid_gradient_stable_relu"
    "residual_gradient_stable_relu"
    "scale_aware_gradient_stable_relu"
    "dynamic_gradient_stable_relu"
    # Add more functions as needed
)

# Loop through each function
for func in "${FUNCTIONS[@]}"
do
    echo "Starting training with activation function: $func"
    python train.py $func
    
    # Optional: add a delay between runs
    sleep 5
done