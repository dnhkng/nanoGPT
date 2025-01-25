#!/bin/bash

# List of activation functions to test
FUNCTIONS=(
    # "softmax"
    # "dynamic_hybrid_activation2"
    # "residual_hybrid_activation2"
    # "hybrid_relu_bounded_activation2"
    # "gated_hybrid_activation2"
    "enhanced_gradient_stable_relu"  #scale_factor
    # "adaptive_gradient_stable_relu"
    # "hybrid_gradient_stable_relu"     # alpha
    # "residual_gradient_stable_relu"   # scale factor
    # "scale_aware_gradient_stable_relu"  #temperature
    # "dynamic_gradient_stable_relu"
    # Add more functions as needed
)

# Loop through each function
for func in "${FUNCTIONS[@]}"
do
    echo "Starting training with activation function: $func"
    python train2.py $func
    
    # Optional: add a delay between runs
    sleep 5
done