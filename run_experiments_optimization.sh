#!/bin/sh

# Define the array of rho_safe and rho_danger values
rho_safe_values=(0.0 0.95)
rho_danger_values=(0.97 0.9 0.0)
# models=("no_curtailment" "full" "no_storage" "no_redispatch" "custom")
models=("custom")

# Iterate over the rho_safe and rho_danger values
for rs in "${rho_safe_values[@]}"; do
    for rd in "${rho_danger_values[@]}"; do
        # Only proceed if rho_safe < rho_danger
        if (( $(echo "$rs <= $rd" | bc -l) )); then
            # Iterate over the models
            for model in "${models[@]}"; do
                # Call the script and pipe the output
                python src/python/agents/test_educ14_storage.py --rho_safe $rs --rho_danger $rd --model $model
            done
        fi
    done
done
