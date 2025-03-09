import os
import sys

# Get user input for the number of epochs (only once)
nEpochs = input("Enter the number of epochs: ")

# List of different noise levels
noise_vals = [0.10]

# Run experiments for each noise level with the given number of epochs
for noise in noise_vals:
    print(f"Running experiment with relative noise = {noise}, nEpochs = {nEpochs}")
    os.system(f"python main_demo_my1.py {noise} {nEpochs}")


