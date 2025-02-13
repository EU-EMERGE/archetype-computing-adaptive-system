import torch
import numpy as np
import matplotlib.pyplot as plt
from acds.archetypes.esn import DeepReservoir
from acds.archetypes.ron import DeepRandomizedOscillatorsNetwork
import argparse
# get wikipedia dataset used in Generating Text with     Recurrent Neural Networks

parser = argparse.ArgumentParser(description='Perturbation Experiment')
parser.add_argument('--RON', type=bool, default=False, help='Use Randomized Oscillators Network instead of Deep Reservoir')
parser.add_argument('--length', type=int, default=1000, help='Length of the sequence')
parser.add_argument('--perturbation_step', type=int, default=100, help='Step where the perturbation is introduced')
parser.add_argument('--input_size', type=int, default=10, help='Input size')
parser.add_argument('--tot_units', type=int, default=100, help='Total number of units')
parser.add_argument('--n_layers', type=int, default=10, help='Number of layers')
parser.add_argument('--gamma_range', type=float, default=0.1, help='Gamma range')
parser.add_argument('--epsilon_range', type=float, default=0.1, help='Epsilon range')
parser.add_argument('--gamma', type=float, default=1.0, help='Gamma')
parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon')

args = parser.parse_args()
RON = args.RON


def main():

    length = args.length
    perturbation_step = args.perturbation_step
    input_size = args.input_size
    tot_units = args.tot_units
    n_layers = args.n_layers
    
    if RON == False:
    
        model = DeepReservoir(input_size=input_size, tot_units=tot_units,
                            n_layers=n_layers, concat=True, spectral_radius=0.95, input_scaling=0.1, inter_scaling=0.1, leaky=1, 
                            connectivity_recurrent=int(tot_units/n_layers), 
                            connectivity_input=int(tot_units/n_layers),
                            connectivity_inter=int(tot_units/n_layers), all=True).to(dtype=torch.float64)
    else:
        gamma = (args.gamma - args.gamma_range / 2.0, args.gamma + args.gamma_range / 2.0)
        epsilon = (
            args.epsilon - args.epsilon_range / 2.0,
            args.epsilon + args.epsilon_range / 2.0,
        )
                
        model = DeepRandomizedOscillatorsNetwork(n_inp=input_size, total_units=tot_units,
                                            dt=0.02, gamma=gamma, epsilon=epsilon, 
                                            input_scaling=0.2, inter_scaling=0.2,
                                            connectivity_input=int(tot_units/n_layers), 
                                            connectivity_inter=int(tot_units/n_layers),
                                            rho=0.95, n_layers=n_layers).to(dtype=torch.float64)
                                            

    s1 = torch.from_numpy(np.random.randint(0, 10, length)).to(dtype=torch.float64)
    s2 = s1.clone() 
    # perturb s2 to still be in the same 0-9 range
    s2[perturbation_step] = (s2[perturbation_step] + 1) % 10
    # one hot encoding
    s1 = torch.nn.functional.one_hot(s1.to(torch.int64), num_classes=10).to(dtype=torch.float64)
    s2 = torch.nn.functional.one_hot(s2.to(torch.int64), num_classes=10).to(dtype=torch.float64)
    
    # 2. Collect states
    with torch.no_grad():
       if RON == False:
            # Deep Reservoir
            _, _, states_un1 = model(s1.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
            _, _, states_un2 = model(s2.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
       else:
           # Randomized Oscillators Network
            _, _, states_ron1 = model(s1.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
            _, _, states_ron2 = model(s2.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
        
    if RON == True:
        states_un1 = states_ron1
        states_un2 = states_ron2 
        
    # so we will with distances for layer 0 of len s1,s2 steps and so on
    distances = np.zeros((n_layers, length))
    
    for layer in range(n_layers):
        for t in range(length):
            distances[layer, t] = np.linalg.norm(states_un1[layer][0][t] - states_un2[layer][0][t])
            
    #normalize
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    # Plot distance
    plt.figure(figsize=(12, 6))
    for l in range(n_layers):
        plt.plot(distances[l], label=f'Layer {l}', alpha=0.3 + 0.03 * l, color = (0, 0, 1 - l/n_layers), linewidth=2)   
    plt.axvline(x=perturbation_step, color='r', linestyle='--', label='Error introduced at step 100')
    plt.xlabel('Time step')
    plt.ylabel('Euclidean distance between states')
    plt.title(f'Distance between Reservoir States of s1 and s2 ({n_layers} Layers) of "{model.__class__.__name__}"')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    plt.savefig('distance_reservoir_states.png')
    
if __name__ == "__main__":
    main()
