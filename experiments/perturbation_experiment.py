import torch
import numpy as np
import matplotlib.pyplot as plt
from acds.archetypes.esn import DeepReservoir
from acds.archetypes.ron import DeepRandomizedOscillatorsNetwork

# get wikipedia dataset used in Generating Text with     Recurrent Neural Networks


#from acds.archetypes.ron import RandomizedOscillatorsNetwork, RandomizedOscillatorsNetworkLayer

def main():
    length = 1000
    perturbation_step = 100
    input_size = 10
    tot_units = 100
    n_layers = 10
    
    #s1, s2 = generate_sequences_artificial_dataset(length, perturbation_step)

    model = DeepReservoir(input_size=input_size, tot_units=tot_units,
                          n_layers=n_layers, concat=True, spectral_radius=0.95, input_scaling=0.1, inter_scaling=0.1, leaky=1, 
                          connectivity_recurrent=int(tot_units/n_layers), 
                          connectivity_input=int(tot_units/n_layers),
                          connectivity_inter=int(tot_units/n_layers), all=True).to(dtype=torch.float64)
    
    ron = DeepRandomizedOscillatorsNetwork(n_inp=input_size, n_hid=tot_units, dt=0.1, gamma=0.1, epsilon=0.1) 
    
    ron.eval()                      
    model.eval()

    s1 = torch.from_numpy(np.random.randint(0, 10, length)).to(dtype=torch.float64)
    print(s1.dtype)
    s2 = s1.clone() 
    # perturb s2 to still be in the same 0-9 range
    s2[perturbation_step] = (s2[perturbation_step] + 1) % 10
    # one hot encoding
    s1 = torch.nn.functional.one_hot(s1.to(torch.int64), num_classes=10).to(dtype=torch.float64)
    s2 = torch.nn.functional.one_hot(s2.to(torch.int64), num_classes=10).to(dtype=torch.float64)
    
    # 2. Collect states
    with torch.no_grad():
       
        # Deep Reservoir
        # the input is len 10
        _, _, states_un1 = model(s1.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
        _, _, states_un2 = model(s2.to('cpu').reshape(1, length, 10).to(dtype=torch.double))
        # Randomized Oscillators Network
        """ 
        states_s1 = ron(s1.to('cpu').reshape(1,-1,1).float())[0].cpu().numpy()
        states_s2 = ron(s2.to('cpu').reshape(1,-1,1).float())[0].cpu().numpy()
        states_s1 = states_s1.reshape(-1, tot_units)
        states_s2 = states_s2.reshape(-1, tot_units) 
        """
   
    # so we will with distances for layer 0 of len s1,s2 steps and so on
    distances = np.zeros((n_layers, length))
    
    for layer in range(n_layers):
        for t in range(length):
            print(f"Layer {layer} - Step {t}")
            print("Comparing states:", states_un1[layer][0][t], states_un2[layer][0][t])
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
    plt.title(f'Distance between Reservoir States of s1 and s2 ({n_layers} Layers)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    plt.savefig('distance_reservoir_states.png')
    
if __name__ == "__main__":
    main()
