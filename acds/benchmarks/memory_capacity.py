import numpy as np
import matplotlib.pyplot as plt
import torch 
import os
from torch.utils.data import DataLoader, TensorDataset


def generate_memory_capacity_dataset(delay, signal_length=6000, train_length=5000, test_length=1000):
    """
    Generates a dataset for the memory capacity task of a recurrent network.
    
    Parameters:
    - signal_length: Total length of the input signal.
    - train_length: Length of the training set.
    - test_length: Length of the test set.
    - delay: Current delay to consider for the task.
    
    Returns:
    - u: Input signal.
    - y: Target signal with delays.
    """
    # Generate input signal
    u = np.random.uniform(low= -0.8, high= 0.8, size = (signal_length + delay, 1), seed=42)
    
    u_input = u[delay:]
    u_target = u[:-delay]
 
    u_train = u_input[:train_length]
    y_train = u_target[:train_length]
    
    u_test = u_input[train_length:]
    y_test = u_target[train_length:]
    
    return (u_train, y_train), (u_test, y_test)
    

def get_memory_capacity(delay, train_ratio: float = 0.8, test_size: int = 1000):
    """
    Returns the memory capacity dataset as torch tensors.
    
    In this following format:
    - u_train: Training input signal.
    - y_train: Training target signal.
    - u_test: Test input signal.
    - y_test: Test target signal.

    - apply washout to the training set
    
    Returns: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """
    
    (u_train, y_train), (u_test, y_test) = generate_memory_capacity_dataset(delay)
    
    assert len(u_train) == len(y_train), "Input and target signals must have the same length."
    
    test_start_idx = len(u_train) - test_size
    u_test, y_test = u_train[test_start_idx:], y_train[test_start_idx:]
    
    u_train_valid, y_train_valid = u_train[:test_start_idx], y_train[:test_start_idx]
    
    train_size = int(train_ratio * len(u_train_valid))
    u_train, y_train = u_train_valid[:train_size], y_train_valid[:train_size]
    u_val, y_val = u_train_valid[train_size:], y_train_valid[train_size:]
   
    # as numpy arrays
    return (torch.from_numpy(u_train).float(), torch.from_numpy(y_train).float()), (torch.from_numpy(u_val).float(), torch.from_numpy(y_val).float()), (torch.from_numpy(u_test).float(), torch.from_numpy(y_test).float())

if __name__ == "__main__":
    
    debug = True
    
    if debug:
        
        (u_train, y_train), (u_val, y_val), (u_test, y_test) = get_memory_capacity(1, train_ratio=0.8, test_size=1000)
        # make them into numpy arrays
        u_train, y_train = u_train.numpy(), y_train.numpy()
        u_val, y_val = u_val.numpy(), y_val.numpy()
        u_test, y_test = u_test.numpy(), y_test.numpy()
        
        
        # plot the dataset
        plt.figure(figsize=(12, 6))
        # original signal
        y = np.concatenate([y_train, y_val, y_test])
        plt.plot(y, label="Target signal", alpha=0.7)
        plt.plot(range(len(y_train), len(y_train) + len(y_val)), y_val, label="Validation signal", alpha=0.7)
        plt.plot(range(len(y_train) + len(y_val), len(y)), y_test, label="Test signal", alpha=0.7)
        plt.legend()
        
        plt.savefig("memory_capacity_dataset.png")
        
        # visualize the delayed signal against the input signal
        plt.figure(figsize=(12, 6))
        plt.plot(u_train, label="Input signal", alpha=0.7)
        plt.plot(y_train, label="Target signal", alpha=0.7)
        plt.legend()
        
        plt.savefig("memory_capacity_input_target.png")
        
    