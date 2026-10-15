from functools import partial
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from acds.archetypes import UnicycleReservoir
from utils import get_mnist_data
import time

# Objective function for Optuna
def objective(trial, aligned_orientations=None, ang_input=None, ang_connections=None, n_units=None):
    # Suggest hyperparameters for Optuna to search
    aligned_orientations = trial.suggest_categorical("aligned_orientations", [True, False]) if aligned_orientations is None else aligned_orientations
    if not aligned_orientations:
        ang_input = trial.suggest_categorical("ang_input", [True, False]) if ang_input is None else ang_input
        ang_connections = trial.suggest_categorical("ang_connections", [True, False]) if ang_connections is None else ang_connections
    else:
        ang_input = False
        ang_connections = False

    n_units = n_units#trial.suggest_int('n_units', 10, 200, step=10)
    lin_stiff_min = trial.suggest_float('lin_stiff_min', 0.1, 1.0)
    lin_stiff_max = trial.suggest_float('lin_stiff_max', lin_stiff_min, 10.0)  # lin_stiff_max >= lin_stiff_min
    ang_stiff_min = trial.suggest_float('ang_stiff_min', 0.1, 1.0)
    ang_stiff_max = trial.suggest_float('ang_stiff_max', ang_stiff_min, 2.0)  # ang_stiff_max >= ang_stiff_min
    lin_damping_min = trial.suggest_float('lin_damping_min', 0.1, 1.0)
    lin_damping_max = trial.suggest_float('lin_damping_max', lin_damping_min, 5.0)
    ang_damping_min = trial.suggest_float('ang_damping_min', 0.1, 10.0)
    ang_damping_max = trial.suggest_float('ang_damping_max', ang_damping_min, 20.0)
    bs = 500
    dt = trial.suggest_float("dt", 0.0001, 0.01, step=0.0005)
    inp_bias = trial.suggest_float("inp_bias", -1,1)
    n_connections_anchor_fraction = trial.suggest_float("anchor_con_fraction", 0, 1.0, step=0.1)

    # Randomized lin_input_map
    lin_input_map = torch.zeros(1, n_units)
    num_non_zero = trial.suggest_int("non_zero_elements", 1, n_units)
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]  # Randomly select indices
    non_zero_values = trial.suggest_float("magnitude_min", -10.0, 0.0)
    magnitude_max = trial.suggest_float("magnitude_max", non_zero_values, 20)
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max- non_zero_values) + non_zero_values  # Random magnitudes

    # Randomized ang_input_map
    ang_input_map = torch.zeros(1, n_units)
    if ang_input:
        num_non_zero_ang = trial.suggest_int("non_zero_elements_ang", 1, n_units)
        non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]  # Randomly select indices
        non_zero_values_ang = trial.suggest_float("magnitude_min_ang", -10, 0.0)
        magnitude_max_ang = trial.suggest_float("magnitude_max_ang", non_zero_values_ang, 10)
        ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang- non_zero_values_ang) + non_zero_values_ang  # Random magnitudes

    n_connections_fraction = trial.suggest_float("n_connections_fraction", 0.2, 1.0, step=0.1)    
    n_connections = int(n_units*n_connections_fraction)
    washup = trial.suggest_int('washup_steps', 0, 4000, step=1000)
    n_connections_anchor = int(n_connections_anchor_fraction * n_units)
    n_steps_readout = 0#trial.suggest_int("steps_readout", 0, 100, step=10)
    if not ang_connections:
        n_connections_ang = 0
        n_connections_anchor_ang = 0
    else:
        n_connections_ang_fraction = trial.suggest_float("n_connections_ang_fraction", 0.2, 1.0, step=0.1)
        n_connections_ang = int(n_units*n_connections_ang_fraction)
        n_connections_anchor_fraction_ang = trial.suggest_float("anchor_con_fraction_ang", 0, 1.0, step=0.1)
        n_connections_anchor_ang = int(n_connections_anchor_fraction_ang * n_units)

    eq_dist_min = trial.suggest_float("eq_dist_min", 0.2,1.0)
    eq_dist_max = trial.suggest_float("eq_dist_max", eq_dist_min, 2.0)
    eq_dist_min_ang = trial.suggest_float("eq_dist_min_ang", -2*torch.pi, 0.0)
    eq_dist_max_ang = trial.suggest_float("eq_dist_max_ang", eq_dist_min_ang, 2*torch.pi)

    # Initialize the model with the suggested hyperparameters
    classes = [0,1,2,3,4,5,6,7,8,9]
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=dt, n_out=len(classes),
        lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
        eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, 
        eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
    n_connections=n_connections, inp_bias=inp_bias, lin_input_map=lin_input_map, 
    n_connections_anchor=n_connections_anchor, ang_input_map=ang_input_map,
    n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang, 
    n_past_steps_readout=n_steps_readout).to(device)

    bs_train, bs_test = bs, bs
    train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=classes, new_fraction=0.4, test_fraction=0.4)
    # n_epochs = trial.suggest_int("n_epochs", 1,30)
    model.set_init_states_random(bs_train)

    # Move initial states to GPU if available
    model.x_init = model.x_init.to(device)
    model.z_init = model.z_init.to(device)
    model.theta_init = model.theta_init.to(device)
    model.s_init = model.s_init.to(device)
    model.omega_init = model.omega_init.to(device)
    model.lin_input_map = model.lin_input_map.to(device)
    model.ang_input_map = model.ang_input_map.to(device)
    model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
    model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
    model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
    model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

    model.s_init[:,0] = 0
    model.omega_init[:,:] = 0
    if not aligned_orientations:
        model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (2*torch.pi - (-2*torch.pi)) + -2*torch.pi
    else:
        print("aligned orientations")
        model.theta_init[:,:] = torch.rand(1) * (2*torch.pi - (-2*torch.pi)) + -2*torch.pi
    x = model.x_init[0:1,:]
    z = model.z_init[0:1,:]
    theta = model.theta_init[0:1,:]
    s = model.s_init[0:1,:]
    omega = model.omega_init[0:1,:]
    states_list = []

    u_lin = torch.zeros((1, washup, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)

    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map

        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

        concatenated_states = torch.hstack((x, z, theta, s, omega))
        states_list.append(concatenated_states)
    
    # Debug: Check initial states for NaN
    print(f"Initial states - x: {torch.isnan(x).any()}, z: {torch.isnan(z).any()}, theta: {torch.isnan(theta).any()}, s: {torch.isnan(s).any()}, omega: {torch.isnan(omega).any()}")
    print(f"Initial state ranges - x: [{x.min():.3f}, {x.max():.3f}], s: [{s.min():.3f}, {s.max():.3f}]")
    
    model.set_init_states(bs_train, x,z,theta,s,omega)
    
    # Debug: Check model parameters for extreme values
    print(f"dt: {dt}, lin_stiff_max: {lin_stiff_max}, ang_stiff_max: {ang_stiff_max}")
    print(f"lin_damping_max: {lin_damping_max}, ang_damping_max: {ang_damping_max}")
    print(f"Input map range: [{lin_input_map.min():.3f}, {lin_input_map.max():.3f}]")
    
    # Logistic regression ESN training loop
    progress_bar = tqdm(train_loader)
    activations, ys = [], []
    for images, labels in progress_bar:
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0,2,1)
        images = images.to(device)
        labels = labels.to(device)
        # model should return mid_states as third output
        states_list, output, mid_states = model(images, images)
        
        # Debug: Check for NaN in model outputs
        if torch.isnan(mid_states).any():
            print(f"NaN detected in mid_states for batch {len(activations)}")
            print(f"Input range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}] (if not all NaN)")
            return 0.0  # Early termination
            
        activations.append(mid_states.detach().cpu())
        ys.append(labels.cpu())
    activations = torch.cat(activations, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    
    # Check for NaN values in activations
    import numpy as np
    if np.isnan(activations).any():
        print("NaN values detected in activations, ending trial")
        return 0.0  # Return low score to indicate failed trial
    
    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(activations)
    activations = scaler.transform(activations)
    classifier = LogisticRegression(max_iter=1000).fit(activations, ys)

    # Validation and test scores using ESN
    validation_accuracy = test_esn(valid_loader, model, classifier, scaler, bs_test)
    # test_accuracy = test_esn(test_loader, model, classifier, scaler, bs_test)
    print(f"Validation score (ESN): {validation_accuracy}")
    # print(f"Test score (ESN): {test_accuracy}")
    # Optuna will aim to maximize validation accuracy
    return validation_accuracy


# ESN evaluation function for validation and testing accuracy
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler, bs_test):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.reshape(bs_test, 1, 784)
        images = images.permute(0, 2, 1)
        images = images.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(images, images)
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)

# Define the search space and start the optimization
if __name__ == '__main__':
    # Define the GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = "cpu"
    # Load your data (train_loader, valid_loader, and test_loader)

    n_units_list = [4,8,12,16,20]


    database_name = "unicycle_mnist_increasing_n_smaller"
    for n_units in n_units_list:
        print(f"Starting optimization for n_units={n_units}")
        study_name = f"{n_units}_units"
        storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
        study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
    # study_name = "not_aligned_w_input_w_connections_actual_100_units_last_readout_no_tanh"
    # storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
    # study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
    # for trial in study.trials:
    #     if trial.datetime_complete and trial.datetime_start:
    #         duration = trial.datetime_complete - trial.datetime_start
    #         print(f"Trial {trial.number} took {duration.total_seconds()} seconds")
        study.optimize(partial(objective, aligned_orientations=False, ang_input=True, ang_connections=True, n_units=n_units), n_trials=100)

    # Get the best hyperparameters
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")

    # database_name = "unicycle_mnist_increasing_n"
    # for n_units in [60]:
    #     print(f"Starting optimization for n_units={n_units}")
    #     study_name = f"{n_units}_units_v2"
    #     storage_name = "sqlite:///optuna_databases/{}.db".format(database_name)
    #     study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
    #     study.optimize(partial(objective, aligned_orientations=False, ang_input=True, ang_connections=True, n_units=n_units), n_trials=300)

    # # Test the model with the best hyperparameters on the test set
    # test_accuracy = test(test_loader)
    # print(f"Test accuracy with the best model: {test_accuracy}")