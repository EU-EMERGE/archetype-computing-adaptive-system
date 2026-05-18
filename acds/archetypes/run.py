from torch import nn
import torch
import numpy as np


class UnicycleNetwork(nn.Module):
    def __init__(self, n_inp, n_units, dt, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0, eq_dist_min_ang=0.0,
                 eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=None, n_connections_anchor=2,
                 n_connections_ang=None, n_connections_anchor_ang=2):
        super().__init__()
        self.n_units = n_units
        self.dt = dt
        self.lin_damping = torch.rand(1,n_units, requires_grad=False) * (lin_damping_max - lin_damping_min) + lin_damping_min
        self.ang_damping = torch.rand(1,n_units, requires_grad=False) * (ang_damping_max - ang_damping_min) + ang_damping_min
        self.mass_vector = torch.ones(1, n_units, requires_grad=False)
        self.mass_vector[0,0] = 0.
        self.j_vector = torch.ones(1, n_units, requires_grad=False)
        self.j_vector[0,0] = 0.

        # Create coupling matrices with their equilibrium distances in one pass
        stiff_matrix, eq_dist_matrix = self._create_sparse_coupling_with_eq_distances(
            n_units, n_connections, n_connections_anchor,
            lin_stiff_min, lin_stiff_max, eq_dist_min, eq_dist_max
        )
        self.stiffness_coupling_matrix = stiff_matrix
        self.eq_distances_matrix = nn.Parameter(eq_dist_matrix.reshape(n_units, n_units, 1), requires_grad=False)
        
        ang_matrix, eq_ang_matrix = self._create_sparse_coupling_with_eq_distances(
            n_units, n_connections_ang, n_connections_anchor_ang,
            ang_stiff_min, ang_stiff_max, eq_dist_min_ang, eq_dist_max_ang, 
            antisymmetric=True
        )
        self.dist_ang_coupling = ang_matrix
        self.eq_distances_mat_ang = nn.Parameter(eq_ang_matrix, requires_grad=False)
    
    def _create_sparse_coupling_with_eq_distances(self, n_units, n_connections, n_connections_anchor, 
                                                   stiff_min, stiff_max, eq_min, eq_max, antisymmetric=False):
        """
        Create sparse coupling matrix and equilibrium distance matrix in a single pass.
        
        Args:
            antisymmetric: If True, equilibrium distances are antisymmetric (negated for j,i)
        
        Returns:
            coupling_matrix: nn.Parameter sparse matrix
            eq_distances: Tensor of equilibrium distances
        """
        coupling = torch.zeros((n_units, n_units))
        eq_dist = torch.zeros((n_units, n_units))
        
        for i in range(n_units):
            for j in range(i + 1, n_units):
                max_conn = n_connections_anchor if i == 0 else n_connections
                if np.abs(i - j - 1) < max_conn:
                    # Generate both coupling stiffness and equilibrium distance
                    stiff = torch.rand(1).item() * (stiff_max - stiff_min) + stiff_min
                    eq = torch.rand(1).item() * (eq_max - eq_min) + eq_min
                    
                    coupling[i, j] = stiff
                    coupling[j, i] = stiff
                    
                    # You need antisymmetric equilibrium distances for angular coupling, but not for linear coupling
                    if antisymmetric:
                        eq_dist[i, j] = eq
                        eq_dist[j, i] = -eq
                    else:
                        eq_dist[i, j] = eq
                        eq_dist[j, i] = eq
        
        return nn.Parameter(coupling, requires_grad=False), eq_dist

        
    def forward(self, u_lin, u_ang, x, z, theta, s, omega):
        bs = u_lin.shape[0]
        linear_inp_forces = u_lin
        coords_2d = torch.stack((x, z), dim=-1)  # (b, n_units, 2)  # Stack x and z into a 3D tensor
        theta_unit_vectors = self.angle_to_unit_vector(theta)  # (b, n_units, 2)
        distance_vectors = self.pairwise_differences(coords_2d)  # (b, n_units, n_units, 2)

        # Compute the distance magnitudes batch-wise (torch.norm instead of np.linalg.norm)
        distance_magnitudes = torch.norm(distance_vectors, dim=-1, keepdim=True)  # (b, n_units, n_units, 1)

        # Normalize the distance vectors, avoid division by zero
        distance_vectors_normalized = torch.nan_to_num(distance_vectors / distance_magnitudes)  # (b, n_units, n_units, 2)

        # Forces computation, keeps shape(b, n_units, n_units, 2), i.e. forces per pair as 2D vectors
        forces_before_projection = self.stiffness_coupling_matrix[None, :, :, None] * (self.eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized  # (b, n_units, n_units, 2)

        # Project forces along the theta direction using einsum (b, n_units, n_units, 2) -> (b, n_units)
        projected_forces = torch.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)  # (b, n_units)

        v_dot = (linear_inp_forces + projected_forces - (s * self.lin_damping)) * self.mass_vector

        s = s + v_dot*self.dt

        inp_term_theta = u_ang
        # Expand theta for pairwise differences
        theta_expanded_1 = theta[:, :, None]   # shape (b, n_units, 1)
        theta_expanded_2 = theta[:, None, :]   # shape (b, 1, n_units)
        ang_distances = theta_expanded_1 - theta_expanded_2
        coupling_term_ang = torch.sum(self.dist_ang_coupling[None, :, :] * (self.eq_distances_mat_ang.repeat(bs,1,1)-ang_distances), dim=2, keepdim=False)  # shape (b, n_units, 1)
        omega_dot = ((inp_term_theta + coupling_term_ang) - omega * self.ang_damping) * self.j_vector
        omega = omega + omega_dot*self.dt

        theta = theta + self.dt*omega
        x = x + torch.cos(theta) * s * self.dt
        z = z + torch.sin(theta) * s * self.dt

        return x, z, theta, s, omega
    
    def pairwise_differences(self, arr):
        """
        Computes all pairwise differences between vectors in the given 2D array.
        
        Parameters:
        arr (numpy.ndarray): A 2D array of shape (n, m), where n is the number of vectors
                            and m is the dimension of each vector.
                            
        Returns:
        numpy.ndarray: A 3D array of shape (n, n, m) where the element at [i, j, :] is the 
                    difference between the i-th and j-th vectors.
        """
        b, n, m = arr.shape
        # Expand dimensions to broadcast the subtraction over the pairs within each batch
        expanded_arr1 = arr.unsqueeze(2)  # (b, n_units, 1, 2)
        expanded_arr2 = arr.unsqueeze(1)  # (b, 1, n_units, 2)
        
        # Calculate pairwise differences
        differences = expanded_arr1 - expanded_arr2  # (b, n_units, n_units, 2)
        
        return differences
    
    def angle_to_unit_vector(self, angle):
        cos_angle = torch.cos(angle)  # (b, n_units)
        sin_angle = torch.sin(angle)  # (b, n_units)
    
        # Stack cos and sin to create the unit vector (b, n_units, 2)
        return torch.stack((cos_angle, sin_angle), dim=-1)
    
    def set_eq_distances_from_positions(self, x, z):
        """
        Set equilibrium distances based on actual distances between connected robots.
        Only updates springs that have non-zero stiffness (i.e., actual connections).
        
        Args:
            x: numpy array or torch tensor of x positions, shape (n_units,)
            z: numpy array or torch tensor of z positions, shape (n_units,)
        """
        # Convert to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        
        # Compute pairwise distances
        n_units = len(x)
        for i in range(n_units):
            for j in range(i + 1, n_units):
                # Only update if there's actually a connection (non-zero stiffness)
                if self.stiffness_coupling_matrix[i, j] > 0:
                    # Calculate Euclidean distance
                    dist = torch.sqrt((x[i] - x[j])**2 + (z[i] - z[j])**2)
                    # Update equilibrium distance for this connection
                    self.eq_distances_matrix.data[i, j, 0] = dist
                    self.eq_distances_matrix.data[j, i, 0] = dist
        
        print(f"Updated equilibrium distances based on initial positions")
        print(f"  Distance range: [{self.eq_distances_matrix.data.min():.4f}, {self.eq_distances_matrix.data.max():.4f}]")


class UnicycleReservoir(nn.Module):
    def __init__(self, n_inp, n_units, dt, n_out, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0,
                 eq_dist_min_ang=0.0, eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=None, inp_bias=0, n_connections_anchor=2, 
                 n_connections_ang=None, n_connections_anchor_ang=2, n_past_steps_readout=0) -> None:
        super().__init__()
        self.n_inp = n_inp
        self.n_units = n_units
        self.unicycle_network = UnicycleNetwork(n_inp, n_units, dt, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max, 
                 ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
                 ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max,
                 eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
                 lin_input_map=None, ang_input_map=None, n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                 n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang)
        self.readout = nn.Linear(n_units*5*(n_past_steps_readout+1), n_out)

        self.inp_bias=inp_bias
        self.n_past_steps_readout = n_past_steps_readout

        if lin_input_map is None:
            lin_input_map = torch.rand(n_inp, n_units)
            self.lin_input_map = nn.Parameter(lin_input_map, requires_grad=False)
        else:
            self.lin_input_map = lin_input_map
        if ang_input_map is None:
            ang_input_map = torch.rand(n_inp, n_units)
            self.ang_input_map = nn.Parameter(ang_input_map, requires_grad=False)
        else:
            self.ang_input_map = ang_input_map
    
    def forward(self, u_lin, u_ang):
        #start = time.time()

        x = self.x_init
        z = self.z_init
        theta = self.theta_init
        s = self.s_init
        omega = self.omega_init
        states_list = []

        for t in range(u_lin.size()[1]):
            linear_input = (u_lin[:, t] +self.inp_bias) @ self.lin_input_map
            angular_input = (u_ang[:, t]) @ self.ang_input_map
            x, z, theta, s, omega = self.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

            concatenated_states = torch.hstack((x, z, theta, s, omega))
            states_list.append(concatenated_states)

        if self.n_past_steps_readout > 0:
            mid_states_idxs = [(int(u_lin.size()[1] / self.n_past_steps_readout) - 1)*k for k in range(1,self.n_past_steps_readout+1)]
            mid_states = torch.hstack(([states_list[idx] for idx in mid_states_idxs]))
        else:
            mid_states = states_list[-1]
        output = None
        return states_list, output, mid_states