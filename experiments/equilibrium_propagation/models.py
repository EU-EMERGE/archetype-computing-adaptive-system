"""
Adapted from
https://github.com/Laborieux-Axel/Equilibrium-Propagation/blob/master/model_utils.py
"""

import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

def my_sigmoid(x):
    return 1 / (1 + torch.exp(-4 * (x - 0.5)))


def hard_sigmoid(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5


def ctrd_hard_sig(x):
    return (F.hardtanh(2 * x)) * 0.5


def my_hard_sig(x):
    return (1 + F.hardtanh(x - 1)) * 0.5


def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy


def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p] == 'm':
            pools.append(torch.nn.MaxPool2d(2, stride=2))
        elif letters[p] == 'a':
            pools.append(torch.nn.AvgPool2d(2, stride=2))
        elif letters[p] == 'i':
            pools.append(torch.nn.Identity())
    return pools


def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)

    return my_scaled_init

'''
MODELS
'''
# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False  # Softmax readout is only defined for CNN and VFCNN
        self.nc = self.archi[-1]

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

    def Phi(self, x, y, neurons, beta, criterion):
        # Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss
        x = x.view(x.size(0), -1)  # flattening the input

        layers = [x] + neurons  # concatenate the input to other layers

        # Primitive function computation
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layers[idx]) * layers[idx + 1],
                             dim=1).squeeze()  # Scalar product s_n.W.s_n-1

        if beta != 0.0:  # Nudging the output layer when beta is non zero
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layers[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)  # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)  # Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads)  # dPhi/ds

            for idx in range(len(neurons) - 1):
                neurons[idx] = self.activation(grads[idx])  # s_(t+1) = sigma( dPhi/ds )
                neurons[idx].requires_grad = True

            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation(grads[-1])

            neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


# Random Oscillator Network

class RON(torch.nn.Module):
    def __init__(self, archi, device, activation=torch.tanh, tau=1, epsilon_min=0, epsilon_max=1, gamma_min=0, gamma_max=1, learn_oscillators=True):
        super(RON, self).__init__()

        self.activation = activation
        self.archi = archi
        self.softmax = False
        self.same_update = False
        self.nc = self.archi[-1]
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau = tau
        print("learn oscillator = ", learn_oscillators)
        self.learn_oscillators = learn_oscillators
        self.device = device

        self.gamma = torch.rand(archi[1], device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(archi[1], device=device) * (epsilon_max - epsilon_min) + epsilon_min
        self.gamma = torch.nn.Parameter(self.gamma, requires_grad=learn_oscillators)
        self.epsilon = torch.nn.Parameter(self.epsilon, requires_grad=learn_oscillators)
        assert len(archi) > 2, "The architecture must have at least 1 hidden layer"
        assert all([archi[1] == a for a in archi[2:-1]]), "The hidden layers must have the same number of neurons"

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))

    def Phi_statez(self, x, y, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)

        layersy = [x] + neuronsy
        
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx + 1],
                             dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def Phi_statey(self, neuronsz, neuronsy):
        phi = 0.0
        for idx in range(len(neuronsz)):
            phi += 0.5 * (torch.einsum('ij,ij->i', neuronsy[idx], neuronsy[idx]) +
                          self.tau * torch.einsum('ij,ij->i', neuronsz[idx], neuronsz[idx]))
        return phi

    def Phi(self, x, y, neuronsz, neuronsy, beta, criterion):
        x = x.view(x.size(0), -1)
        
        layersz = [x] + neuronsz
        layersy = [x] + neuronsy
        
        phi = torch.sum(0.5 * self.tau * self.synapses[0](x) * layersy[1], dim=1).squeeze()
        for idx in range(1, len(self.synapses) - 1):
            phiz = ((-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersz[idx], torch.diag(self.epsilon).to(self.device)), layersz[idx]))
                    + (-0.5 * torch.einsum('ij,ij->i', torch.einsum('ij,jj->ij', layersy[idx], torch.diag(self.gamma).to(self.device)), layersy[idx]))
                    + (0.5 * torch.einsum('ij,ij->i', layersz[idx], layersz[idx]))
                    + (self.tau * torch.sum(self.synapses[idx](layersy[idx]) * layersy[idx+1], dim=1).squeeze()))

            phi += 0.5 * (torch.einsum('ij,ij->i', layersy[idx], layersy[idx]) + self.tau * phiz)
        phi += torch.sum(0.5 * self.tau * self.synapses[-1](layersy[-2]) * layersy[-1], dim=1).squeeze()

        if beta != 0.0:
            if criterion.__class__.__name__.find('MSE') != -1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5 * criterion(layersy[-1].float(), y.float()).sum(dim=1).squeeze()
            else:
                L = criterion(layersy[-1].float(), y).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neuronsz, neuronsy, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none')):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi_statez(x, y, neuronsy, beta, criterion)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            grads = torch.autograd.grad(phi, neuronsy, grad_outputs=init_grads)

            for idx in range(len(neuronsz)):
                oscillator = neuronsz[idx] - self.tau * self.epsilon * neuronsz[idx] - self.tau * self.gamma * neuronsy[idx]
                neuronsz[idx] = (self.activation(grads[idx]) * self.tau + oscillator).detach()
                neuronsz[idx].requires_grad = True

            if not_mse:
                neuronsy[-1] = grads[-1]
            else:
                neuronsy[-1] = self.activation(grads[-1])
            neuronsy[-1].requires_grad = True

            phi = self.Phi_statey(neuronsz, neuronsy)
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device,
                                      requires_grad=True)
            gradsz = torch.autograd.grad(phi, neuronsz, grad_outputs=init_grads, retain_graph=True)
            gradsy = torch.autograd.grad(phi, neuronsy[:-1], grad_outputs=init_grads)
            grads = [gz + gy for gz, gy in zip(gradsz, gradsy)]

            for idx in range(len(neuronsy) - 1):
                neuronsy[idx] = grads[idx]
                neuronsy[idx].requires_grad = True

        return neuronsz, neuronsy

    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neuronsz, neuronsy = [], []
        for size in self.archi[1:-1]:
            neuronsz.append(torch.zeros(mbs, size, requires_grad=True, device=device))
            neuronsy.append(torch.zeros(mbs, size, requires_grad=True, device=device))
        neuronsy.append(torch.zeros(mbs, self.archi[-1], requires_grad=True, device=device))
        return neuronsz, neuronsy

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        neurons_1z, neurons_1y = neurons_1
        neurons_2z, neurons_2y = neurons_2

        self.zero_grad()  # p.grad is zero
        phi_1 = self.Phi(x, y, neurons_1z, neurons_1y, beta_1, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2z, neurons_2y, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem

'''
TRAIN
'''

"""
Modified training and evaluation functions to support both P_MLP and RON models.
For P_MLP we use a single state (neurons) while for RON we use two states (neuronsz, neuronsy).
"""

def train_epoch(model, optimizer, epoch_number, train_loader, T1, T2, betas, device, criterion, alg='EP',
          random_sign=False, thirdphase=False, cep_debug=False, ron=False, id=None):
    mbs = train_loader.batch_size
    iter_per_epochs = math.ceil(len(train_loader.dataset) / mbs)
    beta_1, beta_2 = betas

    run_correct = 0
    run_total = 0
    model.train()

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # if alg=='CEP' and cep_debug:
        #    x = x.double()

        if ron:
            neuronsz, neuronsy = model.init_neurons(x.size(0), device)
        else:
            neurons = model.init_neurons(x.size(0), device)
        if alg == 'EP' or alg == 'CEP':
            # First phase
            if ron:
                neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T1, beta=beta_1, criterion=criterion)
                neurons_1 = (copy(neuronsz), copy(neuronsy))
                neurons = neuronsy
            else:
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
        elif alg == 'BPTT':
            assert not ron, "RON not implemented for BPTT"
            neurons = model(x, y, neurons, T1 - T2, beta=0.0, criterion=criterion)
            # detach data and neurons from the graph
            x = x.detach()
            x.requires_grad = True
            for k in range(len(neurons)):
                neurons[k] = neurons[k].detach()
                neurons[k].requires_grad = True

            neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion)  # T2 time step

        # Predictions for running accuracy
        with torch.no_grad():
            if not model.softmax:
                pred = torch.argmax(neurons[-1], dim=1).squeeze()
            else:
                # WATCH OUT: prediction is different when softmax == True
                pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0), -1)), dim=1),
                                    dim=1).squeeze()

            run_correct += (y == pred).sum().item()
            run_total += x.size(0)

        if alg == 'EP':
            # Second phase
            if random_sign and (beta_1 == 0.0):
                rnd_sgn = 2 * np.random.randint(2) - 1
                betas = beta_1, rnd_sgn * beta_2
                beta_1, beta_2 = betas

            if ron:
                neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=beta_2, criterion=criterion)
                neurons_2 = (copy(neuronsz), copy(neuronsy))
            else:
                neurons = model(x, y, neurons, T2, beta=beta_2, criterion=criterion)
                neurons_2 = copy(neurons)

            # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
            if thirdphase:
                if ron:
                    neuronsz, neuronsy = copy(neurons_1[0]), copy(neurons_1[1])
                    neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T2, beta=- beta_2, criterion=criterion)
                    neurons_3 = (copy(neuronsz), copy(neuronsy))
                else:
                # come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta=- beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                if not (isinstance(model, VF_CNN)):
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                else:
                    if model.same_update:
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                    else:
                        model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion,
                                                neurons_3=neurons_3)
            else:
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

            optimizer.step()

        elif alg == 'CEP':
            if random_sign and (beta_1 == 0.0):
                rnd_sgn = 2 * np.random.randint(2) - 1
                betas = beta_1, rnd_sgn * beta_2
                beta_1, beta_2 = betas

            # second phase
            if cep_debug:
                prev_p = {}
                for (n, p) in model.named_parameters():
                    prev_p[n] = p.clone().detach()
                for i in range(len(model.synapses)):
                    prev_p['lrs' + str(i)] = optimizer.param_groups[i]['lr']
                    prev_p['wds' + str(i)] = optimizer.param_groups[i]['weight_decay']
                    optimizer.param_groups[i]['lr'] *= 6e-5
                    # optimizer.param_groups[i]['weight_decay'] = 0.0

            for k in range(T2):
                neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # one step
                neurons_2 = copy(neurons)
                model.compute_syn_grads(x, y, neurons_1, neurons_2, betas,
                                        criterion)  # compute cep update between 2 consecutive steps
                for (n, p) in model.named_parameters():
                    p.grad.data.div_((1 - optimizer.param_groups[int(n[9])]['lr'] *
                                      optimizer.param_groups[int(n[9])]['weight_decay']) ** (T2 - 1 - k))
                optimizer.step()  # update weights
                neurons_1 = copy(neurons)

            if thirdphase:
                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion)  # come back to s*
                neurons_2 = copy(neurons)
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta=-beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                    optimizer.step()
                    neurons_2 = copy(neurons)

        elif alg == 'BPTT':
            assert not ron, "RON not implemented for BPTT"
            # final loss
            if criterion.__class__.__name__.find('MSE') != -1:
                loss = 0.5 * criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(
                    dim=1).mean().squeeze()
            else:
                if not model.softmax:
                    loss = criterion(neurons[-1].float(), y).mean().squeeze()
                else:
                    loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0), -1)).float(),
                                     y).mean().squeeze()
            # setting gradients field to zero before backward
            model.zero_grad()

            # Backpropagation through time
            loss.backward()
            optimizer.step()
        if ((idx % (iter_per_epochs // 10) == 0) or (idx == iter_per_epochs - 1)):
            run_acc = run_correct / run_total
            if (id != None):
                # print("Trial ", id, '-> Epoch :', round(epoch_number + (idx / iter_per_epochs), 2),
                #   '\tRun train acc :', round(run_acc, 3), '\t(' + str(run_correct) + '/' + str(run_total) + ')')
                pass
            else:
                print('Epoch :', round(epoch_number + (idx / iter_per_epochs), 2),
                  '\tRun train acc :', round(run_acc, 3), '\t(' + str(run_correct) + '/' + str(run_total) + ')')


def train_epoch_TS(
    model,
    optimizer,
    epoch_number,
    train_loader,
    T1,            
    T2,            
    betas,         
    device,
    criterion,
    reset_factor=0.0,
    id=None,
    ron=False
):
    """
    Train an epoch on time-series data, updating weights at every timestep.
    Modified to support MLP (with a single state) in addition to RON (with two states).
    """

    model.train()
    beta_1, beta_2 = betas
    run_correct = 0
    run_total = 0
    
    mbs = train_loader.batch_size
    iter_per_epoch = math.ceil(len(train_loader.dataset) / mbs)

    for idx, (x, y) in enumerate(train_loader):
        # x shape: [B, T, D] and y: [B] (or possibly [B, T])
        x = x.to(device)
        y = y.to(device)

        B, T_seq, D = x.shape
        
        if not ron:
            neurons = None  # single state for MLP
            for t in range(T_seq):
                x_t = x[:, t, :]
                y_t = y  # adjust if label changes per timestep

                # Reset (or initialize) state
                if neurons is None:
                    neurons = model.init_neurons(B, device)
                else:
                    # partial reset of neurons (scaling by reset_factor)
                    neurons = [n.detach() * reset_factor for n in neurons]
                    neurons = [n.clone().requires_grad_() for n in neurons]

                model.zero_grad()
                neurons_1 = model(x_t, y_t, copy(neurons), T=T1, beta=beta_1, criterion=criterion)
                with torch.no_grad():
                    pred = torch.argmax(neurons_1[-1], dim=1).squeeze()
                    run_correct += (pred == y_t).sum().item()
                    run_total   += B

                model.zero_grad()
                neurons_2 = model(x_t, y_t, copy(neurons), T=T2, beta=beta_2, criterion=criterion)
                model.compute_syn_grads(x_t, y_t, neurons_1, neurons_2, (beta_1, beta_2), criterion)
                optimizer.step()

                # Carry forward the state from the nudged phase.
                neurons = neurons_2

        else:
            # RON branch: use two states (neuronsz, neuronsy)
            neuronsz, neuronsy = None, None
            for t in range(T_seq):
                x_t = x[:, t, :]
                y_t = y

                if neuronsz is None or neuronsy is None:
                    neuronsz, neuronsy = model.init_neurons(B, device)
                else:
                    neuronsz = [nz.detach() * reset_factor for nz in neuronsz]
                    neuronsz = [nz.clone().requires_grad_() for nz in neuronsz]
                    neuronsy = [ny.detach() * reset_factor for ny in neuronsy]
                    neuronsy = [ny.clone().requires_grad_() for ny in neuronsy]

                model.zero_grad()
                neuronsz_1, neuronsy_1 = model(x_t, y_t, copy(neuronsz), copy(neuronsy), T=T1, beta=beta_1, criterion=criterion)
                with torch.no_grad():
                    pred = torch.argmax(neuronsy_1[-1], dim=1).squeeze()
                    run_correct += (pred == y_t).sum().item()
                    run_total   += B

                model.zero_grad()
                neuronsz_2, neuronsy_2 = model(x_t, y_t, copy(neuronsz), copy(neuronsy), T=T2, beta=beta_2, criterion=criterion)
                model.compute_syn_grads(x_t, y_t, (neuronsz_1, neuronsy_1), (neuronsz_2, neuronsy_2), (beta_1, beta_2), criterion)
                optimizer.step()

                neuronsz, neuronsy = neuronsz_2, neuronsy_2

        if ((idx % (iter_per_epoch // 10) == 0) or (idx == iter_per_epoch - 1)):
            run_acc = run_correct / run_total if run_total > 0 else 0.0
            if id is not None:
                pass
            else:
                print(
                    'Epoch :', round(epoch_number + (idx / iter_per_epoch), 2),
                    '\tRun train acc :', round(run_acc, 3),
                    '\t(', run_correct, '/', run_total, ')'
                )


'''
EVALUATE
'''

def evaluate(model, loader, T, device, ron=False):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if ron:
            neuronsz, neuronsy = model.init_neurons(x.size(0), device)
            neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, T)  # dynamics for T time steps
            neurons = neuronsy
        else:
            neurons = model.init_neurons(x.size(0), device)
            neurons = model(x, y, neurons, T)  # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1],
                                dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else:  # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0), -1)), dim=1), dim=1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct / len(loader.dataset)
    return acc


def evaluate_TS(model, loader, T, device, ron=False):
    """
    Evaluate the model on time-series data.
    - For a single-state network (e.g. P_MLP), we use one state.
    - For models like RON with two states, we use both states.
    If labels are provided per time step (shape: [B, T]), then the label for the current time step
    is used; otherwise, the same label is applied at every step.
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        B, T_seq, D = x.shape

        if not ron:
            # Single-state network (e.g., P_MLP)
            neurons = model.init_neurons(B, device)
            for t in range(T_seq):
                x_t = x[:, t, :]
                # Use per-timestep label if available; otherwise, use the entire y
                y_t = y[:, t] if (y.ndim > 1 and y.size(1) == T_seq) else y
                neurons = model(x_t, y_t, neurons, T, beta=0.0)
            output = neurons[-1]
        else:
            # Two-state branch (e.g., RON)
            neuronsz, neuronsy = model.init_neurons(B, device)
            for t in range(T_seq):
                x_t = x[:, t, :]
                y_t = y[:, t] if (y.ndim > 1 and y.size(1) == T_seq) else y
                neuronsz, neuronsy = model(x_t, y_t, neuronsz, neuronsy, T, beta=0.0)
            output = neuronsy[-1]

        pred = torch.argmax(output, dim=1).squeeze()
        correct += (pred == y).sum().item()
        total += B

    acc = correct / total
    return acc


'''
CONVERGENCE
'''

def visualize_convergence(model, loader, T_ep, device, ron=False, name=None):
    """
    Visualize the convergence of a non-time-series model's state to a fixed point,
    storing differences between consecutive states.
    """
    model.eval()  # Set model to evaluation mode

    # Retrieve one batch from the loader for visualization.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        break

    differences = []

    if not ron:
        # Single-state model
        # 1) Initialize the state and store as 'prev_state'
        neurons = model.init_neurons(x.size(0), device)
        prev_state = neurons[-1].clone().detach()  # The initial state (iteration 0)

        # 2) Run T_ep extra iterations
        for _ in range(T_ep):
            # A single EP step
            neurons = model(x, y, neurons, 1, beta=0.0)
            current_state = neurons[-1]
            
            # Difference between current and previous
            diff = torch.norm(current_state - prev_state, p=2, dim=1).mean().item()
            differences.append(diff)
            
            prev_state = current_state.clone().detach()

    else:
        # Two-state (RON)
        # 1) Initialize the states
        neuronsz, neuronsy = model.init_neurons(x.size(0), device)
        prev_state = neuronsy[-1].clone().detach()  # The initial Y-state (iteration 0)

        # 2) Run T_ep iterations
        for _ in range(T_ep):
            # A single EP step
            neuronsz, neuronsy = model(x, y, neuronsz, neuronsy, 1, beta=0.0)
            current_state = neuronsy[-1]
            
            # Difference
            diff = torch.norm(current_state - prev_state, p=2, dim=1).mean().item()
            differences.append(diff)
            
            prev_state = current_state.clone().detach()

    # ------------------------- PLOTTING PART -------------------------
    iterations = np.arange(1, T_ep + 1)

    plt.figure(figsize=(10, 6))  # Larger figure
    plt.plot(iterations, differences, marker='o', markersize=3, linestyle='-')
    plt.xlabel('EP Iteration (Step)', fontsize=12)
    plt.ylabel('Mean L2 Norm Difference', fontsize=12)
    if name:
        plt.title(name, fontsize=14)
    else:
        plt.title('Convergence of Model to a Fixed Point', fontsize=14)

    # Log scale on the y-axis to reveal exponential decay
    plt.yscale('linear')

    # Show fewer x-axis ticks (up to 10 evenly spaced)
    num_ticks = min(T_ep, 10)
    xtick_positions = np.linspace(1, T_ep, num_ticks, dtype=int)
    plt.xticks(xtick_positions, rotation=5)

    # Add a grid
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Tighten layout and display
    plt.tight_layout()
    plt.show()
    # ----------------------------------------------------------------

    return differences


def visualize_convergence_TS(model, loader, T_ep, device, ron=False, name=None):
    """
    Visualize the convergence of a time-series model's state to a fixed point
    (in the same way 'evaluate_TS' processes data).
    
    Args:
        model: The time-series neural network model (single-state or RON).
        loader: Dataloader providing evaluation samples.
        T_ep: Number of "mini-steps" (EP iterations) per time step to measure convergence.
        device: Torch device to run the model on.
        ron: Whether the model is RON (two states: z, y) or not (single state).
        
    Returns:
        A list (or 1D array) of mean L2 norm differences between consecutive
        states across all time steps and EP iterations.
    """
    model.eval()
    
    # Grab a single batch for visualization
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    B, T_seq, D = x.shape
    
    # We store the difference at each "mini-step" across the entire sequence
    differences = []
    
    if not ron:
        # ------------------ Single-state TS model ------------------
        # 1) Initialize the state only once, as in 'evaluate_TS'
        neurons = model.init_neurons(B, device)
        
        # 2) Loop over the time dimension
        for t in range(T_seq):
            # Extract the time-slice
            x_t = x[:, t, :]
            # Use per-timestep label if available
            if (y.ndim > 1) and (y.size(1) == T_seq):
                y_t = y[:, t]
            else:
                y_t = y
            
            # 3) For each time step, do T_ep "mini-steps" at beta=0
            #    measuring the difference between consecutive states
            for _ in range(T_ep):
                prev_state = neurons[-1].clone().detach()
                # One step of EP dynamics
                neurons = model(x_t, y_t, neurons, 1, beta=0.0)
                current_state = neurons[-1]
                # Measure norm of difference
                diff = torch.norm(current_state - prev_state, p=2, dim=1).mean().item()
                differences.append(diff)
                
    else:
        # ------------------ Two-state (RON) TS model ------------------
        neuronsz, neuronsy = model.init_neurons(B, device)
        
        for t in range(T_seq):
            x_t = x[:, t, :]
            if (y.ndim > 1) and (y.size(1) == T_seq):
                y_t = y[:, t]
            else:
                y_t = y
            
            for _ in range(T_ep):
                prev_state = neuronsy[-1].clone().detach()
                # One step of EP (RON) dynamics
                neuronsz, neuronsy = model(x_t, y_t, neuronsz, neuronsy, 1, beta=0.0)
                current_state = neuronsy[-1]
                diff = torch.norm(current_state - prev_state, p=2, dim=1).mean().item()
                differences.append(diff)
    
    # ------------------------- PLOTTING PART -------------------------
    iterations = np.arange(len(differences)) + 1  # 1-based indexing
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, differences, marker='o', markersize=3, linestyle='-')
    plt.xlabel('Global EP Step (across all time steps)', fontsize=12)
    plt.ylabel('Mean L2 Norm Difference', fontsize=12)
    plt.title('Time-Series Convergence to a Fixed Point', fontsize=14)
    
    plt.yscale('linear')
    # Show fewer x-axis ticks
    max_ticks = 10
    if len(iterations) > max_ticks:
        xtick_positions = np.linspace(1, len(iterations), max_ticks, dtype=int)
        plt.xticks(xtick_positions, rotation=5)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # ----------------------------------------------------------------
    
    return differences