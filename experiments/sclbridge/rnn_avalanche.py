# execute with PYTHONPATH=/path/to/archetype-computing-adaptive-system/ python rnn_avalanche.py 

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from wrapper import AvalancheRON

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = SimpleMLP(num_classes=10)
# model 
model = AvalancheRON(
    n_inp=1,n_hid=256,dt=0.017,
    gamma=1.0, epsilon=1.0, rho=0.99, 
    input_scaling=1.0, device=device, n_classes=10) # type: ignore


smnist = SplitMNIST(n_experiences=5)
train_stream = smnist.train_stream
test_stream = smnist.test_stream

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()])

cl_strategy = Naive(
    model=model, optimizer=optimizer, criterion=criterion, train_mb_size=64, train_epochs=2, 
    eval_mb_size=128, evaluator=eval_plugin, device=device)

results = []
for train_task in train_stream:
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.eval(test_stream))
