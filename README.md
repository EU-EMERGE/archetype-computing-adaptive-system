# Archetype Computing and Adaptive System (ACDS)
<img src="https://raw.githubusercontent.com/EU-EMERGE/archetype-computing-adaptive-system/main/credit.svg" alt="EMERGE project funding acknowledgements">

Archetype Computing and Adapting System for the [EMERGE project](https://eic-emerge.eu/), a project funded by the European Innovation Council (EIC) of the European Union (EU) under Grant Agreement 101070918.

The APIdoc can be found at [this link.](https://eu-emerge.github.io/archetype-computing-adaptive-system/)

### Archetypes
The archetype package implements the Echo State Network model, the Long Short-Term memory model, and the Random Oscillators Network model.
Each model is implemented as a Python class using the PyTorch library . The archetypes package defines the way each archetype performs computation and it is therefore at the center of the Archetype Computing System.

The archetype package also provided utility functions that influence the archetype computation. For example, currently available functions can create sparsely connected Archetype Networks or control the topology of the connections (ring topology, Toeplitz topology and others).

### Benchmarks
The benchmarks package allows to easily create an instance of popular datasets and benchmarks.
Currently, the package features the MNIST dataset, the Mackey-Glass chaotic dynamical system and the Adiac dataset (a real-world time series processing task).

The datasets are currently implemented via helper functions. Each dataset has its own helper functions that returns the training set, the validation set and the test set randomly sampled from the dataset.
Each set is a PyTorch compatible dataset or dataloader, that automatically manages multiple iterations with mini-batches.
Each dataset can implement a custom way of splitting the examples in the three sets.
From the point of view of the user, this does not impact neither on the archetypes nor on the experiment where the dataset is then used.

### Experiments
The experiments package currently provides three experiments.
Each experiment is a main Python script that can be executed by directly running it with a Python interpreter.
The Adiac experiment uses the Adiac dataset and trains either an RON model or an ESN model.
The choice of the model can be controlled by parameters provided as input to the main script.
The experiments logs the results to a text file with the train and validation metrics.
The test metrics can be computed when the corresponding flag is activated (to distinguish between model selection and model assessment).

The sMNIST experiment follows the same approach, but it leverages the MNIST dataset, taken one pixel at a time (sequential MNIST) due to the use of recurrent archetypes.

Finally, the Mackey-Glass experiment uses the Mackey-Glass dataset and trains an RON model or an ESN model to predict future states given the current one.
The logging functionalities remain the same across the three experiments. Each experiment describes all the accepted parameters at the beginning of the code.

### Demo
The `robots_demo` package provides the code to run the EMERGE demo. Please, note that data is not shared through GitHub. 

## How to use
The project requires Python 3.11 or newer. Install from a checkout with pip:

```bash
git clone https://github.com/EU-EMERGE/archetype-computing-adaptive-system.git
cd archetype-computing-adaptive-system
python -m pip install .
```

For development, install the package in editable mode with the development tools:

```bash
python -m pip install -e ".[dev]"
```

`make install` also installs the package in editable mode into the active Python
environment. With conda, activate the environment first (for example,
`conda activate tomato`) and use `make install PYTHON=python`.

The first PyPI release will be **1.0.0**. Once it is published, install it with:

```bash
python -m pip install acds
# Or pin the first release:
python -m pip install "acds==1.0.0"
```

The distribution and Python import are both named `acds`. These PyPI commands
become available after the first release; use the checkout instructions above
until then.

The `evolutionary` extra installs dependencies for `acds.evolutionary`:

```bash
python -m pip install ".[evolutionary]"
# After the PyPI release:
python -m pip install "acds[evolutionary]"
```

Graphviz rendering also requires the system Graphviz installation and its `dot`
executable. Experiments and robot demos run from the source checkout and may need
additional experiment-specific dependencies and external data.

For example, a model can be created without downloading any datasets:

```python
import torch
from acds.archetypes import RandomizedOscillatorsNetwork

model = RandomizedOscillatorsNetwork(
    n_inp=1, n_hid=100, dt=0.01, gamma=1.0, epsilon=1.0
)
states, final_state = model(torch.randn(8, 50, 1))
```

If you want to run the sMNIST experiment with the RON model you can run:
```bash
    python experiments/smnist.py --ron --dataroot /path/where/to/download/mnist
```

## Dataset directory
The library bundles raw datasets in `acds/benchmarks/raw` in both the wheel and
source distribution.
The directory contains the `adiac` folder where both train and test examples for Adiac are stored. 
It also contains the `mackey_glass.csv` file, where the Mackey-Glass dataset is saved. 
Locate these files after installing the package with:

```python
from pathlib import Path
import acds.benchmarks

raw = Path(acds.benchmarks.__file__).parent / "raw"
train, validation, test = acds.benchmarks.get_adiac_data(
    raw / "adiac", bs_train=32, bs_test=64
)
```

Use a writable directory outside the installed package for downloaded datasets
such as MNIST, and pass that directory as the experiment's `dataroot`.

## Results
Results of the experiments are saved within the `resultroot` directory, provided as parameter to each script. Remember to create the directory before using it.


