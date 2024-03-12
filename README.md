# Archetype Computing and Adaptive System (ACDS)
<img src="./credit.svg">

Archetype Computing and Adapting System for the [EMERGE project](https://eic-emerge.eu/), a project funded by the European Innovation Council (EIC) of the European Union (EU) under Grant Agreement 101070918.

## Main structure
The library offers three packages (within the main acds package): archetypes, benchmarks and experiments.

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

## How to use
The project requires Python 3, PyTorch, Torchvision, Numpy, Scipy and Scikit-learn (alongside minor packages like tqdm).
You can clone the project and use it by adding it to the PYTHONPATH. For example, if you want to run the sMNIST experiment with the RON model you can run:
```bash
PYTHONPATH=/path/to/project/folder python experiments/smnist.py --ron
```


