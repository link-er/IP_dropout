# Information Plane Analysis for Dropout Neural Networks

This is the official implementation repository for [Information Plane Analysis for Dropout Neural Networks (Adilova, Geiger, Fischer, ICLR 2023)](https://openreview.net/forum?id=bQB6qozaBw). 

There are training procedures for CIFAR10 and MNIST experiments, each of which includes the script for training itself and Jupyter notebook for plotting information planes after training.
There is also a script for the checkup of the properties of the estimator under multiplicative Gaussian noise (estimator_validation folder).

MI estimators include NPEET from https://github.com/gregversteeg/NPEET and EDGE from https://github.com/mrtnoshad/EDGE.
Dropout MI ([dropout_MI.py](https://github.com/link-er/IP_dropout/blob/master/mi_estimators/dropout_MI.py)) estimator is an implementation of the algorithm proposed in the paper.


## 1. Training 
For training run either mi_cifar_dropout.py or mi_mnist_dropout.py.
The training script creates two folders: representations and IP, that are later used in the notebook for plotting information planes.
Representations contains numpy arrays of the representations during epochs (callback in [collect_repr_callback.py](https://github.com/link-er/IP_dropout/blob/master/utils/collect_repr_callback.py)) and numpy arrays of data and labels for ease of access during plotting.
IP contains dictionaries of mutual informations saved during training.
Note, that only in the case of information dropout the dictionary of MI(X;Z) is not filled with zeros.

## 2. Plotting
For plotting run corresponding notebook _draw_IP.
It has a part for plotting the information dropout information planes, Gaussian dropout ones, and binning information planes.

## 3. Estimator validation experiments
Running known_MI.py requires two parameters: --dim and --fdim, dimensionality of the input and dimensionality of the representation correspondingly.
The values for Difference-of-Entropies (DoE) Estimator can be verified by running the [official code](https://github.com/karlstratos/doe) with corresponding data used.

Both of mc_convergence.py and upper_bound_check.py require the same parameters to run.