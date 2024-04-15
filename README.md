# Symmetric Positive Definite Convolutional Network for Surrogate Modeling and Optimization of Modular Structures

This is the code and model weights used in the paper "Symmetric Positive Definite Convolutional Network for Surrogate Modeling and Optimization of Modular Structures".

Authors: Liya Gaynutdinova $^1$, Marin Doškář $^1$, Ivana Pultarová $^1$, Ondřej Rokoš $^2$

$^1$ Czech Technical University in Prague, Thákurova 7, Prague, 16629, Czech Republic

$^2$ Eindhoven University of Technology, 5600 MB, Eindhoven, P.O. Box 513, The Netherlands

The training data can be downloaded from Zenodo, doi:10.5281/zenodo.10909619.

## The network

The network in SPD_CNN_modular.py can be instantiated for a different resolutions $n \times n$ (parameter 'res'), and different kernel size (list parameter 'kernels'). The 'training_mode' parameter can be switched between 'stiffness' and 'compliance'. In the first case, the model will only output the condensed stiffness matrix $K_{grid}$, otherwise it will also output the compliance matrix $C_{grid}$. It is recommended to order the kernels in the ascending order, e.g. '[1, 3]', '[1, 3, 5]'. The weights of the models for different resolutions are mutually compatible. The pre-trained networks have a following naming convention ('resolution' indicates the largest resolution the network has been trained on):

### Dataset 1 (anisotropic modules)

NN_anisotropic_{resolution}x{resolution}_{kernel list}

### Dataset 2 (varied density modules)

NN_{resolution}x{resolution}_{kernel list}

## Data generator

New data for the dataset 2 can be generated in the "data generator demo" Jupyter Notebook.

## Training

Training process is demonstrated on the dataset 1 in the "training demo" Jupyter Notebook.

## Optimization experiments

The optimization experiments on the anisotropic module dataset can be calculated with the "Optimization + brute search" Jupyter Notebook.

The optimization experiments on the varied density dataset can be calculated with the "Optimization simple beam" and "Optimization L-beam" Jupyter Notebook.



