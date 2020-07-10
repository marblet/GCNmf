# GCNmf
This is a PyTorch implementation of "Graph Convolutional Networks for Graphs Containing Missing Features".
https://arxiv.org/abs/2007.04583

## Requirements
- pytorch
- scikit-learn
- networkx
- optuna
- tqdm

## Run codes
To run GCNmf, you can use main.py with some options to specify dataset, missing type, missing rate, hyper-parameters:
```
$ python main.py --dataset citeseer --type struct --rate 0.1 --verbose
```
The following command shows the arguments of main.py:
```
$ python main.py --help
```
You can optimize hyperparameters (dropout, learning rate, weight_decay) using tuning.py:
```
$ python tuning.py --dataset cora --type struct --rate 0.1
```
