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
### Node Classification
To run GCNmf for node classification, you can use run_node_cls.py with some options to specify dataset, missing type, missing rate, and hyper-parameters:
```
$ python run_node_cls.py --dataset citeseer --type struct --rate 0.1 --verbose
```
The following command shows the arguments of run_node_cls.py:
```
$ python run_node_cls.py --help
```
You can optimize hyper-parameters (dropout, learning rate, weight_decay) using tune_node_cls.py:
```
$ python tune_node_cls.py --dataset amaphoto --type struct --rate 0.1
```

### Link Prediction
We take VGAE as the base model for link prediction and define a graph autoencoder, which employs GCNmf as an encoder.
(For here, we call the link prediction model VGAEmf to distinguish the model for node classification.)
To run the model for link prediction, you can use run_link_pred.py with some options:
```
$ python run_link_pred.py --dataset cora --type bias --rate 0.5 --verbose
```
You can optimize hyper-parameters (dropout, learning rate, weight_decay) using tune_link_pred.py:
```
$ python tune_link_pred.py --dataset citeseer --type struct --rate 0.1
```
