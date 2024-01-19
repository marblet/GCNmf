"""
version 1.0
date 2021/02/04
"""

import argparse
import random

import numpy as np
import optuna
import torch

from numpy import mean
from tqdm import tqdm

from torch_geometric.data import Batch
from models import GCNmf
from models import GCN
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask, ABMInMemoryDataset
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='abm',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--data_path', default='data/abm')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=8, type=int, help='the number of hidden units')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--epoch', default=1000, type=int, help='the number of training epoch')
parser.add_argument('--niter', default=1, type=int, help='Number of training iterations')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--seed', default=17, type=int)
parser.add_argument('--batch_size', default=32, type=int)

args = parser.parse_args()
TRIAL_SIZE = 100
TIMEOUT = 60 * 60 * 3

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

print(args.dataset, args.type, args.rate)
print("num of components:", args.ncomp)
print("nhid:", args.nhid)
print("epochs:", args.epoch)
print("patience:", args.patience)

# generate all masks for the experiment
# tmpdata = NodeClsData(args.data_path)
# masks = [generate_mask(tmpdata.features, args.rate, args.type) for _ in range(5)]


def objective(trial):
    # Tune hyperparameters (dropout, weight decay, learning rate) using Optuna
    dropout = trial.suggest_uniform('dropout', 0.4, 0.8)
    lr = trial.suggest_loguniform('lr', 5e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-1)

    # prepare data and model
    # data = NodeClsData(args.data_path)
    train_data = ABMInMemoryDataset(args.data_path + '/train')
    val_data = ABMInMemoryDataset(args.data_path + '/val')
    test_data = ABMInMemoryDataset(args.data_path + '/test')
    # train_data, val_data, test_data =  data.train_data, data.val_data, data.test_data
    model = GCN(5, 2, args.nhid, dropout)
    # model = GCNmf(data, args.nhid, dropout, args.ncomp)

    # run model
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    # train_data = Batch.from_data_list(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    trainer = NodeClsTrainer(train_loader, val_loader, test_loader, model, params, niter=args.niter)
    result = trainer.run()
    return - result['val_acc']


def tune_hyperparams():
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE, timeout=TIMEOUT)
    return study.best_params


def evaluate_model(hyperparams):
    means = []
    dropout = hyperparams['dropout']
    for mask in tqdm(masks):
        # generate missing data, model and trainer
        data = NodeClsData(args.data_path)
        apply_mask(data.features, mask)  # convert masked number to nan
        model = GCNmf(data, args.nhid, dropout, args.ncomp)
        params = {
            'lr': hyperparams['lr'],
            'weight_decay': hyperparams['weight_decay'],
            'epochs': args.epoch,
            'patience': args.patience,
            'early_stopping': True
        }
        trainer = NodeClsTrainer(data, model, params, niter=20)

        # run the model
        result = trainer.run()
        means.append(result['test_acc'])

    return mean(means)

def evaluate_model_gcn(hyperparams):
    means = []
    dropout = hyperparams['dropout']

    # generate missing data, model and trainer
    train_data, val_data, test_data = NodeClsData(args.data_path)
    model = GCN(train_data, args.nhid, dropout)
    params = {
        'lr': hyperparams['lr'],
        'weight_decay': hyperparams['weight_decay'],
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    
    train_loader = DataLoader(train_data.data_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data.data_list, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data.data_list, batch_size=args.batch_size, shuffle=True)
    
    trainer = NodeClsTrainer(train_loader, val_loader, test_loader, model, params, niter=20)

    # run the model
    result = trainer.run()
    means.append(result['test_acc'])

    return mean(means)

def main():
    hyper_params = tune_hyperparams()
    result = evaluate_model_gcn(hyper_params)
    print(result)


if __name__ == '__main__':
    main()