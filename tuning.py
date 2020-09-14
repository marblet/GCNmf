import argparse
import numpy as np
import optuna
import random
import torch

from numpy import mean
from tqdm import tqdm

from models import GCNmf
from train import Trainer
from utils import NodeClsData, apply_mask, generate_mask


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='random',
                    choices=['random', 'struct'],
                    help="randomly missing or structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--seed', default=17, type=int)

args = parser.parse_args()
TRIAL_SIZE = 100
TIMEOUT = 60 * 60 * 3

patience, epochs = 100, 10000

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

print(args.dataset, args.type, args.rate)
print("num of components:", args.ncomp)
print("nhid:", args.nhid)
print("patience:", patience)
print("epochs:", epochs)

# generate all masks for the experiment
tmpdata = NodeClsData(args.dataset)
masks = [generate_mask(tmpdata.features, args.rate, args.type) for _ in range(5)]


def objective(trial):
    # Tune hyperparameters (dropout, weight decay, learning rate) using Optuna
    dropout = trial.suggest_uniform('dropout', 0.4, 0.8)
    lr = trial.suggest_loguniform('lr', 5e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-1)

    # prepare data and model
    data = NodeClsData(args.dataset)
    apply_mask(data.features, masks[0])
    model = GCNmf(data, args.nhid, dropout, args.ncomp)

    # run model
    trainer = Trainer(data, model, lr, weight_decay, epochs=epochs, niter=20, early_stopping=True, patience=patience)
    result = trainer.run()
    return - result['val_acc']


def tune_hyperparams():
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE, timeout=TIMEOUT)
    return study.best_params


def evaluate_model(params):
    means = []
    dropout = params['dropout']
    lr = params['lr']
    weight_decay = params['weight_decay']
    for mask in tqdm(masks):
        # generate missing data, model and trainer
        data = NodeClsData(args.dataset)
        apply_mask(data.features, mask)  # convert masked number to nan
        model = GCNmf(data, args.nhid, dropout, args.ncomp)
        trainer = Trainer(data, model, lr, weight_decay, epochs=epochs, niter=20, patience=patience)

        # run the model
        result = trainer.run()
        means.append(result['test_acc'])

    return mean(means)


def main():
    params = tune_hyperparams()
    result = evaluate_model(params)
    print(result)


if __name__ == '__main__':
    main()
