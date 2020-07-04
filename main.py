import argparse

from models import *
from train import Trainer
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type', default='random', choices=['random', 'struct'],
                    help="randomly missing or structurally missing")
parser.add_argument('--rate', default=0.1, help='missing rate')
parser.add_argument('--nhid', default=16, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, help='dropout rate')
parser.add_argument('--ncomp', default=5, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, help='learning rate')
parser.add_argument('--wd', default=1e-2, help='weight decay')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--seed', default=17)

args = parser.parse_args()
missing_rate = float(args.rate)
nhid = int(args.nhid)
dropout = float(args.dropout)
n_components = int(args.ncomp)
lr = float(args.lr)
wd = float(args.wd)
verbose = args.verbose
SEED = int(args.seed)


if __name__ == '__main__':
    data = load_data(args.dataset)
    mask = generate_mask(data.features, missing_rate, args.type)
    apply_mask(data.features, mask)
    model = GCNmf(data, nhid=nhid, dropout=dropout, n_components=n_components)
    trainer = Trainer(data, model, lr=lr, weight_decay=wd, niter=20, patience=100, epochs=10000, verbose=verbose)
    trainer.run()
