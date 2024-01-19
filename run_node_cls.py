"""
version 1.0
date 2021/02/04
"""

import argparse
import socket
import wandb
import random
import numpy as np
from collections import defaultdict
from models import GCNmf, GCN
from train import NodeClsTrainer
from torch_geometric.loader import DataLoader
from utils import apply_mask, generate_mask, ABMInMemoryDataset, set_seed, logger, WandbDummy
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data/abm_sep_obs')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=1, type=int, help='the number of training epoch')
parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the dataloader')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--seed', default=30, type=int, help='Seed value for all packages')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--run_mf', action='store_true', help='If true, will run the multivariate gaussian model')
parser.add_argument('--niter', default=1, type=int, help='Number of training iterations')
parser.add_argument('--n_layers', default=2, type=int, help='Number of layers for the models')
parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))

args = parser.parse_args()

if __name__ == '__main__':
     # Set random seed
    seed_value = args.seed
    if seed_value < 0:
        seed_value = random.randint(0, 999)
        
    set_seed(seed_value)        
    
    #Start WandB
    config_wandb = defaultdict(None)
    config_wandb['type'] = args.type
    config_wandb['run_mf'] = args.run_mf
    config_wandb['epochs'] = args.epoch
    config_wandb['missing_rate'] = args.rate
    config_wandb['nhid'] = args.nhid
    config_wandb['gaussian_comp'] = args.ncomp
    config_wandb['weight_decay'] = args.wd
    config_wandb['lr'] = args.lr
    config_wandb['batch_size'] = args.batch_size
    config_wandb['patience'] = args.patience
    config_wandb['early_stopping'] = False
    
    do_wandb = 'int6' not in socket.gethostname()
    if do_wandb:
        runner_global = wandb.init(
            project='GCN_mf',
            notes = " ",
            name=args.name,
            config=config_wandb
        )
    else:
        runner_global = WandbDummy()
    
    train_data = ABMInMemoryDataset(args.data_path + '/train')
    val_data = ABMInMemoryDataset(args.data_path + '/val')
    test_data = ABMInMemoryDataset(args.data_path + '/test')
    # args.run_mf = True
    if args.run_mf:
        mask_train = generate_mask(train_data.x, args.rate, args.type)
        apply_mask(train_data.x, mask_train)
        
        mask_val = generate_mask(val_data.x, args.rate, args.type)
        apply_mask(val_data.x, mask_val)
        
        mask_test = generate_mask(test_data.x, args.rate, args.type)
        apply_mask(test_data.x, mask_test)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    if args.run_mf:
        logger.info('Running GCN_mf model')
        model = GCNmf(train_data, args.nhid, args.dropout, args.ncomp)
    else:
        logger.info('Running GCN model')
        model = GCN(train_data, args.nhid, args.dropout, args.n_layers)
    
    trainer = NodeClsTrainer(
        train_loader,
        val_loader,
        test_loader,
        model,
        config_wandb,
        runner_global,
        niter=args.niter,
        verbose=args.verbose
        )
    trainer.run()
        
    runner_global.finish()

