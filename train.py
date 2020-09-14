from copy import deepcopy

import torch
import torch.nn.functional as F

from numpy import mean, std
from torch.optim import Adam
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience, verbose):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def reset(self):
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        if evals['val_loss'] <= self.best_val_loss or evals['val_acc'] >= self.best_val_acc:
            if evals['val_loss'] <= self.best_val_loss and evals['val_acc'] >= self.best_val_acc:
                self.state_dict = deepcopy(model.state_dict())
            self.best_val_loss = min(self.best_val_loss, evals['val_loss'])
            self.best_val_acc = max(self.best_val_acc, evals['val_acc'])
            self.counter = 0
        else:
            self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            if self.verbose:
                print("Stop training, epoch:", epoch)
            model.load_state_dict(self.state_dict)
        return stop


class Trainer:
    def __init__(self, data, model, params, niter=100, verbose=False):
        self.data = data
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.epochs = params['epochs']
        self.niter = niter
        self.verbose = verbose
        self.early_stopping = params['early_stopping']
        if self.early_stopping:
            self.stop_checker = EarlyStopping(params['patience'], verbose)

        self.data.to(device)

    def train(self):
        data, model, optimizer = self.data, self.model, self.optimizer
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.labels[data.train_mask])
        loss.backward()
        optimizer.step()

    def evaluate(self):
        data, model = self.data, self.model
        model.eval()

        with torch.no_grad():
            output = model(data)

        outputs = {}
        for key in ['train', 'val', 'test']:
            if key == 'train':
                mask = data.train_mask
            elif key == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            loss = F.nll_loss(output[mask], data.labels[mask]).item()
            pred = output[mask].max(dim=1)[1]
            acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

            outputs['{}_loss'.format(key)] = loss
            outputs['{}_acc'.format(key)] = acc

        return outputs

    def reset(self):
        self.model.to(device).reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.early_stopping:
            self.stop_checker.reset()

    def print_verbose(self, epoch, evals):
        print('epoch: {: 5d}'.format(epoch),
              'train loss: {:.5f}'.format(evals['train_loss']),
              'train acc: {:.5f}'.format(evals['train_acc']),
              'val loss: {:.5f}'.format(evals['val_loss']),
              'val acc: {:.5f}'.format(evals['val_acc']))

    def run(self):
        val_acc_list = []
        test_acc_list = []

        for _ in tqdm(range(self.niter)):
            self.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in range(1, self.epochs + 1):
                self.train()
                evals = self.evaluate()

                if self.verbose:
                    self.print_verbose(epoch, evals)

                if self.early_stopping:
                    if self.stop_checker.check(evals, self.model, epoch):
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            evals = self.evaluate()
            if self.verbose:
                for met, val in evals.items():
                    print(met, val)

            val_acc_list.append(evals['val_acc'])
            test_acc_list.append(evals['test_acc'])

        print(mean(test_acc_list))
        print(std(test_acc_list))
        return {
            'val_acc': mean(val_acc_list),
            'test_acc': mean(test_acc_list),
            'test_acc_std': std(test_acc_list)
        }
