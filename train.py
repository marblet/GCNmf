"""
version 1.0
date 2021/02/04
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from numpy import mean, std
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from torch.optim import Adam
from tqdm import tqdm
from utils import logger

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

    def check(self, val_loss, val_acc, model, epoch):
        if val_loss <= self.best_val_loss or val_acc >= self.best_val_acc:
            if val_loss <= self.best_val_loss and val_acc >= self.best_val_acc:
                self.state_dict = deepcopy(model.state_dict())
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.best_val_acc = max(self.best_val_acc, val_acc)
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


class NodeClsTrainer:
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader,
        model,
        params,
        runner,
        niter=100,
        verbose=False
        ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.runner = runner
        self.optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.criterion = torch.nn.BCELoss()
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.epochs = params['epochs']
        self.niter = niter
        self.verbose = verbose
        self.early_stopping = params['early_stopping']
        if self.early_stopping:
            self.stop_checker = EarlyStopping(params['patience'], verbose)

    def reset(self):
        self.model.to(device).reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.early_stopping:
            self.stop_checker.reset()

    def train(self):
        model = self.model
        total_loss = 0
        model.train()
        for data in self.train_loader:
            data = data.to(device)
            self.optimizer.zero_grad()                
            output = model(data)        
            # if output.squeeze(1).isnan().any():
            #     print('Nan value encountered in output')
            #     print(output)
            #     for name, param in model.named_parameters():
            #         print(name, torch.isfinite(param.grad).all())
                
            loss = self.criterion(output.squeeze(1), data.y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            
            torch.autograd.set_detect_anomaly(True)
            total_loss += loss.item() / data.num_graphs

        return total_loss

    def evaluate(self, loader, final_eval=False):
        model = self.model
        model.eval()
        total_loss = 0
        correct = 0
        #Initialize arrays for calculating avg_p, auroc and recall
        all_outputs = []
        all_labels=[]
        all_preds = []
        
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                output = self.model(data).squeeze(1)
                labels = data.y.to(torch.float)
                loss = self.criterion(output, labels)
                total_loss += loss.item() * data.num_graphs
                # pred = output.max(dim=1)[1]
                pred = (output >= 0.5).float()
                
                correct += pred.eq(labels).sum().item()
                all_labels.append(labels.cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                if final_eval:
                    all_outputs.append(output.cpu().numpy())
                
        total_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        
        #Recall calculations
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels).astype(np.int32)
        recall = recall_score(all_labels, all_preds)

        if final_eval:
            all_outputs = np.concatenate(all_outputs)
            average_precision = average_precision_score(all_labels, all_outputs)
            auroc = roc_auc_score(all_labels, all_outputs)
            return {
                'loss': total_loss,
                'acc': acc,
                'avg_prec': average_precision,
                'auroc': auroc,
                'recall': recall
            }
        
        return {
            'loss': total_loss,
            'acc': acc,
            'avg_prec': None,
            'auroc': None,
            'recall': recall
        }
        

    def print_verbose(self, epoch, evals):
        print('epoch: {: 5d}'.format(epoch),
              'train loss: {:.5f}'.format(evals['train_loss']),
              'train acc: {:.5f}'.format(evals['train_acc']),
              'val loss: {:.5f}'.format(evals['val_loss']),
              'val acc: {:.5f}'.format(evals['val_acc']))

    def run(self):
        val_acc_list = []
        test_acc_list = []
        
        if device == 'cuda':
            logger.info(f'Initial GPU memory allocated: {torch.cuda.memory_allocated()}')
            logger.info((
            f"CUDA device {torch.cuda.current_device()} "
            f"out of {torch.cuda.device_count()}"))

        for _ in tqdm(range(self.niter)):
            logger.info('Resetting model parameters')
            self.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in tqdm(range(1, self.epochs + 1)):
                train_loss = self.train()
                logger.info(f'Evaluation after epoch {epoch}')
                train_eval = self.evaluate(self.train_loader)
                val_eval = self.evaluate(self.val_loader)
                test_eval = self.evaluate(self.test_loader)
                
                self.runner.log({
                    "train_loss": train_eval['loss'],
                    "train_acc": train_eval['acc'],
                    "val_loss": val_eval['loss'],
                    "val_acc": val_eval['acc'],
                    "val_recall": val_eval['recall'],
                    "test_acc": test_eval['acc'],
                    "test_recall": test_eval['recall'],
                    })
                
                if self.verbose:
                    self.print_verbose(epoch, train_loss, val_eval['loss'], val_eval['acc'])

                if self.early_stopping:
                    if self.stop_checker.check(val_eval['loss'], val_eval['acc'], self.model, epoch):
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            #Final evaluation after training for a single iteration
            val_eval = self.evaluate(self.val_loader, True)
            test_eval = self.evaluate(self.test_loader, True)
            
            val_acc_list.append(val_eval['acc'])
            test_acc_list.append(test_eval['acc'])

            self.runner.log({
                "final_test_acc": test_eval['acc'],
                "val_average_prec": val_eval['avg_prec'],
                "val_auroc": val_eval['auroc'],
                "val_recall": val_eval['recall'],
                "test_average_prec": test_eval['avg_prec'],
                "test_auroc": test_eval['auroc'],
                "test_recall": test_eval['recall'],
                })
            
        avg_test_acc = mean(test_acc_list)
        std_test_acc = std(test_acc_list)
        avg_val_acc = mean(val_acc_list)
        print(f'Average Test Accuracy: {avg_test_acc}, Std Dev: {std_test_acc}')
        
        
        self.runner.log({
                    "avg_val_acc": avg_val_acc,
                    "avg_test_acc": avg_test_acc,
                    "std_test_acc": std_test_acc,
                    })
        
        fname_model = self.runner.name
        torch.save(self.model.state_dict(), "results/models/" + fname_model)
        logger.info(f"Saved PyTorch Model State to {fname_model}")
        
        return {
            'val_acc': mean(val_acc_list),
            'test_acc': avg_test_acc,
            'test_acc_std': std_test_acc
        }

def reconstruction_loss(data, output):
    adj_recon = output['adj_recon']
    return data.norm * F.binary_cross_entropy_with_logits(adj_recon, data.adjmat, pos_weight=data.pos_weight)
