import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import SGD, Adam
from torch.autograd import Variable
import sys
sys.path.append("../../CellTemplate")
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_loader import data_loaders as module_data
from model import loss as module_loss
from model import metric as module_metric
from model import model as module_arch
from trainer import Trainer
from utils import Logger
from utils import util
from utils import torchsummary
from utils import viewTraining
from utils import lr_finder
import logging
from utils import classActivationMap

'''
args
- epoch_scale
- max iter
- eta
- num_gpu


params = {
    "num_feature": [4, 8, 16,32],
    "batch_size": [4, 8, 16, 32, 64, 128, 256],
    #"optim_type": ['adam', 'sgd'],
    "lr": [0.00001, 0.1],
    "weight_decay": [0.00001, 0.01],
    "step_size": [20, 150],
    "gamma": [0.05, 0.5]
}
'''

class HyperOptim():
    def __init__(self, args, params, config):
        self.args = args
        self.params = params
        self.config = config
        logging.basicConfig(format='%(asctime)s - %(message)s', filename = 'optimization.log', filemode = 'w', level=logging.INFO)
        logger = logging.getLogger(__name__)
        self.logger = logger
        # hyperband params
        self.num_gpu = args.num_gpu
        self.epoch_scale = args.epoch_scale
        self.max_iter = args.max_iter
        self.eta = args.eta
        self.s_max = int(np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter

        print(
            "[*] max_iter: {}, eta: {}, B: {}".format(
                self.max_iter, self.eta, self.B
            )
        )
        logger.info("[*] max_iter: {}, eta: {}, B: {}".format(
                self.max_iter, self.eta, self.B
            ))
        # device
        self.device = torch.device("cuda" if self.num_gpu > 0 else "cpu")
        
    def get_random_config(self):
        rand_params = {}
        for key in self.params:
            value = self.params[key]
            if len(value) > 2:
                ran = np.random.randint(0, high = len(value)-1) #pick a random element from a list
                rand_params[key] = value[ran]
            else:
                ranf = np.random.uniform(low = self.params[key][0], high = self.params[key][1]) #pick a random number from a range
                rand_params[key] = ranf
        return rand_params        
        


    def tune(self, skipLast, earlyStop=False):
        """
        Tune the hyperparameters of the pytorch model
        using Hyperband.
        """
        best_configs = []
        results = {}
        list_s = reversed(range(self.s_max+1))
        if earlyStop:
            skipLast = 0
            list_s = [self.s_max]
        # finite horizon outerloop
        
        for s in list_s:
            # initial number of configs
            n = int(
                np.ceil(
                    int(self.B / self.max_iter / (s + 1)) * self.eta ** s
                )
            )
            # initial number of iterations to run the n configs for
            r = self.max_iter * self.eta ** (-s)          

            # finite horizon SH with (n, r)
            T = [self.get_random_config() for i in range(n)]

            tqdm.write("s: {}".format(s))

            for i in range(s + 1 - int(skipLast)): #remove +1 to get rid of last iteration (e.g. 1 model trained for max iterations)
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))

                tqdm.write(
                    "[*] {}/{} - running {} configs for {} iters each".format(
                        i+1, s+1, len(T), r_i)
                )
                self.logger.info("[*] {}/{} - running {} configs for {} iters each".format(
                        i+1, s+1, len(T), r_i)
                )
                
                # Todo: add condition for all models early stopping

                # run each of the n_i configs for r_i iterations
                val_losses = []
                with tqdm(total=len(T)) as pbar:
                    for t in T:
                        val_loss = self.run_config(t, r_i)
                        val_losses.append(val_loss)
                        pbar.update(1)

                # remove early stopped configs and keep the best n_i / eta
                if i < s - 1:
                    sort_loss_idx = np.argsort(
                        val_losses
                    )[0:int(n_i / self.eta)]
                    #T = [T[k] for k in sort_loss_idx if not T[k].early_stopped]
                    T = [T[k] for k in sort_loss_idx]
                    tqdm.write("Left with: {}".format(len(T)))

            best_idx = np.argmin(val_losses)
            # i could get the parameters from T[idx].params, then use that to predict what the starting values should be for S
            best_configs.append([T[best_idx], val_losses[best_idx]])

        best_idx = np.argmin([b[1] for b in best_configs])
        best_model = best_configs[best_idx]
        results["val_loss"] = best_model[1]
        results["params"] = best_model #best_model[0].params
        results["str"] = best_model[0].__str__()
        self.logger.info(results)
        return results
    
    
    def run_config(self, t, r_i):# Initialize random trainer
        rand_params = t
        print(t)
        train_logger = Logger()
        data_loader = getattr(module_data, self.config['data_loader']['type'])(
            self.config['data_loader']['args']['hdf5_path'],
            batch_size= rand_params["batch_size"],
            shape = self.config['data_loader']['args']['shape'],
            shuffle=True,
            validation_split=self.config['data_loader']['args']['validation_split'],
            training=True,
            num_workers=self.config['data_loader']['args']['num_workers']
        )
        valid_data_loader = data_loader.split_validation()
        model = getattr(module_arch, self.config['arch']['type'])(
                        num_classes = self.config['arch']['args']['num_classes'],
                        num_feature = rand_params["num_feature"])
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        loss = getattr(module_loss, self.config['loss']) #looks in model/loss.py for criterion function specified in config
        criterion = loss(data_loader.dataset.weight.to(self.device)) # for imbalanced datasets
        
        optimizer = getattr(torch.optim, self.config['optimizer']['type'])(trainable_params, lr = self.config['optimizer']['args']['lr'] * rand_params["batch_size"] / 64, momentum = rand_params["momentum"], weight_decay = rand_params["weight_decay"], nesterov= True)
        
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['type'])(optimizer, step_size = rand_params["step_size"], gamma = rand_params["gamma"])
        metrics = [getattr(module_metric, met) for met in self.config['metrics']]\
        
        
        #not sure if i can use this bc num epochs is defined in config
        trainer = Trainer(model, criterion, metrics, optimizer,
                          resume=False,
                          config=self.config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          train_logger=train_logger)
        
        '''
        mutated.trainer = trainer
        
        mutated.model = model
        mutated.data_loader = data_loader
        mutated.val_data_loader = valid_data_loader
        mutated.criterion = criterion
        mutated.metrics = metrics
        mutated.optimizer = optimizer
        mutated.lr_scheduler = lr_scheduler
        '''
        del data_loader, valid_data_loader,  #clear memory
        trainer.epochs = r_i #adjust number of epochs accordingly
        trainer.train()
        val_loss = trainer.val_loss
        return val_loss 
        
        