import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger
from utils import util
from utils import torchsummary
from utils import viewTraining
from utils import lr_finder
from utils import classActivationMap
import importlib
import math
import torchvision
from torch.nn import functional as F
from torch import topk
import skimage.transform
from utils import hyperband as HypOpt
import pickle
print("Modules loaded")


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    print("GPUs available: " + str(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to train")
    
    params = {
        "num_feature": [4, 8,16,32,64],
        "batch_size": [4, 8, 16, 32, 64, 128, 256],
        "momentum": [0.8, 0.85, 0.9, 0.95],
        #"optim_type": ['adam', 'sgd'],
        #"lr": [0.00001, 0.1],
        "weight_decay": [0.00001, 0.01],
        "step_size": [1, 20],
        "gamma": [0.05, 0.5]
    }
    args = argparse.ArgumentParser()
    args.epoch_scale = 1
    args.max_iter = 27
    args.eta = 3
    args.num_gpu = 1
    
    ho = HypOpt.HyperOptim(args, params, config)
    results = ho.tune(1, earlyStop = True)
    print(results)
    pickle_out = open("optimization.txt", "w")
    pickle_out.write(str(results))
    pickle_out.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
        
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
