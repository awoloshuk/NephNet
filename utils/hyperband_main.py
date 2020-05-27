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
from utils import classActivationMap
import importlib
import math
import torchvision
from torch.nn import functional as F
from torch import topk
import skimage.transform
import jupyter
from IPython import display
from ipywidgets import *
import hyperband as HypOpt
import argparse
print("Modules loaded")


importlib.reload(module_data) #load recent changes to data_loaders.py
importlib.reload(module_arch)
importlib.reload(module_loss)
importlib.reload(module_metric)
importlib.reload(util)
importlib.reload(viewTraining)
importlib.reload(lr_finder)
importlib.reload(classActivationMap)
importlib.reload(HypOpt)
print("Reload complete")

print("GPUs available: " + str(torch.cuda.device_count()))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


params = {
    "num_feature": [4, 8, 16,32],
    "batch_size": [4, 8, 16, 32, 64, 128, 256],
    #"optim_type": ['adam', 'sgd'],
    "lr": [0.00001, 0.1],
    "weight_decay": [0.00001, 0.01],
    "step_size": [20, 150],
    "gamma": [0.05, 0.5]
}

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

config_file = '../../CellTemplate/configs/config_hdf5_hypop.json'

# load config file
with open(config_file) as handle:
    config = json.load(handle)
# setting path to save trained models and log files
path = os.path.join(config['trainer']['save_dir'], config['name'])
print("loaded")

args = argparse.ArgumentParser()
args.epoch_scale = 1
args.max_iter = 100
args.eta = 10
args.num_gpu = 1

ho = HypOpt.HyperOptim(args, params, config)

ho.tune()