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
print("Modules loaded")


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(savedDir, resume):
    torch.backends.cudnn.benchmark = True
    print("GPUs available: " + str(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setup data_loader instances

    
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to train")

    viewTraining.graphLoss(savedDir, best=False) #imported from utils
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-s', '--savedDir', default=None, type=str,
                        help='saved directory (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.savedDir:
        print("now loading training")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args.savedDir, args.resume)
