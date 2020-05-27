# These imports enhance Python2/3 compatibility.
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import data_loader.databases as module_datasets
from base import BaseDataLoader
from data_loader import databases
import os
import importlib
import torch
from utils import transforms3d as t3d
import numpy as np
import importlib
importlib.reload(t3d) #used to update changes from the 3D transforms in Jupyter

from sklearn.model_selection import StratifiedKFold
import copy
import numpy as np


def main(config):
    cv_n_folds = 4
    myfiles = []
    for i in range(cv_n_folds):
        print("=======FOLD {} / {} ========".format(i, cv_n_folds))
        filename = one_fold(config, cv_n_folds, i)
        myfiles.append(filename)
    combine_folds(config, myfiles)


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def one_fold(config, cv_n_folds, cv_fold):
    print("GPUs available: " + str(torch.cuda.device_count()))
    train_logger = Logger()
    cv_seed = 1234
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    num_classes = config['arch']['args']['num_classes']
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Initialize training dataset
    rs = np.random.RandomState()
    mean = config['data_loader']['args']['mean']
    stdev = config['data_loader']['args']['stdev']
    trsfm_train = [
                   t3d.RandomRotate90(rs), 
                   t3d.Normalize(mean, stdev), 
                   t3d.ToTensor(True)]
    train_dataset = getattr(module_datasets, 
        'hdf5dataset', config['data_loader']['args']['hdf5_path'],
        shape = config['data_loader']['args']['shape'], 
        transforms = trsfm_train,
        training = True)

    labels = [label for img, label in train_dataset]
    # Split train into train and holdout for particular cv_fold.
    kf = StratifiedKFold(n_splits = cv_n_folds, shuffle = True, random_state = cv_seed)
    cv_train_idx, cv_holdout_idx = list(kf.split(range(len(labels)), labels))[cv_fold]
    np.random.seed(cv_seed)
    # Seperate datasets        
    holdout_dataset = copy.deepcopy(train_dataset)
    holdout_dataset.imgs = [train_dataset.imgs[i] for i in cv_holdout_idx]
    holdout_dataset.samples = holdout_dataset.imgs
    # Subset of holdout used to choose the best model.
    val_dataset = copy.deepcopy(holdout_dataset)
    val_size = int(len(cv_holdout_idx) / 5)
    val_imgs_idx = np.random.choice(range(len(holdout_dataset.imgs)), size=val_size, replace=False,)
    val_dataset.imgs = [holdout_dataset.imgs[i] for i in val_imgs_idx]
    val_dataset.samples = val_dataset.imgs
    train_dataset.imgs = [train_dataset.imgs[i] for i in cv_train_idx]
    train_dataset.samples = train_dataset.imgs
    print('Train size:', len(cv_train_idx), len(train_dataset.imgs))
    print('Holdout size:', len(cv_holdout_idx), len(holdout_dataset.imgs))
    print('Val size (subset of holdout):', len(val_imgs_idx), len(val_dataset.imgs))


   #create data_loaders
    train_loader = hdf5_3d_dataloader(
        config['data_loader']['args']['hdf5_path'],
        config['data_loader']['args']['batch_size'],
        shuffle = config['data_loader']['args']['shuffle'],
        shape = config['data_loader']['args']['shape'],
        num_workers = config['data_loader']['args']['num_workers'],
        training = config['data_loader']['args']['training'],
        dataset = train_dataset,
        mean = config['data_loader']['args']['mean'],
        stdev = config['data_loader']['args']['stdev'])

    val_loader = hdf5_3d_dataloader(
        config['data_loader']['args']['hdf5_path'],
        config['data_loader']['args']['batch_size'],
        shuffle = config['data_loader']['args']['shuffle'],
        shape = config['data_loader']['args']['shape'],
        num_workers = config['data_loader']['args']['num_workers'],
        training = config['data_loader']['args']['training'],
        dataset = val_dataset,
        mean = config['data_loader']['args']['mean'],
        stdev = config['data_loader']['args']['stdev'])
    
    model = get_instance(module_arch, 'arch', config)
    train_logger = Logger()
    loss = getattr(module_loss, config['loss']) #looks in model/loss.py for criterion function specified in config
    criterion = loss(data_loader.dataset.weight.to(device)) # for imbalanced datasets
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()

    # Load model best and make predictions on holdout dataset
    holdout_loader = hdf5_3d_dataloader(
        config['data_loader']['args']['hdf5_path'],
        config['data_loader']['args']['batch_size'],
        shuffle = config['data_loader']['args']['shuffle'],
        shape = config['data_loader']['args']['shape'],
        num_workers = config['data_loader']['args']['num_workers'],
        training = config['data_loader']['args']['training'],
        dataset = holdout_dataset,
        mean = config['data_loader']['args']['mean'],
        stdev = config['data_loader']['args']['stdev'])
    
    saved_dir = trainer.checkpoint_dir
    best_model = os.path.join(saved_dir, "model_best.pth")
    print("=> loading {}".format(best_model))

    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['state_dict'])
    print("Running forward pass on holdout set of size:", len(holdout_dataset.imgs))
    probs = get_probs(holdout_loader, model)
    filename = os.path.join( saved_dir, 'model_{}__fold_{}__probs.npy'.format(config['name'], cv_fold))
    
    np.save(filename, probs)    
    return filename




def get_probs(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # switch to evaluate mode
    model.eval()
    ntotal = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / ntotal), end = "")
            if torch.cuda.is_available():
                input = input.to(device)
            target = target.to(device)

            # compute output
            outputs.append(model(input))
    
    # Prepare outputs as a single matrix
    probs = np.concatenate([
        torch.nn.functional.softmax(z, dim = 1) if not torch.cuda.is_available() else 
        torch.nn.functional.softmax(z, dim = 1).cpu().numpy() 
        for z in outputs
    ])
    
    return probs


def combine_folds(config, files):
    cv_seed = 1234
    wfn = os.path.join(config['saved_dir'], 'combined_folds_{}.npy'.format(config['name']))
    print('This method will overwrite file: {}'.format(wfn))
    print('Computing fold indices. This takes 15 seconds.')
    # Prepare labels
    rs = np.random.RandomState()
    mean = config['data_loader']['args']['mean']
    stdev = config['data_loader']['args']['stdev']
    trsfm_train = [
                   t3d.Normalize(mean, stdev), 
                   t3d.ToTensor(True)]
    train_dataset = getattr(module_datasets, 
        'hdf5dataset', config['data_loader']['args']['hdf5_path'],
        shape = config['data_loader']['args']['shape'], 
        transforms = trsfm_train,
        training = True)

    labels = [label for img, label in train_dataset]
    num_classes = config['arch']['args']['num_classes']
    # Intialize pyx array (output of trained network)
    pyx = np.empty((len(labels), num_classes))
    
    # Split train into train and holdout for each cv_fold.
    kf = StratifiedKFold(n_splits = args.cvn, shuffle = True, random_state = cv_seed)
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(range(len(labels)), labels)):
        probs = np.load(files[k]) 
        pyx[cv_holdout_idx] = probs[:, :num_classes]
    print('Writing final predicted probabilities.')
    np.save(wfn, pyx) 
    
    # Compute overall accuracy
    print('Computing Accuracy.', flush=True)
    acc = sum(np.array(labels) == np.argmax(pyx, axis = 1)) / float(len(labels))
    print('Accuracy: {:.25}'.format(acc))

class hdf5_3d_dataloader(BaseDataLoader):
    '''
    3D augmentations use a random state to peform augmentation, which is organized as a List rather than torch.Compose()
    '''
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True, dataset = None, mean = 0, stdev = 1):
        rs = np.random.RandomState()
        trsfm_train = [
                       t3d.RandomRotate90(rs), 
                       t3d.Normalize(mean, stdev), 
                       t3d.ToTensor(True)]
        
        trsfm_test = [t3d.Normalize(mean, stdev), 
                      t3d.ToTensor(True)]
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        if dataset == None:
        	self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm)
        else:
        	self.dataset = dataset
        super(hdf5_3d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




