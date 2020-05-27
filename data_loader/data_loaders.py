from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import databases
import os
import importlib
import torch
from utils import transforms3d as t3d
import numpy as np
import importlib
importlib.reload(t3d) #used to update changes from the 3D transforms in Jupyter


class hdf5_2d_dataloader(BaseDataLoader):
    '''
    Data loader use create a dataset from the hdf5 path, set the parameters for processing the images (including preprocessing / augmentation parameters). Note that normalization to a mean and standard deviation is variable based on the dataset
    '''
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True, projected = False):
        
        if projected:
            rs = 256
            cc = 224
        else:
            rs = 90
            cc = 64
        trsfm_train = transforms.Compose([
            transforms.Resize(rs),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, translate=(.05,.05), scale=(0.95,1.05), shear=None, resample=False, fillcolor=0),
            transforms.CenterCrop(cc),
            transforms.ToTensor(),
            transforms.Normalize(mean=[8.18], std=[19.91]), 
        ])
        trsfm_test = transforms.Compose([
            transforms.Resize(rs),
            transforms.CenterCrop(cc),
            transforms.ToTensor(),
            transforms.Normalize(mean=[8.18], std=[19.91]), 
        ])
        
        '''
        means and std for IMPRS dataset
        allDAPI_volume 141.42 18.85
        mask_avgproj 128.16, 0.54
        mask_maxproj 128.51, 1.23
        mask_sumproj 126.145, 31.49
        mask_volume 128.23, 0.84
        '''
        
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm, projection = projected)
        super(hdf5_2d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        #base data loader requires (dataset, batchsize, shuffle, validation_split, numworkers)

class hdf5_3d_dataloader(BaseDataLoader):
    '''
    3D augmentations use a random state to peform augmentation, which is organized as a List rather than torch.Compose()
    '''
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True, projected = False, mean = 15, stdev = 18.4):
        rs = np.random.RandomState()
        self.mean = mean
        self.stdev = stdev
        
        trsfm_train = [#t3d.shotNoise(rs, alpha = 0.8, execution_prob = 0.2), 
                       #t3d.Downsample(rs, factor = 2.1, order=2),
                       t3d.RandomFlip(rs),
                       t3d.RandomRotate90(rs),
                       t3d.RandomRotate(rs, angle_spectrum=25, axes=[(1,2)], mode='constant', order=1),
                       t3d.Translate(rs, pixels = 5, execution_prob = 0.2),
                       t3d.RandomContrast(rs, factor = 0.8, execution_probability=0.2), 
                       #t3d.ElasticDeformation(rs, 3, alpha=20, sigma=3, execution_probability=0.2), 
                       #t3d.GaussianNoise(rs, 3), 
                       #t3d.Normalize(mean, stdev), 
                       t3d.RangeNormalize(),
                       t3d.ToTensor(True)]
        
        trsfm_test = [#t3d.Normalize(mean, stdev),
                      t3d.RangeNormalize(),
                      t3d.ToTensor(True)]
        
        '''
        means and std for IMPRS dataset
        allDAPI_volume 141.42 18.85
        mask_avgproj 128.16, 0.54
        mask_maxproj 128.51, 1.23
        mask_sumproj 126.145, 31.49
        mask_volume 128.23, 0.84
        allDAPI_volume0701 140.61 17.60
        '''
        if training == True:
            trsfm = trsfm_train
        else:
            trsfm = trsfm_test       
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shape = shape
        importlib.reload(databases) #used to get load any recent changes from database class
        self.dataset = databases.hdf5dataset(hdf5_path, shape = self.shape, training = training, transforms=trsfm, projection = projected)
        super(hdf5_3d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class hdf5_1d_dataloader(BaseDataLoader):
    '''
    Data loader use create a dataset from the hdf5 path, set the parameters for processing the images (including preprocessing / augmentation parameters). Note that normalization to a mean and standard deviation is variable based on the dataset
    '''
    def __init__(self, hdf5_path, batch_size, shuffle=True, shape = [7,32,32], validation_split=0.0, num_workers=1, training=True):
        mean = 14.5
        stdev = 17
        rs = np.random.RandomState()
        trsfm_train = [#t3d.shotNoise(rs, alpha = 0.7, execution_prob = 0.2), 
                       #t3d.Downsample(rs, factor = 4.0, order=2),
                       #t3d.RandomFlip(rs),
                       #t3d.RandomRotate90(rs), 
                       #t3d.RandomContrast(rs, factor = 0.8, execution_probability=0.2), 
                       #t3d.ElasticDeformation(rs, 3, alpha=20, sigma=3, execution_probability=0.2), 
                       t3d.GaussianNoise(rs, 3), 
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
        self.dataset = databases.hdf5dataset1D(hdf5_path, shape = self.shape, training = training, transforms=trsfm)
        super(hdf5_1d_dataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        #base data loader requires (dataset, batchsize, shuffle, validation_split, numworkers)        
        
