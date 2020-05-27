from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageEnhance
from utils import transforms3d
import h5py
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class hdf5dataset(Dataset):
    '''
    Reads an HDF5 file with the following keys: train_data, train_labels, test_data, test_labels, test_ids. Train_data is a vector which is reshped to fit the shape variable. Works for both 2D and 3D input
    '''
    def __init__(self, h5_path, shape = (7,32,32), training = True, transforms=None, projection = False):
        st = pd.HDFStore(h5_path)
        self.store = st
        if training:
            self.data = st['train_data'].values
            self.label = st['train_labels'].values.squeeze()
        else:
            self.data = st['test_data'].values
            self.label = st['test_labels'].values.squeeze()
        self.transforms = transforms
        self.data_len = self.data.shape[0]
        self.shape = shape
        self.projection = projection
        #print(type(self.data))
        # weight the classes by 1/#examples * maximum #examples, note that labels should start at 1
        # for example if the number of examples = [10, 50, 100] then self.weight = [10, 2, 1]
        count_labels = {}
        for item in self.label:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        mysort = []
        for key in count_labels:
            mysort.append(key)
        mysort.sort(key = natural_keys)
        
        for key in mysort:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
            
    def getIds(self):
        #get test serial ID for mapping back to the original image. 
        #NOTE: IDs should NOT be shuffled in order to keep ground truth label with the correct ID
        ids = self.store['test_ids'].values
        return ids
    
    def __getitem__(self, index):
        num_pixels = 1
        for dim in self.shape: num_pixels = num_pixels*dim
        #print(len(self.data))
        #print(num_pixels)
        #print(type(self.data))
        img = self.data[index, 0:num_pixels]
        img = np.reshape(img, self.shape, order = 'C') #last index of shape changes fastest
        img = img.astype(float)
        img = np.float32(img)
        #print(str(np.amin(img)))
        '''
        if self.projection:
            img = np.amax(img, axis = 0)
        '''
        label = self.label[index] - 1 #labeling starts at 0 for CNN
        
        
        # Perform augmentation. The data loader differentiates between training and test transformations, this simply sends the batch to receive the transformations
        if self.transforms is not None:
            if len(self.shape) > 2 and not self.projection: #3D input
                for transform in self.transforms:
                    img = transform(img)
                img_as_tensor = img
                #img_as_tensor = img_as_tensor.type(torch.FloatTensor)
            elif self.projection:
                img_as_img = Image.fromarray(img) #convert to PIL
                img_as_img = img_as_img.resize((60,60), resample=Image.NEAREST)
                img_as_img = img_as_img.resize((100,100), resample=Image.NEAREST)
                img_as_img = img_as_img.resize((256,256), resample=Image.NEAREST)
                rgbimg = Image.new("RGB", img_as_img.size, color=(0,0,0))
                rgbimg.paste(img_as_img)
                values =  list(rgbimg.getdata())
                new_image= rgbimg.point(lambda argument: argument*1)
                img_as_tensor = self.transforms(new_image)
            else: #2D input
                image = img.astype(float)
                img_as_img = Image.fromarray(image)
                img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label

    def __len__(self):
        return self.data_len

class templateDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.csv_data = pd.read_csv(csv_path, header = 0)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.csv_data)
        
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.csv_data.iloc[index, 7]) #path data is in the 7th column
        image = io.imread(img_name)
        img_as_img = Image.fromarray(image) #convert to PIL
        label = self.csv_data.iloc[index, 6]  #label is in the 6th column
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len


class mnistDataset(Dataset):
    def __init__(self, csv_path, root_dir, transforms=None):
        self.data = torch.load(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_len = len(self.data[1])
        
    def __getitem__(self, index):
        img = self.data[0][index]
        trans = transforms.ToPILImage()
        img_as_img = trans(img)
        img_as_img = img_as_img.resize((60,60), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((100,100), resample=Image.NEAREST)
        #img_as_img = img_as_img.resize((150,150), resample=Image.NEAREST)
        img_as_img = img_as_img.resize((256,256), resample=Image.NEAREST)
        rgbimg = Image.new("RGB", img_as_img.size, color=(0,0,0))
        rgbimg.paste(img_as_img)
        values =  list(rgbimg.getdata())
        new_image= rgbimg.point(lambda argument: argument*1)
        label = self.data[1][index]
        if self.transforms is not None:
            img_as_tensor = self.transforms(new_image)
        return img_as_tensor, label
        
    
    def __len__(self):
        return self.data_len

class hdf5dataset2D(Dataset):
    '''
    Reads an HDF5 file with the following keys: train_data, train_labels, test_data, test_labels, test_ids. Train_data is a vector which is reshped to fit the shape variable. Works for both 2D and 3D input
    '''
    def __init__(self, h5_path, shape = (7,32,32), training = True, transforms=None):
        st = pd.HDFStore(h5_path)
        self.store = st
        if training:
            self.data = st['train_data'].values
            self.label = st['train_labels'].values
        else:
            self.data = st['test_data'].values
            self.label = st['test_labels'].values
        self.transforms = transforms
        self.data_len = self.data.shape[0]
        self.shape = shape
        
        # weight the classes by 1/#examples * maximum #examples, note that labels should start at 1
        # for example if the number of examples = [10, 50, 100] then self.weight = [10, 2, 1]
       # weight the classes by 1/#examples * maximum #examples, note that labels should start at 1
        # for example if the number of examples = [10, 50, 100] then self.weight = [10, 2, 1]
        count_labels = {}
        for item in self.label:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        mysort = []
        for key in count_labels:
            mysort.append(key)
        mysort.sort(key = natural_keys)
        
        for key in mysort:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
            
    def getIds(self):
        #get test serial ID for mapping back to the original image. 
        #NOTE: IDs should NOT be shuffled in order to keep ground truth label with the correct ID
        ids = self.store['test_ids'].values
        return ids
    
    def __getitem__(self, index):
        num_pixels = 1
        for dim in self.shape: num_pixels = num_pixels*dim
        img = self.data[index, 0:num_pixels]
        img = np.reshape(img, self.shape, order = 'C') #last index of shape changes fastest
        img = img.astype(float)
        label = self.label[index] - 1 #labeling starts at 0 for CNN
        
        # Perform augmentation. The data loader differentiates between training and test transformations, this simply sends the batch to receive the transformations
        if self.transforms is not None:
            if len(self.shape) > 2: #3D input
                for transform in self.transforms:
                    img = transform(img)
                img_as_tensor = img
                img_as_tensor = img_as_tensor.type(torch.FloatTensor)
            else: #2D input
                image = img.astype(float)
                img_as_img = Image.fromarray(image)
                img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, label
    
class hdf5dataset1D(Dataset):
    '''
    Reads an HDF5 file with the following keys: train_data, train_labels, test_data, test_labels, test_ids. Train_data is a vector which is reshped to fit the shape variable. flattens the input to 1xnum_pixels
    '''
    def __init__(self, h5_path, shape = (7,32,32), training = True, transforms=None):
        st = pd.HDFStore(h5_path)
        self.store = st
        if training:
            self.data = st['train_data'].values
            self.label = st['train_labels'].values
        else:
            self.data = st['test_data'].values
            self.label = st['test_labels'].values
        self.transforms = transforms
        self.data_len = self.data.shape[0]
        self.shape = shape
        
        # weight the classes by 1/#examples * maximum #examples, note that labels should start at 1
        # for example if the number of examples = [10, 50, 100] then self.weight = [10, 2, 1]
        count_labels = {}
        for item in self.label:
            if item: 
                key = "class_" + str(item)
                if not key in count_labels: count_labels[key] = 0
                count_labels[key] += 1 
        weights = []
        weights2 = []
        mysort = []
        for key in count_labels:
            mysort.append(key)
        mysort.sort(key = natural_keys)
        
        for key in mysort:
            weights.append(1. / count_labels[key])
            weights2.append(count_labels[key])
        
        weightsnp = np.asarray(weights)
        weights2np = np.asarray(weights2)
        maxnum = np.amax(weights2np)
        weightsnp = weightsnp*maxnum
        self.weight = torch.FloatTensor(weightsnp)
            
    def getIds(self):
        #get test serial ID for mapping back to the original image. 
        #NOTE: IDs should NOT be shuffled in order to keep ground truth label with the correct ID
        ids = self.store['test_ids'].values
        return ids
    
    def __getitem__(self, index):
        num_pixels = 1
        for dim in self.shape: num_pixels = num_pixels*dim
        img = self.data[index, 0:num_pixels]
        img = np.reshape(img, self.shape, order = 'C') #last index of shape changes fastest
        img = img.astype(float)
        img = np.float32(img)
        #print(str(np.amin(img)))
        '''
        if self.projection:
            img = np.amax(img, axis = 0)
        '''
        label = self.label[index] - 1 #labeling starts at 0 for CNN
        
        # Perform augmentation. The data loader differentiates between training and test transformations, this simply sends the batch to receive the transformations
        if self.transforms is not None:
            for transform in self.transforms:
                img = transform(img)
            img_as_tensor = img
            #img_as_tensor = img_as_tensor.type(torch.FloatTensor)
        return img_as_tensor, label
                
    def __len__(self):
        return self.data_len