from __future__ import print_function, with_statement, division
import copy
import os
import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import importlib
import math
import torchvision
from torch.nn import functional as F
from torch import topk
import skimage.transform
import data_loader.data_loaders as module_data
import numpy as np




class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()
        
def getCAM(feature_conv, weight_fc, class_idx):
    print(feature_conv.shape)
    print(np.squeeze(weight_fc[class_idx]).shape)
    _, nc, h, w = feature_conv.shape
    cam = np.squeeze(weight_fc[class_idx]).dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]    

class CAMgenerator():
    def __init__(self, hm_layers, config, model ):
        '''
        Takes the model and feeds in an image, tracks the CNN layer activation and maps the last convolution to the original input. Used to visualize where to model is "looking" in the image. Note that the last activation is usually much smaller (after multiple pooling layers), and therefore the resize operation can have artificts. Works for 2D images. 
        
        hm_layers = {final_layer: string, conv_num: (int)layer of last convolution, fc_layer: string}
        num_images = number of images to show
        
        '''
        
        data_loader = getattr(module_data, config['data_loader_test']['type'])(
            config['data_loader_test']['args']['csv_path'],
            config['data_loader_test']['args']['data_dir'],
            batch_size=1,
            shuffle=True,
            validation_split=0.0,
            training=False,
            num_workers=2)
    
        self.data_loader = data_loader
        
        self.final_layer = model._modules.get(hm_layers['final_layer'])
        self.fc_layer = hm_layers['fc_layer']
        self.fc_num = hm_layers['fc_num']
        self.model = model.eval()
        self.activated_features = SaveFeatures(self.final_layer[hm_layers['conv_num']])
        
        
    
    def generateImage(self, num_images = 5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.enable_grad(): #speed up calculations, unable to perform back propogation
            for i, (data, target) in enumerate(tqdm(self.data_loader)): #tqdm is a progress bar
                if i < num_images:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    image = np.squeeze(data[0].cpu().data.numpy())
                    self.activated_features.remove()
                    pred_probabilities = F.softmax(output.cpu(), dim=1).data.squeeze()
                    class_idx = topk(pred_probabilities,1)[1].int()
                    weight_softmax_params = list(self.model._modules.get(self.fc_layer)[self.fc_num].parameters())
                    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                    overlay = getCAM(self.activated_features.features, weight_softmax, class_idx )
                    plt.title("Prediction = " + str(class_idx[0].data.numpy()) + \
                              " | Actual = " + str(target[0].cpu().data.numpy()) )
                    plt.imshow(image, cmap="gray")
                    plt.imshow(skimage.transform.resize(overlay[0], image.shape), alpha=0.4, cmap='RdBu') #hot is good
                    plt.pause(.1)
                    
                    
def getCAM3d(feature_conv, weight_fc, class_idx):
    feature_conv = feature_conv[0]
    new_features = np.empty((feature_conv.shape[0]//2, feature_conv.shape[1], feature_conv.shape[2], feature_conv.shape[3]))
    for i in range(feature_conv.shape[0]//2):
        new_features[i] = (feature_conv[i*2, 0 ,:,:] + feature_conv[i*2+1, 0 ,:,:])/2
    nc, _, h, w = new_features.shape
    cam = np.squeeze(weight_fc[class_idx]).dot(np.squeeze(new_features).reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]  

class CAMgenerator3d():
    def __init__(self, hm_layers, config, model ):
        '''
        Takes the model and feeds in an image, tracks the CNN layer activation and maps the last convolution to the original input. Used to visualize where to model is "looking" in the image. Note that the last activation is usually much smaller (after multiple pooling layers), and therefore the resize operation can have artificts. Works for 3D images. 
        
        hm_layers = {final_layer: string, conv_num: (int)layer of last convolution, fc_layer: string}
        num_images = number of images to show
        
        '''
        
        data_loader = getattr(module_data, config['data_loader_test']['type'])(
            config['data_loader_test']['args']['hdf5_path'],
            batch_size=1,
            shuffle=True,
            validation_split=0.0,
            training=False,
            num_workers=2)
    
        self.data_loader = data_loader
        
        #print(hm_layers['final_layer'])
        self.final_layer = model._modules.get(hm_layers['final_layer'])
        self.fc_layer = hm_layers['fc_layer']
        self.fc_num = hm_layers['fc_num']
        self.model = model.eval()
        if hm_layers['conv_num'] != 0: 
            self.activated_features = SaveFeatures(self.final_layer[hm_layers['conv_num']])
        else:
            self.activated_features = SaveFeatures(self.final_layer)
        
        
    
    def generateImage(self, num_images = 5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.enable_grad(): #speed up calculations, unable to perform back propogation
            for i, (data, target) in enumerate(tqdm(self.data_loader)): #tqdm is a progress bar
                if i < num_images:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    image = np.squeeze(data[0].cpu().data.numpy())
                    num_slices, _, _ = image.shape
                    self.activated_features.remove()
                    pred_probabilities = F.softmax(output.cpu(), dim=1).data.squeeze()
                    class_idx = topk(pred_probabilities,1)[1].int()
                    if self.fc_num != 0: 
                        weight_softmax_params = list(self.model._modules.get(self.fc_layer)[self.fc_num].parameters())
                    else:
                        weight_softmax_params = list(self.model._modules.get(self.fc_layer).parameters())
                    
                    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                    overlay = getCAM3d(self.activated_features.features, weight_softmax, class_idx )
                    plt.title("Prediction = " + str(class_idx[0].data.numpy()) + \
                              " | Actual = " + str(target[0].cpu().data.numpy()) )
                    plt.imshow(image[num_slices//2], cmap="gray") #shows the middle slice of the volume
                    plt.imshow(skimage.transform.resize(overlay[0], image[num_slices//2].shape), alpha=0.4, cmap='RdBu') 
                    plt.pause(.1)