import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
import re

'''
Reads the files in a saved directory and shows the training / validation accuracy and loss for every epoch. 
'''

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def graphLoss(saved_dir, best=True):
    config_saved_filename = os.path.join(saved_dir, "config.json")
    losses = []
    val_losses = []
    metric = []
    val_metric = []
    checkpoints = []
    
    for filename in os.listdir(saved_dir):
        name = os.path.join(saved_dir,filename) 
        if os.path.isfile(name) and name.endswith(".pth"):
            checkpoints.append(filename)
        else:
            print("skipped file: " + filename)
    
        
    file_name = os.path.join(saved_dir, "model_best.pth")
    if not best: 
        checkpoints.sort(key = natural_keys)
        file_name = os.path.join(saved_dir,checkpoints[-2])
        
    checkpoint = torch.load(file_name)
    logger = checkpoint['logger']
    for item in logger.entries:
        losses.append(logger.entries[item]['loss'])
        val_losses.append(logger.entries[item]['val_loss'])
        metric.append(logger.entries[item]['balanced_accuracy'])
        val_metric.append(logger.entries[item]['val_balanced_accuracy'])
    fig1 = plt.figure() 
    plt.plot(range(len(losses)), losses, 'r--', range(len(losses)), val_losses, 'b--')
    plt.title('Training and validation loss - red and blue respectively')
    plt.ylim(0,2.0)
    plt.show()
    
    fig2 = plt.figure()
    plt.plot(range(len(metric)), metric, 'r--', range(len(metric)), val_metric, 'b--')
    plt.title('Training and validation accuracy - red and blue respectively')
    plt.show()
    return {'loss': losses, 'val_loss': val_losses}
    
    
    