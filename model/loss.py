import torch.nn.functional as F
import torch.nn as nn

'''
Functions return an instance of a loss function with the given weights. In order to use the function, it must be 1) called from the config loss = getattr(module_loss, config['loss']) and then 2) supplied the dataset weights criterion = loss(data_loader.dataset.weight.to(device)). Criterion is then fed to the trainer

New loss functions can be implemented, simply define the function with the respective torch.nn class
'''

def nll_loss(weights):
    # model must have softmax layer 
    return nn.NLLLoss(weight = weights)

def cross_entropy_loss(weights):
    return nn.CrossEntropyLoss(weight = weights)

def bce_loss(output, target):
    lossF = nn.BCELoss()
    return lossF(output,target)
