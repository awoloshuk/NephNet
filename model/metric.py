import torch
import sklearn.metrics as skm
import torch.nn as nn

'''
Define new metrics for training here. While all metrics in the config are tracked, only the monitor metric is used for determining the model's best performance
'''

def MSE(recon_x, x):
    reconstruction_function = nn.MSELoss(size_average=False)
    return reconstruction_function(recon_x, x).cpu().detach().numpy()

def accuracy(output, target): #accuracy
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def topKaccuracy(output, target, k=3): #top 3 accuracy
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1_c = 0
        f1_c = skm.f1_score(target.detach().numpy(), pred, average = 'weighted')
    return f1_c

def roc_auc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        roc = 0
        roc = skm.roc_auc_score(target.detach().numpy(), pred)
    return roc

def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1_c = 0
        f1_c = skm.recall_score(target.detach().numpy(), pred, average = 'weighted')
    return f1_c

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1_c = 0
        f1_c = skm.precision_score(target.detach().numpy(), pred, average = 'weighted')
    return f1_c

def balanced_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1_c = 0
        f1_c = skm.balanced_accuracy_score(target.detach().numpy(), pred)
    return f1_c
     