import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import numpy as np
import torchvision
from torch.nn import functional as F
from torch import topk
import skimage.transform
from torch.optim import lr_scheduler
from tqdm import tqdm
import math
from utils import util
import pandas as pd

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def set_instance(module, name, config, *args):
    setattr(module, config[name]['type'])(*args, **config[name]['args'])
    
def main(config, resume):
    # set visualization preference
    print("GPUs available: " + str(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputOverlaycsv = True
    showHeatMap = False
    
    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader_test', config)
    
    '''
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['hdf5_path'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0,
        projected = True,
        shape = [32,32]
    )
    '''

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    print(model)
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU to test")
        
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    criterion = loss_fn(None)
    #criterion = loss_fn(data_loader.dataset.weight.to(device)) # for imbalanced datasets
    
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    
    #classes = ('endothelium', 'pct', 'vasculature')
    classes = ('S1', 'PCT', 'TAL', 'DCT', 'CD', 'cd45', 'nestin', 'cd31_glom', 'cd31_inter')
    all_pred = []
    all_pred_k = []
    all_true = []
    all_softmax = []
    
    
    if showHeatMap:
        hm_layers = {'final_layer': 'layer', 'fc_layer': 'fc_layer', 'conv_num': 17, 'fc_num': 3} #need to set based on model
        heatmapper = classActivationMap.CAMgenerator(hm_layers, config, model)
        #heatmapper = classActivationMap.CAMgenerator3d(hm_layers, config, model)  #for 3d data
        heatmapper.generateImage(num_images=10)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            k=2
            data, target = data.to(device), target.to(device)
            output = model(data)
            image = np.squeeze(data[0].cpu().data.numpy())
            label = np.squeeze(target[0].cpu().data.numpy())
            all_true.extend(target.cpu().data.numpy())
            all_pred.extend(np.argmax(output.cpu().data.numpy(), axis=1))
            mypred = torch.topk(output, k, dim=1)[1]
            all_pred_k.extend( mypred.cpu().data.numpy())
            m = torch.nn.Softmax(dim=0)
            for row in output.cpu():
                sm = m(row)
                all_softmax.append(sm.data.numpy())
                
            if i < 1:
                m = torch.nn.Softmax(dim=0)
                print("prediction percentages")
                print(m(output.cpu()[0]))
                print(all_true[i])
                plt.figure()
                plt.imshow(image[3], cmap = 'gray')
                plt.title("Label is " + classes[np.argmax(m(output.cpu()[0]))])
                plt.pause(0.1)
            
                #all_softmax.extend(m(output.cpu()))
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output.cpu(), target.cpu()) * batch_size
        
        correct = 0
        all_pred_k = np.array(all_pred_k)
        print(all_pred_k.shape)
        all_pred_k = torch.from_numpy(all_pred_k)
        all_true_k = torch.from_numpy(np.array(all_true))
        for i in range(k):
            correct += torch.sum(all_pred_k[:, i] == all_true_k).item()
        print("TOP 2 ACCURACY: {}".format(correct / len(all_true)))    
    if outputOverlaycsv:
        ids = data_loader.dataset.getIds()
        softmax = pd.DataFrame(all_softmax)
        #ids = ids[:,1].reshape(ids.shape[0], 1)
        num_test = len(all_true)
        ids = ids[:num_test]
        print(ids[0:5])
        print(ids.shape)
        print(softmax.shape)
        print(len(all_true))
        frames = [ids, softmax, pd.DataFrame(all_true)]
        output_data= np.concatenate(frames, axis=1)
        print(output_data.shape)
        output_df = pd.DataFrame(output_data)
        filename = "overlaycsv_" + config['name'] + ".csv"
        output_df.to_csv(filename, index=False,  header=False)
        
    n_samples = len(data_loader.sampler)
    print("num test images = " + str(n_samples))
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    for key in log:
        print("{} = {:.4f}".format(key, log[key]))
    #print(log)
    log['classes'] = classes
    log['test_targets'] = all_true
    log['test_predictions'] = all_pred
    print("CM will show aggregation of PCT classes")
    
    util.plot_confusion_matrix_combinePCT(all_true, all_pred, classes=classes, normalize=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-t', '--test', default=None, type=str,
                        help='path to test h5 (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.test:
        config['data_loader_test']['args']['hdf5_path'] = args.test
        config['data_loader_test']['args']['mean'] = 15.59
        config['data_loader_test']['args']['stdev'] = 15.59
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
