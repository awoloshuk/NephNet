#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:19:12 2019

@author: andre
"""

#python3 splitDataFolder.py -r "../data/" -c "../data/GroundTruth_052219/Label_1.csv" -s 0.1
import argparse
import os, os.path, shutil
import pathlib
import random
import pandas as pd
import numpy as np
from collections import Counter

def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]


def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_file() and f.name.endswith('.tiff')]

def getForwardProbabilty(train_list, test_list):
    train_labels = {}
    for item in train_list:
        if item: 
            key = "class_" + str(item[1])
            if not key in train_labels: train_labels[key] = 0
            train_labels[key] += 1 
    for key in train_labels.keys():
        print("forward probability for " + key + " is: " + str(train_labels[key]/(len(train_list)-1)))
    print(train_labels)
    
    
    test_labels = {}
    for item in test_list:
        if item: 
            key = "class_" + str(item[1])
            if not key in test_labels: test_labels[key] = 0
            test_labels[key] += 1 
    print("Testing class breakdown:")
    print(test_labels)

def splitTrainTest(root_dir, csv_path, split_fraction):
    folder_path = root_dir
    print(folder_path)
    
    images = list_files(folder_path) #returns all tiff files in folder regardless of csv info

    
    csv_data = pd.read_csv(csv_path, header = 0, engine='python')
    all_files = csv_data.iloc[:,0] # first column of data frame 
    
    print("Total number of images = " + str(len(all_files)))
    #random.seed(0)
    idx = list(range(len(all_files)))
    random.shuffle(idx)
    test_num = int(split_fraction*len(all_files))
    test_idx = idx[0:test_num-1]
    train_idx = idx[test_num:len(images)-1]
    
    test_list = [[]]
    train_list = [[]]
    count = 0
    for i in test_idx:
        #image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        test_list.append([filename, label])
        #edit file name 
        #add to new data list 
        
        
        folder_name = "Test"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    count = 0    
    for i in train_idx:
        #image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        train_list.append([filename, label])
        folder_name = "Train"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    print("Training images = " + str(len(train_list)))
    print("Test images = " + str(len(test_list)))
    getForwardProbabilty(train_list, test_list)
    cols = ['Filename', 'Label']
    csv_Test = pd.DataFrame(test_list, columns = cols) 
    csv_Train = pd.DataFrame(train_list, columns = cols)
    
    csv_Test.to_csv(path_or_buf = rootd+"/Test.csv", index=False)
    csv_Train.to_csv(path_or_buf = rootd+"/Train.csv", index=False)

def splitTrainTest_upsample(root_dir, csv_path, split_fraction):
    ''' Deprecated in favor of weighted random sampler
    '''
    folder_path = root_dir
    print(folder_path)
    
    images = list_files(folder_path) #returns all tiff files in folder regardless of csv info

    
    csv_data = pd.read_csv(csv_path, header = 0, engine='python')
    all_labels = csv_data.iloc[:,1] # first column of data frame 
 
    
    print("Total number of images = " + str(len(all_labels)))
    random.seed(0)
    idx = list(range(len(all_labels)))
    random.shuffle(idx)
    test_num = int(split_fraction*len(all_labels))
    test_idx = idx[0:test_num-1]
    train_idx = idx[test_num:len(images)-1]
    
    test_list = [[]]
    train_list = [[]]
    count = 0
    for i in test_idx:
        #image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        test_list.append([filename, label])
        #edit file name 
        #add to new data list 
        
        
        folder_name = "Test"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    count = 0    
    for i in train_idx:
        #image = images[i]
        filename = csv_data.iloc[i, 0]
        label = csv_data.iloc[i, 1]
        train_list.append([filename, label])
        folder_name = "Train"
        new_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print("New Folder " + folder_name + " created")
    
        old_image_path = os.path.join(folder_path, filename)
        new_image_path = os.path.join(new_path, filename)
        #shutil.move(old_image_path, new_image_path)
        count = count + 1
    
    
    
    #make a dictionary with key = label name, value = count of that label in the training dataset 
    count_labels = {}
    train_labels = train_list[:,1]
    for item in train_labels:
        if item: 
            key = "class_" + str(item[1])
            if not key in train_labels: train_labels[key] = 0
            train_labels[key] += 1 
    
    # determine biggest class in training dataset
    max_count = 0
    max_label = ""
    train_list_class = {} #value of this dictionary is a list of all filenames, label with label == key
    for key in count_labels:
        train_list_class[key] = train_list[:,1] == key
        if max_count < count_labels[key]:
            max_count = count_labels[key]
            max_label = key
    # resample other classes to match biggest class by just repeating other classes 
    # data augmentation should prevent overtraining
    for key in count_labels:
        if max_count > count_labels[key]:
            for i in range(max_count - count_labels[key]):
                train_list_class[key].append(train_list_class[key][i]) 
    
    #combine train_list_class into one big train_list
    train_list_upsampled = [[]]
    for key in train_list_class:
        for row in train_list_class[key]:
            train_list_upsampled.append(row)
        
    
    
    
    print("Training images = " + str(len(train_list_upsampled)))
    print("Test images = " + str(len(test_list)))
    getForwardProbabilty(train_list_upsampled, test_list)
    cols = ['Filename', 'Label']
    csv_Test = pd.DataFrame(test_list, columns = cols) 
    csv_Train = pd.DataFrame(train_list_upsampled, columns = cols)
    
    csv_Test.to_csv(path_or_buf = rootd+"/Test.csv", index=False)
    csv_Train.to_csv(path_or_buf = rootd+"/Train_upsampled.csv", index=False)    

def main(rootd, csv, split):
    splitTrainTest(rootd, csv, split)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--root', default=None, type=str,
                        help='root directory file path (default: None)')
    parser.add_argument('-c', '--csv2', default=None, type=str,
                        help='path to csv (default: None)')
    parser.add_argument('-s', '--split', default=0.1, type=float,
                        help='split percentage for test, e.g. 0.1')
    args = parser.parse_args()
    
    
    
    if args.root:
        rootd = args.root    
    if args.csv2:
        csv1 = args.csv2
    if args.split:
        split1 = args.split
        
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c Train/Train.csv', for example.")
    main(rootd, csv1, split1)
