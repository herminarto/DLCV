# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:28:06 2018

@author: herminarto.nugroho
"""

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_interval = 100

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
#transformations = transforms.Compose([transforms.ToTensor()])

#fire_datasets =  CustomDatasetFromImages('../data/fire_labels.csv')
        
#fire_dataset_loader = torch.utils.data.DataLoader(dataset=fire_datasets,
#                                                    batch_size=100,
#                                                    shuffle=False)

def CustomSplitLoader(datasets, batch_size, train_percentage, test_percentage, valid_percentage):
    # Split the datasets into training, testing, and validation
    num_train = len(datasets)
    indices = list(range(num_train))
    split_test = int(np.floor(train_percentage/100 * num_train))
    split_valid = int(np.floor(valid_percentage/100 * num_train))
    train_indices, test_idx = indices[split_test:], indices[:split_test]
    train_idx, valid_idx = train_indices[split_valid:], train_indices[:split_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # create train, validation and test loader
    train_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=train_sampler, num_workers=4, pin_memory=False,)
    test_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=test_sampler, num_workers=4, pin_memory=False,)
    valid_loader = torch.utils.data.DataLoader(datasets, batch_size,
                    sampler=valid_sampler, num_workers=4, pin_memory=False,)
    
    return train_loader, test_loader, valid_loader
    