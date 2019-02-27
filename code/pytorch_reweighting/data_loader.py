# Note: a lot of code referenced from https://github.com/danieltan07/learning-to-reweight-examples
# as well as https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math

from constants import *

class ImbalancedData(): 
    def __init__(self, dataset_name=MNIST, n_items = 5000, classes=[9, 4], weights=[0.9, 0.1], 
            n_val_per_class=5, random_seed=1, mode="train", train_type="reweight"):
        self.dataset_name = dataset_name
        self.n_items = n_items
        self.classes = classes
        self.weights = weights
        self.n_val_per_class = n_val_per_class
        self.mode = mode
        self.train_type = train_type

        n_class = self.get_dataset(random_seed)
        self.build_data_sampler(n_class)

    def get_dataset(self, random_seed):
        '''
            Get corresponding dataset
        '''
        np.random.seed(random_seed)

        # Define dataset and corresponding transform function
        if self.dataset_name == MNIST:
            self.dataset = datasets.MNIST

            self.transform = transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == CIFAR:
            self.dataset = datasets.CIFAR10

            self.transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        if self.mode == "train":
            self.dataset = self.dataset(DATA_DIR, train=True, download=True, transform=self.transform)
        else:
            self.dataset = self.dataset(DATA_DIR, train=False, download=True, transform=self.transform)
            
            self.weights = [1.0] * len(self.classes)
            n_val_per_class = 0

        self.weights = np.divide(self.weights, np.sum(self.weights))

        n_class = list(map(int, np.floor(self.n_items * self.weights)))
        return n_class


    def build_data_sampler(self, n_class):
        '''
            Build the data samplers for train/val/test data
        '''
        data_indices = np.array([])
        val_indices = np.array([])

        data_source = self.dataset

        if type(data_source[0][1]) is int:
            labels = np.array([datum[1] for datum in data_source])
        else:
            labels = np.array([datum[1].item() for datum in data_source])

        # Identify indices of datasets per class
        for ind, c in enumerate(self.classes):
            tmp_idx = np.where(labels == c)[0]
            np.random.shuffle(tmp_idx)

            # if not reweighting, to be fair, we add validation items to training data
            if self.train_type == 'reweight':
                data_indices = np.concatenate((data_indices, tmp_idx[:n_class[ind]]))

                if self.mode == 'train':
                    val_indices = np.concatenate((val_indices, tmp_idx[n_class[ind]:n_class[ind] + self.n_val_per_class]))    
            else:
                data_indices = np.concatenate((data_indices, tmp_idx[:n_class[ind] + self.n_val_per_class]))

                        
        # create loader for train/test data
        self.data_sampler = SubsetRandomSampler(data_indices.astype(int))

        print(f"Gathered {len(data_indices)} {self.mode} samples")

        # create loader for validation data
        if self.mode == "train":
            print(f"Gathered {len(val_indices)} validation samples")
            self.val_sampler = SubsetRandomSampler(val_indices.astype(int))


def get_data_loader(dataset_name, batch_size, classes=[9, 4], n_items=5000, weights=[0.9, 0.1], 
        n_val_per_class=5, mode='train', train_type="reweight"):
    """Build and return data loader."""

    wrapper = ImbalancedData(dataset_name=dataset_name, classes=classes, n_items=n_items, weights=weights, 
        n_val_per_class=n_val_per_class, mode=mode, train_type=train_type)
    
    data_loader = DataLoader(
        wrapper.dataset, batch_size=batch_size, sampler=wrapper.data_sampler
    )

    val_loader = None if mode != 'train' else DataLoader(
        wrapper.dataset, batch_size=n_val_per_class * len(classes), sampler=wrapper.val_sampler
    )     

    return (data_loader, val_loader)
