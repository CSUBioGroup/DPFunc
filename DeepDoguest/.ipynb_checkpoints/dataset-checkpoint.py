from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
from collections import defaultdict
from tqdm.auto import trange, tqdm

class Dataset(TensorDataset):    
    def __init__(self, feature, label):
        assert feature.size(0) == label.size(0)
        self.feature = feature
        self.label = label

    def __getitem__(self, index): 
        feature = self.feature[index]
        label = self.label[index]
        
        return (feature, label)

    def __len__(self):
        return self.feature.size(0)

class Dataset_graph_interpro(TensorDataset):    
    def __init__(self, label):
        assert feature.size(0) == label.size(0)
        self.feature = feature
        self.label = label

    def __getitem__(self, index): 
        feature = self.feature[index]
        label = self.label[index]
        
        return (feature, label)

    def __len__(self):
        return self.feature.size(0)