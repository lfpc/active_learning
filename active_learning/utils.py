import numpy as np
import torch
import NN_utils
from uncertainty import comparison as unc
from torch.utils.data import DataLoader, Subset
import copy
import torchvision.transforms as transforms


def certain_unlabeled(data,uncertainty_method, n):
    unc.get_most_certain(data,uncertainty_method, n)
    uncertainty = uncertainty_method(data)
    #unc = uncertainty.comparison.()
    unc = torch.argsort(uncertainty, descending = True)
    return idx

def uncertain_unlabeled(data,uncertainty_method, n):
    if 0<n<1:
        n = int(n*len(data))
    uncertainty = uncertainty_method(data)
    unc = torch.argsort(uncertainty, descending = False)
    return unc[0:n]

def dataloader_from_idx(complete_train_dataset,idx, batch_size = 100, train = True):
    dataset = Subset(complete_train_dataset,idx)
    if not train:
        dataset = copy.deepcopy(dataset)
        dataset.dataset.transform =transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = train)
    return dataloader
