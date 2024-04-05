import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

          
class dataset_elast_inf(Dataset):
    def __init__(self, res, res2=8):
        self.path = "C:/Users/gaynuliy/OneDrive/ModularOptimization/ModularOptimization/Truss/data/"
        if res==5:
            self.path += f'elast_{res2}_5x5/'
            N = 5000
        elif res==6:
            self.path += f'elast_{res2}_6x6/'
            N = 40000
        elif res==10:
            self.path += f'elast_{res2}_10x10/'
            N = 14400
        elif res==15:
            self.path += f'elast_{res2}_15x15/'
            N = 1000
        elif res==20:
            self.path += f'elast_{res2}_20x20/'
            N = 1000
        self.data = []
        for i in range(N):
            self.data.append(self.path + f'{i}.pickle')       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as handle:
            data = pickle.load(handle)
        layout = data['layout'].type(torch.float32)
        K = data['K'].type(torch.float32)
        C = data['C'].type(torch.float32)
        zero_map = data['zero_map']
        f = data['F'].type(torch.float32)
        DBC = data['DBC'].type(torch.float32)
        return layout, K, C, zero_map, DBC, f


class dataset_3(Dataset):
    def __init__(self, res):
        self.path = "C:/Users/gaynuliy/OneDrive/ModularOptimization/ModularOptimization/Truss/data/"
        if res==6:
            self.path += f'elast_3_6x6/'
            N = 10000
        elif res==10:
            self.path += f'elast_3_10x10/'
            N = 1000
        self.data = []
        for i in range(N, 20000):
            self.data.append(self.path + f'{i}.pickle')       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as handle:
            data = pickle.load(handle)
        layout = data['layout'].type(torch.float32).view(1, data['layout'].shape[0], data['layout'].shape[1])
        K = data['K'].type(torch.float32)
        C = data['C'].type(torch.float32)
        zero_map = data['zero_map']
        f = data['F'].type(torch.float32)
        DBC = data['DBC'].type(torch.float32)
        return layout, K, C, zero_map, DBC, f


class dataset_opt(Dataset):
    def __init__(self):
        self.path = "optimization data/"
        N = 13
        self.data = []
        for i in range(N):
            self.data.append(self.path + f'{i}.pickle')       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], 'rb') as handle:
            data = pickle.load(handle)
        layout = torch.stack([data['layout'], data['layout']], 0).type(torch.float32)
        support = data['support'].type(torch.float32)
        zero_map = data['zero_map']
        f = data['F'].type(torch.float32)
        force = data['force'].type(torch.float32)
        DBC = data['DBC'].type(torch.float32)
        return layout, support, force, zero_map, DBC, f


def get_loaders(data, batch_size):
    n_train = int(0.8 * data.__len__())
    n_test = (data.__len__() - n_train) // 2
    n_val = data.__len__() - n_train - n_test
    torch.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(data, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    return loaders


