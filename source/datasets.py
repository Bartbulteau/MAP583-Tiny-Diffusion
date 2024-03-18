from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import resample

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OneDataset(Dataset):
    def __init__(self, path, n=2000):
        filenames = [Path(path) / f"one_{i}.png" for i in range(11)]
        images = [255. - np.mean(plt.imread(filename), axis=2) for filename in filenames]
        self.X = np.zeros((len(filenames), 2000, 2))

        for i, image in enumerate(images):
            coordinates = np.argwhere(image < 255)
            coordinates = coordinates / 32 - 1
            coordinates = resample(coordinates, n_samples=2000, replace=True)
            self.X[i, :, :] = coordinates

        self.X[:, :, [0, 1]] = self.X[:, :, [1, 0]]
        self.X[:, :, 1] = -self.X[:, :, 1]
        self.X = self.X.reshape(-1, 2)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float)


class NumsDataset(Dataset):
    def __init__(self, path, n=2000):
        filenames = [Path(path) / f"num_{i}.png" for i in range(10)]
        images = [255. - np.mean(plt.imread(filename), axis=2) for filename in filenames]
        self.X = np.zeros((len(filenames), 2000, 2))
        self.y = np.zeros((len(filenames), 2000, len(filenames)))
        for i, image in enumerate(images):
            coordinates = np.argwhere(image < 255)
            coordinates = coordinates / 32 - 1
            coordinates = resample(coordinates, n_samples=2000, replace=True)
            self.X[i, :, :] = coordinates
            y = np.zeros(len(filenames))
            y[i] = 1
            self.y[i, :, :] = y

        self.X[:, :, [0, 1]] = self.X[:, :, [1, 0]]
        self.X[:, :, 1] = -self.X[:, :, 1]
        self.X = self.X.reshape(-1, 2)
        self.y = self.y.reshape(-1, 10)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)
    
class GrecsDataset(Dataset):
    def __init__(self, path, n=2000):
        filenames = [Path(path) / f"grec-{i}.png" for i in range(24)]
        images = [255. - np.mean(plt.imread(filename), axis=2) for filename in filenames]
        self.X = np.zeros((len(filenames), 2000, 2))
        self.y = np.zeros((len(filenames), 2000, len(filenames)))
        for i, image in enumerate(images):
            coordinates = np.argwhere(image < 255)
            coordinates = coordinates / 32 - 1
            coordinates = resample(coordinates, n_samples=2000, replace=True)
            self.X[i, :, :] = coordinates
            y = np.zeros(len(filenames))
            y[i] = 1
            self.y[i, :, :] = y

        self.X[:, :, [0, 1]] = self.X[:, :, [1, 0]]
        self.X[:, :, 1] = -self.X[:, :, 1]
        self.X = self.X.reshape(-1, 2)
        self.y = self.y.reshape(-1, 24)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)
    
def get_dataloader_one(path, batch_size=32):
    dataset = OneDataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
    
def get_dataloader_nums(path, batch_size=32):
    dataset = NumsDataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def get_dataloader_grecs(path, batch_size=32):
    dataset = GrecsDataset(path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader