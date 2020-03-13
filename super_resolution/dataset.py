import torch

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num = Y.shape[0]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]