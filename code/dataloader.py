import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, csr_train, n_negs):
        # self.csr_train = csr_train
        self.num_edge = csr_train.nnz
        self.num_U, self.num_V = csr_train.shape
        self.num_negs = n_negs
        self.src, self.dst = csr_train.nonzero()
        self.src = torch.from_numpy(self.src)
        self.dst = torch.from_numpy(self.dst)
        
    def __len__(self):
        return self.num_edge
    
    def __getitem__(self, idx):
        neg_idx_V = None
        if self.num_negs > 0:
            neg_idx_V = torch.randint(0, self.num_V, (self.num_negs,))
        return self.src[idx], self.dst[idx], neg_idx_V
    
    @staticmethod
    def collate_fn(data):
        idx_U = torch.stack([_[0] for _ in data], dim=0)
        pos_idx_V = torch.stack([_[1] for _ in data], dim=0)
        if data[0][2] is not None:
            neg_idx_V = torch.stack([_[2] for _ in data], dim=0)
        else:
            neg_idx_V = None
        return idx_U, pos_idx_V, neg_idx_V


def train_mini_batch(edges: torch.LongTensor, batch_size, batch_per_epoch):  # edges: (num_edge, 2)
    num_edge = edges.shape[0]
    perm = torch.randperm(num_edge)  # (num_edge, )
    if batch_per_epoch is not None:
        perm = perm[:batch_per_epoch * batch_size] # (batch_per_epoch * batch_size, )
    actual_edges = edges[perm, :]
    b_begin = 0
    while b_begin < num_edge:
        batch = actual_edges[b_begin:b_begin + batch_size]
        yield batch
        b_begin += batch_size