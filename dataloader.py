"""
@author:32369
@file:dataloader.py
@time:2021/11/05
"""
from torch.utils.data import Dataset


class MindDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        line = self.data[index]
        his_tit = line[0]
        imp_tit = line[1]

        label = line[2]

        return his_tit, imp_tit, label

    def __len__(self):
        return len(self.data)
