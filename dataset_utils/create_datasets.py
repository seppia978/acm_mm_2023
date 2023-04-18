import os
from collections.abc import Iterable
from utils import get_datasets
import torch
from torch.utils.data import DataLoader, random_split, Subset

def get_dataloaders(c_to_del: Iterable = [2, 3, 394, 4, 149]) -> tuple:
    train, val, others = get_datasets(c_to_del)

    size_others = 1
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    others_loader = DataLoader(random_split(others, [size_others, len(others) - size_others])[0], batch_size=64, shuffle=True)
    val_loader = DataLoader(random_split(val, [size_others, len(val) - size_others])[0], batch_size=64, shuffle=True)

    return tuple((train_loader, val_loader, others_loader))


def save_datasets(datasets: Iterable, path=".") -> None:
    train, val, others = datasets

    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(train, os.path.join(path, 'train.pt'))
    torch.save(val, os.path.join(path, 'val.pt'))
    torch.save(others, os.path.join(path, 'others.pt'))

save_datasets(get_datasets(c_to_del=[2, 3, 394, 4, 149]), path=os.path.join('files'))