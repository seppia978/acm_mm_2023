import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import ast
from collections.abc import Iterable
import numpy as np

size=224
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]


T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

_T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.Normalize(means, stds)
])

DeT = transforms.Compose([
    transforms.Normalize(-1 * torch.Tensor(means) / torch.Tensor(stds), 1.0 / torch.Tensor(stds))
])

def get_datasets(c_to_del: Iterable = [2, 3, 394, 4, 149]) -> tuple:
    global size, means, stds

    _train = ImageFolder(
        root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
        transform=T
    )

    id_to_remove = np.where(np.isin(np.array(_train.targets), c_to_del))[0]

    train = Subset(_train, id_to_remove)

    _val = ImageFolder(
        root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
        transform=T
    )

    id_to_remove = np.where(np.isin(np.array(_val.targets), c_to_del))[0]

    val = Subset(_val, id_to_remove)

    # test = ImageFolder(
    #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/test',
    #     transform=T
    # )

    with open('/homes/spoppi/pycharm_projects/inspecting_twin_models/class_names/names.txt', 'r') as f:
        txt = f.read()

    classes = ast.literal_eval(txt)
    train.dataset.names = classes
    val.dataset.names = classes

    id_others = np.where(~ np.isin(np.array(_val.targets), c_to_del))[0]
    others = Subset(_val, id_others)
    others.dataset.names = classes

    return tuple((train, val, others))

