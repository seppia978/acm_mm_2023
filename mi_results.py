import os
import time
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
#from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import math
from datetime import datetime
import wandb
from PIL import Image
# from images_utils import images_utils as IMUT
import matplotlib.pyplot as plt
import ast

import argparse
from torch import nn
import numpy as np

# from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split, Subset
# from sklearn.metrics import accuracy_score

import warnings
from sklearn.metrics import auc

import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# from custom_archs import convert_conv2d_to_alpha, set_label, get_all_alpha_layers, \
#     get_all_layer_norms, set_alpha_val, clip_alpha_val
import sys
sys.path.append('/homes/spoppi/pycharm_projects/inspecting_twin_models')
from custom_archs import WTFCNN
from custom_archs.wtflayer import WTFLayer

arch_name = 'resnet18'
dataset = 'cifar10'
baseline = 'standard'
resroot = '/homes/spoppi/pycharm_projects/inspecting_twin_models/mi_res'

import json
ckpfile = '/mnt/beegfs/work/dnai_explainability/final_ckpts_5-3-23.json'

with open(ckpfile, 'r') as f:
    ckp = json.load(f)

root = ckp[baseline][f'{arch_name}_{dataset}']
if os.path.isfile(os.path.join(root, 'best.pt')):
    path = os.path.join(root, 'best.pt')
elif os.path.isfile(os.path.join(root, 'final.pt')):
    path = os.path.join(root, 'final.pt')
else:
    path = os.path.join(root, 'last_intermediate.pt')

if 'vgg16'in arch_name:
    model=WTFCNN.WTFCNN(
        kind=WTFCNN.vgg16, pretrained=True,
        m=3., resume=path,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFCNN(
        kind=WTFCNN.vgg16, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
    
    nchannel = [64,256,512]
elif 'resnet18' in arch_name:
    model=WTFCNN.WTFCNN(
        kind=WTFCNN.resnet18, pretrained=True,
        m=3., resume=path,
        dataset=dataset.lower()
    )
    general=WTFCNN.WTFCNN(
        kind=WTFCNN.resnet18, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )

    
    nchannel = [64,256,512]
elif 'vit_small_16224' in arch_name:
    model=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_small_16224, pretrained=True,
        m=3., resume=path,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_small_16224, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
    
    nchannel = (1536,) * 3
elif 'vit_tiny_16224' in arch_name:
    model=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_tiny_16224, pretrained=True,
        m=3., resume=path,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_tiny_16224, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )

mkc0 = torch.load(f'{resroot}/{baseline}_{arch_name}_{dataset}_0.pt')
mkc1 = torch.load(f'{resroot}/{baseline}_{arch_name}_{dataset}_1.pt')
mkc2 = torch.load(f'{resroot}/{baseline}_{arch_name}_{dataset}_2.pt')
N = torch.load(f'{resroot}/{baseline}_{arch_name}_{dataset}_normalizator.pt')
bs = torch.load(f'{resroot}/{baseline}_{arch_name}_{dataset}_size_val.pt')

x=0