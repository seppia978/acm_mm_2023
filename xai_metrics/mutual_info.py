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

warnings.filterwarnings('ignore')

# torch imports
import torch
import torch.nn as nn
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

from sklearn.feature_selection import mutual_info_classif as MI

AK = None

def hook(model, input, output):
    global AK
    AK = output.data

arch_name = 'resnet18'
dataset = 'imagenet'
baseline = 'standard'

import json
ckpfile = '/mnt/beegfs/work/dnai_explainability/ok.json'

with open(ckpfile, 'r') as f:
    ckp = json.load(f)

root = ckp[baseline][f'res18_{dataset}']
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
    model.arch.features[44].register_forward_hook(hook)
    general.arch.features[44].register_forward_hook(hook)
    nchannel = 512
elif "resnet34" in arch_name:
    # model = models.resnet18(pretrained=True)
    model=WTFCNN.WTFCNN(
        kind=WTFCNN.resnet34, pretrained=False,
        m=3., resume=None,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFCNN(
        kind=WTFCNN.resnet34, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
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
    model.arch.avgpool.register_forward_hook(hook)
    general.arch.avgpool.register_forward_hook(hook)
    nchannel = 512
elif 'deit_small_16224' in arch_name:
    model=WTFCNN.WTFTransformer(
        kind=WTFCNN.deit_small_16224, pretrained=False,
        m=3., resume=None,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFTransformer(
        kind=WTFCNN.deit_small_16224, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
elif 'vit_small_16224' in arch_name:
    model=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_small_16224, pretrained=False,
        m=3., resume=None,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_small_16224, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
elif 'swin_small_16224' in arch_name:
    model=WTFCNN.WTFTransformer(
        kind=WTFCNN.swin_small_16224, pretrained=False,
        m=3., resume=None,
        dataset=dataset.lower()
    )

    general=WTFCNN.WTFTransformer(
        kind=WTFCNN.swin_small_16224, pretrained=True,
        m=3., resume=None,
        dataset=dataset.lower(),alpha=False
    )
general.arch.requires_grad_(requires_grad=False)


print("Preparing datasets...")


if dataset.lower() == 'imagenet':

    size = 224
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

    # means = [0.5, 0.5, 0.5]
    # stds = [0.5, 0.5, 0.5]
    # vit_transform = transforms.Compose([
    #     transforms.CenterCrop(size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(means, stds)
    # ])

    DeT = transforms.Compose([
        transforms.Normalize(-1 * torch.Tensor(means) / torch.Tensor(stds), 1.0 / torch.Tensor(stds))
    ])

    T = model.T if hasattr(model, 'T') and model.T is not None else T

    # _train = ImageFolder(
    #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
    #     transform=T
    # )

    _val = ImageFolder(
        root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
        transform=T
    )

elif dataset.lower() == 'cifar10':

    T = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    _train = CIFAR10(
        root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
        transform=T, download=True, train=True
    )

    _val = CIFAR10(
        root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/val',
        transform=T, download=True, train=False
    )

elif dataset.lower() == 'cifar100':

    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    _train = CIFAR100(
        root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/train',
        transform=T, download=True, train=True
    )

    _val = CIFAR100(
        root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/val',
        transform=T, download=True, train=False
    )

elif dataset.lower() == 'mnist':

    means, stds = (0.1307,), (0.3081,)
    T  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    _train = MNIST(
        root='/work/dnai_explainability/unlearning/datasets/mnist_classification/train',
        transform=T, download=True, train=True
    )

    _val = MNIST(
        root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val',
        transform=T, download=True, train=False
    )

classes_number = len(_train.classes)


size_val = 64
import tqdm
MKC = torch.zeros(classes_number, nchannel)

for cl in range(classes_number):
    chosen_class = cl
    # val_loader = DataLoader(random_split(_val, [size_val, len(_val) - size_val])[0], batch_size=64, shuffle=True)


    y_train = _val.targets  # train.datasets[0].dataset.targets
                
    weight = 1. / torch.Tensor([.001 for _ in range(classes_number)])

    weight[~np.isin(list(range(len(weight))), np.array([chosen_class]))] = .5 / (len(weight) - len([chosen_class]))
    weight[np.array([chosen_class])] = .5 / len([chosen_class])

    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    val_loader = DataLoader(_val, batch_size=size_val, num_workers=4, sampler=sampler)

    model.arch.eval().cuda()
    general.arch.eval().cuda()


    with torch.inference_mode():
        for x,y in tqdm.tqdm(val_loader):
            x, y = x.cuda(), y.cuda()

            scores = general(x, labels=chosen_class)
            target = torch.zeros_like(y)
            target[y==chosen_class] = 1.
            target = 1 - target

            for i in range(AK.shape[1]):
                MKC[chosen_class, i] += torch.from_numpy(
                    MI(AK[:,i,0,0].unsqueeze(1).detach().cpu().numpy(), target.detach().cpu().numpy(), random_state=0)
                ).squeeze()



import math
MKC_ = MKC / math.ceil(len(_val)/size_val)
torch.save(MKC, f'{baseline}_{arch_name}_{dataset}.pt')