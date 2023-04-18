import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
import sys
import tqdm
 
# setting path
sys.path.append('/homes_nfs/spoppi/pycharm_projects/inspecting_twin_models')
from custom_archs import WTFCNN


dataset='cifar10'


model=WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower()
)

a=WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
)
random_seed = 9999
# torch.manual_seed(random_seed)
b = models.resnet18(num_classes=10)



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

    # _train = CIFAR10(
    #     root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
    #     transform=T, download=True, train=True
    # )

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

    # _train = CIFAR100(
    #     root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/train',
    #     transform=T, download=True, train=True
    # )

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

    # _train = MNIST(
    #     root='/work/dnai_explainability/unlearning/datasets/mnist_classification/train',
    #     transform=T, download=True, train=True
    # )

    _val = MNIST(
        root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val',
        transform=T, download=True, train=False
    )
    b.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

classes_number = len(_val.classes)

id_c = np.where(np.isin(np.array(_val.targets), [1]))[0]
val_c = Subset(_val,id_c)

s=len(val_c)
print(s)
val_loader = DataLoader(
                        random_split(val_c, [s, len(val_c)-s])[0],
                        batch_size=16, num_workers=4
)

for  i in range(10):
    a.arch = a.arch.eval().cuda()
    a.arch.requires_grad_(False)

    b = models.resnet18(num_classes=10)
    b = b.eval().cuda()
    b.requires_grad_(False)

    model.arch = model.arch.cuda().eval()
    model.arch.requires_grad_(False)

    jsu,jsg=0,0
    with torch.inference_mode():
        import time
        t0 = time.time()
        for i,l in tqdm.tqdm(val_loader):
            i,l = i.cuda(),l.cuda()
            ascore = torch.softmax(a(i),-1).cpu()
            bscore = torch.softmax(b(i),-1).cpu()
            cscore = torch.softmax(model(i, labels=l),-1).cpu()
            
            jsu += np.sum(np.power(jensenshannon(cscore,bscore, axis=1),2))
            jsg += np.sum(np.power(jensenshannon(ascore,bscore, axis=1),2))
            
        tf = time.time()
    print(f'ZRF unlearning: {1 - jsu/s}, time: {tf-t0}')
    print(f'ZRF original: {1 - jsg/s}, time: {tf-t0}')
    print(f'ratio: {(1 - jsu/s)/(1 - jsg/s)}, diff: {(1 - jsu/s)-(1 - jsg/s)}')
x=0

