import os
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import wandb
from PIL import Image
# from images_utils import images_utils as IMUT
import matplotlib.pyplot as plt

from torch import nn
import numpy as np

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score

import warnings
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import tqdm


trans = transforms.Compose([
                    transforms.Normalize(
                        mean=[-n/255. for n in [129.3, 124.1, 112.4]],
                        std=[255./n for n in [68.2,  65.4,  70.4]]
                    )
        ])

trans1 = transforms.Compose([
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[n/255. for n in [129.3, 124.1, 112.4]],
                        std=[n/255. for n in [68.2,  65.4,  70.4]]
                    )
        ])


testsetA = CIFAR100(os.getcwd(), train=False, download=True, transform=trans1)
testsetB = CIFAR100(os.getcwd(), train=False, download=True, transform=trans1)

label_to_remove = 2
te_to_remove = np.array(testsetB.targets)[np.array(testsetB.targets)!=label_to_remove]

testsetB = Subset(testsetB, te_to_remove)

modelA = models.resnet18(pretrained=False, num_classes=100)
modelB = models.resnet18(pretrained=False, num_classes=99)

modelA.load_state_dict(torch.load('checkpointA.pt'))
modelB.load_state_dict(torch.load('checkpointB.pt'))
modelA = modelA.cuda().eval()
modelB = modelB.cuda().eval()

def test_loop(
        testset,
        model,
        bs=64,
        is_B=False
):
    test_loader = DataLoader(testset, batch_size=bs, shuffle=True)
    for idx, (imgs, labs) in tqdm.tqdm(enumerate(test_loader)):

        # imgs, labs = data
        imgs, model = \
            imgs.cuda(), model.cuda()

        labs = labs.cpu()
        if is_B:
            labs[labs > label_to_remove] -= 1

        outs = nn.functional.softmax(model(imgs))

        preds = outs.cpu().max(dim=1)

        accs = accuracy_score(preds.indices, labs)

        if idx % 10 == 0:
            img = trans(imgs[idx % min(bs, len(imgs))].unsqueeze(0))
            plt.imshow(img.squeeze().cpu().detach().permute(1, 2, 0).numpy())

            i = idx % min(bs, len(test_loader))
            if is_B:
                plt.title(f'{testset.dataset.classes[preds.indices[i]+1]} vs {testset.dataset.classes[labs[i]+1]}')
            else:
                plt.title(f'{testset.classes[preds.indices[i]]} vs {testset.classes[labs[i]]}')
            plt.show()

    return accs


print(f'''{test_loop(
    testset=testsetA,
    model=modelA
)=}''')

print(f'''{test_loop(
    testset=testsetB,
    model=modelB,
    is_B=True
)=}''')

x=0