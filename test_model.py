import os
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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
from torch.utils.data import DataLoader, Subset

import tqdm

c_to_del = [2, 3, 4, 394, 391, 5, 6, 149, 389, 395, 148, 147, 150, 146, 657, 33, 744, 0, 397, 983, 390, 833, 895, \
            112, 557, 97, 973, 34, 296, 913, 99, 876, 435, 107, 684, 749, 801, 758, 145, 433, 65, 472, 908, 638, 814, \
            356, 327, 728, 719, 29, 103, 360, 124, 36, 121, 337, 396, 403, 329, 971, 977, 978, 976, 437, 405, 461, 344, \
            118, 1, 58, 126, 98, 50, 392, 251, 975, 898, 137, 144, 871, 484, 693, 628, 696, 643, 117, 980, 81, 838, \
            721, 756, 119, 172, 794, 625, 554, 842, 357, 140, 812, 404, 896, 787, 965, 725, 775, 250, 114, 737, 562, \
            460, 972, 408, 639, 463, 792, 540, 132, 795, 279, 21, 691, 701, 818, 828, 89, 71, 777, 499, 596, 591, 692, \
            115, 450, 589, 863, 735, 832, 733, 314, 123, 362, 178, 120, 180, 125, 195, 780, 525, 536, 510, 22, 514, 171, \
            23, 129, 111, 807, 502, 515, 606, 808, 885, 469, 563, 921, 558, 902, 419, 636, 793, 970, 899, 452, 773, \
            412, 700, 518, 51, 552, 797, 30, 122, 35, 32, 162, 26, 28, 133, 242, 116, 245, 471, 86, 161, 31, 467, 914, \
            173, 253, 428, 294, 456, 979, 210, 20, 576, 834, 49, 718, 445, 109, 134, 94, 131, 616, 457, 715, 623, 541, \
            997, 618, 597, 464, 659, 909, 566, 954, 961, 813, 78, 310, 929, 974, 928, 438, 802, 686, 900, 441, 604, 712, \
            901, 714, 473, 845, 849, 786, 631, 740, 995, 723, 994, 673, 999, 466, 545, 947, 674, 459, 710, 641, 363, \
            992, 418, 680, 991, 587, 676, 489, 804, 742, 560, 891, 246, 169, 249, 694, 212, 139, 359, 87, 364, 282, 411, \
            637, 824, 747, 614, 501, 708, 399, 610, 440, 841, 414, 879, 953, 667, 653, 709, 578, 427, 915, 448, 887, 442, \
            88, 862, 439, 748, 917, 689, 981, 96, 907, 496, 868, 93, 619, 524, 570, 750, 494, 385, 645, 699, 883, 400, 738, \
            462, 987, 490, 796, 934] # [2, 3, 4, 5, 394, 6, 395, 391, 389, 149] # [159, 168, 180, 242, 163, 243, 246, 211, 179, 236] #[2, 3, 394, 4, 149]


# c_to_del = [2,4]
c_to_del = c_to_del[:10]

hyperparams = {
    'lr': 1e-3,
    'model': 'resnet18'
}

# run = wandb.init(
#     reinit=True, config=hyperparams, project="unlearn_one_short", name='resnet18_baby_shark'
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


PATH = "/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/short/resnet18_0.9.pt"
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load(PATH))

_train = ImageFolder(
    root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
    transform=T
)

id_c = np.where(np.isin(np.array(_train.targets), c_to_del))[0]


train = Subset(_train, id_c)


_val = ImageFolder(
    root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
    transform=T
)

id_c = np.where(np.isin(np.array(_val.targets), c_to_del))[0]
id_others = np.where(~ np.isin(np.array(_val.targets), c_to_del))[0]
val_c = Subset(_val, id_c)
val_others = Subset(_val, id_others)
# test = ImageFolder(
#     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/test',
#     transform=T
# )

loss = nn.CrossEntropyLoss(reduction='none')
optimizer=torch.optim.SGD(model.parameters(), lr=hyperparams['lr'])

#TRAINING LOOP
def test_loop(
        n_epochs,
        optimizer,
        model,
        loss_fn,
        train,
        val,
        hyp
):
    max_val_size = 20_000
    # others_loader = DataLoader(random_split(val_ot, [max_val_size, len(val_ot) - max_val_size])[0], batch_size=64, shuffle=True)

    model = model.cuda()

    confusion_matrix = torch.zeros(1000)

    x_cls_del = 0
    for epoch in range(n_epochs):

        val_loader = DataLoader(val, batch_size=64, shuffle=True)
        print(f'Val {epoch=} ')
        model.eval()
        with torch.no_grad():
            for idx, (imgs, labels) in tqdm.tqdm(enumerate(val_loader)):
                imgs = imgs.cuda()
                labels = labels.cuda()

                t_out_val = nn.functional.softmax(model(imgs))

                cl_acc = (t_out_val.max(1).indices == labels).sum() / (t_out_val.max(1).indices == labels).shape[0]
                # run.log({'cl_t_acc': cl_acc})

                ot_t_acc = []
            for o, ol in others_loader:
                o = o.cuda()
                ol = ol.cuda()
                test_outs = model(o)
                ot_t_acc.append((test_outs.max(1).indices == ol).sum() / (test_outs.max(1).indices == ol).shape[0])

                # run.log({'others_t_acc': torch.Tensor(ot_t_acc).mean()})


        # if epoch == 1 or epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}, Validation loss {loss_val.item():.4f}")

test_loop(
    n_epochs = 1,
    optimizer = optimizer,
    model = model,
    loss_fn = loss,
    train=train,
    val=val_others,
    hyp=hyperparams
)
