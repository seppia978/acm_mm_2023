import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from custom_archs import wfmodels_lora as wfmodels

unl_class = 0
total_images_number = 5000
dataset = 'cifar10'
model_name = 'vit_tiny'
bs = 128

if 'cifar10' in dataset:
    if 'tiny' in model_name:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    else:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_small_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/alpha_matrices/vit_small_neggrad_64_0.0001_1.0_0.01/2023-04-30/cifar10/difference/1.0-0.01/best_CLASS-{unl_class}.pt"
        # gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    c_number = 10
else:
    if 'tiny' in model_name:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/unlearning/icml2023/alpha_matrices/checkpoints_acm/vit_tiny_16224_lora_zero_3way_fixed_l1_AB_256_0.01_0.001_1.0/2023-05-03/0.001-1.0/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/unlearning/icml2023/alpha_matrices/checkpoints_acm/vit_tiny_16224_neggrad_256_0.0001_1.0_0.005/2023-05-03/1.0-0.005/best_CLASS-{unl_class}.pt"
        gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    else:
        ...
    c_number = 20

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, out_classes),
        )    

    def forward(self, x):
        out = self.classifier(x)
        return out
    
if model_name == 'vit_tiny':
    kind = wfmodels.vit_tiny_16224
else:
    kind = wfmodels.vit_small_16224

model = wfmodels.WFTransformer(
    kind=kind, pretrained=True,
    resume=ours_ckp,
    dataset=dataset.lower(), alpha=False
)
model = model.arch

general = wfmodels.WFTransformer(
    kind=kind, pretrained=False,
    resume=None,
    dataset=dataset.lower(), alpha=False
)
standard = general.arch

gold = wfmodels.WFTransformer(
    kind=wfmodels.vit_tiny_16224, pretrained=True,
    resume=gold_ckp,
    dataset=dataset.lower(), alpha=False
)
gold = gold.arch

disc = Discriminator(10, 64, 2)

if 'cifar10' in dataset:
    size = 32
    T = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    _train = CIFAR10(
        root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
        transform=T, download=False, train=True
    )

    _val = CIFAR10(
        root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/val',
        transform=transform_test, download=False, train=False
    )


    unlearning_ids = np.where(np.isin(np.array(
        _val.targets
        ), unl_class))[0]
    unlearning_set = Subset(_val, unlearning_ids)
    unlearning_set.targets = torch.Tensor(_val.targets).long()[unlearning_ids].tolist()

    retaining_ids = np.where(~np.isin(np.array(
        _val.targets
        ), unl_class))[0]
    retaining_set = Subset(_val, retaining_ids)
    retaining_set.targets = torch.Tensor(_val.targets).long()[retaining_ids].tolist()

    
    # Define the data loader
    unlearning_loader = DataLoader(unlearning_set, batch_size=256, shuffle=True)
    unlearning_loader_iter = iter(unlearning_loader)
    c_number = 10

train_loader = DataLoader(_train, batch_size=bs, shuffle=True)
train_loader_iter = iter(train_loader)

# train_loader = DataLoader(Subset(_train, _train.targets[:500]), batch_size=128, shuffle=True)
val_loader = DataLoader(_val, batch_size=bs, shuffle=True)
val_loader_iter = iter(val_loader)

standard.train()
model.train()


import os
if os.path.isfile('disc_X.pt') and os.path.isfile('disc_Y.pt'):
    disc_X = torch.load('disc_X_nolabel.pt')
    disc_Y = torch.load('disc_Y_nolabel.pt')
else:
    disc_train, disc_val = [], []

    standard=standard.cuda()

    from tqdm import tqdm
    with torch.inference_mode():
        for train_inputs, train_labels in tqdm(train_loader):

            train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()

            train_outputs = standard(train_inputs)

            disc_train.append(train_outputs.detach().cpu())
            # disc_train.append(torch.cat([train_outputs,train_labels.unsqueeze(1)], 1).detach().cpu())

        for val_inputs, val_labels in tqdm(val_loader):

            val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

            val_outputs = standard(val_inputs)

            disc_val.append(val_outputs.detach().cpu())
            # disc_val.append(torch.cat([val_outputs,val_labels.unsqueeze(1)], 1).detach().cpu())


    disc_train_X = torch.cat(disc_train, dim=0)
    disc_train_Y = torch.ones(disc_train_X.shape[0])

    disc_val_X = torch.cat(disc_val, dim=0)
    disc_val_Y = torch.ones(disc_val_X.shape[0]) * (-1)

    disc_X = torch.cat([disc_train_X, disc_val_X], 0)
    disc_Y = torch.cat([disc_train_Y, disc_val_Y], 0)

    torch.save(disc_X, 'disc_X_nolabel.pt')
    torch.save(disc_Y, 'disc_Y_nolabel.pt')

disc_X = disc_X.numpy()
disc_Y = disc_Y.numpy()


lda = LinearDiscriminantAnalysis()
lda.fit(disc_X, disc_Y)

model=model.cuda()

from tqdm import tqdm
# with torch.no_grad():
for inputs, targets in tqdm(val_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    try:
        unlearning_outputs = model(inputs[targets == unl_class])
        unlearning_outputs = unlearning_outputs.detach().cpu()
        # unlearning_outputs = torch.cat([unlearning_outputs,targets[targets == unl_class].unsqueeze(1)],1).detach().cpu()
        unlearning_logl = lda.predict(unlearning_outputs.numpy())
        x=0
    except:
        pass

    try:
        retaining_outputs = model(inputs[targets != unl_class])
        retaining_outputs = retaining_outputs.detach().cpu()
        # retaining_outputs = torch.cat([retaining_outputs,targets[targets != unl_class].unsqueeze(1)],1).detach().cpu()
        retaining_logl = lda.predict(retaining_outputs.numpy())
        x=0
    except:
        pass

    x=0


