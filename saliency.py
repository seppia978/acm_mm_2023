import os
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder, \
    CIFAR10, \
    CIFAR100, \
    MNIST
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import SGD
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import math
from custom_archs import WTFCNN


cid = 444
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

hyperparams = {
    'lr': 1e-1,
    'dataset': 'cifar10'
}

lr = hyperparams['lr']
dataset = hyperparams['dataset']

wdb_name = f'test_{dataset}_{lr}'

run = wandb.init(
    reinit=True, config=hyperparams, project=f"saliency", name=f'{wdb_name}'
)

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

    DeT = transforms.Compose([
        transforms.Normalize(-1 * torch.Tensor(means) / torch.Tensor(stds), 1.0 / torch.Tensor(stds))
    ])

    # _train = ImageFolder(
    #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
    #     transform=T
    # )

    _val = ImageFolder(
        root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
        transform=T
    )

    root='/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16'# root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/test_all_resnet18_1.0_100.0_2022-12-23-22'
    general = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    import ast
    with open('class_names/names.txt',  'r') as f:
        txt = f.read()

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

    root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-29-48'

    general = models.resnet18(pretrained=False, num_classes=10)
    # general.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
    acab = torch.load(
        os.path.join(
            '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
            '2022-12-29_5.pt'
        )
    )
    ac=list(map(lambda x: x[6:], acab.keys()))
    ckp = dict()
    for k1,k2 in zip(acab,ac):
        if k2 == k1[6:]:
            ckp[k2] = acab[k1]
    general.load_state_dict(ckp)

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

    root = '//mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/test_all_resnet18_1.0_100.0_2022-12-28-24'


classes_number = len(_val.classes)



PATH = f"{root}/final.pt"
unlearnt=WTFCNN.WTFCNN(
    kind=WTFCNN.resnet18, pretrained=True,
    m=3, classes_number=classes_number, resume=PATH,
    dataset=dataset.lower()
)
general.requires_grad_(False)

MSE = nn.MSELoss()
imgn = 1
val_loader = DataLoader(_val, batch_size=imgn, num_workers=0, shuffle=True)

for img, lab in val_loader:
    img = img.to(dev)
    lab = lab.to(dev)
    general.to(dev).eval()
    unlearnt.arch.to(dev).eval()

    x = nn.Parameter(torch.ones(
        (img.shape[0],1,img.shape[-2],img.shape[-1]),
        requires_grad=True,
        device=dev
        ))

    optimizer = SGD([x], lr=lr)
    # glogits = general(x * img)
    # ulogits = unlearnt(img, labels=lab)

    loss = 1e4

    while loss > 1:

        glogits = general(torch.sigmoid(x) * img)
        ulogits = unlearnt(img, labels=lab)

        loss = MSE(glogits, ulogits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run.log({'loss': loss})


import matplotlib.pyplot as plt

plt.imshow((DeT(img[0])).permute(1,2,0).detach().cpu().numpy(), cmap='gray')
plt.savefig('img.png')

xn=x-x.min()
xn /= xn.max()

plt.imshow((xn[0,0] * DeT(img[0])).permute(1,2,0).detach().cpu().numpy(), cmap='gray')
plt.savefig('sal.png')

plt.imshow((torch.sigmoid(x[0,0]) * DeT(img[0])).permute(1,2,0).detach().cpu().numpy(), cmap='gray')
plt.savefig('salsig.png')

x=0