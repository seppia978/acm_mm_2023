import os
import time
import argparse
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
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
from torchvision.datasets import ImageFolder, \
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


def main(args):
    dataset = args.dataset
    baseline = args.baseline
    arch_name = args.nn

    bs = 128

    # model=WTFCNN.WTFCNN(
    #         kind=WTFCNN.resnet18, pretrained=True,
    #         m=3., resume=None,
    #         dataset=dataset.lower()
    # )

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

        # T = model.T if hasattr(model, 'T') and model.T is not None else T

        # _train = ImageFolder(
        #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
        #     transform=T
        # )

        _val = ImageFolder(
            root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
            transform=T
        )

        import ast
        with open('/work/dnai_explainability/id_tot.txt', 'r') as f:
            idx_train = ast.literal_eval(f.read())

        val = Subset(_val, idx_train)

        val.targets = torch.Tensor(_val.targets).int()[idx_train].tolist()
        # _val=Subset(_val, range(2))
    elif dataset.lower() == 'cifar10':
        size = 32
        T = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # _train = CIFAR10(
        #     root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
        #     transform=T, download=True, train=True
        # )

        _val = CIFAR10(
            root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/val',
            transform=T, download=False, train=False
        )
        val = _val

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
            transform=T, download=False, train=False
        )

    elif dataset.lower() == 'mnist':
        size = 32
        T = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # _train = MNIST(
        #     root='/work/dnai_explainability/unlearning/datasets/mnist_classification/train',
        #     transform=T, download=True, train=True
        # )

        _val = MNIST(
            root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val',
            transform=T, download=False, train=False
        )
        val = _val

    classes_number = len(_val.classes)
    import json

    arch_dict = {
        'resnet18': 'res18',
        'vgg16': 'vgg',
        'vit_small_16224': 'vit_small',
        'vit_tiny_16224': 'vit_tiny',
    }

    ckpfile = '/mnt/beegfs/work/dnai_explainability/ok.json' if not 'vit' in arch_name else \
        '/mnt/beegfs/work/dnai_explainability/final_ckpts.json'

    with open(ckpfile, 'r') as f:
        ckp = json.load(f)

    root = ckp[baseline][f'{arch_dict[arch_name]}_{dataset}']
    if os.path.isfile(os.path.join(root, 'best.pt')):
        path = os.path.join(root, 'best.pt')
    elif os.path.isfile(os.path.join(root, 'final.pt')):
        path = os.path.join(root, 'final.pt')
    else:
        path = os.path.join(root, 'last_intermediate.pt')

    if 'vgg16' in arch_name:

        general = WTFCNN.WTFCNN(
            kind=WTFCNN.vgg16, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )

    elif "resnet34" in arch_name:
        general = WTFCNN.WTFCNN(
            kind=WTFCNN.resnet34, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )
    elif 'resnet18' in arch_name:
        model = WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,
            m=3., resume=path,
            dataset=dataset.lower()
        )

        general = WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )
    elif 'deit_small_16224' in arch_name:

        general = WTFCNN.WTFTransformer(
            kind=WTFCNN.deit_small_16224, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )
    elif 'vit_small_16224' in arch_name:
        model = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_small_16224, pretrained=True,
            m=3., resume=path,
            dataset=dataset.lower()
        )

        general = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_small_16224, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )
    elif 'vit_tiny_16224' in arch_name:
        model = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_tiny_16224, pretrained=True,
            m=3., resume=path,
            dataset=dataset.lower()
        )

        general = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_tiny_16224, pretrained=True,
            m=3., resume=None,
            dataset=dataset.lower(), alpha=False
        )

    general.arch.eval()
    general.arch.cuda()

    TOT = 10
    nn = 'vgg16'
    baseline = 'standard'
    val_loader = DataLoader(val, shuffle=True, batch_size=bs)
    counter, normal_tot, others_tot, others1_tot = 0., 0., 0., 0.
    scores_array = torch.Tensor([0. for _ in range(TOT)])
    scores_std = torch.Tensor([0. for _ in range(TOT)])
    others_array = torch.Tensor([0. for _ in range(TOT)])
    others_std = torch.Tensor([0. for _ in range(TOT)])

    # import json
    # ckpfile = '/mnt/beegfs/work/dnai_explainability/final_ckpts_5-3-23.json'

    # with open(ckpfile, 'r') as f:
    #     ckp = json.load(f)

    # root = ckp[baseline][f'{nn}_{dataset}']
    # if os.path.isfile(os.path.join(root, 'best.pt')):
    #     path = os.path.join(root, 'best.pt')
    # elif os.path.isfile(os.path.join(root, 'final.pt')):
    #     path = os.path.join(root, 'final.pt')
    # else:
    #     path = os.path.join(root, 'last_intermediate.pt')

    import tqdm
    for idx, (img, lab) in tqdm.tqdm(enumerate(val_loader)):

        img = img.cuda()
        lab = lab.cuda()
        otlab = lab.clone().cuda()

        if 'vgg16' in arch_name:
            deletion_model = WTFCNN.WTFCNN(
                kind=WTFCNN.vgg16, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

            model = WTFCNN.WTFCNN(
                kind=WTFCNN.vgg16, pretrained=True,
                m=3., resume=path,
                dataset=dataset.lower()
            )

        elif "resnet34" in arch_name:
            # model = models.resnet18(pretrained=True)
            deletion_model = WTFCNN.WTFCNN(
                kind=WTFCNN.resnet34, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

        elif 'resnet18' in arch_name:
            deletion_model = WTFCNN.WTFCNN(
                kind=WTFCNN.resnet18, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

            model = WTFCNN.WTFCNN(
                kind=WTFCNN.resnet18, pretrained=True,
                m=3., resume=path,
                dataset=dataset.lower()
            )

        elif 'deit_small_16224' in arch_name:
            deletion_model = WTFCNN.WTFTransformer(
                kind=WTFCNN.deit_small_16224, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

        elif 'vit_small_16224' in arch_name:
            deletion_model = WTFCNN.WTFTransformer(
                kind=WTFCNN.vit_small_16224, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

            model = WTFCNN.WTFTransformer(
                kind=WTFCNN.vit_small_16224, pretrained=True,
                m=3., resume=path,
                dataset=dataset.lower()
            )

        elif 'vit_tiny_16224' in arch_name:
            deletion_model = WTFCNN.WTFTransformer(
                kind=WTFCNN.vit_tiny_16224, pretrained=False,
                m=3., resume=None,
                dataset=dataset.lower()
            )

            model = WTFCNN.WTFTransformer(
                kind=WTFCNN.vit_tiny_16224, pretrained=True,
                m=3., resume=path,
                dataset=dataset.lower()
            )

        deletion_model.arch.eval()
        deletion_model.arch.cuda()

        while torch.any(otlab == lab):
            ids = torch.argwhere(otlab == lab)
            otlab[ids] = torch.randint(0, classes_number, ids.shape).cuda()

        # alpha = [x[lab] for x in tuple(model.get_all_alpha_layers().values())]
        scores, others, others1 = [], [], []

        with torch.inference_mode():
            ids = torch.cat([torch.Tensor(range(lab.shape[0])).unsqueeze(0), lab.cpu().unsqueeze(0)], 0).long().cuda()
            otids = torch.cat([torch.Tensor(range(lab.shape[0])).unsqueeze(0), otlab.cpu().unsqueeze(0)],
                              0).long().cuda()

            uscores = torch.softmax(deletion_model(img, labels=lab), 1)
            oscores = torch.softmax(deletion_model(img, labels=otlab), 1)
            gscores = torch.softmax(general(img, labels=lab), 1)
            sc = uscores[ids[0], ids[1]]
            ot = uscores[otids[0], otids[1]]
            sc /= gscores[ids[0], ids[1]]
            ot /= gscores[ids[0], ids[1]]
            sc[sc > 1.] = 1.
            ot[ot > 1.] = 1.
            scores.append(
                (sc).unsqueeze(0)
            )
            others.append(
                (ot).unsqueeze(0)
            )
            # others1.append(torch.softmax(model(img, labels=otlab),1)[ids[0],ids[1]].unsqueeze(0))

            tot = TOT - 1
            t0 = time.time()
            for i in range(tot):
                for (n1, x1), (n, x) in zip(deletion_model.arch.named_modules(), model.arch.named_modules()):

                    if isinstance(x, WTFLayer):
                        maxids = ((-1) * x.alpha).topk(
                            x.alpha.shape[1] // TOT
                        ).indices

                        for ii in range(x.alpha.data.shape[0]):
                            x1.alpha.data[ii, maxids[ii]] = -3.
                            x.alpha.data[ii, maxids[ii]] = 3.


                # ids = torch.cat([torch.Tensor(range(lab.shape[0])).unsqueeze(0), lab.cpu().unsqueeze(0)],0).long().cuda()
                # otids = torch.cat([torch.Tensor(range(lab.shape[0])).unsqueeze(0), otlab.cpu().unsqueeze(0)],0).long().cuda()
                sc = torch.softmax(deletion_model(img, labels=lab), 1)[ids[0], ids[1]]
                ot = torch.softmax(deletion_model(img, labels=lab), 1)[otids[0], otids[1]]
                sc /= torch.softmax(general(img, labels=lab), 1)[ids[0], ids[1]]
                ot /= torch.softmax(general(img, labels=lab), 1)[ids[0], ids[1]]
                sc[sc > 1.] = 1.
                ot[ot > 1.] = 1.
                scores.append(
                    (sc).unsqueeze(0)
                )
                others.append(
                    (ot).unsqueeze(0)
                )

                # others1.append(
                #     torch.softmax(model(img, labels=otlab),1)[ids[0],ids[1]].unsqueeze(0)
                # )
        tf = time.time()
        scores = torch.cat(scores, 0).squeeze().cpu()
        scores_std += scores.std(dim=1)
        scores = scores.mean(dim=1).detach().numpy()
        scores_array += scores
        others = torch.cat(others, 0).squeeze().cpu()
        others_std += others.std(dim=1)
        others = others.mean(dim=1).detach().numpy()
        others_array += others
        # others1 = torch.cat(others1,0).squeeze().cpu().mean(dim=1).detach().numpy()
        points = np.array(range(scores.shape[0]))
        auc_scores = auc(points, scores)
        auc_others = auc(points, others)
        # auc_others1 = auc(points,others1)
        normal_tot += auc_scores
        others_tot += auc_others
        # others1_tot += auc_others1
        counter += 1

        print('#############################')
        print(f'Elapsed time {idx}: {tf - t0}')
        print(f'normal: {auc_scores / TOT}, others: {auc_others / TOT}')  # , others1: {auc_others1}')
        print('#############################')
        # break
        # if True or idx % 10 == 0 and idx > 0:
        #     break
        # break

        print(f'FINAL: \n\t normal: {normal_tot / counter / TOT}, others: {others_tot / counter / TOT}')
        # scores_array /= counter
        # scores_std /= counter
        # others_array /= counter
        # others_std /= counter

        # import matplotlib.pyplot as plt

        # plt.plot(scores_array)
        # plt.fill_between(points, scores_array-scores_std, scores_array+scores_std, alpha=.3)
        # plt.plot(others_array)
        # plt.fill_between(points, others_array, others_array+others_std, alpha=.3)
        # plt.ylim(-.2, 1)
        # plt.savefig(f'alpha_res/plots/{nn}.png')
        torch.save(scores_array, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_normal.pt')
        torch.save(others_array, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_others.pt')
        torch.save(scores_std, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_nstd.pt')
        torch.save(others_std, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_otstd.pt')
        torch.save(counter, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_counter.pt')
        torch.save(torch.tensor([TOT]), f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_tot.pt')

    print(f'FUCKING FINAL: \n\t normal: {normal_tot / counter / TOT}, others: {others_tot / counter / TOT}')
    print('########################################################################################')
    torch.save(scores_array, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_normal.pt')
    torch.save(others_array, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_others.pt')
    torch.save(scores_std, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_nstd.pt')
    torch.save(others_std, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_otstd.pt')
    torch.save(counter, f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_counter.pt')
    torch.save(torch.tensor([TOT]), f'alpha_res/plots/deletion_{baseline}_{arch_name}_{dataset}_tot.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn", type=str, help='nn', default="resnet18")
    parser.add_argument("--baseline", type=str, help='baseline', default='standard')
    parser.add_argument('-D', "--dataset", type=str, help='dataset', default='imagenet')

    args = parser.parse_args()

    main(args)