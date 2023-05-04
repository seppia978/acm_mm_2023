import os
from typing import Union
from collections.abc import Iterable
import torch

from torch.utils.data import Dataset, ConcatDataset, \
    Subset, WeightedRandomSampler
import torchvision.datasets
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import math
from datetime import datetime
import wandb
from PIL import Image
import matplotlib.pyplot as plt

import argparse
from torch import nn
import numpy as np
import loss_manager

from torch.utils.data import DataLoader, random_split, Subset

import warnings

warnings.filterwarnings('ignore')

# torch imports
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST

# from custom_archs import wfmodels_lora as wfmodels
from custom_archs import wfmodels_lora as wfmodels

import loralib as lora
import random
# import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


from torchvision.transforms.functional import normalize, resize, to_pil_image
import numpy as np
from matplotlib import cm
from PIL import Image


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image
    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)
    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image
    Returns:
        overlayed image
    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def main(args):

    arch_name = args.nn
    bs = args.batch_size
    dataset = args.dataset
    cam_name = args.cam
    unl_class = args.unl_class

    for unl_class in range(10):

        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/debug_instance_prod_64_5e-05_2.5e-05_1.0/2023-05-02/2.5e-05-1.0/best_CLASS-0.pt"
        gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"

        hyperparams = {
            'model': arch_name,
            'batch_size': bs,
            'dataset': dataset,
            'cam_name': cam_name,
            'unl_class': unl_class,
        }

        print([(k,v) for k,v in hyperparams.items()])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'cifar10' in dataset.lower():
            c_number = 10
        elif 'cifar20' in dataset.lower():
            c_number = 20
        else:
            print("Dataset not supported")
            return -1

        try:
            root='/mnt/beegfs/work/dnai_explainability/'
            folder_name = os.path.join(
                root, "cam_viz", str(unl_class),
                f'{cam_name}'
            )
        except:
            root='/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm'
            folder_name = f'{arch_name}_{datetime.today().strftime("%Y-%m-%d")}'
            os.makedirs(os.path.join(root,arch_name))
            folder_name = os.path.join(
                root, "cam_viz",str(unl_class),
                f'{cam_name}'
            )

        run_root = os.path.join(root, folder_name)
        if not os.path.isdir(run_root):
            os.makedirs(run_root)

        if 'vgg16' in arch_name:
            model = wfmodels.WFCNN(
                kind=wfmodels.vgg16, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFCNN(
                kind=wfmodels.vgg16, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch

        elif 'resnet18' in arch_name:
            model = wfmodels.WFCNN(
                kind=wfmodels.resnet18, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFCNN(
                kind=wfmodels.resnet18, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        elif 'vit_small_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.vit_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)

            general = wfmodels.WFTransformer(
                kind=wfmodels.vit_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )

            standard = general.arch
        elif 'vit_tiny_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                resume=ours_ckp,
                dataset=dataset.lower(), alpha=False
            )
            model = model.arch
            target_layer = model.transformer.layers[-2][-1].norm
            # target_layer = model.transformer
            general = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=False,
                resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
            std_target_layer = standard.transformer.layers[-2][-1].norm
            
            gold = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                resume=gold_ckp,
                dataset=dataset.lower(), alpha=False
            )
            gold = gold.arch
            gold_target_layer = gold.transformer.layers[-2][-1].norm

            ldiff = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                resume=loss_diff_ckp,
                dataset=dataset.lower(), alpha=False
            )
            ldiff = ldiff.arch
            ldiff_target_layer = ldiff.transformer.layers[-2][-1].norm
            
            # std_target_layer = standard.transformer
        elif 'swin_small_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.swin_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFTransformer(
                kind=wfmodels.swin_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        elif 'swin_tiny_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.swin_tiny_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = model.arch
            general = wfmodels.WFTransformer(
                kind=wfmodels.swin_tiny_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        general.arch.requires_grad_(requires_grad=True)
        # model = model.train() # Lorenzo
        standard = standard.train()


            # model.set_label(0)
            
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

            from numpy.random import choice
            idx_tot = []
            for i in range(1000):
                chosen = choice(range(50), size=5)
                idx_tot.extend(list(50*i+chosen))

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
            idx_val = [x for x in range(50_000) if x not in idx_train]
            train = Subset(_val, idx_train)
            val = Subset(_val, idx_val)

            train.targets = torch.Tensor(_val.targets).int()[idx_train].tolist()
            
            # TODO: split val in unlearning and retaining
            val.targets = torch.Tensor(_val.targets).int()[idx_val].tolist()
            train.num_classes = 1000
            val[0].num_classes, val[1].num_classes = 1000, 1000


        elif 'cifar10' in dataset.lower():

            # T = transforms.Compose(
            #     [
            #         transforms.ToTensor(),
            #         cifar10_normalization(),
            #     ]
            # )


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

            # _train = CIFAR10(
            #     root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
            #     transform=T, download=True, train=True
            # )

            _val = CIFAR10(
                root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/val',
                transform=transform_test, download=True, train=False
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


        elif 'cifar20' in dataset.lower():

            class_mapping_dict = \
                    {0: 4,
                    1: 1,
                    2: 14,
                    3: 8,
                    4: 0,
                    5: 6,
                    6: 7,
                    7: 7,
                    8: 18,
                    9: 3,
                    10: 3,
                    11: 14,
                    12: 9,
                    13: 18,
                    14: 7,
                    15: 11,
                    16: 3,
                    17: 9,
                    18: 7,
                    19: 11,
                    20: 6,
                    21: 11,
                    22: 5,
                    23: 10,
                    24: 7,
                    25: 6,
                    26: 13,
                    27: 15,
                    28: 3,
                    29: 15,
                    30: 0,
                    31: 11,
                    32: 1,
                    33: 10,
                    34: 12,
                    35: 14,
                    36: 16,
                    37: 9,
                    38: 11,
                    39: 5,
                    40: 5,
                    41: 19,
                    42: 8,
                    43: 8,
                    44: 15,
                    45: 13,
                    46: 14,
                    47: 17,
                    48: 18,
                    49: 10,
                    50: 16,
                    51: 4,
                    52: 17,
                    53: 4,
                    54: 2,
                    55: 0,
                    56: 17,
                    57: 4,
                    58: 18,
                    59: 17,
                    60: 10,
                    61: 3,
                    62: 2,
                    63: 12,
                    64: 12,
                    65: 16,
                    66: 12,
                    67: 1,
                    68: 9,
                    69: 19,
                    70: 2,
                    71: 10,
                    72: 0,
                    73: 1,
                    74: 16,
                    75: 12,
                    76: 9,
                    77: 13,
                    78: 15,
                    79: 13,
                    80: 16,
                    81: 19,
                    82: 2,
                    83: 4,
                    84: 6,
                    85: 19,
                    86: 5,
                    87: 5,
                    88: 8,
                    89: 19,
                    90: 18,
                    91: 1,
                    92: 2,
                    93: 15,
                    94: 6,
                    95: 0,
                    96: 17,
                    97: 8,
                    98: 14,
                    99: 13
                    }

            class CIFAR20(CIFAR100):

                def __init__(self, root, c_to_del=[], train=True, transform=None, target_transform=None, download=False):
                    super().__init__(root, train, transform, _cifar100_to_cifar20, download)

            def _cifar100_to_cifar20(target):
                mapping = class_mapping_dict[target]
                return mapping
            
            def _cifar20_from_cifar100(target):
                mapping = _cifar100_to_cifar20(target)
                demapping = [x for x,v in mapping if v==target[0]]

                return demapping


            means = (0.4914, 0.4822, 0.4465)
            stds = (0.2023, 0.1994, 0.2010)
            size = 32
            T = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

            _train = CIFAR20(
                root='/work/dnai_explainability/unlearning/datasets/cifar20_classification/train',
                transform=T, download=True, train=True, c_to_del=c_to_del
            )

            _val = CIFAR20(
                root='/work/dnai_explainability/unlearning/datasets/cifar20_classification/val',
                transform=transform_test, download=True, train=False, c_to_del=c_to_del
            )

            id_c = np.where(np.isin(np.array(
                tuple(map(_cifar100_to_cifar20, _val.targets))
                ), c_to_del))[0]
            val_c = Subset(_val, id_c)
            val_c.targets = torch.Tensor(_val.targets).long()[id_c].tolist()
            val_c.targets = map(_cifar100_to_cifar20, val_c.targets)

            id_ot = np.where(~np.isin(np.array(
                tuple(map(_cifar100_to_cifar20, _val.targets))
                ), c_to_del))[0]
            val_ot = Subset(_val, id_ot)
            val_ot.targets = torch.Tensor(_val.targets).long()[id_ot].tolist()
            val_ot.targets = map(_cifar100_to_cifar20, val_ot.targets)

            if 'zero' in hyperparams['loss_type']:

                id_c = np.where(np.isin(np.array(
                    tuple(map(_cifar100_to_cifar20, _train.targets))
                    ), c_to_del))[0]
                train_c = Subset(_train, id_c)
                train_c.targets = torch.Tensor(_train.targets).long()[id_c].tolist()
                train_c.targets = map(_cifar100_to_cifar20, train_c.targets)
                
                train = train_c
            else:
                train = _train
            val = (val_c, val_ot)
            train.num_classes = 20
            train.real_targets = list(map(_cifar100_to_cifar20, train.targets))
            val[0].num_classes, val[1].num_classes = 20, 20

        elif dataset.lower() == 'mnist':

            # means, stds = (0.1307,), (0.3081,)
            # T  = transforms.Compose([
            #     transforms.Resize(32),
            #     transforms.ToTensor(),
            #     transforms.Normalize(means, stds)
            # ])
            size = 32
            T = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            _train = MNIST(
                root='/work/dnai_explainability/unlearning/datasets/mnist_classification/train',
                transform=T, download=True, train=True
            )

            _val = MNIST(
                root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val',
                transform=transform_test, download=True, train=False
            )
            train,val=_train,_val
        if 'custom' in dataset.lower():
            root = r'/mnt/beegfs/work/dnai_explainability/unlearning/datasets'
            dset_folder = 'test0'
            train_ = 'train'
            val_ = 'val'
            cl = 'cat'
            train_path = os.path.join(root, dset_folder, train_, cl)
            val_path = os.path.join(root, dset_folder, val_)

            class CustomDataset(Dataset):
                def __init__(self, root, c_to_del, transform=None):
                    self.root_dir = root
                    self.transform = transform
                    self.image_paths = os.listdir(root)
                    self.class_label = c_to_del

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, index):
                    img_path = os.path.join(self.root_dir, self.image_paths[index])
                    image = Image.open(img_path)

                    if self.transform:
                        image = self.transform(image)

                    return image, torch.tensor(self.class_label).squeeze()

            # size = 224
            # means = [0.485, 0.456, 0.406]
            # stds = [0.229, 0.224, 0.225]
            # T = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(size),
            #     # transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize(means, stds),
            #     # transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
            # ])

            # _T = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.CenterCrop(size),
            #     transforms.Normalize(means, stds)
            # ])

            # DeT = transforms.Compose([
            #     transforms.Normalize(-1 * torch.Tensor(means) / torch.Tensor(stds), 1.0 / torch.Tensor(stds))
            # ])

            T = model.T if hasattr(model, 'T') and model.T is not None else T

            _train = CustomDataset(root=train_path, c_to_del=c_to_del, transform=T)
            # # _train = ImageFolder(root=train_path, transform=T)
            # _val = ImageFolder(
            #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
            #     transform=T
            # )
            # _val = ImageFolder(root=val_path, transform=T)

            # _train = ImageFolder(
            #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
            #     transform=T
            # )

            # id_c = np.where(np.isin(np.array(_train.targets), c_to_del))[0]
            # train = Subset(_train, id_c) 
            # train.targets = torch.Tensor(_train.targets).int()[id_c].tolist()
            # _train = train

            train, val = _train, _val
            train.num_classes = 10
            val.num_classes = 10

        classes_number = len(_val.classes)

        unlearning_loader = DataLoader(unlearning_set, batch_size=bs, shuffle=True)
        unlearning_loader_iter = iter(unlearning_loader)

        retaining_loader = DataLoader(retaining_set, batch_size=bs, shuffle=True)
        retaining_loader_iter = iter(retaining_loader)


        for m in model.parameters():
            m.requires_grad_(True)
        for m in standard.parameters():
            m.requires_grad_(True)

        for i in range(15):
            imgs, labs = next(unlearning_loader_iter)

            target = [ClassifierOutputTarget(unl_class)]

            unlearning_cam = GradCAM(model=model, target_layers=[target_layer])
            standard_cam = GradCAM(model=standard, target_layers=[std_target_layer])
            gold_cam = GradCAM(model=gold, target_layers=[gold_target_layer])
            ldiff_cam = GradCAM(model=ldiff, target_layers=[ldiff_target_layer])

            # m = wfmodels.WFTransformer(
            #         kind=wfmodels.vit_tiny_16224, pretrained=False,
            #         resume=None,
            #         dataset='imagenet', alpha=False
            #     )
            # m = m.arch
            # cam = GradCAM(model=m, target_layers=[m.blocks[-1].norm1])
            # resized_tensor = torch.nn.functional.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
            # m_map = cam(input_tensor=resized_tensor, targets=target)


            # targets = model(imgs)

            imgs_up = torch.nn.functional.interpolate(inverse_transform(imgs.squeeze()).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

            unlearning_map = torch.from_numpy(unlearning_cam(input_tensor=imgs, targets=target))
            standard_map = torch.from_numpy(standard_cam(input_tensor=imgs, targets=target))
            gold_map = torch.from_numpy(gold_cam(input_tensor=imgs, targets=target))
            ldiff_map = torch.from_numpy(ldiff_cam(input_tensor=imgs, targets=target))

            unlearning_map_up = torch.nn.functional.interpolate(unlearning_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).detach().numpy()
            unlearning_exp = (imgs_up*unlearning_map_up).detach().squeeze().permute(1,2,0).numpy()
            
            standard_map_up = torch.nn.functional.interpolate(standard_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).detach().numpy()
            standard_exp = (imgs_up*standard_map_up).detach().squeeze().permute(1,2,0).numpy()

            gold_map_up = torch.nn.functional.interpolate(gold_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).detach().numpy()
            gold_exp = (imgs_up*gold_map_up).detach().squeeze().permute(1,2,0).numpy()

            ldiff_map_up = torch.nn.functional.interpolate(ldiff_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).detach().numpy()
            ldiff_exp = (imgs_up*ldiff_map_up).detach().squeeze().permute(1,2,0).numpy()

            x=overlay_mask(to_pil_image(imgs_up.squeeze()), to_pil_image(unlearning_map_up.squeeze()), alpha=0.6)
            x.save(
                os.path.join(run_root, f'{i}_ours.png')
            )

            x=overlay_mask(to_pil_image(imgs_up.squeeze()), to_pil_image(standard_map_up.squeeze()), alpha=0.6)
            x.save(
                os.path.join(run_root, f'{i}_standard.png')
            )
            
            x=overlay_mask(to_pil_image(imgs_up.squeeze()), to_pil_image(gold_map_up.squeeze()), alpha=0.6)
            x.save(
                os.path.join(run_root, f'{i}_gold.png')
            )

            x=overlay_mask(to_pil_image(imgs_up.squeeze()), to_pil_image(ldiff_map_up.squeeze()), alpha=0.6)
            x.save(
                os.path.join(run_root, f'{i}_neg_grad.png')
            )

            to_pil_image(imgs_up.squeeze()).save(
                os.path.join(run_root, f'{i}_original.png')
            )
        
            x=0





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', "--dataset", type=str, help='dataset', default='cifar10')
    parser.add_argument('-C', "--cam", type=str, help='CAM name', default='GradCAM')
    parser.add_argument('-n', "--nn", type=str, help='Backbone to use', default='vgg16')
    parser.add_argument('-b', "--batch-size", type=int, help='Untraining batch size', default=32)
    parser.add_argument("--unl_class", type=int, help='numbero of unlearning class', default=0)
    args = parser.parse_args()

    main(args=args)