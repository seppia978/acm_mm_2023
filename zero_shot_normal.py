import os
import torch
import timm
from torch.utils.data import Dataset
import torchvision.datasets
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
# import ast
# from torchviz import make_dot

import argparse
from torch import nn
import numpy as np
import loss_manager

# from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split, Subset
# from sklearn.metrics import accuracy_score

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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# from custom_archs import convert_conv2d_to_alpha, set_label, get_all_alpha_layers, \
#     get_all_layer_norms, set_alpha_val, clip_alpha_val
from custom_archs import wfmodels

# import tqdm

def random_labels_check(a,b):
    ret = 0.
    for ii in range(a.shape[1]):
        ret += (a[:,ii] == b).sum()
    return ret > 0., ret

def generate_random_idxs(idxs_portion, dim1, classes_number):
    ret = (torch.rand(
        int(idxs_portion.shape[0]), dim1
    ) * classes_number).long().to(idxs_portion.device)

    while random_labels_check(ret, idxs_portion)[0]:                    
        for ii in range(ret.shape[1]):
            idxs = torch.argwhere(ret[:,ii] == idxs_portion).flatten()
            ret[idxs,ii] = (torch.rand(
                int(idxs.shape[0])
            ) * classes_number).long().to(ret.device)
    return ret

def show_img(imgs, idx, DeT):
    plt.imshow(DeT(imgs[idx]).permute(1, 2, 0).cpu().detach().numpy())
    plt.show()

def test_others(idx, others, model,_T):
    img = _T(others[idx][0]).unsqueeze(0).cuda()
    print(nn.functional.softmax(model(img)).max(1),others.dataset.targets[idx])

hooked_tensor = None

def hook_fn(model, input, output):
    global hooked_tensor
    hooked_tensor = output.data

def parameters_distance(model:nn.Module, other:nn.Module, kind:str = 'l2'):
        ret = []
        
        for i, ((n,x),(n1,x1)) in enumerate(zip(model.model.named_parameters(), other.named_parameters())):
            x, x1 = x.cuda(), x1.cuda()
            if kind.lower() == 'l2':
                    ret.append(
                        (torch.linalg.norm(
                            torch.pow(x - x1, 2)
                        ))
                        .unsqueeze(0)
                    )
            elif kind.lower() == 'kl-div':
                    ret.append(
                        torch.kl_div(
                            torch.log(x), x1
                        ).mean().unsqueeze(0)
                    )

        return torch.cat(ret, 0)

class MyModel(nn.Module):
   def __init__(self, model):
     super(MyModel, self).__init__()
     self.model = model
     self.weights = nn.Parameter(torch.ones(len(tuple(model.parameters())), dtype=torch.float))
     

   def forward(self, *args, **kwargs):
     return self.model.forward(*args, **kwargs)

def main(args):

    debug = args.debug
    validation = True
    wdb_proj = args.project
    wdb_name = args.name
    lambda0 = args.lambda_0
    lambda1 = args.lambda_1
    lambda2 = args.lambda_2
    loss_type = args.loss_type
    alpha_init = args.alpha_init
    bs = args.batch_size
    zgf = args.zero_grad_frequency
    ef = args.evaluation_frequency
    patience = args.patience
    val_test_size = args.validation_test_size
    ukr = args.unlearnt_kept_ratio
    dataset = args.dataset
    logits = args.logits
    clamp = args.clamp
    flipped = False

    # print(f'Unlearning class {c_to_del[0]}...')

    arch_name = args.nn.lower()
    print(f'rete, model, network: {arch_name}')
    print(f'wdbproj: {wdb_proj}, wdbname: {wdb_name}')

    perc = args.percentage
    if perc > 1 and perc <= 100:
        perc /= 100
    if perc < 0 or perc > 100:
        raise ValueError(f'The percentage must be in [0,1], or in [0,100]. Found {perc}.')

    ur = args.unlearning_rate 

    hyperparams = {
        'loss_type': loss_type,
        'ur': ur,
        'model': arch_name,
        'lambda0': lambda0,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'g': 1e-2,
        'alpha_init': alpha_init,
        'batch_size': bs,
        'zero_grad_frequency': zgf,
        'evaluation_frequency': ef,
        'initial_patience': patience,
        'max_val_size': val_test_size,
        'unlearnt_kept_ratio': ukr,
        'dataset': dataset,
        'logits': logits,
        'clamp': clamp,
        'flipped': flipped
    }

    if logits:
        cls_loss = nn.MSELoss
    else:
        cls_loss = nn.CrossEntropyLoss

    LM = loss_manager.loss.LossManager(
        loss_type=loss_type,lambdas=[hyperparams[l] for l in hyperparams.keys() if 'lambda' in l],
        classification_loss_fn = cls_loss, logits=logits
    )

    print([(k,v) for k,v in hyperparams.items()])

    wdb_name = f'{wdb_name}_{arch_name}_{perc}_{ur}'
    if not debug:
        run = wandb.init(
            settings=wandb.Settings(start_method="fork"),
            reinit=True, config=hyperparams, project=f"{wdb_proj}", name=f'{wdb_name}', entity="unl4xai"
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    root = '//mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16'
    PATH = f"{root}/final.pt"

    ret_acc, unl_acc = torch.zeros(1), torch.zeros(1)

    if 'cifar10' in dataset.lower():
        c_number = 10
    elif 'cifar20' in dataset.lower():
        c_number = 20
    else:
        print("Dataset not supported")
        return -1
    
    all_classes = range(c_number)

    for class_to_delete in all_classes:

        print(f'############################ Class to unlearn: {class_to_delete} ############################')

        if 'vgg16' in arch_name:
            model = wfmodels.WFCNN(
                kind=wfmodels.vgg16, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFCNN(
                kind=wfmodels.vgg16, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        elif "resnet34" in arch_name:
            # model = models.resnet18(pretrained=True)
            model = wfmodels.WFCNN(
                kind=wfmodels.resnet34, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower()
            )

            general = wfmodels.WFCNN(
                kind=wfmodels.resnet34, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
        elif 'resnet18' in arch_name:
            model = wfmodels.WFCNN(
                kind=wfmodels.resnet18, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFCNN(
                kind=wfmodels.resnet18, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        elif 'deit_small_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.deit_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower()
            )

            general = wfmodels.WFTransformer(
                kind=wfmodels.deit_small_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
        elif 'vit_small_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.vit_small_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)

            general = wfmodels.WFTransformer(
                kind=wfmodels.vit_small_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )

            standard = general.arch
        elif 'vit_tiny_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            model = MyModel(model.arch)
            general = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
            standard = general.arch
        elif 'swin_small_16224' in arch_name:
            model = wfmodels.WFTransformer(
                kind=wfmodels.swin_small_16224, pretrained=False,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower()
            )

            general = wfmodels.WFTransformer(
                kind=wfmodels.swin_small_16224, pretrained=True,
                m=hyperparams['alpha_init'], resume=None,
                dataset=dataset.lower(), alpha=False
            )
        general.arch.requires_grad_(requires_grad=False)
        model = model.train()
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

            transform_test = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            _train = CIFAR10(
                root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
                transform=T, download=True, train=True
            )

            _val = CIFAR10(
                root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/val',
                transform=transform_test, download=True, train=False
            )

            c_to_del = [class_to_delete]

            id_c = np.where(np.isin(np.array(
                _val.targets
                ), c_to_del))[0]
            val_c = Subset(_val, id_c)
            val_c.targets = torch.Tensor(_val.targets).long()[id_c].tolist()

            id_ot = np.where(~np.isin(np.array(
                _val.targets
                ), c_to_del))[0]
            val_ot = Subset(_val, id_ot)
            val_ot.targets = torch.Tensor(_val.targets).long()[id_ot].tolist()

            if 'zero' in hyperparams['loss_type']:

                id_c = np.where(np.isin(np.array(_train.targets), c_to_del))[0]
                train_c = Subset(_train, id_c)
                train_c.targets = torch.Tensor(_train.targets).int()[id_c].tolist()

                train = train_c
            else:
                train = _train
            val = (val_c, val_ot)
            train.num_classes = 10
            train.real_targets = train.targets
            val[0].num_classes, val[1].num_classes = 10, 10


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

            c_to_del = [class_to_delete]

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

        
        # model.requires_grad_(requires_grad=False)
        # model = convert_conv2d_to_alpha(model, m=hyperparams['alpha_init'])
        
        root='//mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/'
        # root='//mnt/beegfs/work/dnai_explainability/ssarto/alpha_matrices/'

        if not os.path.isdir(os.path.join(root,wdb_proj)):
            os.mkdir(os.path.join(root,wdb_proj))

        folder_name = f'{wdb_name}_{datetime.today().strftime("%Y-%m-%d")}'
        folder_name += f'-{len(os.listdir(os.path.join(root,wdb_proj)))}'
        run_root = os.path.join(root, wdb_proj, folder_name)
        if not os.path.isdir(os.path.join(root,wdb_proj,folder_name)):
            os.mkdir(run_root)

        with open(f"{run_root}/config", 'w') as f:
            f.write(str(hyperparams))
        # else:
        #     folder_name = root
        #     run_root = folder_name
        #     with open(f"{folder_name}/config_2", 'w') as f:
        #         f.write(str(hyperparams))


        # optimizer=Adam((x for n,x in model.named_parameters() if 'alpha' in n), lr=hyperparams['ur'])
        # optimizer=SGD((x for n,x in model.named_parameters() if 'alpha' in n), lr=hyperparams['ur'])
        optimizer=torch.optim.SGD(model.parameters(), lr=hyperparams['ur'])


        print("Untraining...")
        #TRAINING LOOP
        def train_loop(
                n_epochs,
                optimizer,
                model,
                loss_fn,
                train,
                val,
                hyp,
                general=None
        ):
            should_stop = False
            patience = hyp['initial_patience']
            best_acc = 0.
            model.eval()

            save_checkpoint_frequency = 50
            validation_frequency = int(len(train)/1000) if int(len(train)/1000) > 0 else 10
            # evaluation_frequency = hyp['evaluation_frequency']
            evaluation_frequency = 0 \
                if hyp['evaluation_frequency'] == 0 \
                else int(len(train)/ (hyp['evaluation_frequency'] * hyp['batch_size']))
            if evaluation_frequency:
                validation_frequency = evaluation_frequency
            elif evaluation_frequency == 0:
                evaluation_frequency = validation_frequency
            best_found = False
            # c_to_del = [0]
            # id_c = np.where(np.isin(np.array(train.targets), c_to_del))[0]
            # id_others = np.where(~ np.isin(np.array(train.targets), c_to_del))[0]

            # train_c = Subset(train, id_c)
            # train_others = Subset(train, id_others)

            # train_c.targets = torch.Tensor(train.targets).int()[id_c].tolist()
            # train_others.targets = torch.Tensor(train.targets).int()[id_others].tolist()
            
            # concat_train = data.ConcatDataset((train_c,train_others))
            # concat_train.targets = [*train_c.targets, *train_others.targets]

            with open('class_names/names.txt', 'r') as f:
                txt = f.read()
            
            import ast
            classes = ast.literal_eval(txt)

            batch_train = batch_val = hyp['batch_size']
            # train_loader = DataLoader(train, batch_size=batch_train, shuffle=True)

            # val_c,val_ot=val
            # max_val_size=20_000
            # size_val = min(max_val_size, int(len(val_c)))
            # size_val = max(max_val_size, int(.1 * len(val_ot)))
            # otval_loader = DataLoader(random_split(val_ot, [size_val, len(val_ot) - size_val])[0], batch_size=64, shuffle=True)

            # otval_loader.dataset.dataset.names = classes
            # cval_loader.dataset.dataset.names = classes

            model.to(device)

            # confusion_matrix = torch.zeros(1000)

            # x_cls_del = 0
            # pl = 0
            for epoch in range(n_epochs):
                if should_stop:
                    break

                print(f'Untrain {epoch=} ')
                if not debug:
                    run.log({'epoch': epoch})

                # * balancing the batch with 50% samples from THE class and 50% from the others
                if 'zero' not in hyp['loss_type']:
                    y_train = train.real_targets  # train.datasets[0].dataset.targets
                    
                    weight = 1. / torch.Tensor([1/train.num_classes for _ in range(train.num_classes)])
                    
                    weight[~np.isin(list(range(len(weight))), np.array(c_to_del))] = .5 / (len(weight) - len(c_to_del))
                    weight[np.array(c_to_del)] = .5 / len(c_to_del)
                    
                    samples_weight = np.array([weight[t] for t in y_train])
                    samples_weight = torch.from_numpy(samples_weight)
                    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

                    train_loader = DataLoader(train, batch_size=batch_train, num_workers=4, sampler=sampler)
                else:
                    train_loader = DataLoader(train, batch_size=batch_train, shuffle=True)

                # loss_ctrain_list, loss_ottrain_list, loss_otval_list, loss_cval_list = [], [], [], []

                for idx, (imgs, labels) in enumerate(train_loader):
                    print(f'Untraining: {round((100 * batch_train * idx)/len(train_loader.dataset),2)}%')

                    # * setting images and labels to device
                    imgs=imgs.to(device)#.requires_grad_(True)
                    labels=labels.to(device)
                    # c_to_del = [torch.unique(labels).squeeze().tolist()]

                    # * reordering images and labels to have [unlearning_images, retaining_images]
                    # imgsn = []
                    # labelsn = []

                    # for i in range(imgs.shape[0]):
                    #     if labels[i] == c_to_del[0]:
                    #         labelsn.append(labels[i].unsqueeze(0))
                    #         imgsn.append(imgs[i].unsqueeze(0))

                    # half = len(labelsn)
                    # for i in range(imgs.shape[0]):
                    #     if labels[i] != c_to_del[0]:
                    #         labelsn.append(labels[i].unsqueeze(0))
                    #         imgsn.append(imgs[i].unsqueeze(0))
                    
                    # imgs = torch.cat(imgsn, 0).to(device)
                    # labels = torch.cat(labelsn, 0).to(device)
                    # kept_labels = (torch.rand(
                    #     int(labels.shape[0] / 2), hyp['unlearnt_kept_ratio']
                    # ) * classes_number).long()


                    # * saving unlearning labels and images
                    # unlearnt_labels = labels[:half]
                    # unlearnt_imgs = imgs[:half]

                    # * saving retaining images and labels.
                    # * NB: for the single class case the retaining labels must be
                    # * the same as unlearning labels! (i.e. corresponging to the class to forget)

                    # kept_portion = unlearnt_labels.clone()
                    # if not 'zero' in hyp['loss_type']:
                        # kept_portion = labels.cpu()[:int(labels.cpu().shape[0] / 2)].clone()
                    # while random_labels_check(kept_labels, kept_portion)[0]:                    
                    #     for ii in range(kept_labels.shape[1]):
                    #         idxs = torch.argwhere(kept_labels[:,ii] == kept_portion).flatten()
                    #         kept_labels[idxs,ii] = (torch.rand(
                    #             int(idxs.shape[0])
                    #         ) * classes_number).long()
                    
                    # kept_labels = generate_random_idxs(
                    #     torch.Tensor([c_to_del[0] for _ in labels[half:]]),
                    #     hyp['unlearnt_kept_ratio'], classes_number
                    # )
                    if not 'zero' in hyp['loss_type']:
                        # kept_labels = generate_random_idxs(
                        #     kept_portion, hyp['unlearnt_kept_ratio'], classes_number
                        # )
                        kept_labels = labels[labels!=c_to_del[0]]
                        kept_imgs = imgs[labels!=c_to_del[0]]

                        unlearnt_labels = labels[labels==c_to_del[0]]
                        unlearnt_imgs = imgs[labels==c_to_del[0]]
                    else:
                        unlearnt_labels = labels
                        unlearnt_imgs = imgs

                    # * NB: for the single class case the retaining labels must be
                    # * the same as unlearning labels (i.e. corresponging to the class to forget)
                    # kept_labels = torch.ones_like(kept_labels) * c_to_del[0]

                    # kept_imgs = imgs[half:]
                    # kept_labels = labels[half:]
                    # kept_imgs = imgs[:int(labels.shape[0] / 2)]

                    # kept_labels = ((labels[:int(labels.shape[0] / 2)] + 1) % classes_number).view(-1,1)

                    # unlearnt_labels = labels[int(labels.shape[0] / 2):]

                    # set_label(model, unlearnt_labels)

                    # if exists, make a forward step in a normal model
                    # for debug only
                    # if general is not None:
                    #     general..cuda().eval()
                    #     general_score = general(unlearnt_imgs)

                    
                    # * forward step and loss computation for unlearning
                    unlearnt_score = model(unlearnt_imgs)
                    # unlearnt_loss = loss_fn(unlearnt_score, unlearnt_labels)
                    unlearnt_loss = LM.classification_loss_selector(
                        unlearnt_score, unlearnt_imgs, 
                        unlearnt_labels, None
                    )

                    # * forward steps and loss computations for retaining
                    # * eventually the losses are averaged
                    if not 'zero' in hyp['loss_type']:
                        # kept_loss_list=[]

                        # if general is not None:
                        #         general.cuda().eval()
                        #         general_score = general(kept_imgs)

                        # for k in range(hyp['unlearnt_kept_ratio']):
                            # set_label(model, kept_labels[:,k].squeeze())
                        kept_score = model(kept_imgs)
                            # kept_loss_list.append(
                            #     loss_fn(
                            #         kept_score,
                            #         labels[:int(labels.shape[0] / 2)
                            #         ].squeeze()
                            #     ).unsqueeze(0)
                            # )
                        # kept_loss_list.append(
                        kept_loss = LM.classification_loss_selector(
                            kept_score, kept_imgs,
                            kept_labels, general
                        )
                        # )

                        # kept_loss = torch.cat(kept_loss_list).mean(dim=0)
                        # kept_loss = torch.cat(kept_loss_list).mean(dim=0)

                    # * computes the final loss
                    # * loss = lambda_0 * loss_ret^2 + lambda_1 * 1 / (loss_unl) + lambda_2 * alpha_norm

                    if hyp['loss_type'] == 'difference':
                        keep = kept_loss.clone()
                        unlearn = unlearnt_loss.clone()
                        loss_cls = hyp['lambda0'] * keep.mean() - hyp['lambda1'] * unlearn.mean()
                        loss_train = loss_cls.mean()
                        # loss_train += alpha_norm
                    elif '3way' in hyp['loss_type']:
                        if 'zero' not in hyp['loss_type']:
                            keep = kept_loss.clone()
                        unlearn = unlearnt_loss.clone()
                        if 'multiplication' in hyp['loss_type']:
                            loss_cls = (hyp['lambda0'] * keep.mean() / (hyp['lambda1'] * torch.abs(unlearn.mean())))
                            # loss_train = loss_cls + alpha_norm
                        elif 'sum' in hyp['loss_type']:
                            loss_cls = torch.pow(hyp['lambda1']/(unlearn.mean()+1e-8),1)
                            loss_reg = torch.pow(hyp['lambda0'] * keep.mean(),2)
                            loss_reg_weighted = loss_reg
                            loss_train = loss_cls + loss_reg
                        elif 'zero' in hyp['loss_type']:
                            loss_cls = torch.pow(1. / (unlearn.mean() + 1e-8), 1)
                            loss_reg = torch.pow(parameters_distance(model, standard, kind='l2'), 2)

                            if 'fixed' in hyp['loss_type']:
                                weights = 1.
                            elif 'learnable' in hyp['loss_type']:
                                weights = model.weights
                            elif 'inv-square' in hyp['loss_type']:
                                weights = torch.tensor(tuple(
                                    math.pow(
                                    len(tuple(model.model.parameters())) - i, 2
                                    ) for i in range(len(tuple(model.model.parameters())))
                                ), device=device)
                                
                            loss_reg_weighted = (loss_reg * weights).sum()
                            # loss_reg_weighted = loss_reg
                            alpha_norm = 0 # hyp['lambda2'] * (model.get_all_layer_norms(m=1.)).mean().to('cuda')
                            loss_train = hyp['lambda0'] * loss_reg_weighted + hyp['lambda1'] * loss_cls

                        elif 'third' in hyp['loss_type']:
                            loss_cls = hyp['lambda0'] * keep.mean() * (1 - hyp['lambda1'] / torch.abs(unlearn.mean())) # l+ * (1 - lambda1/l-)
                            # loss_train = loss_cls + alpha_norm
                    else:
                        loss_train = loss_cls.mean()

                    # import pdb; breakpoint()


                    # loss_ctrain_list.append(mean_loss_ctrain.cpu().item())
                    # loss_ottrain_list.append(mean_loss_ottrain.cpu().item())

                    # * zeroing and backwarding the grads, followed by an optimizer step
                    # * NB: zeroing and stepping frequencies are handled by
                    # * the zero_grad_frequency hyperparameter
                    # if (idx + 1)  % hyp['zero_grad_frequency'] == 0:
                    #     optimizer.zero_grad()

                    loss_train.backward()
                    # print(model.features[0].alpha.grad.mean())

                    if (idx + 1) % hyp['zero_grad_frequency'] == 0 or (idx + 1) * hyp['batch_size'] > len(train):

                        if hyp['clamp'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyp['clamp'])
                        optimizer.step()
                        optimizer.zero_grad()

                    # * clipping alpha values between max and min
                    # model.clip_alphas() # clip alpha vals
                    # model.weights.data = torch.relu(model.weights)

                    # * wandb loggings
                    if not debug:
                        # run.log({'alpha_norm': alpha_norm})
                        run.log({'loss': loss_train})
                        run.log({'unlearning_loss': loss_cls})
                        # run.log({'train_keep': keep.mean()})
                        run.log({'retaining_loss': loss_reg_weighted})
                        run.log({'layer_distance_sum': loss_reg.sum()})
                        run.log({'unlearning_factor': torch.abs(unlearn.mean())})
                        run.log({'weights_mean': model.weights.abs().mean()})
                        # run.log({'l1c1_alpha_max': tuple(model.get_all_alpha_layers().values())[0].max()})
                        # run.log({'l1c1_alpha_min': tuple(model.get_all_alpha_layers().values())[0].min()})

                    # * validation steps:
                    # * firstly the unlearning validation step
                    # * secondly the retaining validation step
                    if idx % validation_frequency == 0 and validation:
                        
                        # * saving CUDA memory
                        unlearnt_loss.cpu()
                        loss_reg.cpu()
                        loss_train.cpu()

                        unlearnt_score.cpu()
                        # kept_score.cpu()

                        unlearnt_labels.cpu()
                        # kept_labels.cpu()

                        loss_cls.cpu()
                        # alpha_norm.cpu()

                        imgs.cpu()
                        labels.cpu()

                        print(f'Validation step {idx}')

                        # * defining max number of validation images
                        max_val_size = min(hyp['max_val_size'], len(val[0]))

                        # val_loader = DataLoader(
                        #     random_split(val, [max_val_size, len(val)-max_val_size])[0],
                        #     batch_size=batch_val, num_workers=1, shuffle=True
                        # )

                        # * selecting the unlearning and the retaining validation sets
                        # * useless in the multiclass case
                        # id_c = np.where(np.isin(np.array(val.targets), c_to_del))[0]
                        # id_others = np.where(~ np.isin(np.array(val.targets), c_to_del))[0]

                        # val_c = Subset(val, id_c)
                        # val_others = Subset(val, id_others)

                        # val_c.targets = torch.Tensor(val.targets).int()[id_c].tolist()
                        # val_others.targets = torch.Tensor(val.targets).int()[id_others].tolist()

                        # concat_val = data.ConcatDataset((val_c,val_others))
                        # concat_val.targets = [*val_c.targets, *val_others.targets]


                        val_c, val_others = val
                        u_val_loader = DataLoader(
                            random_split(val_c, [max_val_size, len(val_c)-max_val_size])[0],
                            batch_size=batch_val, num_workers=4, shuffle=True
                        )
                        k_val_loader = DataLoader(
                            random_split(val_others, [max_val_size, len(val_others)-max_val_size])[0],
                            batch_size=batch_val, num_workers=4, shuffle=True
                        )

                        mean_acc_forget = mean_acc_keep = 0.

                        with torch.inference_mode():
                            for ival, (ims, labs) in enumerate(u_val_loader):
                                ims=ims.cuda()
                                # labs=labs.cuda()

                                # set_label(model, labs)

                                if len(c_to_del) > 0:
                                    outs = torch.softmax(
                                        model(ims),
                                        -1
                                    ).cpu()
                                else:
                                    outs = torch.softmax(model(ims, labels=torch.zeros(1, device='cuda')), -1).cpu()

                                mean_acc_forget += (outs.max(1).indices == labs).sum() / \
                                                    labs.shape[0]

                            mean_acc_forget /= (ival + 1)
                            ims.cpu()
                            for ival, (ims, labs) in enumerate(k_val_loader):
                                ims = ims.cuda()

                                if len(c_to_del) > 0:
                                    outs = torch.softmax(
                                        model(ims),
                                        -1
                                    ).cpu()
                                else:
                                    outs = torch.softmax(model(ims), -1).cpu()

                                mean_acc_keep += (outs.max(1).indices == labs).sum() / \
                                                    labs.shape[0]

                            mean_acc_keep /= (ival + 1)
                            ims.cpu()
                        if not debug:
                            run.log({'acc_on_unlearnt': mean_acc_forget})
                            run.log({'acc_of_kept': mean_acc_keep})
                        # if general is not None and False:
                        #     mean_acc_gen=0.
                        #     general.cuda()
                        #     general.eval()
                        #     for ival, (ims, labs) in enumerate(val_loader):
                        #             ims = ims.cuda()
                        #             # labs = labs.cuda()

                        #             labs_portion = labs.clone()

                        #             CLASS = generate_random_idxs(
                        #                 labs_portion, 1, classes_number
                        #             ).squeeze()
                        #             # set_label(model, CLASS.cuda())

                        #             outs = torch.softmax(general(ims,labels=labs.cuda()), -1).cpu()

                        #             mean_acc_gen += (outs.max(1).indices == labs).sum() / \
                        #                                 labs.shape[0]

                        #     mean_acc_gen /= (ival + 1)

                        # current_acc = 0.5 * (1-mean_acc_forget)
                        current_acc = 0.5 * ((1-mean_acc_forget) + mean_acc_keep)
                        if evaluation_frequency and idx % evaluation_frequency == 0:
                            if current_acc < best_acc:
                                patience -= 1
                                currents.append(current_acc)
                                if patience == 0:
                                    should_stop = True
                            else: 
                                best_acc = current_acc
                                best_acc_both = (mean_acc_keep, mean_acc_forget)
                                currents = list()
                                patience = hyp['initial_patience']
                                best_found = True
                        if not debug:
                            run.log({'best_val_acc': best_acc})
                            run.log({'current_val_acc': current_acc})
                            run.log({'best_acc_ret': best_acc_both[0]})
                            run.log({'best_acc_unl': best_acc_both[1]})

                        if should_stop:
                            print(f'mean_unl: {mean_acc_forget}, current: {currents}, best: {best_acc}, patience: {patience}')
                            return best_acc_both
                    
                    # * saving intermediate checkpoints
                    if idx % save_checkpoint_frequency == 0 or best_found:
                        #root = '/work/dnai_explainability/unlearning/icml2023'
                        PATH = f"{run_root}/last_intermediate.pt" if not best_found else f"{run_root}/best.pt"
                        # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/all_classes/{arch_name}_{perc}_{ur}_class-{c_to_del[0]}_alpha.pt"
                        # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/short/class_100-classes_freeze-30perc_untraining_{hyperparams['model']}.pt"
                        torch.save(model.state_dict(), PATH)
                        best_found = False
                        print(f'Saved at {PATH}')

        if flipped:
            trainset, valset = val, train,
        else:
            trainset, valset = train, val

        best_acc_both = train_loop(
            n_epochs = 100_000,
            optimizer = optimizer,
            model = model,
            loss_fn = cls_loss,
            train=trainset,
            val=valset,
            hyp=hyperparams,
            general=standard
        )

        ret_acc += best_acc_both[0]
        unl_acc += best_acc_both[1]
        PATH = f"{run_root}/final.pt"
        torch.save(model.state_dict(), PATH)
        print(f'Saved at {PATH}')
        # print(model.weights)
        if not debug:
            wandb.log({
                "mean_ret_acc": ret_acc/(class_to_delete+1),
                "mean_unl_acc": unl_acc/(class_to_delete+1)
            })

        x=0

    print(f'ret: {ret_acc/len(tuple(all_classes))}, unl: {unl_acc/len(tuple(all_classes))}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', "--debug", type=bool, help='debug', default=False)
    parser.add_argument('-p', "--percentage", type=float, help='dataset', default=100)
    parser.add_argument('-D', "--dataset", type=str, help='dataset', default='imagenet')
    parser.add_argument('-0', "--lambda-0", type=float, help='lambda0', default=1.)
    parser.add_argument('-1', "--lambda-1", type=float, help='lambda1', default=1.)
    parser.add_argument('-2', "--lambda-2", type=float, help='lambda2', default=1.)
    parser.add_argument('-u', "--unlearning-rate", type=float, help='Unlearning Rate', default=1e1)
    parser.add_argument('-L', "--loss-type", type=str, help='Loss function type', default='sum')
    parser.add_argument('-P', "--project", type=str, help='WanDB project', required=True)
    parser.add_argument('-N', "--name", type=str, help='WandDB name', required=True)
    parser.add_argument('-n', "--nn", type=str, help='Backbone to use', default='vgg16')
    parser.add_argument('-c', "--tounlearn", type=int, help='Idx of the class to unlearn', default=2)
    parser.add_argument('-a', "--alpha-init", type=float, help='Initialization value for alpha', default=5.)
    parser.add_argument('-b', "--batch-size", type=int, help='Untraining batch size', default=32)
    parser.add_argument('-z', "--zero-grad_frequency", type=int, help='Zero grady frequency', default=1)
    parser.add_argument("--evaluation-frequency", type=int, help='Evaluation frequency', default=5)
    parser.add_argument("--patience", type=int, help='Initial patience', default=15)
    parser.add_argument('-T', "--validation-test-size", type=int, help='Validation test size', default=20_000)
    parser.add_argument('-R', "--unlearnt-kept-ratio", type=int, help='Unlearnt-kept ratio', default=5)
    parser.add_argument('-l', "--logits", type=bool, help='Compute loss over logits or labels', default=False)
    parser.add_argument("--clamp", type=float, help='Gradient clamping val', default=-1.0)
    args = parser.parse_args()

    main(args=args)
