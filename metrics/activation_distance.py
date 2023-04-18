import os
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from datetime import datetime
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import ast
from torchviz import make_dot

import argparse
from torch import nn
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset

import warnings
import sys
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
from scipy.spatial.distance import jensenshannon

# setting path
sys.path.append('/work/dnai_explainability/ssarto/Unlearning4XAI')
import custom_archs
from custom_archs import WTFCNN
from custom_archs import WTFLayer
from torch import linalg as LA
import timm
import torchvision.transforms as transforms
import torchvision
# from custom_archs.non_imagenet_models.vgg import VGG
from custom_archs.non_imagenet_models.vgg import VGG
from custom_archs.non_imagenet_models.resnet import ResNet18, ResNet34
from custom_archs.non_imagenet_models.swin import swin_s
from custom_archs.non_imagenet_models.vit import ViT 
import json

def load_gold_model(arch_name, dataset, gold_checkpoint_path, size):
    if 'vgg'in arch_name:
        # gold version --> trained without a specific class
        gold = VGG('VGG16')
        random_model = VGG('VGG16')
        if dataset == "MNIST":
            gold.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
            random_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
    elif 'res18' in arch_name:
        gold = ResNet18()
        random_model = ResNet18()
        if dataset == "MNIST":
            gold.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            random_model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif 'res34' in arch_name:
        gold = ResNet34()
        random_model = ResNet34()
        if dataset == "MNIST":
            gold.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            random_model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif 'swin' in arch_name:
        if dataset == "MNIST":
            n_channels = 1
        else:
            n_channels = 3
        gold = swin_s(
            window_size=4,
            channels=n_channels,
            num_classes=10,
            downscaling_factors=(2,2,2,1)
        )
        random_model = swin_s(
            window_size=4,
            channels=n_channels,
            num_classes=10,
            downscaling_factors=(2,2,2,1)
        )
    elif 'vit_small' in arch_name:
        if dataset == "MNIST":
            n_channels = 1
        else:
            n_channels = 3
        # ViT for cifar10
        gold = ViT(
            channels=n_channels,
            image_size = size,
            patch_size = 4,
            num_classes = 10,
            dim = 384,
            depth = 12,
            heads = 8,
            mlp_dim = 384,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        random_model = ViT(
            channels=n_channels,
            image_size = size,
            patch_size = 4,
            num_classes = 10,
            dim = 384,
            depth = 12,
            heads = 8,
            mlp_dim = 384,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    elif 'vit_tiny' in arch_name:
        if dataset == "MNIST":
            n_channels = 1
        else:
            n_channels = 3
        # ViT for cifar10
        gold = ViT(
            channels=n_channels,
            image_size = size,
            patch_size = 4,
            num_classes = 10,
            dim = 192,
            depth = 12,
            heads = 8,
            mlp_dim = 192,
            dropout = 0.1,
            emb_dropout = 0.1
        )  
        random_model = ViT(
            channels=n_channels,
            image_size = size,
            patch_size = 4,
            num_classes = 10,
            dim = 192,
            depth = 12,
            heads = 8,
            mlp_dim = 192,
            dropout = 0.1,
            emb_dropout = 0.1
        )  
    
    loaded = torch.load(gold_checkpoint_path)
    loaded = loaded['model']
    ac=list(map(lambda x: x[7:], loaded.keys()))
    ckp = dict()
    for k1,k2 in zip(loaded,ac):
        if k2 == k1[7:]:
            ckp[k2] = loaded[k1]
    gold.load_state_dict(ckp)
    
    gold.eval()

    random_model.eval()
    return gold, random_model

def load_models(args):
    hyperparams = {
        'alpha_init': 3,
    }
    arch_name = args.arch
    dataset = args.dataset
    baseline = args.baseline
    with open('/mnt/beegfs/work/dnai_explainability/final_ckpts.json', 'r') as fp:
        data = json.load(fp)

    checkpoint_path = data[baseline][f'{arch_name}_{dataset.lower()}']
    
    # if os.path.exists(os.path.join(checkpoint_path, 'best.pt')):
    #     checkpoint_path = os.path.join(checkpoint_path, 'best.pt')
    # elif os.path.exists(os.path.join(checkpoint_path, 'final.pt')):
    #     checkpoint_path = os.path.join(checkpoint_path, 'final.pt')
    # else:
    checkpoint_path = os.path.join(checkpoint_path, 'last_intermediate.pt')
        
    if 'vgg'in arch_name:
        # DA TROVARE CHECKPOINT
        # if dataset == "CIFAR10":
        #     checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/vgg16-1-10--no-logits-0_vgg16_1.0_100.0_2023-03-01-243/best.pt'
        # else:
        #     pass
        #     # checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/vgg16-1-10--no-logits-0_vgg16_1.0_100.0_2023-03-01-243/best.pt'

        # unlearned model
        model_unl = WTFCNN.WTFCNN(
            kind=WTFCNN.vgg16, pretrained=True,
            m=hyperparams['alpha_init'], resume=checkpoint_path,
            dataset=dataset.lower()
        )
        
        # general version == a zrf 
        general=WTFCNN.WTFCNN(
            kind=WTFCNN.vgg16, pretrained=True,alpha=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )
    # ! DO NOT USE IT
    elif "res34" in arch_name:
        # model = models.resnet18(pretrained=True)
        
        checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-0'
        # unlearned model
        model_unl=WTFCNN.WTFCNN(
            kind=WTFCNN.resnet34, pretrained=False,
            m=hyperparams['alpha_init'], resume=checkpoint_path,
            dataset=dataset.lower()
        )
        
        general=WTFCNN.WTFCNN(
            kind=WTFCNN.resnet34, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )
    elif 'res18' in arch_name:
        # if dataset == "CIFAR10":
        #     checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/resnet18-1-10--no-logits-0_resnet18_1.0_100.0_2023-03-01-240/best.pt'
        # else:
        #     pass
        
        # unlearned model
        model_unl = WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,
            m=hyperparams['alpha_init'], resume=checkpoint_path,
            dataset=dataset.lower()
        )
        
        # general version == a zrf 
        general=WTFCNN.WTFCNN(
            kind=WTFCNN.resnet18, pretrained=True,alpha=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

    elif 'vit_small' in arch_name: # DA FINIRE --> manca vit tiny
        # if dataset == "CIFAR10":
        #     checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/vit_small_16224-1-100--no-logits-0_vit_small_16224_1.0_100.0_2023-03-01-244/best.pt'
        # else:
        #     checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/vit_small_16224-1-100--no-logits-0_vit_small_16224_1.0_100.0_2023-03-01-257/best.pt'
        # unlearned model
        model_unl = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_small_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=checkpoint_path,
            dataset=dataset.lower()
        )
        
        # general version == a zrf 
        general=WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_small_16224, pretrained=True,alpha=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )
    elif 'vit_tiny' in arch_name: # DA FINIRE --> manca vit tiny
        # if dataset == "CIFAR10":
        #     checkpoint_path = '/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/debug-vitt-cifar10_vit_tiny_16224_1.0_100.0_2023-03-01-211/best.pt'
        # else:
        #     checkpoint_path = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/vit_tiny_16224-1-100--no-logits-0_vit_tiny_16224_1.0_100.0_2023-03-02-266/best.pt'
         # unlearned model
        model_unl = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_tiny_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=checkpoint_path,
            dataset=dataset.lower()
        )
        
        # general version == a zrf 
        general = WTFCNN.WTFTransformer(
            kind=WTFCNN.vit_tiny_16224, pretrained=True,alpha=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )
    elif 'swin' in arch_name: # DA FINIRE
        model_unl = WTFCNN.WTFTransformer(
            kind=WTFCNN.swin_small_16224, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general = WTFCNN.WTFTransformer(
            kind=WTFCNN.swin_small_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )

        general.arch.requires_grad_(requires_grad=False)


    model_unl.arch.eval()
    general.arch.eval()

    return model_unl, general


def load_dataset(dataset, size):
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

            _train = ImageFolder(
                root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/train',
                transform=T
            )

            _val = ImageFolder(
                root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/val',
                transform=T
            )

    if dataset.lower() == "cifar10":
        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    elif dataset.lower() == 'mnist':
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
        testset = MNIST(root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val/',
                                train=False, transform=transform_test)

    classes_number = len(testset.classes)
    return testset, classes_number

def compute_activation_distance(model, gold, val_loader, device, args):
    running_sum = 0
    counter = 0
    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            gold.to(device)
            out_gold = gold(x)

            if args.with_unlearned:
                model.arch.to(device)
                out_unlearned = model(x, labels=args.gold_class)
                # activations = activation_unlearned
            else:
                model.to(device)
                out_unlearned = model(x)
                # activations = activation_random
            
            out_gold = torch.softmax(out_gold, dim=1)
            out_unlearned = torch.softmax(out_unlearned, dim=1)
            # for val_gold, val_unl in zip(activation_gold.values(), activations.values()):
            # for val_gold, val_unl in zip(activation_gold.values(), activation_random.values()):
            running_sum += (out_gold-out_unlearned).pow(2)\
                .sum(dim=tuple(int(i)+1 for i in range(len(out_gold.shape[1:]))))\
                .sqrt().sum()
            counter += len(y)

    
    activation_distance = running_sum / counter
    return activation_distance

def compute_js_divergence(model, gold, val_loader, device, args):
    jsu = 0
    len_dataset = args.len_val_dataset # len val dataset
    with torch.inference_mode():
        for x,y in val_loader:
            x, y = x.to(device), y.to(device)
            gold.to(device)
            if args.with_unlearned:
                model.arch.to(device)
                cscore = torch.softmax(model(x, labels=args.gold_class),-1).cpu()
            else:
                model.to(device)
                cscore = torch.softmax(model(x),-1).cpu()
            # model.arch.to(device)
            bscore = torch.softmax(gold(x),-1).cpu()
            # cscore = torch.softmax(model(x, labels=args.gold_class),-1).cpu()
            
            jsu += np.nansum(np.power(jensenshannon(cscore+1e-15,bscore+1e-15, axis=1),2))
    final_jsu = jsu / len_dataset
    return final_jsu

def compute_zrf(model_unl, general, random_model, val ):

    general.arch.to(device)
    random_model.to(device)
    model_unl.arch.to(device)
    zrf_original_total = 0
    zrf_unlearned_total = 0
    for i in range(10):
        id_c = np.where(np.isin(np.array(val.targets), [i]))[0]
        val_c = Subset(val, id_c)

        s=len(val_c)
        val_loader = DataLoader(random_split(val_c, [s, len(val_c)-s])[0], batch_size=16, num_workers=4, drop_last=True)
        jsu,jsg=0,0
        with torch.inference_mode():
            import time
            t0 = time.time()
            for i,l in val_loader:
                i,l = i.cuda(),l.cuda()
            
                ascore = torch.softmax(general(i),-1).cpu()
                bscore = torch.softmax(random_model(i),-1).cpu()
                cscore = torch.softmax(model_unl(i, labels=l),-1).cpu()
                
                jsu += np.nansum(np.power(jensenshannon(cscore+1e-15,bscore+1e-15, axis=1),2))
                jsg += np.nansum(np.power(jensenshannon(ascore+1e-15,bscore+1e-15, axis=1),2))
                
            tf = time.time()
        zrf_unlearning = 1 - jsu/s
        zrf_original = 1 - jsg/s

        zrf_original_total += zrf_original
        zrf_unlearned_total += zrf_unlearning
        # print(f'ZRF unlearning: {zrf_unlearning}, time: {tf-t0}')
        # print(f'ZRF original: {zrf_original}, time: {tf-t0}')

        # print(f'ratio: {(1 - jsu/s)/(1 - jsg/s)}, diff: {(1 - jsu/s)-(1 - jsg/s)}')
    return zrf_unlearned_total, zrf_original_total



def compute_accuracy(model, val_loader, device, args, class_to_delete, kind='unlearned'):
    correct_deleted = 0
    correct_retained = 0
    total_deleted = 0
    total_retained = 0
    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
 
            if kind == 'unlearned':
                model.arch.to(device)
                out = model(x, labels=class_to_delete)
            else:
                model.to(device)
                out = model(x)

            _, predicted = torch.max(out.data, 1)
            total_deleted += y[y == class_to_delete].size(0)
            total_retained += y[y != class_to_delete].size(0)
            
            correct_deleted += (predicted[y == class_to_delete] == class_to_delete).sum().item()
            correct_retained += (predicted[y != class_to_delete] == y[y != class_to_delete]).sum().item()

    accuracy_deleted = (correct_deleted / total_deleted)*100
    accuracy_retained = (correct_retained / total_retained)*100
    return accuracy_deleted, accuracy_retained

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unlearning')

    parser.add_argument('--arch', type=str, default='res18', choices=['res18', 'res34', 'vgg', 'swin', 'vit_small', 'vit_tiny'])
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'IMAGENET'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--with_unlearned', action='store_true')
    parser.add_argument('--baseline', type=str, default='standard', choices=['standard', 'flipped', 'difference', 'logits'])

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = 32
    val, n_classes = load_dataset(args.dataset, size)

    if args.dataset == 'MNIST':
        checkpoints_path = f'/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/MNIST/{args.arch}' 
    elif args.dataset == 'CIFAR10':
        checkpoints_path = f'/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/{args.arch}' 
    
    model_unl, general = load_models(args)
    
    activation_unlearned = {}

    # forward pass -- getting the outputs
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, drop_last=True)
    args.len_val_dataset = len(val)

    activation_distance_total = 0
    js_divergence_total = 0
    zrf_total = 0
    accuracy_deleted_total = 0
    accuracy_retained_total = 0
    for gold_class in range(1,10):
        gold_checkpoint_path = os.path.join(checkpoints_path, f'{args.dataset.lower()}_{args.arch}-4-ckpt_original_{gold_class}.t7')
        gold, random_model = load_gold_model(args.arch, args.dataset, gold_checkpoint_path, size)

        args.gold_class = gold_class

        # if args.with_unlearned:
        #     activation_distance = compute_activation_distance(model_unl, gold, val_loader, device, args) # tra gold e unlearned
        # else:
        #     activation_distance = compute_activation_distance(random_model, gold, val_loader, device, args) # tra gold e random - PROVA
        # print(f"Activation Distance gold class {gold_class}: {activation_distance}")
        # activation_distance_total += activation_distance

        # if args.with_unlearned:
        #     js_divergence = compute_js_divergence(model_unl, gold, val_loader, device, args)
        # else:
        #     js_divergence = compute_js_divergence(random_model, gold, val_loader, device, args)
        # print(f"JS Divergence gold class {gold_class}: {js_divergence}")
        # js_divergence_total += js_divergence

        # compute accuracy without class gold_class
        class_to_delete = gold_class
        accuracy_deleted, accuracy_retained = compute_accuracy(model_unl, val_loader, device, args, class_to_delete)
        print(f"Accuracy deleted gold class {gold_class}: {accuracy_deleted}")
        print(f"Accuracy retained gold class {gold_class}: {accuracy_retained}")
        accuracy_deleted_total += accuracy_deleted
        accuracy_retained_total += accuracy_retained

    # gold_checkpoint_path = os.path.join(checkpoints_path, f'{args.dataset.lower()}_{args.arch}-4-ckpt_original_{2}.t7')
    # gold, random_model = load_gold_model(args.arch, args.dataset, gold_checkpoint_path, size)
    
    # fare media dei risultati
    print(f"Accuracy retained: {accuracy_retained_total/n_classes}")
    print(f"Accuracy deleted: {accuracy_deleted_total/n_classes}")
    print(f"Activation Distance: {activation_distance_total/n_classes}")
    print(f"JS Divergence: {js_divergence_total/n_classes}")

    zrf_unlearned, zrf_original = compute_zrf(model_unl, general, random_model, val) # questa l'unica su imagent
    print(f"ZRF Unlearned: {zrf_unlearned/n_classes}")
    print(f"ZRF Original: {zrf_original/n_classes}")