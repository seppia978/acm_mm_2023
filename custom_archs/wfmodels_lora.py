from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

import timm
from timm.models.vision_transformer import \
    Attention
    
from .non_imagenet_models.vgg_lora import VGG
from .non_imagenet_models.resnet_lora import ResNet18, ResNet34
from .non_imagenet_models.vit_lora import ViT
from .non_imagenet_models.swin_lora import swin_s, swin_t

from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

from .wflayer import WFLayer
from . import wfconv2d as AC2D
from . import wfhead as AWH

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12
resnet18='resnet18'
resnet34='resnet34'
vgg16='vgg16'
deit_small_16224='deit_small_224_16'
imagenet='imagenet'
cifar10='cifar10'
cifar20='cifar20'
mnist='mnist'
vit_small_16224='vit_small_224_16'
vit_tiny_16224='vit_tiny_224_16'
swin_small_16224='swin_small_224_16'
swin_tiny_16224='swin_tiny_16224'

__all__ = {
    'cnn': (
        resnet18,
        resnet34,
        vgg16
    ),

    'transformer': (
        deit_small_16224
    )
}

def which_baseline(name) -> callable:
    if name in __all__['cnn']:
        return 'cnn'
    elif name in __all__['transformer']:
        return 'transformer'

class WFModel_Path:
    
    resnet18_imagenet = os.path.join(
        '//work/dnai_explainability/unlearning/icml2023/alpha_matrices/unl4xai/alpha_resnet18-100-0_resnet18_1.0_100.0_2023-02-07-18',
        'last_intermediate.pt'
    )

    vgg16_imagenet = os.path.join(
        '/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_vgg16-100-0_vgg16_1.0_100.0_2023-01-10-223',
        'last_intermediate.pt'
    )

    deit_small_16244_imagenet = os.path.join(
        '/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_deit_small_224_16-100-0_deit_small_224_16_1.0_100.0_2023-01-11-236',
        'final.pt'
    )

    resnet18_cifar10 = os.path.join(
        '/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-0',
        'final.pt'
    )

    resnet18_mnist = os.path.join(
        '/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-1',
        'final.pt'
    )


class WFModel:

    # alpha_layers = (
    #     nn.Conv2d,
    #     Attention,
    #
    # )

    def __init__(
        self,
        kind=resnet18,
        pretrained=True,
        alpha=True,
        m=.5,
        classes_number=1000,
        resume=None,
        dataset=imagenet
    
    ) -> None:
        self.kind,self.pretrained,self.alpha,\
            self.m,self.classes_number,self.resume,self.dataset = \
                kind,pretrained,alpha,m,classes_number,resume,dataset
        self.T, self.arch,self.layer_type = (None,) * 3

    # def is_alpha_layer(self, name=None, layer=None):
    #
    #     for l in self.alpha_layers:
    #         if isinstance(layer, l):
    #             return True
    #
    #     return False

    def parameters_distance(self, label:Tensor = Tensor([-1]), kind:str = 'l2'):
        ret = []

        for (n,x) in self.arch.named_modules():
            if isinstance(x, WFLayer):
                if kind.lower() == 'l2':
                    if label >-1:
                        ret.append(
                            torch.linalg.norm(
                                torch.pow(torch.sigmoid(x.alpha[label]) - torch.ones_like(x.alpha[label]), 2)
                            ).unsqueeze(0)
                        )
                    else:
                        ret.append(
                            torch.linalg.norm(
                                torch.pow(torch.sigmoid(x.alpha) - torch.ones_like(x.alpha), 2)
                            ).unsqueeze(0)
                        )
                elif kind.lower() == 'kl-div':
                    if label >-1:
                        ret.append(
                            torch.kl_div(
                                torch.log(torch.sigmoid(x.alpha[label])), torch.ones_like(x.alpha[label])
                            ).mean().unsqueeze(0)
                        )
                    else:
                        ret.append(
                            torch.kl_div(
                                torch.log(torch.sigmoid(x.alpha)), torch.ones_like(x.alpha)
                            ).mean().unsqueeze(0)
                        )

        return torch.cat(ret, 0)

    def minimum_by_layer(self, label:Tensor = Tensor([-1])):
        ret = []
        for (n,x) in self.arch.named_modules():
            if isinstance(x, WFLayer):
                if label >-1:
                    ...
                else:
                    ret.append(
                        torch.sigmoid(x.alpha).min().unsqueeze(0)
                    )

        return torch.cat(ret, 0)
    
    def maximum_by_layer(self, label:Tensor = Tensor([-1])):
        ret = []
        for (n,x) in self.arch.named_modules():
            if isinstance(x, WFLayer):
                if label >-1:
                    ...
                else:
                    ret.append(
                        torch.sigmoid(x.alpha).max().unsqueeze(0)
                    )

        return torch.cat(ret, 0)

    def set_register_forward_hook(self, register_forward_hook:bool = False) -> None:
        if register_forward_hook:
            for n,x in self.arch.named_modules():
                if isinstance(x, WFLayer):
                    x.register_forward_hook(x.save_activations)

    def get_all_alpha_layers(self, sigmoid=False) -> dict:
        ret = {}
        for name, x in self.arch.named_modules():
            if isinstance(x, self.layer_type):
                if not sigmoid:
                    ret[name] = x.alpha.data
                else:
                    ret[name] = torch.sigmoid(x.alpha.data)
        return ret

    def get_all_layer_norms(self, label:Tensor = Tensor([-1]), m:float = 1.) -> Tensor:

        ret = []
        for i, (name, val) in enumerate(self.arch.named_modules()):
            if isinstance(val, self.layer_type):
                x = val.alpha.data
                ret.append(
                    (m - torch.sigmoid(
                        x
                    ))[0].unsqueeze(0).mean().unsqueeze(0)
                )

        return torch.cat(ret, dim=0)

    def params_count(self):
        import numpy as np
        return sum([np.prod(p.size()) for p in self.arch.parameters()])/ \
            1_000_000
    
    def alpha012(self):
        F, S, T = (None,) * 3
        self.f, self.s, self.t = (None,) * 3
        if self.alpha:
            s = 0
            f,t = 0,0
            wcs = 0
            WF = False
            for i, (n,x) in enumerate(self.arch.named_modules()):
                if WF and 'conv' in n or 'qkv' in n:
                    if self.f is None:
                        self.f = '.'.join(n.split('.')[:-1])
                    self.t = '.'.join(n.split('.')[:-1])
                    WF = False
                if isinstance(x, WFLayer):
                    s+=1
                    if F is None:
                        F = n
                    if s % 2 == 0:
                        wcs += 1
                    T = n
                    WF = True
            WF = False
            for i,(n,x) in enumerate(self.arch.named_modules()):
                if WF and 'conv' in n or 'qkv' in n:
                    self.s = '.'.join(n.split('.')[:-1])
                    WF = False
                if isinstance(x, WFLayer):
                    if wcs==0:
                        S = n
                        WF = True

                    wcs -= 1

        return F, S, T


    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class WFCNN(WFModel):
    def __init__(
            self,
            kind=resnet18, pretrained=True, alpha=True, m=.5,
            classes_number=1000, resume=None, dataset=imagenet,
            baseline_pretrained = True, lora_r = 4
    ) -> None:

        super(WFCNN, self).__init__(
            kind=kind, pretrained=pretrained, alpha=alpha, m=m,\
            classes_number=classes_number, resume=resume, dataset=dataset
        )

        self.layer_type = AC2D.WFConv2d
        if imagenet in dataset:
            if kind==resnet18:
                if baseline_pretrained:
                    weights='ResNet18_Weights.DEFAULT'
                    self.arch = models.resnet18(weights=weights)
                else:
                    self.arch = models.resnet18()
            elif kind==resnet34:
                if baseline_pretrained:
                    weights='ResNet34_Weights.DEFAULT'
                    self.arch = models.resnet34(weights=weights)
                else:
                    ...
            elif kind==vgg16:
                if baseline_pretrained:
                    weights='VGG16_Weights.DEFAULT'
                    self.arch = models.vgg16(weights=weights)
                else:
                    ...
        elif cifar10 in dataset:
            classes_number = 10
            if kind == resnet18:
                self.arch = ResNet18(lora_r=lora_r)
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                            'res18-4-ckpt_original.t7'
                        )
                    )
            elif kind == resnet34:
                self.arch = ResNet34()
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                            'res34-4-ckpt_original.t7'
                        )
                    )
            elif kind == vgg16:
                self.arch = VGG(vgg_name='VGG16', lora_r=lora_r)
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                            'vgg-4-ckpt_original.t7'
                        )
                    )
            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp, strict=False) # lora
        elif cifar20 in dataset:
            classes_number = 20
            ckp_root = '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar20'

            if kind == resnet18:
                self.arch = ResNet18(num_classes=20, lora_r=lora_r)
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            ckp_root,
                            'res18-4-ckpt_original.t7'
                        )
                    )

            elif kind == vgg16:
                self.arch = VGG(vgg_name='VGG16', num_classes=20, lora_r=lora_r)
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            ckp_root,
                            'vgg-4-ckpt_original.t7'
                        )
                    )

            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp, strict=False) # lora
        elif mnist in dataset:
            classes_number = 10
            if kind == resnet18:
                self.arch = ResNet18()
                self.arch.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_mnist',
                            'res18-4-ckpt_original.t7'
                        )
                    )
            elif kind == resnet34:
                self.arch = ResNet34()
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_mnist',
                            'res34-4-ckpt_original.t7'
                        )
                    )
            elif kind == vgg16:
                self.arch = VGG(vgg_name='VGG16')
                self.arch.features[0] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1, 1), padding=(1,1))
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_mnist',
                            'vgg-4-ckpt_original.t7'
                        )
                    )
            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp)

        if self.alpha:
            self.arch.requires_grad_(requires_grad=False)
            self.arch=AC2D.convert_conv2d_to_alpha(
                self.arch, m=m, classes_number=classes_number
            )

            if self.pretrained:
                if self.resume is None:
                    self.resume = getattr(WFModel_Path(), f'{kind}_{dataset}')
                assert isinstance(self.resume, str), f'Path must be a str. Found {type(self,resume)}'
                self.arch.load_state_dict(torch.load(self.resume))

        self.arch.eval()
                

    # def get_all_layer_norms(self,m=1.):
    #     return AC2D.get_all_layer_norms(self.arch, m=1.)
    # def get_all_alpha_layers(self, sigmoid=False):
    #     return self.get_all_alpha_layers()
    def clip_alphas(self):
        AC2D.clip_alpha_val(self.arch)

    def set_label(self, labels) -> None :
        AC2D.set_label(self.arch, labels)

    def forward(self, x, labels=None, register_forward_hook=False):
        if not self.alpha:
            return self.arch(x)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor([labels])
            AC2D.set_label(self.arch, labels)
            self.set_register_forward_hook(register_forward_hook=register_forward_hook)
            return self.arch(x)

    # def set_alpha(self, layer, idx, vals):
    #     self.arch.layer[idx] = vals

    # def set_n_alphas(self, idx, n=.1):
    #     model = self.arch
    #
    #     def reset_n_alpha(x, n): # internal function
    #         x2 = 1 - torch.sigmoid(x)
    #         i = x2.topk(int(n * x.squeeze().shape[0])).indices
    #         x[i] = 3.
    #
    #         return x
    #
    #     if isinstance(model, models.VGG):
    #         for name, l in enumerate(model.features):
    #             if isinstance(l, AC2D.AlphaWeightedConv2d):
    #                 l.alpha.data[idx] = reset_n_alpha(l.alpha.data[idx], n)
    #     elif isinstance(model, models.ResNet):
    #         for name, x in model.named_children():
    #             if isinstance(x, AC2D.AlphaWeightedConv2d):
    #                 x.alpha.data[idx] = reset_n_alpha(x.alpha.data[idx], n)
    #             else:
    #                 for name2, x2 in x.named_children():
    #                     if isinstance(x2, models.resnet.BasicBlock):
    #                         for n3, x3 in x2.named_children():
    #                             if isinstance(x3, AC2D.AlphaWeightedConv2d):
    #                                 x3.alpha.data[idx] = reset_n_alpha(x3.alpha.data[idx], n)
    #
    #     return model

    def statistics(self, rev=False):
        if not rev:
            ret = f'''Less than .3 -------------------------------------------------
    {str([x[x<.3].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Less than .2 -------------------------------------------------
    {str([x[x<.2].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Less than .1 -------------------------------------------------
    {str([x[x<.1].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .7 -------------------------------------------------
    {str([x[x>.7].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .8 -------------------------------------------------
    {str([x[x>.8].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .9 -------------------------------------------------
    {str([x[x>.9].shape[0]/x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
            '''
        else:
            ret = f'''Less than .3 -------------------------------------------------
    {str([(1-x)[(1-x) < .3].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Less than .2 -------------------------------------------------
    {str([(1-x)[(1-x) < .2].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Less than .1 -------------------------------------------------
    {str([(1-x)[(1-x) < .1].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .7 -------------------------------------------------
    {str([(1-x)[(1-x) > .7].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .8 -------------------------------------------------
    {str([(1-x)[(1-x) > .8].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
Greater than .9 -------------------------------------------------
    {str([(1-x)[(1-x) > .9].shape[0] / x.numel() for x in self.get_all_alpha_layers(sigmoid=True)])}
            '''
        return ret

class WFTransformer(WFModel):
    def __init__(
        self, 
        kind=vit_tiny_16224, pretrained=True, alpha=True, m=.5,
        classes_number=1000, resume=None, dataset=imagenet, baseline_pretrained=True, lora_r = 4
    ) -> None:
        
        super(WFTransformer, self).__init__(
            kind=kind, pretrained=pretrained, alpha=alpha, m=m,\
            classes_number=classes_number, resume=resume, dataset=dataset
        )

        self.layer_type = AWH.WFHead
        if imagenet in dataset:
            if kind==deit_small_16224:
                vittype='deit_small_patch16_224'
                self.arch = timm.create_model(
                    vittype,
                    pretrained=True,
                )
            elif kind==vit_small_16224:
                vittype='vit_small_patch16_224'
                # self.arch = timm.create_model(
                #     vittype,
                #     pretrained=True,
                # )

                self.arch = ViT(
                    image_size = 224,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 384,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 384,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )

                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                            'vit_small_equivalent_timm-4-ckpt_original_with_augm.t7'
                        )
                    )
            elif kind==swin_small_16224:
                vittype='swin_small_patch4_window7_224'
                self.arch = timm.create_model(
                    vittype,
                    pretrained=True,
                )
            elif kind==vit_tiny_16224:
                vittype='vit_tiny_patch16_224'
                # self.arch = timm.create_model(
                #     vittype,
                #     pretrained=True,
                # )
                self.arch = ViT(
                    channels=3,
                    image_size = 224,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 192,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 192,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                        'vit_tiny_equivalent_timm-4-ckpt_original_with_augm.t7'
                    )
                )
                
        elif cifar10 in dataset:
            classes_number = 10
            if kind==vit_small_16224:
                self.arch = ViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 384,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 384,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                        'vit_small_equivalent_timm-4-ckpt_original_with_augm.t7'
                    )
                )
            elif kind==vit_tiny_16224:
                self.arch = ViT(
                    channels=3,
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 192,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 192,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                        'vit_tiny_equivalent_timm-4-ckpt_original_with_augm.t7'
                    )
                )
            elif kind==swin_small_16224:
                self.arch = swin_s(
                    window_size=4,
                    num_classes=classes_number,
                    downscaling_factors=(2,2,2,1)
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                        'swin_s_lr_1e-4_CIFAR10.t7'
                    )
                )
            elif kind==swin_tiny_16224:
                self.arch = swin_t(
                    window_size=4,
                    num_classes=classes_number,
                    downscaling_factors=(2,2,2,1)
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar10',
                        'swin_t_lr_1e-4_CIFAR10.t7'
                    )
                )

            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp, strict=False) # lora

        elif cifar20 in dataset:
            classes_number = 20
            ckp_root = '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar20'
            
            if kind == vit_small_16224:
                self.arch = ViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 384,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 384,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            ckp_root,
                            'vit_small_lr_1e-4_CIFAR20_dropout_0.1.t7'
                        )
                    )

            elif kind == vit_tiny_16224:
                self.arch = ViT(
                    channels=3,
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 192,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 192,
                    dropout = 0.1,
                    emb_dropout = 0.1,
                    lora_r = lora_r
                )
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            ckp_root,
                            'vit_tiny_lr_1e-4_CIFAR20_dropout_0.1.t7'
                        )
                    )

            elif kind==swin_small_16224:
                self.arch = swin_s(
                    window_size=4,
                    num_classes=classes_number,
                    downscaling_factors=(2,2,2,1)
                )
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar20',
                            'swin_s_lr_1e-4_CIFAR20.t7'
                        )
                    )
            elif kind==swin_tiny_16224:
                self.arch = swin_s(
                    window_size=4,
                    num_classes=classes_number,
                    downscaling_factors=(2,2,2,1)
                )
                if baseline_pretrained:
                    acab = torch.load(
                        os.path.join(
                            '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_cifar20',
                            'swin_t_lr_1e-4_CIFAR20.t7'
                        )
                    )

            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp, strict=False) # lora
        elif mnist in dataset:
            classes_number = 10
            if kind==vit_small_16224:
                self.arch = ViT(
                    channels=1,
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 384,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 384,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_mnist',
                        'vit_small_equivalent_timm-4-ckpt_original_with_augm.t7'
                    )
                )
            elif kind==vit_tiny_16224:
                self.arch = ViT(
                    channels=1,
                    image_size = 32,
                    patch_size = 4,
                    num_classes = classes_number,
                    dim = 192,
                    depth = 12,
                    heads = 8,
                    mlp_dim = 192,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint_mnist',
                        'vit_tiny_equivalent_timm-4-ckpt_original_with_augm.t7'
                    )
                )
            elif kind==deit_small_16224:
                pass
            elif kind==swin_small_16224:
                self.arch = swin_s(
                    window_size=4,
                    channels=1,
                    num_classes=classes_number,
                    downscaling_factors=(2,2,2,1)
                )

                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/ssarto/checkpoints_full/checkpoint',
                        '' #NOT YET
                    )
                )
            if baseline_pretrained:
                acab=acab['model']
                ac=list(map(lambda x: x[7:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[7:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp)

        if self.alpha:
            self.arch.requires_grad_(requires_grad=False)
            self.arch=AWH.convert_head_to_alpha(
                self.arch, m=m, classes_number=classes_number
            )

        if self.pretrained:
            if self.resume is None:
                self.resume = getattr(WFModel_Path(), kind)
            assert isinstance(self.resume, str), f'Path must be a str. Found {type(self,resume)}'
            acab = torch.load(self.resume)
            if 't7' in self.resume:
                acab=acab['model']
                k = 7
            else:
                k = 6
            ac=list(map(lambda x: x[k:], acab.keys()))
            ckp = dict()
            for k1,k2 in zip(acab,ac):
                if k2 == k1[k:]:
                    ckp[k2] = acab[k1]
            if 's' in ckp.keys():
                del ckp['s']
            self.arch.load_state_dict(ckp, strict=False)


        config = timm.data.resolve_data_config({}, model=self.arch)
        self.T = timm.data.transforms_factory.create_transform(**config)

    # def get_all_layer_norms(self,m=1.):
    #     return AWH.get_all_layer_norms(self.arch, m=1.)
    # def get_all_alpha_layers(self, sigmoid=False):
    #     return self.get_all_alpha_layers()
    def clip_alphas(self):
        AWH.clip_alpha_val(self.arch)
    def set_label(self, labels) -> None :
        AWH.set_label(self.arch, labels)

    def forward(self, x, labels=None, register_forward_hook=False):
        if not self.alpha:
            return self.arch(x)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor([labels])
            AWH.set_label(self.arch, labels, bs=x.shape[0])
            return self.arch(x)
