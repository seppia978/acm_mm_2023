import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

import timm
from timm.models.vision_transformer import \
    Attention
    

from collections.abc import Iterable
from typing import Optional, List, Tuple, Union
from . import wtfconv2d as AC2D
from . import wtfhead as AWH

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12
resnet18='resnet18'
resnet50='resnet50'
vgg16='vgg16'
deit_small_16224='deit_small_224_16'
imagenet='imagenet'
cifar10='cifar10'
cifar100='cifar100'
mnist='mnist'

__all__ = {
    'cnn': (
        resnet18,
        resnet50,
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

class WTFModel_Path:
    
    resnet18 = os.path.join(
        #'/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2023-01-12-273',
        '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2023-01-10-228',
        #'/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_resnet18-1-0_resnet18_1.0_1.0_2023-01-13-290',
        'last_intermediate.pt'
    )

    vgg16 = os.path.join(
        '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_vgg16-100-0_vgg16_1.0_100.0_2023-01-10-223',
        'last_intermediate.pt'
    )

    deit_small_16244 = os.path.join(
        # '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_deit_small_16224-100-0_deit_small_16224_1.0_100.0_2023-01-10-229',
        # '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_deit_small_16224-100-0_deit_small_16224_1.0_100.0_2023-01-10-232',
        '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-13-30/alpha_matrices/alpha_resnet18-100-0_resnet18_1.0_100.0_2022-12-19-16/alpha_matrices/alpha_deit_small_224_16-100-0_deit_small_224_16_1.0_100.0_2023-01-11-236',
        'final.pt'
    )


class WTFModel:

    alpha_layers = (
        nn.Conv2d,
        Attention,

    )

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
        self.T=None

    def is_alpha_layer(self, name=None, layer=None):

        for l in self.alpha_layers:
            if isinstance(layer, l):
                return True
            
        return False


class WTFCNN(WTFModel):
    def __init__(
            self,
            kind=resnet18,
            pretrained=True,
            alpha=True,
            m=.5,
            classes_number=1000,
            resume=None,
            dataset=imagenet
    ):
        super(WTFCNN, self).__init__(
            kind=kind, pretrained=pretrained, alpha=alpha,\
            m=m, classes_number=classes_number, resume=resume, dataset=dataset
        )

        if dataset == imagenet:
            if kind==resnet18:
                weights='ResNet18_Weights.DEFAULT'
                self.arch = models.resnet18(pretrained=True)
            elif kind==resnet50:
                weights='ResNet50_Weights.DEFAULT'
                self.arch = models.resnet50(pretrained=True)
            elif kind==vgg16:
                weights='VGG16_Weights.DEFAULT'
                self.arch = models.vgg16(pretrained=True)
            
        elif dataset == cifar10:
            classes_number = 10
            self.arch = models.resnet18(pretrained=False, num_classes=10)
            acab = torch.load(
                os.path.join(
                    # '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
                    # '23_12_22_0.pt'
                    '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
                    '2022-12-29_5.pt'
                )
            )
            # acab=acab['state_dict']
            ac=list(map(lambda x: x[6:], acab.keys()))
            ckp = dict()
            for k1,k2 in zip(acab,ac):
                if k2 == k1[6:]:
                    ckp[k2] = acab[k1]
            self.arch.load_state_dict(ckp)
        elif dataset == cifar100:
            classes_number = 100
            self.arch = models.resnet18(num_classes=100)
            self.arch.load_state_dict(torch.load(
                os.path.join(
                    '/work/dnai_explainability/unlearning/datasets/cifar100_classification/checkpoints',
                    '23_12_22_0.pt'
                )
            ))
        elif dataset == mnist:
            classes_number = 10
            self.arch = models.resnet18(pretrained=False, num_classes=10)
            self.arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            acab = torch.load(
                os.path.join(
                    '/work/dnai_explainability/unlearning/datasets/mnist_classification/checkpoints',
                    '2022-12-29_2.pt'
                )
            )
            # acab=acab['state_dict']
            ac=list(map(lambda x: x[6:], acab.keys()))
            ckp = dict()
            for k1,k2 in zip(acab,ac):
                if k2 == k1[6:]:
                    ckp[k2] = acab[k1]
            self.arch.load_state_dict(ckp)

        if self.alpha:
            self.arch.requires_grad_(requires_grad=False)
            self.arch=AC2D.convert_conv2d_to_alpha(
                self.arch, m=m, classes_number=classes_number
            )

            if self.pretrained:
                if self.resume is None:
                    self.resume = getattr(WTFModel_Path(), kind)
                assert isinstance(self.resume, str), f'Path must be a str. Found {type(self,resume)}'
                self.arch.load_state_dict(torch.load(self.resume))

    def get_all_layer_norms(self,m=1.):
        return AC2D.get_all_layer_norms(self.arch, m=1.)
    def get_all_alpha_layers(self, sigmoid=False):
        return AC2D.get_all_alpha_layers(self.arch, sigmoid=sigmoid)
    def clip_alphas(self):
        AC2D.clip_alpha_val(self.arch)

    def set_label(self, labels) -> None :
        AC2D.set_label(self.arch, labels)

    def forward(self, x, labels=None):
        if not self.alpha:
            return self.arch(x)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor([labels])
            AC2D.set_label(self.arch, labels)
            return self.arch(x)

    def set_alpha(self, layer, idx, vals):
        self.arch.layer[idx] = vals

    def set_n_alphas(self, idx, n=.1):
        model = self.arch

        def reset_n_alpha(x, n): # internal function
            x2 = 1 - torch.sigmoid(x)
            i = x2.topk(int(n * x.squeeze().shape[0])).indices
            x[i] = 3.

            return x

        if isinstance(model, models.VGG):
            for name, l in enumerate(model.features):
                if isinstance(l, AC2D.AlphaWeightedConv2d):
                    l.alpha.data[idx] = reset_n_alpha(l.alpha.data[idx], n)
        elif isinstance(model, models.ResNet):
            for name, x in model.named_children():
                if isinstance(x, AC2D.AlphaWeightedConv2d):
                    x.alpha.data[idx] = reset_n_alpha(x.alpha.data[idx], n)
                else:
                    for name2, x2 in x.named_children():
                        if isinstance(x2, models.resnet.BasicBlock):
                            for n3, x3 in x2.named_children():
                                if isinstance(x3, AC2D.AlphaWeightedConv2d):
                                    x3.alpha.data[idx] = reset_n_alpha(x3.alpha.data[idx], n)

        return model

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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class WTFTransformer(WTFModel):
    def __init__(
        self, 
        kind=resnet18, pretrained=True, alpha=True, m=0.5, 
        classes_number=1000, resume=None, dataset=imagenet
    ) -> None:
        
        super(WTFTransformer, self).__init__()
        super().__init__(kind, pretrained, alpha, m, classes_number, resume, dataset)

        if pretrained:
            if dataset == imagenet:
                if kind==deit_small_16224:
                    vittype='deit_small_patch16_224'
                    self.arch = timm.create_model(
                        vittype,
                        pretrained=True,
                    )
                elif kind==resnet50:
                    weights='ResNet50_Weights.DEFAULT'
                    self.arch = models.resnet50(weights=weights)
                
            elif dataset == cifar10:
                classes_number = 10
                self.arch = models.resnet18(pretrained=False, num_classes=10)
                acab = torch.load(
                    os.path.join(
                        # '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
                        # '23_12_22_0.pt'
                        '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
                        '2022-12-29_5.pt'
                    )
                )
                # acab=acab['state_dict']
                ac=list(map(lambda x: x[6:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[6:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp)

            elif dataset == cifar100:
                classes_number = 100
                self.arch = models.resnet18(num_classes=100)
                self.arch.load_state_dict(torch.load(
                    os.path.join(
                        '/work/dnai_explainability/unlearning/datasets/cifar100_classification/checkpoints',
                        '23_12_22_0.pt'
                    )
                ))
            elif dataset == mnist:
                classes_number = 10
                self.arch = models.resnet18(pretrained=False, num_classes=10)
                self.arch.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                acab = torch.load(
                    os.path.join(
                        '/work/dnai_explainability/unlearning/datasets/mnist_classification/checkpoints',
                        '2022-12-29_2.pt'
                    )
                )
                # acab=acab['state_dict']
                ac=list(map(lambda x: x[6:], acab.keys()))
                ckp = dict()
                for k1,k2 in zip(acab,ac):
                    if k2 == k1[6:]:
                        ckp[k2] = acab[k1]
                self.arch.load_state_dict(ckp)
        else:
            self.arch = models.resnet18()

        if self.alpha:
            self.arch.requires_grad_(requires_grad=False)
            self.arch=AWH.convert_head_to_alpha(
                self.arch, m=m, classes_number=classes_number
            )

        if self.resume is not None:
            assert isinstance(resume, str)
            self.arch.load_state_dict(torch.load(resume))

        config = timm.data.resolve_data_config({}, model=self.arch)
        self.T = timm.data.transforms_factory.create_transform(**config)

    def get_all_layer_norms(self,m=1.):
        return AWH.get_all_layer_norms(self.arch, m=1.)
    def get_all_alpha_layers(self, sigmoid=False):
        return AWH.get_all_alpha_layers(self.arch, sigmoid=sigmoid)
    def clip_alphas(self):
        AWH.clip_alpha_val(self.arch)
    def set_label(self, labels) -> None :
        AWH.set_label(self.arch, labels)

    def forward(self, x, labels=None):
        if not self.alpha:
            return self.arch(x)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor([labels])
            AWH.set_label(self.arch, labels)
            return self.arch(x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)