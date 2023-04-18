from .wflayer import WFLayer
import torch
import torch.nn as nn
from torch.nn.modules import Linear

from timm.models.vision_transformer import \
    VisionTransformer, \
    Attention, \
    Block

from .non_imagenet_models.vit import ViT
from .non_imagenet_models.vit import PreNorm, Attention as ATT

from .non_imagenet_models.swin import swin_s

from torch.nn import functional as F

from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12

class WFHead(WFLayer):
    def __init__(self, qkv:Linear, m:float = .5, classes_number=1000):
        global amax, amin
        super(WFHead, self).__init__()

        amax, amin = m, -m
        self.qkv = qkv
        self.classes_number = classes_number

        self.alpha = nn.Parameter(torch.ones(self.classes_number, self.qkv.out_features, dtype=torch.float) *amax)

    def _qkv_forward(self, x, w, b):

        t = []

        for i in range(self.bs):
            # if len(x[i].shape) < 3:
            #     _w = w[i].unsqueeze(0)
            # else:
            #     _w = w[i]
            if b is not None:
                t.append(F.linear(
                    x[i].unsqueeze(0),
                    w[i],
                    b[i]
                ))
            else:
                t.append(F.linear(
                    x[i].unsqueeze(0),
                    w[i],
                    None
                ))

        return torch.cat(t, dim=0)

    def forward(self, x):

        dims = self.qkv.weight.ndim - 1

        if len(self.label.shape) == 0:
            self.label = self.label.unsqueeze(0)
        if self.bs > self.label.shape[0]:
            self.label = self.label.repeat(x.shape[0]).long()
        
        weight = self.qkv.weight * \
                 torch.sigmoid(self.alpha[self.label].squeeze(-1))[(...,) + (None,) * dims]

        if self.qkv.bias is not None:
            bias = self.qkv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
        else:
            bias = None

        x = self._qkv_forward(x, weight, bias)

        return x



def get_all_layer_norms(model:nn.Module, norm:int = 1, m:float=0.) -> Tensor :
    ret = []

    for name, x in model.named_modules():
        if isinstance(x, WFHead):
            val = x.alpha
            ret.append(
                (m - torch.sigmoid(
                    val
                )).mean().unsqueeze(0)
            )

    return torch.cat(ret, dim=0)

# INSERISCE LE LABELS NEI LAYER ALPHA PER I FORWARD PASS

def set_label(model:nn.Module, label:int, bs:int) -> list :

    for name, x in model.named_modules():
        if isinstance(x, WFHead):
            x.label = label
            x.bs = bs
        pass


def clip_alpha_val(model:nn.Module) -> None :
    for name, x in model.named_modules():
        if isinstance(x, WFHead):
            x.alpha.data[
                x.alpha.data < amin
            ] = amin
            x.alpha.data[
                x.alpha.data > amax
            ] = amax


def convert_head_to_alpha(
        model:nn.Module,
        m:float = .5, classes_number=1000
) -> nn.Module :

    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                x.attn.qkv = WFHead(
                    qkv=x.attn.qkv,m=m,classes_number=classes_number
                )

    elif isinstance(model, ViT):
        for name, x in model.named_modules():
            if isinstance(x, PreNorm) and isinstance(x.fn, ATT):
                x.fn.to_qkv = WFHead(
                    qkv=x.fn.to_qkv,m=m,classes_number=classes_number,
                )

    else:
        raise NotImplementedError(f'{type(model)} not implemented yet. Only ViT-like are.')

    return model

def invert_alphas(model:nn.Module) -> nn.Module :
    for name, x in model.named_modules():
        if isinstance(x, WFHead):
            x.alpha *= (-1.)
    return model
