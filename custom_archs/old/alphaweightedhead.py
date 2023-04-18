import torch
import torch.nn as nn

from timm.models.vision_transformer import \
    VisionTransformer, \
    Attention, \
    Block
from timm.models.layers.mlp import Mlp

from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12

class AlphaWeightedHead(nn.Module):
    def __init__(self, attn: nn.Module, m:float = .5, classes_number=1000):
        global amax, amin
        super(AlphaWeightedHead, self).__init__()
        # assert isinstance(torch.Tensor,alpha), f'alpha must be of type torch.Tensor. Found {type(alpha)}'
        # assert alpha.shape == torch.Size([kernel_size, 1]), f'alpha must have shape [kernel_size, 1].Found {alpha.shape}'
        amax, amin = m, -m
        self.attn = attn
        self.classes_number = classes_number
        # self.alpha = nn.Parameter(torch.randn(self.linear.in_channels, device='cuda', dtype=torch.FloatTensor))
        self.alpha = nn.Parameter(torch.ones(self.classes_number, self.attn.qkv.out_features, requires_grad=True, dtype=torch.float) *amax)
        
        # self.alpha = nn.Parameter(torch.randn(self.classes_number, self.attn.qkv.out_features, requires_grad=True, dtype=torch.float))
        # self.alpha.data *= 2.
        # self.alpha.data -= 1.
        # self.alpha.data *= amax
        # self.alpha = nn.Parameter(torch.ones(1000, self.linear.out_channels, dtype=torch.float) * m)
        # self.register_parameter(name='Alpha', param=self.alpha)

    def _qkv_forward(self, x, w, b):

        t = []

        for i in range(x.shape[0]):

            if b[i] is not None:
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
    
    def _attn_forward(self, x, w, b):
        B, N, C = x.shape

        qkv = self._qkv_forward(x,w,b).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        # attn = torch.sigmoid(self.alpha)[label][(None,)+(...,)+(None,)*2]\
        #      * attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x
    
    def forward(self, x):
        dims = self.attn.qkv.weight.ndim - 1
        # alpha = self.alpha[self.label].squeeze()

        # self.alpha = nn.Parameter(torch.ones_like(self.alpha))
        # alpha = nn.Parameter(1 - torch.sigmoid(self.alpha))
        # weight = self.linear.weight * alpha[(...,) + (None,) * dims]

        if len(self.label.shape) == 0:
            self.label = self.label.unsqueeze(0)
        if x.shape[0] > self.label.shape[0]:
            self.label = torch.Tensor(
                [self.label for _ in range(x.shape[0])]
            ).long()

        weight = self.attn.qkv.weight * \
                 torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]

        if self.attn.qkv.bias is not None:
            # bias = self.conv.bias * alpha[self.label].squeeze()
            bias = self.attn.qkv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
            # bias = self.conv.bias * self.alpha
        else:
            bias = None
        
        self._attn_forward(x, weight, bias)

        return x

def get_all_alpha_layers(model:nn.Module, sigmoid=False) -> dict :
    ret = {}
    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                if not sigmoid:
                    ret[x.attn.alpha] = x.attn.alpha.data
                else:
                    ret[x.attn.alpha] = torch.sigmoid(x.attn.alpha.data)
    return ret

def get_all_leyer_norms(model:nn.Module, norm:int = 1, m:float=0.) -> Tensor :
    ret = []

    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                val = x.attn.alpha
                ret.append(
                    (m - torch.sigmoid(
                        val
                    )).mean().unsqueeze(0)
                )

    return torch.cat(ret,dim=0)

# INSERISCE LE LABELS NEI LAYER ALPHA PER I FORWARD PASS

def set_label(model:nn.Module, label:int) -> list :

    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                x.attn.label = label
            pass


# RIPORTA GLI ALPHA FRA -M E M SE MIN E MAX SONO DIVERSI

# def set_alpha_val(model:nn.Module, label:int, val:float=None) -> list :
    # ret = []
    # if isinstance(model, models.VGG):
    #     for name, l in enumerate(model.features):
    #         if isinstance(l, AlphaWeightedConv2d):
    #             m,M = \
    #                 model.features[name].alpha.data[label].min(), \
    #                 model.features[name].alpha.data[label].max()

    #             if m < M:
    #                 model.features[name].alpha.data[label] -= m
    #                 model.features[name].alpha.data[label] /= (M + eps)

    #                 model.features[name].alpha.data[label] *= 2*amax
    #                 model.features[name].alpha.data[label] -= amax
    #             # model.features[name].alpha.data[label] /= \
    #             #     model.features[name].alpha.data[label].max()

    #             # model.features[name].alpha.data[label] = torch.relu(model.features[name].alpha.data[label])

    # elif isinstance(model, models.ResNet):
    #     for name, x in model.named_children():
    #         if isinstance(x, AlphaWeightedConv2d):
    #             m,M = x.alpha.data[label].min(), x.alpha.data[label].max()
    #             if m < M:

    #                 x.alpha.data[label] -= m
    #                 x.alpha.data[label] /= (M + eps)

    #                 x.alpha.data[label] *= 2*amax
    #                 x.alpha.data[label] -= amax

    #             # x.alpha.data[label] /= x.alpha.data[label].max()
    #             # x.alpha.data[label] = torch.relu(x.alpha.data[label])
    #         else:
    #             for name2, x2 in x.named_children():
    #                 if isinstance(x2, models.resnet.BasicBlock) or isinstance(x2, models.resnet.Bottleneck):
    #                     for n3, x3 in x2.named_children():
    #                         if isinstance(x3, AlphaWeightedConv2d):
    #                             m, M = x3.alpha.data[label].min(), x3.alpha.data[label].max()

    #                             if m < M:
    #                                 x3.alpha.data[label] -= m
    #                                 x3.alpha.data[label] /= (M + eps)

    #                                 x3.alpha.data[label] *= 2 * amax
    #                                 x3.alpha.data[label] -= amax

    #                             else:
    #                                 for n4,x4 in x3.named_children():
    #                                     if isinstance(x4, AlphaWeightedConv2d) and False:
    #                                         m, M = x4.alpha.data[label].min(), x4.alpha.data[label].max()

    #                                         if m < M:
    #                                             x4.alpha.data[label] -= m
    #                                             x4.alpha.data[label] /= (M + eps)

    #                                             x4.alpha.data[label] *= 2 * amax
    #                                             x4.alpha.data[label] -= amax


    #                             # x3.alpha.data[label] /= x3.alpha.data[label].max()
    #                             # x3.alpha.data[label] = torch.relu(x3.alpha.data[label])

    # return ret

def clip_alpha_val(model:nn.Module) -> None :
    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                x.attn.alpha.data[
                    x.attn.alpha.data < amin
                ] = amin
                x.attn.alpha.data[
                    x.attn.alpha.data > amax
                ] = amax


def convert_head_to_alpha(
        model:nn.Module,
        m:float = .5, standard=False, classes_number=1000
) -> nn.Module :

    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                x.attn = AlphaWeightedHead(
                    attn=x.attn,m=m,classes_number=classes_number
                )
            pass

    else:
        raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')

    return model

def invert_alphas(model:nn.Module) -> nn.Module :
    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Attention):
                x.attn.alpha *= (-1.)
    return model

# def normalize_alphas(model:nn.Module, r:list=[-3,3]) -> nn.Module :
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, AlphaWeightedConv2d):
#                 norm_a = model.features[name].alpha.data

#                 norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
#                                                        torch.tensor(10e-8).to(device=norm_a_max_non_zero.device))
#                 norm_a /= norm_a_max_non_zero
#                 norm_a *= 2*r[1]
#                 norm_a -= r[1]

#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, AlphaWeightedConv2d):
#                 norm_a = x.alpha.data

#                 norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
#                                                   torch.tensor(10e-8).to(device=norm_a_max_non_zero.device))
#                 norm_a /= norm_a_max_non_zero
#                 norm_a *= 2 * r[1]
#                 norm_a -= r[1]
#             else:
#                 for name2, x2 in x.named_children():
#                     if isinstance(x2, models.resnet.BasicBlock):
#                         for n3, x3 in x2.named_children():
#                             if isinstance(x3, AlphaWeightedConv2d):
#                                 norm_a = x3.alpha.data

#                                 norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
#                                 norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(
#                                     -1).unsqueeze(-1)
#                                 norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
#                                                                   torch.tensor(10e-8).to(
#                                                                       device=norm_a_max_non_zero.device))
#                                 norm_a /= norm_a_max_non_zero
#                                 norm_a *= 2 * r[1]
#                                 norm_a -= r[1]
#     return model
