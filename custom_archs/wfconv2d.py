from .wflayer import WFLayer
import torch
import torch.nn as nn
from torch.nn.modules import Conv2d
from .non_imagenet_models.vgg import VGG
from .non_imagenet_models.resnet import ResNet
import torchvision.models as models
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12

def conv_output_shape(h, w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h + (2 * pad) - (dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((w + (2 * pad) - (dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

class WFConv2d(WFLayer):
    def __init__(self, conv:Conv2d, m:float = .5, classes_number:int = 1000) -> None:
        global amax, amin
        super(WFConv2d, self).__init__()
        # assert isinstance(torch.Tensor,alpha), f'alpha must be of type torch.Tensor. Found {type(alpha)}'
        # assert alpha.shape == torch.Size([kernel_size, 1]), f'alpha must have shape [kernel_size, 1].Found {alpha.shape}'
        amax, amin = m, -m
        self.conv = conv
        self.classes_number = classes_number
        # self.alpha = nn.Parameter(torch.randn(self.conv.in_channels, device='cuda', dtype=torch.FloatTensor))
        self.alpha = nn.Parameter(torch.ones(self.classes_number, self.conv.out_channels, dtype=torch.float) *amax)
        # self.alpha = nn.Parameter(torch.ones(1000, self.conv.out_channels, dtype=torch.float) * m)
        # self.register_parameter(name='Alpha', param=self.alpha)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.conv.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self.conv._reversed_padding_repeated_twice, mode=self.conv.padding_mode),
                            weight, bias, self.conv.stride,
                            _pair(0), self.conv.dilation, self.conv.groups)
        # t = torch.zeros(input.shape[0], self.conv.out_channels, \
        #                 *(conv_output_shape(
        #                     input.shape[-2], input.shape[-1],
        #                     kernel_size=self.conv.kernel_size,
        #                     stride=self.conv.stride[0],
        #                     pad=self.conv.padding[0],
        #                     dilation=self.conv.dilation[0]
        #                 ))).cuda()

        t=[]
        for i in range(input.shape[0]):
            if bias is not None:
                t.append(F.conv2d(input[i].unsqueeze(0), weight[i], torch.full((weight[i].shape[0],),bias[i]).to(input.device), self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups))
            else:
                t.append(F.conv2d(input[i].unsqueeze(0), weight[i], bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups))

        return torch.cat(t,0)

    def forward(self, x):
        dims = self.conv.weight.ndim - 1
        
        if len(self.label.shape) == 0:
            self.label = self.label.unsqueeze(0)
        if x.shape[0] > self.label.shape[0]:
            self.label = torch.Tensor(
                [self.label for _ in range(x.shape[0])]
            ).long()

        weight = self.conv.weight * \
                 torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]

        if self.conv.bias is not None:
            bias = self.conv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
        else:
            bias = None

        weight = weight.unsqueeze(0) if weight.ndim < 5 else weight
        x = self._conv_forward(x, weight, bias)
        return x

# INSERISCE LE LABELS NEI LAYER ALPHA PER I FORWARD PASS

def set_label(model:nn.Module, label:int) -> list :

    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WFConv2d):
            val.label = label


def clip_alpha_val(model:nn.Module) -> None :
    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WFConv2d):
            val.alpha.data[
                val.alpha.data > amax
            ] = amax

            val.alpha.data[
                val.alpha.data < amin
            ] = amin


def convert_conv2d_to_alpha(
        model:nn.Module,
        m:float = .5, classes_number:int = 1000
) -> nn.Module :

    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WFConv2d) or (i+1) == len(tuple(model.named_modules())):
            continue

        elif isinstance(tuple(model.named_modules())[i+1][1], nn.Conv2d):
            if isinstance(tuple(model.named_modules())[i][1], nn.modules.container.Sequential):
                val[0] = WFConv2d(
                    conv=val[0],m=m,classes_number=classes_number
                )
            elif isinstance(model, models.VGG):
                idx = int(name.split('.')[-1])+1
                model.features[idx] = WFConv2d(
                    conv=model.features[idx],m=m,classes_number=classes_number
                )
            elif isinstance(model, VGG):
                idx = int(name.split('.')[-1])+1
                model.features[idx] = WFConv2d(
                    conv=model.features[idx],m=m,classes_number=classes_number
                )
            else:
                if hasattr(val ,'conv1') or hasattr(val ,'conv2'):
                    x=val
                    attr = tuple(model.named_modules())[i+1][0].split('.')[-1]
                    setattr(
                        x, attr,\
                        WFConv2d(
                            conv=getattr(x,attr),m=m,classes_number=classes_number
                        )
                    )
                else:
                    attr = tuple(model.named_modules())[i+1][0].split('.')[-1]
                    setattr(
                        x, attr,\
                        WFConv2d(
                            conv=getattr(x,attr),m=m,classes_number=classes_number
                        )
                    )

        pass

    return model

def invert_alphas(model:nn.Module) -> nn.Module :
    if isinstance(model, models.VGG):
        for name, l in enumerate(model.features):
            if isinstance(l, WFConv2d):
                model.features[name].alpha.data *= -1

    elif isinstance(model, models.ResNet):
        for name, x in model.named_children():
            if isinstance(x, WFConv2d):
                x.alpha.data *= -1
            else:
                for name2, x2 in x.named_children():
                    if isinstance(x2, models.resnet.BasicBlock):
                        for n3, x3 in x2.named_children():
                            if isinstance(x3, WFConv2d):
                                x3.alpha.data *= -1
    return model

# def normalize_alphas(model:nn.Module, r:list=[-3,3]) -> nn.Module :
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, WFConv2d):
#                 norm_a = model.features[name].alpha.data
#
#                 norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
#                 norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
#                                                        torch.tensor(10e-8).to(device=norm_a_max_non_zero.device))
#                 norm_a /= norm_a_max_non_zero
#                 norm_a *= 2*r[1]
#                 norm_a -= r[1]
#
#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, WFConv2d):
#                 norm_a = x.alpha.data
#
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
#                             if isinstance(x3, WFConv2d):
#                                 norm_a = x3.alpha.data
#
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
