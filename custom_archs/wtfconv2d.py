from .wtflayer import WTFLayer
import torch
import torch.nn as nn
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

class WTFConv2d(WTFLayer):
    def __init__(self, conv: nn.modules.Conv2d, m:float = .5, classes_number=1000):
        global amax, amin
        super(WTFConv2d, self).__init__()
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
        # alpha = self.alpha[self.label].squeeze()

        # self.alpha = nn.Parameter(torch.ones_like(self.alpha))
        # alpha = nn.Parameter(1 - torch.sigmoid(self.alpha))
        # weight = self.conv.weight * alpha[(...,) + (None,) * dims]
        
        if len(self.label.shape) == 0:
            self.label = self.label.unsqueeze(0)
        if x.shape[0] > self.label.shape[0]:
            self.label = self.label.repeat(x.shape[0]).long()
        
        weight = self.conv.weight * \
                 torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]

        # alpha = (self.alpha - self.alpha.min())/(self.alpha.max() - self.alpha.min() + eps)
        # weight = self.conv.weight * \
        #                  alpha[self.label].squeeze()[(...,) + (None,) * dims]

        if self.conv.bias is not None:
            # bias = self.conv.bias * alpha[self.label].squeeze()
            bias = self.conv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
            # bias = self.conv.bias * self.alpha
        else:
            bias = None

        weight = weight.unsqueeze(0) if weight.ndim < 5 else weight
        x = self._conv_forward(x, weight, bias)
        return x

class AWC2d(WTFConv2d):

    def __int__(self,  conv: nn.modules.Conv2d, m:float = .5):
        super(WTFConv2d, self).__init__(conv=conv, m=m)

    def forward(self, x):
        dims = self.conv.weight.ndim - 1

        alpha = nn.Parameter(torch.ones_like(self.alpha))
        # self.alpha = nn.Parameter(1 - torch.sigmoid(self.alpha))
        weight = self.conv.weight * alpha[(...,) + (None,) * dims]
        # weight = self.conv.weight * torch.sigmoid(self.alpha)[(...,) + (None,) * dims]

        if self.conv.bias is not None:
            bias = self.conv.bias * alpha
            # bias = self.conv.bias * torch.sigmoid(self.alpha)
        else:
            bias = None

        x = self.conv._conv_forward(x, weight, bias)
        return x

def get_all_alpha_layers(model:nn.Module, sigmoid=False) -> dict :
    ret = {}
    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WTFConv2d):
            ret[name] = val.alpha

    return ret

def get_all_layer_norms(model:nn.Module, norm:int = 1, m:float=0.) -> Tensor :
    
    ret = []
    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WTFConv2d):
            x = val.alpha.data
            ret.append(
                (m - torch.sigmoid(
                        x
                    ))[0].unsqueeze(0).mean().unsqueeze(0)
            )

    return torch.cat(ret, dim=0)

# INSERISCE LE LABELS NEI LAYER ALPHA PER I FORWARD PASS

def set_label(model:nn.Module, label:int) -> list :

    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WTFConv2d):
            val.label = label


# RIPORTA GLI ALPHA FRA -100 E 100 SE MIN E MAX SONO DIVERSI

# def set_alpha_val(model:nn.Module, label:int, val:float=None) -> list :
#     ret = []
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, WTFConv2d):
#                 m,M = \
#                     model.features[name].alpha.data[label].min(), \
#                     model.features[name].alpha.data[label].max()

#                 if m < M:
#                     model.features[name].alpha.data[label] -= m
#                     model.features[name].alpha.data[label] /= (M + eps)

#                     model.features[name].alpha.data[label] *= 2*amax
#                     model.features[name].alpha.data[label] -= amax
#                 # model.features[name].alpha.data[label] /= \
#                 #     model.features[name].alpha.data[label].max()

#                 # model.features[name].alpha.data[label] = torch.relu(model.features[name].alpha.data[label])

#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, WTFConv2d):
#                 m,M = x.alpha.data[label].min(), x.alpha.data[label].max()
#                 if m < M:

#                     x.alpha.data[label] -= m
#                     x.alpha.data[label] /= (M + eps)

#                     x.alpha.data[label] *= 2*amax
#                     x.alpha.data[label] -= amax

#                 # x.alpha.data[label] /= x.alpha.data[label].max()
#                 # x.alpha.data[label] = torch.relu(x.alpha.data[label])
#             else:
#                 for name2, x2 in x.named_children():
#                     if isinstance(x2, models.resnet.BasicBlock) or isinstance(x2, models.resnet.Bottleneck):
#                         for n3, x3 in x2.named_children():
#                             if isinstance(x3, WTFConv2d):
#                                 m, M = x3.alpha.data[label].min(), x3.alpha.data[label].max()

#                                 if m < M:
#                                     x3.alpha.data[label] -= m
#                                     x3.alpha.data[label] /= (M + eps)

#                                     x3.alpha.data[label] *= 2 * amax
#                                     x3.alpha.data[label] -= amax
                                
#                                 else:
#                                     for n4,x4 in x3.named_children():
#                                         if isinstance(x4, WTFConv2d) and True:
#                                             m, M = x4.alpha.data[label].min(), x4.alpha.data[label].max()

#                                             if m < M:
#                                                 x4.alpha.data[label] -= m
#                                                 x4.alpha.data[label] /= (M + eps)

#                                                 x4.alpha.data[label] *= 2 * amax
#                                                 x4.alpha.data[label] -= amax


#                                 # x3.alpha.data[label] /= x3.alpha.data[label].max()
#                                 # x3.alpha.data[label] = torch.relu(x3.alpha.data[label])

#     return ret

def clip_alpha_val(model:nn.Module) -> None :
    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WTFConv2d):
            val.alpha.data[
                val.alpha.data > amax
            ] = amax

            val.alpha.data[
                val.alpha.data < amin
            ] = amin

    # if isinstance(model, models.VGG):
    #     for name, l in enumerate(model.features):
    #         if isinstance(l, WTFConv2d):

    #             model.features[name].alpha.data[
    #                 model.features[name].alpha.data > amax
    #             ] = amax
    #             model.features[name].alpha.data[
    #                 model.features[name].alpha.data < amin
    #             ] = amin

    # elif isinstance(model, models.ResNet):
    #     for name, x in model.named_children():
    #         if isinstance(x, WTFConv2d):
    #             x.alpha.data[
    #                 x.alpha.data > amax
    #             ] = amax
    #             x.alpha.data[
    #                 x.alpha.data < amin
    #             ] = amin

    #         else:
    #             for name2, x2 in x.named_children():
    #                 if isinstance(x2, models.resnet.BasicBlock) or isinstance(x2, models.resnet.Bottleneck):
    #                     for n3, x3 in x2.named_children():
    #                         if isinstance(x3, WTFConv2d):
    #                             x3.alpha.data[
    #                                 x3.alpha.data > amax
    #                                 ] = amax
    #                             x3.alpha.data[
    #                                 x3.alpha.data < amin
    #                             ] = amin
    #                         else:
    #                             for n4,x4 in x3.named_children():
    #                                 if isinstance(x4, WTFConv2d) and True:
    #                                     x4.alpha.data[
    #                                         x4.alpha.data > amax
    #                                     ] = amax
    #                                     x4.alpha.data[
    #                                         x4.alpha.data < amin
    #                                     ] = amin

def convert_conv2d_to_alpha(
        model:nn.Module,
        m:float = .5, classes_number:int = 1000
) -> nn.Module :

    for i, (name, val) in enumerate(model.named_modules()):
        if isinstance(val, WTFConv2d) or (i+1) == len(tuple(model.named_modules())):
            continue

        elif isinstance(tuple(model.named_modules())[i+1][1], nn.Conv2d):
            if isinstance(tuple(model.named_modules())[i][1], nn.modules.container.Sequential):
                val[0] = WTFConv2d(
                    conv=val[0],m=m,classes_number=classes_number
                )
            elif isinstance(model, models.VGG):
                idx = int(name.split('.')[-1])+1
                model.features[idx] = WTFConv2d(
                    conv=model.features[idx],m=m,classes_number=classes_number
                )
            elif isinstance(model, VGG):
                idx = int(name.split('.')[-1])+1
                model.features[idx] = WTFConv2d(
                    conv=model.features[idx],m=m,classes_number=classes_number
                )
            else:
                if hasattr(val ,'conv1') or hasattr(val ,'conv2'):
                    x=val
                    attr = tuple(model.named_modules())[i+1][0].split('.')[-1]
                    setattr(
                        x, attr,\
                        WTFConv2d(
                            conv=getattr(x,attr),m=m,classes_number=classes_number
                        )
                    )
                else:
                    attr = tuple(model.named_modules())[i+1][0].split('.')[-1]
                    setattr(
                        x, attr,\
                        WTFConv2d(
                            conv=getattr(x,attr),m=m,classes_number=classes_number
                        )
                    )

             

        pass

    return model

    # if not standard:

    #     if isinstance(model, models.VGG):
    #         for name, l in enumerate(model.features):
    #             if isinstance(l, nn.Conv2d):
    #                 model.features[name] = WTFConv2d(
    #                     conv=l,m=m,classes_number=classes_number
    #                 )
    #     elif isinstance(model, models.ResNet):
    #         for name, x in model.named_children():
    #             if isinstance(x, nn.Conv2d):
    #                 setattr(model, name, WTFConv2d(
    #                     conv=x,m=m,classes_number=classes_number
    #                 ))
    #             else:
    #                 for name2, x2 in x.named_children():
    #                     if isinstance(x2, models.resnet.BasicBlock) or isinstance(x2, models.resnet.Bottleneck):
    #                         for n3, x3 in x2.named_children():
    #                             if isinstance(x3, nn.Conv2d):
    #                                 setattr(
    #                                     getattr(
    #                                         getattr(model, name),
    #                                         name2
    #                                     ),
    #                                     n3,
    #                                     WTFConv2d(
    #                                         conv=x3,m=m,classes_number=classes_number
    #                                     )
    #                                 )
    #                             else:
    #                                 for n4,x4 in x3.named_children():
    #                                     if isinstance(x4, nn.Conv2d) and True:
    #                                         setattr(
    #                                             getattr(
    #                                                 getattr(model, name),
    #                                                 name2
    #                                             ),
    #                                             n3,
    #                                             WTFConv2d(
    #                                                 conv=x4,m=m,classes_number=classes_number
    #                                             )
    #                                         )

    #     else:
    #         raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')

    # else:
    #     if isinstance(model, models.VGG):
    #         for name, l in enumerate(model.features):
    #             if isinstance(l, nn.Conv2d):
    #                 model.features[name] = AWC2d(conv=l, m=m)
    #     elif isinstance(model, models.ResNet):
    #         for name, x in model.named_children():
    #             if isinstance(x, nn.Conv2d):
    #                 setattr(model, name, AWC2d(conv=x, m=m))
    #             else:
    #                 for name2, x2 in x.named_children():
    #                     if isinstance(x2, models.resnet.BasicBlock):
    #                         for n3, x3 in x2.named_children():
    #                             if isinstance(x3, nn.Conv2d):
    #                                 setattr(
    #                                     getattr(
    #                                         getattr(model, name),
    #                                         name2
    #                                     ),
    #                                     n3,
    #                                     AWC2d(conv=x3, m=m)
    #                                 )

    #     else:
    #         raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')

    # return model

def invert_alphas(model:nn.Module) -> nn.Module :
    if isinstance(model, models.VGG):
        for name, l in enumerate(model.features):
            if isinstance(l, WTFConv2d):
                model.features[name].alpha.data *= -1

    elif isinstance(model, models.ResNet):
        for name, x in model.named_children():
            if isinstance(x, WTFConv2d):
                x.alpha.data *= -1
            else:
                for name2, x2 in x.named_children():
                    if isinstance(x2, models.resnet.BasicBlock):
                        for n3, x3 in x2.named_children():
                            if isinstance(x3, WTFConv2d):
                                x3.alpha.data *= -1
    return model

def normalize_alphas(model:nn.Module, r:list=[-3,3]) -> nn.Module :
    if isinstance(model, models.VGG):
        for name, l in enumerate(model.features):
            if isinstance(l, WTFConv2d):
                norm_a = model.features[name].alpha.data

                norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
                norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
                norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
                                                       torch.tensor(10e-8).to(device=norm_a_max_non_zero.device))
                norm_a /= norm_a_max_non_zero
                norm_a *= 2*r[1]
                norm_a -= r[1]

    elif isinstance(model, models.ResNet):
        for name, x in model.named_children():
            if isinstance(x, WTFConv2d):
                norm_a = x.alpha.data

                norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
                norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)
                norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
                                                  torch.tensor(10e-8).to(device=norm_a_max_non_zero.device))
                norm_a /= norm_a_max_non_zero
                norm_a *= 2 * r[1]
                norm_a -= r[1]
            else:
                for name2, x2 in x.named_children():
                    if isinstance(x2, models.resnet.BasicBlock):
                        for n3, x3 in x2.named_children():
                            if isinstance(x3, WTFConv2d):
                                norm_a = x3.alpha.data

                                norm_a -= norm_a.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
                                norm_a_max_non_zero = norm_a.flatten(start_dim=-2).max(-1).values.unsqueeze(
                                    -1).unsqueeze(-1)
                                norm_a_max_non_zero = torch.where(norm_a_max_non_zero != 0, norm_a_max_non_zero,
                                                                  torch.tensor(10e-8).to(
                                                                      device=norm_a_max_non_zero.device))
                                norm_a /= norm_a_max_non_zero
                                norm_a *= 2 * r[1]
                                norm_a -= r[1]
    return model
# def train_only_alphas(model:nn.Module) -> nn.Module :
#
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, WTFConv2d):
#                 model.features[name].alpha.requires_grad_()
#                 model.features[name].conv.requires_grad_(requires_grad=False)
#             else:
#                 model.features[name].requires_grad = False
#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, WTFConv2d):
#                 setattr(getattr(model, name), "requires_grad", True)
#             elif len(list(x.named_children())) == 0:
#                 setattr(getattr(model, name), "requires_grad", False)
#             else:
#                 for name2, x2 in x.named_children():
#                     if isinstance(x2, models.resnet.BasicBlock):
#                         for n3, x3 in x2.named_children():
#                             if isinstance(x3, WTFConv2d):
#                                 setattr(
#                                     getattr(
#                                         getattr(
#                                             getattr(model, name)
#                                             ,
#                                             name2
#                                         ),
#                                         n3
#                                     ),
#                                     "requires_grad" ,
#                                     True
#                                 )
#                             else:
#                                 setattr(
#                                     getattr(
#                                         getattr(
#                                             getattr(model, name)
#                                             ,
#                                             name2
#                                         ),
#                                         n3
#                                     ),
#                                     "requires_grad",
#                                     False
#                                 )
#
#     else:
#         raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')
#
#     return model



#########################################

# import torch
# import torch.nn as nn
# import torchvision.models as models
#
# class AlphaWeight(nn.Module):
#     def __init__(self, weight: torch.Tensor):
#         super(AlphaWeight, self).__init__()
#
#         self.weight = nn.Parameter(weight)
#
#     def forward(self):
#         return self.weight
#
# class WTFConv2d(nn.Module):
#     def __init__(self, conv: nn.modules.Conv2d):
#         super(WTFConv2d, self).__init__()
#         # assert isinstance(torch.Tensor,alpha), f'alpha must be of type torch.Tensor. Found {type(alpha)}'
#         # assert alpha.shape == torch.Size([kernel_size, 1]), f'alpha must have shape [kernel_size, 1].Found {alpha.shape}'
#         self.conv = conv
#         # self.alpha = nn.Parameter(to
#         # rch.randn(self.conv.in_channels, device='cuda', dtype=torch.FloatTensor))
#         self.alpha = nn.Parameter(torch.ones(self.conv.out_channels, dtype=torch.float))
#
#
#     def forward(self, x):
#         dims = self.conv.weight.ndim - 1
#         self.conv.weight.data *= nn.functional.sigmoid(self.alpha)[(...,) + (None,) * dims]
#         if type(self.conv.bias) == None:
#             self.conv.bias.data *= nn.functional.sigmoid(self.alpha)
#         # weights = self.conv.weight * self.alpha[(...,) + (None,) * dims]
#         # biases = self.conv.bias * self.alpha
#         x = self.conv._conv_forward(x, self.conv.weight, self.conv.bias)
#         return x
#
#
# def convert_conv2d_to_alpha(model:nn.Module) -> nn.Module :
#
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, nn.Conv2d):
#                 model.features[name] = WTFConv2d(conv=l)
#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, nn.Conv2d):
#                 setattr(model, name, WTFConv2d(conv=x))
#             else:
#                 for name2, x2 in x.named_children():
#                     if isinstance(x2, models.resnet.BasicBlock):
#                         for n3, x3 in x2.named_children():
#                             if isinstance(x3, nn.Conv2d):
#                                 setattr(
#                                     getattr(
#                                         getattr(model, name),
#                                         name2
#                                     ),
#                                     n3,
#                                     WTFConv2d(conv=x3)
#                                 )
#
#     else:
#         raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')
#
#     return model

# def train_only_alphas(model:nn.Module) -> nn.Module :
#
#     if isinstance(model, models.VGG):
#         for name, l in enumerate(model.features):
#             if isinstance(l, WTFConv2d):
#                 model.features[name].alpha.requires_grad_()
#                 model.features[name].conv.requires_grad_(requires_grad=False)
#             else:
#                 model.features[name].requires_grad = False
#     elif isinstance(model, models.ResNet):
#         for name, x in model.named_children():
#             if isinstance(x, WTFConv2d):
#                 setattr(getattr(model, name), "requires_grad", True)
#             elif len(list(x.named_children())) == 0:
#                 setattr(getattr(model, name), "requires_grad", False)
#             else:
#                 for name2, x2 in x.named_children():
#                     if isinstance(x2, models.resnet.BasicBlock):
#                         for n3, x3 in x2.named_children():
#                             if isinstance(x3, WTFConv2d):
#                                 setattr(
#                                     getattr(
#                                         getattr(
#                                             getattr(model, name)
#                                             ,
#                                             name2
#                                         ),
#                                         n3
#                                     ),
#                                     "requires_grad" ,
#                                     True
#                                 )
#                             else:
#                                 setattr(
#                                     getattr(
#                                         getattr(
#                                             getattr(model, name)
#                                             ,
#                                             name2
#                                         ),
#                                         n3
#                                     ),
#                                     "requires_grad",
#                                     False
#                                 )
#
#     else:
#         raise NotImplementedError(f'{type(model)} not implemented yet. Only VGG-like and ResNet-like are.')
#
#     return model

#########################################