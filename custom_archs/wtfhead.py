from .wtflayer import WTFLayer
import torch
import torch.nn as nn

from timm.models.vision_transformer import \
    VisionTransformer, \
    Attention, \
    Block

from timm.models.swin_transformer import \
    SwinTransformer, \
    SwinTransformerBlock

from .non_imagenet_models.vit import ViT
from .non_imagenet_models.vit import PreNorm, Attention as ATT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .non_imagenet_models.swin import swin_s

from timm.models.layers.mlp import Mlp

from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from collections.abc import Iterable
from typing import Optional, List, Tuple, Union

Tensor = torch.Tensor
amax,amin=1e1,-1e1
eps=1e-12

class WTFHead(WTFLayer):
    def __init__(self, attn: nn.Module, m:float = .5, classes_number=1000, ncols=1, kind='vit'):
        global amax, amin
        super(WTFHead, self).__init__()
        # assert isinstance(torch.Tensor,alpha), f'alpha must be of type torch.Tensor. Found {type(alpha)}'
        # assert alpha.shape == torch.Size([kernel_size, 1]), f'alpha must have shape [kernel_size, 1].Found {alpha.shape}'
        
        self.kind = kind
        amax, amin = m, -m
        self.attn = attn
        self.classes_number = classes_number
        # self.alpha = nn.Parameter(torch.randn(self.linear.in_channels, device='cuda', dtype=torch.FloatTensor))
        # self.alpha = nn.Parameter(torch.ones(self.classes_number, ncols, dtype=torch.float) *amax)
        
        if not hasattr(self.attn, 'num_heads'):
            self.attn.num_heads = self.attn.heads

        if hasattr(self.attn, 'qkv'):
            self.alpha = nn.Parameter(torch.ones(self.classes_number, self.attn.qkv.out_features, dtype=torch.float) *amax)
        elif hasattr(self.attn, 'to_qkv'):
            self.alpha = nn.Parameter(torch.ones(self.classes_number, self.attn.to_qkv.out_features, dtype=torch.float) *amax)
        
        # self.alpha = nn.Parameter(torch.randn(self.classes_number, self.attn.qkv.out_features, requires_grad=True, dtype=torch.float))
        # self.alpha.data *= 2.
        # self.alpha.data -= 1.
        # self.alpha.data *= amax
        # self.alpha = nn.Parameter(torch.ones(1000, self.linear.out_channels, dtype=torch.float) * m)
        # self.register_parameter(name='Alpha', param=self.alpha)

    def _qkv_forward(self, x, w, b):

        t = []

        for i in range(self.bs):

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

    def _qkv_swin_forward(self, x, w, b):

        t = []

        for i in range(self.bs):

            if b is not None:
                t.append(F.linear(
                    x.reshape(self.bs, -1, x.shape[-2], x.shape[-1])[i].unsqueeze(0),
                    w[i],
                    b[i]
                ))
            else:
                t.append(F.linear(
                    x.reshape(self.bs, -1, x.shape[-2], x.shape[-1])[i].unsqueeze(0),
                    w[i],
                    None
                ))

        ret = torch.cat(t, dim=0)
        return ret.reshape(-1, ret.shape[-2], ret.shape[-1])
        
    def _proj_forward(self, x, w, b):

        t = []

        for i in range(self.bs):

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
    
    def _attn_forward(self, x, w, b):
        B, N, C = x.shape

        if not isinstance(self.attn, nn.Linear):
            qkv = self._qkv_forward(x,w,b).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
            # qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.attn.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn.attn_drop(attn)

            # attn = attn.flatten() * w[(...,)]
            # ! YOU SHALL NOT mask activations!
            # * (bs, nh, it, ot) -> (bs, it, nh, ot) -> (bs, it, -1)
            # attn = (attn.permute(0,2,1,3).flatten(start_dim=-2,end_dim=-1) * w.unsqueeze(1))\
            #     .reshape(
            #         x.shape[0],
            #         self.alpha.shape[1]//self.attn.num_heads,
            #         self.attn.num_heads,
            #         self.alpha.shape[1]//self.attn.num_heads
            #     ).permute(0,2,1,3)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # x = self._proj_forward(x, w, b)
            x = self.attn.proj(x)
            x = self.attn.proj_drop(x)
        else:
            x = self._qkv_forward(x,w,b)
        

        return x
    
    def _attn_new_forward(self, x, w, b):
        qkv = self._qkv_forward(x,w,b).chunk(3, dim = -1)
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.attn.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.attn.scale

        attn = self.attn.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.attn.to_out(out)
        

        return x
    
    def _attn_swin_forward(self, x, w, b, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self._qkv_swin_forward(x,w,b).reshape(B_, N, 3, self.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.attn.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + self.attn._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.attn.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.attn.num_heads, N, N)
            attn = self.attn.softmax(attn)
        else:
            attn = self.attn.softmax(attn)

        attn = self.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if isinstance(self.attn, nn.Linear):
            dims = self.attn.weight.ndim - 1
        else:
            if hasattr(self.attn, 'qkv'):
                dims = self.attn.qkv.weight.ndim - 1
            elif hasattr(self.attn, 'to_qkv'):
                dims = self.attn.to_qkv.weight.ndim - 1
        # alpha = self.alpha[self.label].squeeze()

        # self.alpha = nn.Parameter(torch.ones_like(self.alpha))
        # alpha = nn.Parameter(1 - torch.sigmoid(self.alpha))
        # weight = self.linear.weight * alpha[(...,) + (None,) * dims]

        if len(self.label.shape) == 0:
            self.label = self.label.unsqueeze(0)
        if self.bs > self.label.shape[0]:
            self.label = torch.Tensor(
                [self.label for _ in range(x.shape[0])]
            ).long()

        if isinstance(self.attn, nn.Linear):
            weight = self.attn.weight * \
                       torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]
        else:
            if hasattr(self.attn, 'qkv'):
                weight = self.attn.qkv.weight * \
                        torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]
                
                if isinstance(self.attn, nn.Linear):
                    if self.attn.bias is not None:
                        # bias = self.conv.bias * alpha[self.label].squeeze()
                        bias = self.attn.bias * torch.sigmoid(self.alpha[self.label].squeeze())
                        # bias = self.conv.bias * self.alpha
                else:
                    if self.attn.qkv.bias is not None:
                        # bias = self.conv.bias * alpha[self.label].squeeze()
                        bias = self.attn.qkv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
                    else:
                        bias = None
            elif hasattr(self.attn, 'to_qkv'): 
                weight = self.attn.to_qkv.weight * \
                        torch.sigmoid(self.alpha[self.label].squeeze())[(...,) + (None,) * dims]
                if isinstance(self.attn, nn.Linear):
                    if self.attn.bias is not None:
                        # bias = self.conv.bias * alpha[self.label].squeeze()
                        bias = self.attn.bias * torch.sigmoid(self.alpha[self.label].squeeze())
                        # bias = self.conv.bias * self.alpha
                else:
                    if self.attn.to_qkv.bias is not None:
                        # bias = self.conv.bias * alpha[self.label].squeeze()
                        bias = self.attn.to_qkv.bias * torch.sigmoid(self.alpha[self.label].squeeze())
                    else:
                        bias = None
        
        if self.kind == 'swin':
            x = self._attn_swin_forward(x, weight, bias, mask)
        elif self.kind == 'vit':
            x = self._attn_forward(x, weight, bias)
        elif self.kind == 'vit_new':
            x = self._attn_new_forward(x, weight, bias)

        return x

def get_all_alpha_layers(model:nn.Module, sigmoid=False) -> dict :
    ret = {}
    # if isinstance(model, VisionTransformer):
    for name, x in model.named_modules():
        if isinstance(x, WTFHead):
            if not sigmoid:
                ret[name] = x.alpha.data
            else:
                ret[name] = torch.sigmoid(x.alpha.data)
    return ret

def get_all_layer_norms(model:nn.Module, norm:int = 1, m:float=0.) -> Tensor :
    ret = []

    # if isinstance(model, VisionTransformer):
    for name, x in model.named_modules():
        if isinstance(x, WTFHead):
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
        if isinstance(x, WTFHead):
            x.label = label
            x.bs = bs
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
    # if isinstance(model, VisionTransformer):
    for name, x in model.named_modules():
        if isinstance(x, WTFHead):
            x.alpha.data[
                x.alpha.data < amin
            ] = amin
            x.alpha.data[
                x.alpha.data > amax
            ] = amax


def convert_head_to_alpha(
        model:nn.Module,
        m:float = .5, standard=False, classes_number=1000
) -> nn.Module :

    if isinstance(model, VisionTransformer):
        for name, x in model.named_modules():
            if isinstance(x, Block):
                x.attn = WTFHead(
                    attn=x.attn,m=m,classes_number=classes_number, ncols = x.attn.num_heads * model.pos_embed.shape[1]
                )
                # x.attn.attn.qkv = WTFHead(
                #     attn=x.attn.attn.qkv,m=m,classes_number=classes_number, ncols = x.attn.attn.qkv.out_features
                # )
            pass
    
    elif isinstance(model, SwinTransformer):
        for name, x in model.named_modules():
            if isinstance(x, SwinTransformerBlock):
                x.attn = WTFHead(
                    attn=x.attn,m=m,classes_number=classes_number,kind='swin'
                )
                # x.attn.attn.qkv = WTFHead(
                #     attn=x.attn.attn.qkv,m=m,classes_number=classes_number, ncols = x.attn.attn.qkv.out_features
                # )
            pass
    elif isinstance(model, ViT):
        for name, x in model.named_modules():
            if isinstance(x, PreNorm) and isinstance(x.fn, ATT):
                x.fn = WTFHead(
                    attn=x.fn,m=m,classes_number=classes_number,kind='vit_new'
                )
                # x.attn.attn.qkv = WTFHead(
                #     attn=x.attn.attn.qkv,m=m,classes_number=classes_number, ncols = x.attn.attn.qkv.out_features
                # )
            pass
    elif isinstance(model, swin_s):
        for name, x in model.named_modules():
            if isinstance(x, PreNorm) and isinstance(x.fn, ATT):
                x.fn = WTFHead(
                    attn=x.fn,m=m,classes_number=classes_number,kind='vit_new'
                )
                # x.attn.attn.qkv = WTFHead(
                #     attn=x.attn.attn.qkv,m=m,classes_number=classes_number, ncols = x.attn.attn.qkv.out_features
                # )
            pass

    else:
        raise NotImplementedError(f'{type(model)} not implemented yet. Only ViT-like and Swin-like are.')

    return model

def invert_alphas(model:nn.Module) -> nn.Module :
    # if isinstance(model, VisionTransformer):
    for name, x in model.named_modules():
        if isinstance(x, WTFHead):
            x.alpha *= (-1.)
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
