import os
import matplotlib.pyplot as plt
import torch
import random
import warnings
import numpy as np
import ast
# import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
from torch.utils.data import DataLoader, random_split, Subset
from collections import Counter
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import custom_archs.WTFCNN as WTFCNN
import json

hyperparams = {
        'alpha_init': 3,
    }
# with open('/mnt/beegfs/work/dnai_explainability/ok.json', 'r') as fp:
    #     data = json.load(fp)
baseline = 'standard'
arch_name = ('vgg',)
dataset = ('cifar10',)

colori = {
    'vgg': ('#548235', '#70AD47', '#70AD47'),
    'res18': ('#84D6F0', '#1896BE', '#1896BE'),
    'vit_tiny': ('#C00000', '#FB7F03', '#FB7F03')
}

nn = arch_name[0]
ds = dataset[0]

if 'res' not in nn and 'vgg' not in nn:

    with open('/mnt/beegfs/work/dnai_explainability/final_ckpts.json', 'r') as fp:
        data = json.load(fp)
else:
    with open('/mnt/beegfs/work/dnai_explainability/ok.json', 'r') as fp:
        data = json.load(fp)

checkpoint_path = data[baseline][f'{nn}_{ds.lower()}']
# root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-0'
# root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-1'
if os.path.exists(os.path.join(checkpoint_path, 'best.pt')):
    checkpoint_path = os.path.join(checkpoint_path, 'best.pt')
elif os.path.exists(os.path.join(checkpoint_path, 'final.pt')):
    checkpoint_path = os.path.join(checkpoint_path, 'final.pt')
else:
    checkpoint_path = os.path.join(checkpoint_path, 'last_intermediate.pt')

num_classes=10

if 'vgg'in nn:
    # unlearned model
    model_unl = WTFCNN.WTFCNN(
        kind=WTFCNN.vgg16, pretrained=True,
        m=hyperparams['alpha_init'], resume=checkpoint_path,
        dataset=ds.lower()
    )
    layer = model_unl.arch.features[40]

elif 'res18' in nn:
    # unlearned model
    model_unl = WTFCNN.WTFCNN(
        kind=WTFCNN.resnet18, pretrained=True,
        m=hyperparams['alpha_init'], resume=checkpoint_path,
        dataset=ds.lower()
    )

    layer = model_unl.arch.layer4[1].conv1

elif 'vit_small' in nn: # DA FINIRE --> manca vit tiny
    model_unl = WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_small_16224, pretrained=True,
        m=hyperparams['alpha_init'], resume=checkpoint_path,
        dataset=ds.lower()
    )
    layer = model_unl.arch.transformer.layers[1][0].fn


elif 'vit_tiny' in nn: # DA FINIRE --> manca vit tiny
    model_unl = WTFCNN.WTFTransformer(
        kind=WTFCNN.vit_tiny_16224, pretrained=True,
        m=hyperparams['alpha_init'], resume=checkpoint_path,
        dataset=ds.lower()
    )
    layer = model_unl.arch.transformer.layers[11][0].fn

model_unl.arch.eval()

# for nn in arch_name:
#     for ds in dataset:
#
#         print(f'{nn}_{ds}')
#         if 'res' not in nn and 'vgg' not in nn:
#
#             with open('/mnt/beegfs/work/dnai_explainability/final_ckpts.json', 'r') as fp:
#                 data = json.load(fp)
#         else:
#             with open('/mnt/beegfs/work/dnai_explainability/ok.json', 'r') as fp:
#                 data = json.load(fp)
#
#         checkpoint_path = data[baseline][f'{nn}_{ds.lower()}']
#         # root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-0'
#         # root = '/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/alpha_matrices/test_all_resnet18_1.0_1.0_2023-01-21-1'
#         if os.path.exists(os.path.join(checkpoint_path, 'best.pt')):
#             checkpoint_path = os.path.join(checkpoint_path, 'best.pt')
#         elif os.path.exists(os.path.join(checkpoint_path, 'final.pt')):
#             checkpoint_path = os.path.join(checkpoint_path, 'final.pt')
#         else:
#             checkpoint_path = os.path.join(checkpoint_path, 'last_intermediate.pt')
#
#         num_classes=10
#
#         if 'vgg'in nn:
#             # unlearned model
#             model_unl = WTFCNN.WTFCNN(
#                 kind=WTFCNN.vgg16, pretrained=True,
#                 m=hyperparams['alpha_init'], resume=checkpoint_path,
#                 dataset=ds.lower()
#             )
#             layer = model_unl.arch.features[37]
#
#         elif 'res18' in nn:
#             # unlearned model
#             model_unl = WTFCNN.WTFCNN(
#                 kind=WTFCNN.resnet18, pretrained=True,
#                 m=hyperparams['alpha_init'], resume=checkpoint_path,
#                 dataset=ds.lower()
#             )
#
#             layer = model_unl.arch.layer4[1].conv1
#
#         elif 'vit_small' in nn: # DA FINIRE --> manca vit tiny
#             model_unl = WTFCNN.WTFTransformer(
#                 kind=WTFCNN.vit_small_16224, pretrained=True,
#                 m=hyperparams['alpha_init'], resume=checkpoint_path,
#                 dataset=ds.lower()
#             )
#             layer = model_unl.arch.transformer.layers[1][0].fn
#
#
#         elif 'vit_tiny' in nn: # DA FINIRE --> manca vit tiny
#             model_unl = WTFCNN.WTFTransformer(
#                 kind=WTFCNN.vit_tiny_16224, pretrained=True,
#                 m=hyperparams['alpha_init'], resume=checkpoint_path,
#                 dataset=ds.lower()
#             )
#             layer = model_unl.arch.transformer.layers[11][0].fn
#
#         model_unl.arch.eval()
#
#
#         # general = models.resnet18(weights='ResNet18_Weights.DEFAULT')
#         # general.cuda().eval()
#         # unlearnt = WTFCNN.WTFCNN(
#         #     kind=WTFCNN.resnet18, pretrained=True,
#         #     m=3, classes_number=num_classes, resume=PATH,
#         #     dataset=dataset.lower()
#         # )
#         # unlearnt.arch.cuda().eval()
#
#         import pandas as pd
#         import plotly.graph_objects as go
#         import plotly.express as px
#         # import plotly.io as pio
#         # pio.renderers.default = "browser"
#
#         # layer = model_unl.arch.layer4[0].conv1  # layer to be analyzed
#         a_ = torch.sigmoid(layer.alpha.cpu().detach())
#         _a=1-a_
#         f = []
#         aidx = _a.topk(7, axis=1).indices
#         aidx=torch.cat((torch.Tensor([i for i in range(aidx.shape[0])]).unsqueeze(1), aidx), dim=1).long()
#         aidx = torch.cat(
#             [torch.Tensor([k,x]).long().unsqueeze(0)\
#                 for k in aidx[:,0] for x in aidx[k,1:]],
#             dim=0
#         )
#
#         # aidx = torch.argwhere(_a>.90)
#         # aidx = torch.cat(             BELLO MA INUTILE
#         #     [torch.Tensor(aidx[aidx[:,0] == i][:,1]).unsqueeze(0) \
#         #         for i in torch.unique(aidx[:,0])],
#         #     dim=0
#         # )
#
#         c = Counter(aidx[:,1:].cpu().flatten().tolist())
#         if 'res' in nn or 'vgg' in nn:
#             aid = torch.Tensor([k for k,v in c.items() if v>1]).long().to(a_.device)
#         else:
#             aid = torch.Tensor([k for k,v in c.items() if v>1]).long().to(a_.device)
#         aid = aidx[torch.isin(aidx[:,1], aid)]
#
#         oldaidx = aidx.clone()
#         aidx = aid if aid.numel() else aidx
#
#         # for i in range(len(aidx[:,0])):
#         #     f.append(_a[aidx[i][0], aidx[i][1]].unsqueeze(0).unsqueeze(0))
#
#         for i in torch.unique(aidx[:,0]):
#             f.append(_a[i, aidx[aidx[:,0] == i,1:]].unsqueeze(0))
#
#         f = torch.cat(f, dim=1)
#
#         a = f.numpy()
#         # a = a[torch.argwhere(a.topk(3))]
#         # df = pd.DataFrame(a, columns=[*[f'a_{i}' for i in range(a.shape[1])]])
#         d = []
#         d = [pd.DataFrame(a[i], columns=['val']) for i in range(a.shape[0])]
#         C = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"] \
#             if ds == 'cifar10' else ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#         for i in range(a.shape[0]):
#             d[i]['a_{ci}'] = aidx[:,1] # d[i].index
#             d[i]['class'] = aidx[:,0]
#
#         df = pd.concat(d)
#         df['names'] = df['class'].map(lambda x:C[x])
#         # df = df[:15]
#         # df = df[50:150]
#
#         acidim=go.parcats.Dimension(values=df['a_{ci}'], label='#')
#         classdim=go.parcats.Dimension(values=df['names'], label='class')
#         # colorscale=['rgba(0,0,255,0.1)','#1896BE']
#         colorscale = [[0.0, 'lightsteelblue'], [0.5, 'mediumseagreen'], [1.0, 'red']]
#         # colorscale=[ 0, 'sunset']
#         color=df['val']
#         val=df['val']
#         val -= val.min()
#         val /= (val.max()+1e-8)
#
#         if val.max() == 0:
#             val += 1
#         from plotly.subplots import make_subplots
#         colorscale = [
#             [0, colori[nn][0]],
#             [0.5, colori[nn][1]],
#             [1, colori[nn][2]]
#         ]
#
#
#         fig = go.Figure(
#             data=[
#                 go.Parcats(
#                     tickfont={'size':17},
#                     labelfont={'size':23},
#                     counts=[v for (i,v) in enumerate(val)], dimensions=[acidim,classdim],
#                     line={'color':color,'colorscale': colorscale,  'shape':'hspline'},
#                 )
#             ]
#         )
#         fig.write_image(f"parcats/{ds}_{model_unl.kind}.pdf")
dataset = 'imagenet'
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

    # T = model.T if hasattr(model, 'T') and model.T is not None else T

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

    val = Subset(_val, idx_train)

    val.targets = torch.Tensor(_val.targets).int()[idx_train].tolist()
    # _val=Subset(_val, range(2))
elif dataset.lower() == 'cifar10':
    size = 32
    T = transforms.Compose([
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
        transform=T, download=False, train=False
    )
    val = _val

elif dataset.lower() == 'cifar100':

    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    # _train = CIFAR100(
    #     root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/train',
    #     transform=T, download=True, train=True
    # )

    _val = CIFAR100(
        root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/val',
        transform=T, download=False, train=False
    )

elif dataset.lower() == 'mnist':
    size = 32
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # _train = MNIST(
    #     root='/work/dnai_explainability/unlearning/datasets/mnist_classification/train',
    #     transform=T, download=True, train=True
    # )

    _val = MNIST(
        root='/work/dnai_explainability/unlearning/datasets/mnist_classification/val',
        transform=T, download=False, train=False
    )
    val = _val

classes_number = len(_val.classes)

unlearnt = model_unl
unlearnt.arch.cuda().eval()

lab=[i for i in range(classes_number)]
lab=[281]

id_c = np.where(np.isin(np.array(_val.targets), [lab]))[0]
# id_others = np.where(~ np.isin(np.array(_val.targets), c_to_del))[0]

val_c = Subset(_val, id_c)

with open('class_names/names.txt',  'r') as f:
    txt = f.read()

classes = ast.literal_eval(txt)
val_c.names=classes

val_loader = DataLoader(val_c, batch_size=1, num_workers=0, shuffle=True)
AH=[]

def register_hook(module, inp, out):
    AH.append(out)


layer.register_forward_hook(register_hook)

with torch.inference_mode():
    for img,lab in val_loader:
        img,lab=img.cuda(), lab.cuda()
        lab=torch.tensor([3]).cuda()

        unlearnt(img,labels=lab)
        # print(C[lab[0]])

        fin = int(input())
        f=layer.conv.weight[fin].data.squeeze()
        f=FF.interpolate(
            f[(None,)*(4-f.ndim)+(...,)], (7,7), mode='bilinear',align_corners=False
        )
        try:    
            if f.shape[1]>1:
                plt.imshow(f.detach().cpu().squeeze().permute(1,2,0))
            else:
                plt.imshow(f.detach().cpu().squeeze())
            plt.savefig('filter.png')
        except:
            pass
        # _a=a.clone()
        a = FF.interpolate(AH[-1][0,fin][(None,)*2+(...,)], (224,224), mode='bilinear',align_corners=False)
        a-=a.min()
        a/=(a.max()+1e-8)
        ups = FF.interpolate(img[(...,)], (224,224), mode='bilinear',align_corners=False)
        d=(ups*a)
        d[d<0]=0
        # plt.imshow(a.squeeze().detach().cpu().numpy(), cmap='jet')
        # plt.savefig("test.png")
        plt.imshow(img[0].detach().cpu().permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(d.squeeze().detach().cpu().permute(1,2,0).numpy())
        plt.show()
        plt.imshow(a.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.show()


        x=0


# x_idx = torch.argwhere(a<-2).tolist()

# minidx = int(unlearnt.arch.layer1[0].conv1.alpha[lab].argmin())

# n_features = a.shape[0]
# feature_idxs = list(range(n_features))
# # feature_idxs.remove(minidx)
# marginal_contributions = []

# with torch.no_grad():
#     for idx, (imgs, labels) in enumerate(val_loader):
#         for _ in range(1000):
#             imgs = imgs.cuda()
#             LABELS = labels.tolist()
#             labels = labels.cuda()

#             z = unlearnt.arch.layer1[0].conv1.alpha[random.randint(0, 1000)]

#             x_idx = random.sample(
#                 torch.argwhere(a>1).squeeze().tolist(),
#                 random.randint(int(.2 * n_features), int(0.8*n_features))
#             )

#             x_idx[random.choice([i for i in range(len(x_idx))])] = minidx
#             # z_idx = [idx for idx in feature_idxs if idx not in x_idx]

#             z_idx = random.sample(
#                 torch.argwhere(a > 1).squeeze().tolist(),
#                 random.randint(int(.2 * n_features), int(0.8 * n_features))
#             )

#             x_i = [xx for xx in x_idx]
#             z_i = [xx for xx in z_idx]


#             random.shuffle(x_idx)
#             random.shuffle(z_idx)

#             xpj = a.clone()
#             xpj[x_i] = a[x_idx]
#             # xpj[z_idx] = z[z_idx]

#             xmj = a.clone()
#             xmj[z_i] = a[z_idx]
#             # xmj[x_idx] = a[x_idx]

#             p = WTFCNN.WTFCNN(kind=WTFCNN.resnet18, m=3, resume=PATH)
#             p.arch.cuda().eval()
#             m = WTFCNN.WTFCNN(kind=WTFCNN.resnet18, m=3, resume=PATH)
#             m.arch.cuda().eval()

#             p.arch.layer1[0].conv1.alpha[lab] = xpj
#             m.arch.layer1[0].conv1.alpha[lab] = xmj

#             marginal_contribution = \
#                 torch.softmax(p(imgs, labels=labels),1)[:,labels] - torch.softmax(m(imgs, labels=labels),1)[:,labels]

#             marginal_contributions.append(marginal_contribution)

#             x=0


