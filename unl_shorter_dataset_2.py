import os
import torch
import torchvision.models as models
import torch.nn.functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import math
from datetime import datetime
import wandb
from PIL import Image
# from images_utils import images_utils as IMUT
import matplotlib.pyplot as plt
import ast
from torchviz import make_dot

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

def main(args):

    debug = args.debug
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
        loss_type='3way_sum',lambdas=[hyperparams[l] for l in hyperparams.keys() if 'lambda' in l],
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

    if 'vgg16'in arch_name:
        model=wfmodels.WFCNN(
            kind=wfmodels.vgg16, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFCNN(
            kind=wfmodels.vgg16, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif "resnet34" in arch_name:
        # model = models.resnet18(pretrained=True)
        model=wfmodels.WFCNN(
            kind=wfmodels.resnet34, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFCNN(
            kind=wfmodels.resnet34, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif 'resnet18' in arch_name:
        model=wfmodels.WFCNN(
            kind=wfmodels.resnet18, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )
        general=wfmodels.WFCNN(
            kind=wfmodels.resnet18, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif 'deit_small_16224' in arch_name:
        model=wfmodels.WFTransformer(
            kind=wfmodels.deit_small_16224, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFTransformer(
            kind=wfmodels.deit_small_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif 'vit_small_16224' in arch_name:
        model=wfmodels.WFTransformer(
            kind=wfmodels.vit_small_16224, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFTransformer(
            kind=wfmodels.vit_small_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif 'vit_tiny_16224' in arch_name:
        model=wfmodels.WFTransformer(
            kind=wfmodels.vit_tiny_16224, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFTransformer(
            kind=wfmodels.vit_tiny_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    elif 'swin_small_16224' in arch_name:
        model=wfmodels.WFTransformer(
            kind=wfmodels.swin_small_16224, pretrained=False,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower()
        )

        general=wfmodels.WFTransformer(
            kind=wfmodels.swin_small_16224, pretrained=True,
            m=hyperparams['alpha_init'], resume=None,
            dataset=dataset.lower(),alpha=False
        )
    general.arch.requires_grad_(requires_grad=False)
        # model.arch.head.register_forward_hook(hook_fn)
        # general.arch.head.register_forward_hook(hook_fn)


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

        import numpy as np
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
        val.targets = torch.Tensor(_val.targets).int()[idx_val].tolist()


    elif dataset.lower() == 'cifar10':

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
        train, val = _train, _val

    elif dataset.lower() == 'cifar100':

        means = [0.5, 0.5, 0.5]
        stds = [0.5, 0.5, 0.5]
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

        _train = CIFAR100(
            root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/train',
            transform=T, download=True, train=True
        )

        _val = CIFAR100(
            root='/work/dnai_explainability/unlearning/datasets/cifar100_classification/val',
            transform=T, download=True, train=False
        )

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

    classes_number = len(_val.classes)

    
    # model.requires_grad_(requires_grad=False)
    # model = convert_conv2d_to_alpha(model, m=hyperparams['alpha_init'])

    root='//mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/'

    if model.resume is None:
        if not os.path.isdir(os.path.join(root,wdb_proj)):
            os.mkdir(os.path.join(root,wdb_proj))

        folder_name = f'{wdb_name}_{datetime.today().strftime("%Y-%m-%d")}'
        folder_name += f'-{len(os.listdir(os.path.join(root,wdb_proj)))}'
        run_root = os.path.join(root, wdb_proj, folder_name)
        if not os.path.isdir(os.path.join(root,wdb_proj,folder_name)):
            os.mkdir(run_root)

        with open(f"{run_root}/config", 'w') as f:
            f.write(str(hyperparams))
    else:
        folder_name = root
        run_root = folder_name
        with open(f"{folder_name}/config_2", 'w') as f:
            f.write(str(hyperparams))


    # model.load_state_dict(torch.load(PATH))

    # model = train_only_alphas(model)
    # model = MVGG.vgg16(pretrained=False)

    # for child in model.named_children():
    #     n, c = child
    #     try:
    #         for x in c:
    #             if 'conv' in x._get_name().lower():
    #                 x = AlphaWeightedConv2d(conv=x)
    #     except Exception as e:
    #         print(e)

    # params1 = standard.named_parameters()
    # params2 = model.named_parameters()
    # params1 = dict(params1)
    # dict_params2 = dict(params2)
    #
    # for name2, param2 in params2:
    #     if 'conv' in dict_params2:
    #         dict_params2[name2].data.copy_(params1[''.join([s for s in name2.split('conv.')])].data)
    # id_c = np.where(np.isin(np.array(_train.targets), c_to_del))[0]
    # # id_c = id_c[:int(perc * len(id_c))]
    # id_others = np.where(~ np.isin(np.array(_train.targets), c_to_del))[0]


    # train_c = Subset(_train, id_c)
    # train_c.targets = torch.Tensor(_train.targets).int()[id_c].tolist()
    # train_others = Subset(_train, id_others)
    # train_others.targets = torch.Tensor(_train.targets).int()[id_others].tolist()
    #
    # concat_train = data.ConcatDataset((train_c,train_others))
    # concat_train.targets = torch.Tensor(_train.targets).int()[id_c].tolist()
    # concat_train.targets = [*train_c.targets, *train_others.targets]

    # id_c = np.where(np.isin(np.array(_val.targets), c_to_del))[0]
    # id_others = np.where(~ np.isin(np.array(_val.targets), c_to_del))[0]
    # val_c = Subset(_val,id_c)
    # val_others = Subset(_val, id_others)

    # test = ImageFolder(
    #     root='/nas/softechict-nas-2/datasets/Imagenet_new/ILSVRC/Data/CLS-LOC/test',
    #     transform=T
    # )

    # layer = 0
    # freeze0=freeze=True
    # perc=.3
    # for i, (n,p) in enumerate(model.named_parameters()):
    #     if i < perc * len(list(model.named_parameters())):
    #         try:
    #             p.weight.requires_grad=False
    #         except:
    #             pass

    # for param in model.named_parameters():
    #     name, p = param
    #     if freeze:
    #         p.requires_grad = False
    #
    #     if str(layer) in name:
    #         freeze = not freeze0
    #     if 'classifier' in name:
    #         freeze = True

    # optimizer=Adam((x for n,x in model.arch.named_parameters() if 'alpha' in n), lr=hyperparams['ur'])
    optimizer=SGD((x for n,x in model.arch.named_parameters() if 'alpha' in n), lr=hyperparams['ur'])
    # optimizer=torch.optim.SGD(model.parameters(), lr=hyperparams['lr'])


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
        model.arch.eval()

        save_checkpoint_frequency = 50
        validation_frequency = int(len(train)/1000)
        # evaluation_frequency = hyp['evaluation_frequency']
        evaluation_frequency = 0 \
            if hyp['evaluation_frequency'] == 0 \
            else int(len(train)/ (hyp['evaluation_frequency'] * hyp['batch_size']))
        if evaluation_frequency:
            validation_frequency = evaluation_frequency
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

        # with open('class_names/names.txt', 'r') as f:
        #     txt = f.read()
        #
        # classes = ast.literal_eval(txt)

        batch_train = batch_val = hyp['batch_size']
        # train_loader = DataLoader(train, batch_size=batch_train, shuffle=True)

        # val_c,val_ot=val
        # max_val_size=20_000
        # size_val = min(max_val_size, int(len(val_c)))
        # size_val = max(max_val_size, int(.1 * len(val_ot)))
        # otval_loader = DataLoader(random_split(val_ot, [size_val, len(val_ot) - size_val])[0], batch_size=64, shuffle=True)

        # otval_loader.dataset.dataset.names = classes
        # cval_loader.dataset.dataset.names = classes

        model.arch.to(device)

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
            # y_train = train.targets  # train.datasets[0].dataset.targets
            
            # weight = 1. / torch.Tensor([.001 for _ in range(1000)])
            
            # weight[~np.isin(list(range(len(weight))), np.array(c_to_del))] = .5 / (len(weight) - len(c_to_del))
            # weight[np.array(c_to_del)] = .5 / len(c_to_del)
            
            # samples_weight = np.array([weight[t] for t in y_train])
            # samples_weight = torch.from_numpy(samples_weight)
            # sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

            # train_loader = DataLoader(concat_train, batch_size=batch_train, num_workers=4, sampler=sampler)
            train_loader = DataLoader(train, batch_size=batch_train, shuffle=True)

            # loss_ctrain_list, loss_ottrain_list, loss_otval_list, loss_cval_list = [], [], [], []

            for idx, (imgs, labels) in enumerate(train_loader):
                print(f'Untraining: {round((100 * batch_train * idx)/len(train_loader.dataset),2)}%')

                # * setting images and labels to device
                imgs=imgs.to(device)#.requires_grad_(True)
                labels=labels.to(device)

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
                kept_portion = labels.cpu()[:int(labels.cpu().shape[0] / 2)].clone()
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

                kept_labels = generate_random_idxs(
                    kept_portion, hyp['unlearnt_kept_ratio'], classes_number
                )
                kept_imgs = imgs[:int(labels.shape[0] / 2)]

                unlearnt_labels = labels[int(labels.shape[0] / 2):]
                unlearnt_imgs = imgs[int(labels.shape[0] / 2):]

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
                #     general.arch.cuda().eval()
                #     general_score = general(unlearnt_imgs)

                
                # * forward step and loss computation for unlearning
                unlearnt_score = model(unlearnt_imgs, labels=unlearnt_labels)
                # unlearnt_loss = loss_fn(unlearnt_score, unlearnt_labels)
                unlearnt_loss = LM.classification_loss_selector(
                    unlearnt_score, unlearnt_imgs, 
                    unlearnt_labels, general
                )

                # * forward steps and loss computations for retaining
                # * eventually the losses are averaged
                kept_loss_list=[]

                # if general is not None:
                #         general.arch.cuda().eval()
                #         general_score = general(kept_imgs)

                for k in range(hyp['unlearnt_kept_ratio']):
                    # set_label(model, kept_labels[:,k].squeeze())
                    kept_score = model(kept_imgs, labels=kept_labels[:,k].squeeze())
                    # kept_loss_list.append(
                    #     loss_fn(
                    #         kept_score,
                    #         labels[:int(labels.shape[0] / 2)
                    #         ].squeeze()
                    #     ).unsqueeze(0)
                    # )
                    kept_loss_list.append(
                        LM.classification_loss_selector(
                            kept_score, kept_imgs,
                            labels[:int(labels.shape[0] / 2)], general
                        ).unsqueeze(0)
                    )

                kept_loss = torch.cat(kept_loss_list).mean(dim=0)

                # t_out_train = nn.functional.softmax(model(imgs), dim=1)
                
                # ! useless!
                # * losses are cat together
                # loss_cls = torch.cat(
                #     [unlearnt_loss, kept_loss]
                # )

                # mean_loss_ctrain = loss_train[
                #     torch.isin(labels, torch.Tensor(c_to_del).to(device))
                # ].mean()

                #
                # mean_loss_ottrain = loss_train[
                #     ~ torch.isin(labels, torch.Tensor(c_to_del).to(device))
                # ].mean()

                # loss_cls[
                #     torch.isin(labels, torch.Tensor(CLASS).to(device))
                # ] *= (-1)

                # * computes the mean alpha norm (regularization factor)
                alpha_norm = hyp['lambda2'] * (model.get_all_layer_norms(m=1.)).mean().to('cuda')
                # print(alpha_norm, torch.cat([m.mean().unsqueeze(0) for m in model.get_all_alpha_layers().values()], 0).mean())

                # * computes the final loss
                # * loss = lambda_0 * loss_ret^2 + lambda_1 * 1 / (loss_unl) + lambda_2 * alpha_norm
                if hyp['loss_type'] == 'difference':
                    keep = kept_loss.clone()
                    unlearn = unlearnt_loss.clone()
                    loss_cls = hyp['lambda0'] * keep.mean() - hyp['lambda1'] * unlearn.mean()
                    loss_train = loss_cls.mean()
                    loss_train += alpha_norm
                elif '3way' in hyp['loss_type']:
                    keep = kept_loss.clone()
                    unlearn = unlearnt_loss.clone()
                    if 'multiplication' in hyp['loss_type']:
                        loss_cls = (hyp['lambda0'] * keep.mean() / (hyp['lambda1'] * torch.abs(unlearn.mean())))
                        loss_train = loss_cls + alpha_norm
                    elif 'sum' in hyp['loss_type']:
                        loss_cls = torch.pow(hyp['lambda0'] * keep.mean(),2) + torch.pow(hyp['lambda1']/(unlearn.mean()+1e-8),1)
                        loss_train = loss_cls + alpha_norm

                        # ! for graphviz visualization only
                        # g = make_dot(
                        #     loss_train, 
                        #     params=dict((n,x) for n,x in model.arch.named_parameters() if 'alpha' in n),
                        #     show_attrs=True, show_saved=True
                        # )
                    elif 'third' in hyp['loss_type']:
                        loss_cls = hyp['lambda0'] * keep.mean() * (1 - hyp['lambda1'] / torch.abs(unlearn.mean())) # l+ * (1 - lambda1/l-)
                        loss_train = loss_cls + alpha_norm
                else:
                    loss_train = loss_cls.mean()


                # loss_ctrain_list.append(mean_loss_ctrain.cpu().item())
                # loss_ottrain_list.append(mean_loss_ottrain.cpu().item())

                # * zeroing and backwarding the grads, followed by an optimizer step
                # * NB: zeroing and stepping frequencies are handled by
                # * the zero_grad_frequency hyperparameter
                # if (idx + 1)  % hyp['zero_grad_frequency'] == 0:
                #     optimizer.zero_grad()

                loss_train.backward()
                # print(model.arch.features[0].alpha.grad.mean())

                if (idx + 1) % hyp['zero_grad_frequency'] == 0:

                    if hyp['clamp'] > 0:
                        nn.utils.clip_grad_norm_(model.arch.parameters(), hyp['clamp'])
                    optimizer.step()
                    optimizer.zero_grad()

                # * clipping alpha values between max and min
                model.clip_alphas() # clip alpha vals

                # * wandb loggings
                if not debug:
                    run.log({'alpha_norm': alpha_norm})
                    run.log({'loss_untrain': loss_train})
                    run.log({'loss_cls': loss_cls})
                    run.log({'train_keep': keep.mean()})
                    run.log({'train_unlearn': torch.abs(unlearn.mean())})
                    run.log({'l1c1_alpha_max': tuple(model.get_all_alpha_layers().values())[0].max()})
                    run.log({'l1c1_alpha_min': tuple(model.get_all_alpha_layers().values())[0].min()})

                # * validation steps:
                # * firstly the unlearning validation step
                # * secondly the retaining validation step
                if idx % validation_frequency == 0 and not debug:
                    
                    # * saving CUDA memory
                    unlearnt_loss.cpu()
                    kept_loss.cpu()
                    loss_train.cpu()

                    unlearnt_score.cpu()
                    kept_score.cpu()

                    unlearnt_labels.cpu()
                    kept_labels.cpu()

                    loss_cls.cpu()
                    alpha_norm.cpu()

                    imgs.cpu()
                    labels.cpu()

                    print(f'Validation step {idx}')

                    # * defining max number of validation images
                    max_val_size = min(hyp['max_val_size'], len(val))

                    # id_c = np.where(np.isin(
                    #     np.array(val.targets),
                    #     unlearnt_labels.cpu().numpy()
                    # ))[0]
                    # id_others = np.where(~ np.isin(
                    #     np.array(val.targets),
                    #     kept_labels.cpu().numpy()
                    # ))[0]
                    #
                    # val_c = Subset(val,id_c)
                    # val_others = Subset(val, id_others)
                    #
                    # c_size_val = min(max_val_size, int(len(val_c)))
                    # ot_size_val = min(max_val_size, int(len(val_others)))
                    #
                    # forget_val_loader = DataLoader(
                    #     random_split(val_c, [c_size_val, len(val_c) - c_size_val])[0],
                    #     batch_size=batch_val, num_workers=4, shuffle=True
                    # )
                    # keep_val_loader = DataLoader(
                    #     random_split(val_others, [ot_size_val, len(val_others) - ot_size_val])[0],
                    #     batch_size=batch_val, num_workers=4, shuffle=True
                    # )

                    val_loader = DataLoader(
                        random_split(val, [max_val_size, len(val)-max_val_size])[0],
                        batch_size=batch_val, num_workers=4, shuffle=True
                    )

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

                    # u_val_loader = DataLoader(
                    #     random_split(val_c, [max_val_size, len(val_c)-max_val_size])[0],
                    #     batch_size=batch_val, num_workers=4, shuffle=True
                    # )
                    # k_val_loader = DataLoader(
                    #     random_split(val_others, [max_val_size, len(val_others)-max_val_size])[0],
                    #     batch_size=batch_val, num_workers=4, shuffle=True
                    # )

                    mean_acc_forget = mean_acc_keep = 0.

                    with torch.inference_mode():
                        for ival, (ims, labs) in enumerate(val_loader):
                            ims=ims.cuda()
                            # labs=labs.cuda()

                            # set_label(model, labs)

                            outs = nn.functional.softmax(model(ims, labels=labs)).cpu()

                            mean_acc_forget += (outs.max(1).indices == labs).sum() / \
                                                labs.shape[0]

                        mean_acc_forget /= (ival + 1)
                        # ims.cpu()
                        for ival, (ims, labs) in enumerate(val_loader):
                            ims = ims.cuda()
                            # labs = labs.cuda()

                            labs_portion = labs.clone()

                            CLASS = generate_random_idxs(
                                labs_portion, 1, classes_number
                            ).squeeze()
                            # set_label(model, CLASS.cuda())

                            outs = torch.softmax(model(ims, labels=CLASS.cuda()),-1).cpu()

                            mean_acc_keep += (outs.max(1).indices == labs).sum() / \
                                                labs.shape[0]

                        mean_acc_keep /= (ival + 1)
                        ims.cpu()
                    run.log({'acc_on_unlearnt': mean_acc_forget})
                    run.log({'acc_of_kept': mean_acc_keep})
                    if general is not None and False:
                        mean_acc_gen=0.
                        general.arch.cuda()
                        general.arch.eval()
                        for ival, (ims, labs) in enumerate(val_loader):
                                ims = ims.cuda()
                                # labs = labs.cuda()

                                labs_portion = labs.clone()

                                CLASS = generate_random_idxs(
                                    labs_portion, 1, classes_number
                                ).squeeze()
                                # set_label(model, CLASS.cuda())

                                outs = torch.softmax(general(ims,labels=labs.cuda()), -1).cpu()

                                mean_acc_gen += (outs.max(1).indices == labs).sum() / \
                                                    labs.shape[0]

                        mean_acc_gen /= (ival + 1)

                    current_acc = 0.5 * ((1-mean_acc_forget) + mean_acc_keep)
                    if evaluation_frequency and idx % evaluation_frequency == 0:
                        if current_acc < best_acc:
                            patience -= 1
                            currents.append(current_acc)
                            if patience == 0:
                                should_stop = True
                        else: 
                            best_acc = current_acc
                            currents = list()
                            patience = hyp['initial_patience']
                            best_found = True
                    if not debug:
                        run.log({'best_val_acc': best_acc})
                        run.log({'current_val_acc': current_acc})

                    if should_stop:
                        print(f'mean_unl: {mean_acc_forget}, mean_ret: {mean_acc_keep}, current: {currents}, best: {best_acc}, patience: {patience}')
                        break
                
                # * saving intermediate checkpoints
                if idx % save_checkpoint_frequency == 0 or best_found:
                    #root = '/work/dnai_explainability/unlearning/icml2023'
                    PATH = f"{run_root}/last_intermediate.pt" if not best_found else f"{run_root}/best.pt"
                    # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/all_classes/{arch_name}_{perc}_{ur}_class-{c_to_del[0]}_alpha.pt"
                    # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/short/class_100-classes_freeze-30perc_untraining_{hyperparams['model']}.pt"
                    torch.save(model.arch.state_dict(), PATH)
                    best_found = False
                    print(f'Saved at {PATH}')

            # print(f'{epoch=} {loss_train=}')
            # for k in gmean:
            #     gmean[k] = torch.nn.functional.relu(gmean[k])
            # gnorm = {}
            # gstoc = {}
            # for k in gmean:
            #     gnorm[k] = (gmean[k] - gmean[k].min()) / (gmean[k].max() - gmean[k].min())
            #     gstoc[k] = gmean[k] / gmean[k].sum()

            # for k in gnorm:
            #     print(k, gnorm[k].mean())



            # _, val_split = random_split(val, [0, 50_000]

            # print(f'Val {epoch=} ')
            # model.eval()
            # with torch.no_grad():
            #     for idx, (imgs, labels) in enumerate(otval_loader):
            #         print(f'Validating: {round(batch_train * idx / len(otval_loader), 2)}%')
            #         imgs = imgs.to(device)
            #         labels = labels.to(device)
            #
            #         t_out_val = nn.functional.softmax(model(imgs))
            #         loss_val = loss_fn(t_out_val, labels).mean()
            #         loss_otval_list.append(loss_val.cpu().item())
            #
            #         cl_acc = (t_out_val.max(1).indices == labels).sum() / (t_out_val.max(1).indices == labels).shape[0]
            #         if not debug:
            #             run.log({'otval_acc': cl_acc})
            #             run.log({'otval_loss': loss_val})

                    # ot_t_acc = []
                    # for o, ol in others_loader:
                    #     o = o.cuda()
                    #     ol = ol.cuda()
                    #     test_outs = model(o)
                    #     ot_t_acc.append((test_outs.max(1).indices == ol).sum() / (test_outs.max(1).indices == ol).shape[0])

                    # run.log({'others_t_acc': torch.Tensor(ot_t_acc).mean()})

                # for idx, (imgs, labels) in enumerate(cval_loader):
                #     print(f'Validating: {round(batch_train * idx / len(cval_loader), 2)}%')
                #     imgs = imgs.to(device)
                #     labels = labels.to(device)
                #
                #     t_out_val = nn.functional.softmax(model(imgs))
                #     loss_val = loss_fn(t_out_val, labels).mean()
                #     loss_cval_list.append(loss_val.cpu().item())
                #
                #     cl_acc = (t_out_val.max(1).indices == labels).sum() / (t_out_val.max(1).indices == labels).shape[0]
                #     if not debug:
                #         run.log({'cval_acc': cl_acc})
                #         run.log({'cval_loss': loss_val})
                #
                #     ot_t_acc = []
                #     # for o, ol in others_loader:
                #     #     o = o.cuda()
                #     #     ol = ol.cuda()
                #     #     test_outs = model(o)
                #     #     ot_t_acc.append((test_outs.max(1).indices == ol).sum() / (test_outs.max(1).indices == ol).shape[0])
                #
                # print(f'{epoch=} {loss_val=}')


            # if epoch == 1 or epoch % 10 == 0:
            #     print(f"Epoch {epoch}")
            #     if not debug:
            #         run.log({'conv1_min': get_all_alpha_layers(model)[0].min()})
            #         run.log({'conv1_max': get_all_alpha_layers(model)[0].max()})
            #         run.log({'conv1_avg': get_all_alpha_layers(model)[0].mean()})
            #         run.log({'conv1_std': get_all_alpha_layers(model)[0].std()})


    if flipped:
        trainset, valset = val, train,
    else:
        trainset, valset = train, val

    train_loop(
        n_epochs = 100_000,
        optimizer = optimizer,
        model = model,
        loss_fn = cls_loss,
        train=trainset,
        val=valset,
        hyp=hyperparams,
        general=general
    )

    PATH = f"{run_root}/final.pt"
    # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/all_classes/{arch_name}_{perc}_{ur}_class-{c_to_del[0]}_alpha.pt"
    # PATH = f"/homes/spoppi/pycharm_projects/inspecting_twin_models/checkpoints/short/class_100-classes_freeze-30perc_untraining_{hyperparams['model']}.pt"
    torch.save(model.arch.state_dict(), PATH)
    print(f'Saved at {PATH}')

    x=0

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
    parser.add_argument("--patience", type=int, help='Initial patience', default=10)
    parser.add_argument('-T', "--validation-test-size", type=int, help='Validation test size', default=20_000)
    parser.add_argument('-R', "--unlearnt-kept-ratio", type=int, help='Unlearnt-kept ratio', default=5)
    parser.add_argument('-l', "--logits", type=bool, help='Compute loss over logits or labels', default=False)
    parser.add_argument("--clamp", type=float, help='Gradient clamping val', default=-1.0)
    args = parser.parse_args()

    main(args=args)
