import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from custom_archs import wfmodels_lora as wfmodels

unl_class = 0
total_images_number = 5000
dataset = 'cifar10'
model_name = 'vit_tiny'

if 'cifar10' in dataset:
    if 'tiny' in model_name:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    else:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_small_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/alpha_matrices/vit_small_neggrad_64_0.0001_1.0_0.01/2023-04-30/cifar10/difference/1.0-0.01/best_CLASS-{unl_class}.pt"
        # gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    c_number = 10
else:
    if 'tiny' in model_name:
        ours_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/unlearning/icml2023/alpha_matrices/checkpoints_acm/vit_tiny_16224_lora_zero_3way_fixed_l1_AB_256_0.01_0.001_1.0/2023-05-03/0.001-1.0/best_CLASS-{unl_class}.pt"
        loss_diff_ckp = f"/mnt/beegfs/work/dnai_explainability/lbaraldi/unlearning/icml2023/alpha_matrices/checkpoints_acm/vit_tiny_16224_neggrad_256_0.0001_1.0_0.005/2023-05-03/1.0-0.005/best_CLASS-{unl_class}.pt"
        gold_ckp = f"/mnt/beegfs/work/dnai_explainability/ssarto/checkpoints_gold/CIFAR10/vit_tiny/cifar10_vit_tiny-4-ckpt_original_{unl_class}.t7"
    else:
        ...
    c_number = 20

# Define the PyTorch model

if model_name == 'vit_tiny':
    kind = wfmodels.vit_tiny_16224
else:
    kind = wfmodels.vit_small_16224

model = wfmodels.WFTransformer(
    kind=kind, pretrained=True,
    resume=ours_ckp,
    dataset=dataset.lower(), alpha=False
)
model = model.arch

if 'cifar10' in dataset and 'tiny' in model_name:
    loss_diff_ckp = \
        f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/debug_instance_prod_64_5e-05_2.5e-05_1.0/2023-05-02/2.5e-05-1.0/best_CLASS-0.pt"
ldiff = wfmodels.WFTransformer(
    kind=kind, pretrained=True,
    resume=loss_diff_ckp,
    dataset=dataset.lower(), alpha=False
)
ldiff = ldiff.arch

general = wfmodels.WFTransformer(
    kind=kind, pretrained=False,
    resume=None,
    dataset=dataset.lower(), alpha=False
)
standard = general.arch

# gold = wfmodels.WFTransformer(
#     kind=wfmodels.vit_tiny_16224, pretrained=True,
#     resume=gold_ckp,
#     dataset=dataset.lower(), alpha=False
# )
# gold = gold.arch


if 'cifar10' in dataset:
    size = 32
    T = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010)),
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


    unlearning_ids = np.where(np.isin(np.array(
        _val.targets
        ), unl_class))[0]
    unlearning_set = Subset(_val, unlearning_ids)
    unlearning_set.targets = torch.Tensor(_val.targets).long()[unlearning_ids].tolist()

    retaining_ids = np.where(~np.isin(np.array(
        _val.targets
        ), unl_class))[0]
    retaining_set = Subset(_val, retaining_ids)
    retaining_set.targets = torch.Tensor(_val.targets).long()[retaining_ids].tolist()

    
    # Define the data loader
    unlearning_loader = DataLoader(unlearning_set, batch_size=256, shuffle=True)
    unlearning_loader_iter = iter(unlearning_loader)
    c_number = 10

elif 'cifar20' in dataset.lower():

        class_mapping_dict = \
                {0: 4,
                1: 1,
                2: 14,
                3: 8,
                4: 0,
                5: 6,
                6: 7,
                7: 7,
                8: 18,
                9: 3,
                10: 3,
                11: 14,
                12: 9,
                13: 18,
                14: 7,
                15: 11,
                16: 3,
                17: 9,
                18: 7,
                19: 11,
                20: 6,
                21: 11,
                22: 5,
                23: 10,
                24: 7,
                25: 6,
                26: 13,
                27: 15,
                28: 3,
                29: 15,
                30: 0,
                31: 11,
                32: 1,
                33: 10,
                34: 12,
                35: 14,
                36: 16,
                37: 9,
                38: 11,
                39: 5,
                40: 5,
                41: 19,
                42: 8,
                43: 8,
                44: 15,
                45: 13,
                46: 14,
                47: 17,
                48: 18,
                49: 10,
                50: 16,
                51: 4,
                52: 17,
                53: 4,
                54: 2,
                55: 0,
                56: 17,
                57: 4,
                58: 18,
                59: 17,
                60: 10,
                61: 3,
                62: 2,
                63: 12,
                64: 12,
                65: 16,
                66: 12,
                67: 1,
                68: 9,
                69: 19,
                70: 2,
                71: 10,
                72: 0,
                73: 1,
                74: 16,
                75: 12,
                76: 9,
                77: 13,
                78: 15,
                79: 13,
                80: 16,
                81: 19,
                82: 2,
                83: 4,
                84: 6,
                85: 19,
                86: 5,
                87: 5,
                88: 8,
                89: 19,
                90: 18,
                91: 1,
                92: 2,
                93: 15,
                94: 6,
                95: 0,
                96: 17,
                97: 8,
                98: 14,
                99: 13
                }

        class CIFAR20(CIFAR100):

            def __init__(self, root, c_to_del=[], train=True, transform=None, target_transform=None, download=False):
                super().__init__(root, train, transform, _cifar100_to_cifar20, download)

        def _cifar100_to_cifar20(target):
            mapping = class_mapping_dict[target]
            return mapping
        
        def _cifar20_from_cifar100(target):
            mapping = _cifar100_to_cifar20(target)
            demapping = [x for x,v in mapping if v==target[0]]

            return demapping


        means = (0.4914, 0.4822, 0.4465)
        stds = (0.2023, 0.1994, 0.2010)
        size = 32
        T = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

        # _train = CIFAR20(
        #     root='/work/dnai_explainability/unlearning/datasets/cifar20_classification/train',
        #     transform=T, download=True, train=True
        # )

        _val = CIFAR20(
            root='/work/dnai_explainability/unlearning/datasets/cifar20_classification/val',
            transform=transform_test, download=True, train=False
        )
        c_number = 20




# whole_loader = DataLoader(Subset(_val, tuple(range(len(_val)))), batch_size=256, shuffle=True)
both = ConcatDataset((_train,_val))
both.datasets[0].targets = torch.ones(len(_train))
both.datasets[1].targets = torch.zeros(len(_val))
whole_loader = DataLoader(both, batch_size=16, shuffle=True)
# whole_loader = DataLoader(Subset(_val, tuple(range(total_images_number))), batch_size=256, shuffle=True)
# total_lenght = len(whole_loader.dataset)


# Set the model to evaluation mode
model.eval()

# Define the selected layer from which to extract activations


# Extract the activations from the selected layer
ours_act, ldiff_act, std_act, gold_act = [], [], [], []

labels = []
print('forwards')

model=model.cuda()
standard = standard.cuda()
ldiff = ldiff.cuda()

from tqdm import tqdm
# with torch.no_grad():
for inputs, targets in tqdm(whole_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs.requires_grad_(True)

    outputs = model(inputs)
    ours_act.append(outputs)

    outputs = standard(inputs)
    std_act.append(outputs)
    
    outputs = ldiff(inputs)
    ldiff_act.append(outputs)

    # outputs = gold(inputs)
    # gold_act.append(outputs)

    labels.append(targets.detach().cpu().numpy())

ours_act = np.concatenate(ours_act, axis=0)
ldiff_act = np.concatenate(ldiff_act, axis=0)
std_act = np.concatenate(std_act, axis=0)
# gold_act = np.concatenate(gold_act, axis=0)
labels = np.concatenate(labels, axis=0)

# Apply t-SNE to the activations to obtain a lower-dimensional representation
tsne = TSNE(n_components=2, random_state=0)

ours_act = ours_act.reshape(total_lenght,-1)
ldiff_act = ldiff_act.reshape(total_lenght,-1)
std_act = std_act.reshape(total_lenght,-1)
# gold_act = gold_act.reshape(total_lenght,-1)

print('tsne')
ours_emb = tsne.fit_transform(ours_act)
ldiff_emb = tsne.fit_transform(ldiff_act)
std_emb = tsne.fit_transform(std_act)
# gold_emb = tsne.fit_transform(gold_act)

# Visualize the embedding with the labels

idx = labels == unl_class

plt.figure()
plt.scatter(ours_emb[~idx, 0], ours_emb[~idx, 1], c=labels[~idx], s=10)
plt.scatter(ours_emb[idx, 0], ours_emb[idx, 1], c='crimson', marker='X', s=10)
# plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig(f'tsne_ours-{unl_class}_{dataset}_{model_name}.pdf', bbox_inches='tight')

plt.figure()
plt.scatter(ldiff_emb[~idx, 0], ldiff_emb[~idx, 1], c=labels[~idx], s=10)
plt.scatter(ldiff_emb[idx, 0], ldiff_emb[idx, 1], c='crimson', marker='X', s=10)
# plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig(f'tsne_ldiff-{unl_class}_{dataset}_{model_name}.pdf', bbox_inches='tight')

plt.figure()
plt.scatter(std_emb[~idx, 0], std_emb[~idx, 1], c=labels[~idx], s=10)
plt.scatter(std_emb[idx, 0], std_emb[idx, 1], c='crimson', marker='X', s=10)
# plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig(f'tsne_std-{unl_class}_{dataset}_{model_name}.pdf', bbox_inches='tight')

# plt.figure()
# plt.scatter(gold_emb[~idx, 0], gold_emb[~idx, 1], c=labels[~idx], s=10)
# plt.scatter(gold_emb[idx, 0], gold_emb[idx, 1], c='crimson', marker='X', s=10)
# # plt.axis('off')
# plt.xticks([])
# plt.yticks([])
# plt.savefig(f'tsne_gold-{unl_class}_{dataset}.pdf', bbox_inches='tight')

