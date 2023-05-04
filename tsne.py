import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder,\
    CIFAR10, \
    CIFAR100, \
    MNIST
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from custom_archs import wfmodels_lora as wfmodels


def hook_fn(model, input, output):
    activations.append(output)

unl_class = 0


ours_ckp = f"/mnt/beegfs/work/dnai_explainability/unlearning/icml2023/alpha_matrices/checkpoints_acm/cifar10-vit_tiny_16224-1-1-0/2023-04-28/0.1-100/best_CLASS-{unl_class}.pt"
# Define the PyTorch model
model = wfmodels.WFTransformer(
                kind=wfmodels.vit_tiny_16224, pretrained=True,
                resume=ours_ckp,
                dataset='cifar10', alpha=False
)
model = model.arch
# model.transformer.layers[-1][-1].norm.register_forward_hook(hook_fn)

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

# _train = CIFAR10(
#     root='/work/dnai_explainability/unlearning/datasets/cifar10_classification/train',
#     transform=T, download=True, train=True
# )

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

whole_loader = DataLoader(Subset(_val, tuple(range(1000))), batch_size=256, shuffle=True)
total_lenght = len(whole_loader.dataset)
# whole_loder = DataLoader(Subset(_val, tuple(range(len(_val)))), batch_size=256, shuffle=True)

# Set the model to evaluation mode
model.eval()

# Define the selected layer from which to extract activations


# Extract the activations from the selected layer
activations = []

labels = []
print('forwards')
with torch.no_grad():
    for inputs, targets in whole_loader:
        outputs = model(inputs)
        activations.append(outputs)
        labels.append(targets.numpy())
activations = np.concatenate(activations, axis=0)
labels = np.concatenate(labels, axis=0)

# Apply t-SNE to the activations to obtain a lower-dimensional representation
tsne = TSNE(n_components=2, random_state=0)
activations_new = activations.reshape(total_lenght,-1)
print('tsne')
embedding = tsne.fit_transform(activations_new)

# Visualize the embedding with the labels

idx = labels == unl_class

plt.scatter(embedding[~idx, 0], embedding[~idx, 1], c=labels[~idx])
plt.scatter(embedding[idx, 0], embedding[idx, 1], c=labels[idx], marker='+')
plt.savefig('tsne.png')
plt.show()