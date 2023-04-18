import os
import requests
import torch
from torchvision.datasets import ImageFolder

from bing_image_downloader import downloader

limit = 50
root = r'/mnt/beegfs/work/dnai_explainability/unlearning/datasets'
dset_folder = f'test{limit}'
dest = 'train'
if not os.path.isdir(os.path.join(root, dset_folder, dest)):
    os.makedirs(os.path.join(root, dset_folder, dest))

classes = ('cat',)
# classes = ('cat', 'dog', 'car')

for query in classes:
    # if not os.path.isdir(os.path.join(root, dset_folder, query)):
    #     os.makedirs(os.path.join(root, dset_folder, query))
    print(f'Downloading {query}...')
    downloader.download(
        query, limit=limit, output_dir=os.path.join(root, dset_folder, dest),
        adult_filter_off=True, force_replace=False, timeout=60
    )

print(f'Dataset at {os.path.join(root, dset_folder, dest)}')