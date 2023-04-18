import os

import pandas as pd
# import seaborn as sn
import numpy as np
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import transforms

# from IPython.core.display import display
from pl_bolts.datamodules import BinaryMNISTDataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

seed_everything(7)
final_acc = list()
test_acc = False

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', "--model", type=str, help='Model', default='resnet18')

args = parser.parse_args()


model_name = args.model

for uc in range(10):
        print(f'################################################## Making {model_name} golden model for class {uc}')

        class_to_delete = uc
        root="/work/dnai_explainability/unlearning/datasets/mnist_classification"
        PATH_DATASETS = f"{root}/train"
        BATCH_SIZE = 64 # 256 if torch.cuda.is_available() else 64
        NUM_WORKERS = 4 # int(os.cpu_count() / 2)


        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        class MNISTDataModule(VisionDataModule):


            def __init__(
                    self, batch_size=64, dir_data=None, num_workers=1,to_del:list = []
            ):
                super().__init__()
                self.batch_size,self.root,self.num_workers = batch_size,dir_data,num_workers
                self.to_del = to_del

                self.transform = train_transforms

            def prepare_data(self):
                # download the MNIST dataset
                MNIST(root=self.root, download=True)

            def setup(self, stage=None):
                mnist_train_full = MNIST(root=self.root, train=True, transform=self.transform)
                mnist_test_full = MNIST(root=self.root, train=False, transform=self.transform)

                _train, _val = random_split(mnist_train_full, [55000, 5000])
                _train.targets = mnist_train_full.targets[_train.indices]
                _val.targets = mnist_train_full.targets[_val.indices]

                # remove digit "9" from the training and validation sets
                id_train = np.where(~np.isin(np.array(_train.targets), self.to_del))[0]
                id_val = np.where(~np.isin(np.array(_val.targets), self.to_del))[0]
                id_test = np.where(~np.isin(np.array(mnist_test_full.targets), self.to_del))[0]

                self.mnist_train = Subset(_train, id_train)
                self.mnist_train.targets = torch.Tensor(_train.targets).int()[id_train].tolist()

                self.mnist_val = Subset(_val, id_val)
                self.mnist_val.targets = torch.Tensor(_val.targets).int()[id_val].tolist()

                self.dataset_test = Subset(mnist_test_full, id_test)
                self.dataset_test.targets = torch.Tensor(mnist_test_full.targets).int()[id_test].tolist()

            def train_dataloader(self):
                return DataLoader(self.mnist_train, batch_size=self.batch_size, drop_last=True)

            def val_dataloader(self):
                return DataLoader(self.mnist_val, batch_size=self.batch_size, drop_last=True)

            def test_dataloader(self):
                return DataLoader(self.dataset_test, batch_size=self.batch_size, drop_last=True)




        # mnist_dm = BinaryMNISTDataModule(
        #     data_dir=PATH_DATASETS,
        #     batch_size=BATCH_SIZE,
        #     num_workers=NUM_WORKERS,
        #     train_transforms=train_transforms,
        #     test_transforms=test_transforms,
        #     val_transforms=test_transforms,
        # )
        class_to_delete = class_to_delete if isinstance(class_to_delete, list) else [class_to_delete]
        class_to_delete = [] if class_to_delete[0] == -1 else class_to_delete
        mnist_dm = MNISTDataModule(
            batch_size=BATCH_SIZE,
            dir_data=PATH_DATASETS,
            num_workers=NUM_WORKERS,
            to_del=class_to_delete
        )


        def create_model():
            if model_name=='resnet18':
                model = torchvision.models.resnet18(pretrained=False, num_classes=10)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif model_name=='resnet34':
                model = torchvision.models.resnet34(pretrained=False, num_classes=10)
                model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif model_name=='vgg16':
                model = torchvision.models.vgg16(pretrained=False, num_classes=10)
                model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif model_name=='deit_small_16224':
                vittype='deit_small_patch16_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=28,num_classes=10
                )
                model.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0))
            elif model_name=='vit_small_16224':
                vittype='vit_small_patch16_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=28,num_classes=10
                )
                model.patch_embed.proj = nn.Conv2d(1, 384, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0))
            elif model_name=='swin_small_16224':
                vittype='swin_small_patch4_window7_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=28,num_classes=10,window_size=4
                )
                #model.patch_embed.proj.in_channels=1
            
            # model.maxpool = nn.Identity()
            return model

        class LitResnet(LightningModule):
            def __init__(self, lr=0.05):
                super().__init__()

                self.save_hyperparameters()
                self.model = create_model()

            def forward(self, x):
                out = self.model(x)
                return F.log_softmax(out, dim=1)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                self.log("train_loss", loss)
                return loss

            def evaluate(self, batch, stage=None):
                global final_acc, test_acc

                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy(preds, y, task="multiclass", num_classes=10)

                if test_acc:
                    final_acc.append(acc.unsqueeze(0))

                if stage:
                    self.log(f"{stage}_loss", loss, prog_bar=True)
                    self.log(f"{stage}_acc", acc, prog_bar=True)

            def validation_step(self, batch, batch_idx):
                self.evaluate(batch, "val")

            def test_step(self, batch, batch_idx):
                self.evaluate(batch, "test")

            def configure_optimizers(self):
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams.lr,
                    momentum=0.9,
                    weight_decay=5e-4,
                )
                steps_per_epoch = 450000000 // BATCH_SIZE
                scheduler_dict = {
                    "scheduler": OneCycleLR(
                        optimizer,
                        0.1,
                        epochs=self.trainer.max_epochs,
                        steps_per_epoch=steps_per_epoch,
                        total_steps=1000000000000000
                    ),
                    "interval": "step",
                }
                return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        model = LitResnet(lr=0.05)

        trainer = Trainer(
            max_epochs=15,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            logger=CSVLogger(save_dir="logs/"),
            callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
            num_sanity_val_steps=0
        )


        n=len(os.listdir(f'{root}/checkpoints'))
        PATH=f'{root}/checkpoints/{datetime.today().strftime("%Y-%m-%d")}_model-{model_name}'

        if len(class_to_delete) > 0:
            PATH+=f'_GOLDEN-{class_to_delete[0]}_{n}.pt'
        else:
            PATH += f'_FULL-TRAINING_{n}.pt'
        
        import yaml
        

        # print(f'Train length: {len(mnist_dm.mnist_train)}, val length: {len(mnist_dm.mnist_val)}')
        trainer.fit(model, mnist_dm)
        test_acc = True 
        trainer.test(model, datamodule=mnist_dm)
        test_acc = False

        final_acc_tensor = str(float(torch.cat(final_acc,0).mean().cpu().detach().numpy()))

        data = {
            'ckp': PATH,
            'model_name': model_name,
            'unl_class': str(uc),
            'test_acc': final_acc_tensor
        }


        print(data)
        final_acc = []
        try:
            torch.save(model.state_dict(), PATH)
            ymlpath = f'golden_checkpoints/{model_name}_MNIST_golden-{str(uc)}.txt'
            # print(f'Saved at {PATH}')
            with open(ymlpath, 'w') as file:
                yaml.dump(data, file)
            print(f'Saved at {ymlpath}')
        except Exception as e:
            print(e)