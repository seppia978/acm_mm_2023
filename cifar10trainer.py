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

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Subset
# from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from datetime import datetime
import wandb

seed_everything(7)
final_acc = list()
test_acc = False

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', "--model", type=str, help='Model', default='resnet18')
parser.add_argument('-l', "--learning-rate", type=float, help='LR', default=0.1)
parser.add_argument('-n', "--name", type=str, help='WanDB Name', default='run')

args = parser.parse_args()


model_name = args.model
learning_rate = args.learning_rate
wdb_proj = 'cifar10_full'

hyp = {
    'model_name': model_name,
    'learning_rate': learning_rate
}

wdb_name = f'{args.name}_{model_name}_{learning_rate}'


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
        )

for uc in range(10):
        print(f'################################################## Making {model_name} full model')# for class {uc}')

        class_to_delete = -1 #uc
        # model_name = mn
        root='/work/dnai_explainability/unlearning/datasets/cifar10_classification'
        PATH_DATASETS = f"{root}/data"
        BATCH_SIZE = 256 # 256 if torch.cuda.is_available() else 64
        NUM_WORKERS = 4 # int(os.cpu_count() / 2)


        class Cifar10DataModule(VisionDataModule):


            def __init__(
                    self, batch_size=64, dir_data=None, num_workers=1,to_del:list = []
            ):
                super().__init__()
                self.batch_size,self.root,self.num_workers = batch_size,dir_data,num_workers
                self.to_del = to_del

                self.transform = train_transforms

            def prepare_data(self):
                # download the MNIST dataset
                CIFAR10(root=self.root, download=True)

            def setup(self, stage=None):
                cifar_train_full = CIFAR10(root=self.root, train=True, transform=self.transform)
                cifar_test_full = CIFAR10(root=self.root, train=False, transform=self.transform)

                _train, _val = random_split(cifar_train_full, [45833, 4167])
                _train.targets = np.array(cifar_train_full.targets)[_train.indices]
                _val.targets = np.array(cifar_train_full.targets)[_val.indices]

                # remove digit "9" from the training and validation sets
                id_train = np.where(~np.isin(np.array(_train.targets), self.to_del))[0]
                id_val = np.where(~np.isin(np.array(_val.targets), self.to_del))[0]
                id_test = np.where(~np.isin(np.array(cifar_test_full.targets), self.to_del))[0]

                self.cifar10_train = Subset(_train, id_train)
                self.cifar10_train.targets = torch.Tensor(_train.targets).int()[id_train].tolist()

                self.cifar10_val = Subset(_val, id_val)
                self.cifar10_val.targets = torch.Tensor(_val.targets).int()[id_val].tolist()

                self.dataset_test = Subset(cifar_test_full, id_test)
                self.dataset_test.targets = torch.Tensor(cifar_test_full.targets).int()[id_test].tolist()

            def train_dataloader(self):
                return DataLoader(self.cifar10_train, batch_size=self.batch_size, drop_last=True)

            def val_dataloader(self):
                return DataLoader(self.cifar10_val, batch_size=self.batch_size, drop_last=True)

            def test_dataloader(self):
                return DataLoader(self.dataset_test, batch_size=self.batch_size, drop_last=True)


        # cifar10_dm = CIFAR10DataModule(
        #     data_dir=PATH_DATASETS,
        #     batch_size=BATCH_SIZE,
        #     num_workers=NUM_WORKERS,
        #     train_transforms=train_transforms,
        #     test_transforms=test_transforms,
        #     val_transforms=test_transforms,
        # )

        class_to_delete = class_to_delete if isinstance(class_to_delete, list) else [class_to_delete]
        class_to_delete = [] if class_to_delete[0] == -1 else class_to_delete
        cifar10_dm = Cifar10DataModule(
            batch_size=BATCH_SIZE,
            dir_data=PATH_DATASETS,
            num_workers=NUM_WORKERS,
            to_del=class_to_delete
        )

        def create_model():
            if model_name=='resnet18':
                model = torchvision.models.resnet18(pretrained=False, num_classes=10)
            elif model_name=='resnet34':
                model = torchvision.models.resnet34(pretrained=False, num_classes=10)
            elif model_name=='vgg16':
                model = torchvision.models.vgg16(pretrained=False, num_classes=10)
            elif model_name=='deit_small_16224':
                vittype='deit_small_patch16_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=32,num_classes=10
                )
            elif model_name=='vit_small_16224':
                vittype='vit_small_patch16_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=32,num_classes=10
                )
            elif model_name=='swin_small_16224':
                vittype='swin_small_patch4_window7_224'
                model = timm.create_model(
                    vittype,
                    pretrained=False,
                    img_size=32,num_classes=10,window_size=7
                )
            # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
            # model.maxpool = nn.Identity()
            return model


        class LitResnet(LightningModule):
            def __init__(self, lr=0.01):
                super().__init__()

                self.save_hyperparameters()
                self.model = create_model()

                # acab = torch.load(
                # os.path.join(
                #     '/work/dnai_explainability/unlearning/datasets/cifar10_classification/checkpoints',
                #     '2022-12-29_3.pt'
                # )
                # )
                # # acab=acab['state_dict']
                # ac=list(map(lambda x: x[6:], acab.keys()))
                # ckp = dict()
                # for k1,k2 in zip(acab,ac):
                #     if k2 == k1[6:]:
                #         ckp[k2] = acab[k1]
                # self.model.load_state_dict(ckp)

            def forward(self, x):
                out = self.model(x)
                return F.log_softmax(out, dim=1)

            def training_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                loss = F.nll_loss(logits, y)
                self.log("train_loss", loss)
                return loss

            def evaluate(self, batch, stage=None):
                global final_acc, test_acc

                x, y = batch
                logits = self(x)
                loss = F.nll_loss(logits, y)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy(preds, y,task='multiclass', num_classes=10)
                print(acc)

                if test_acc:
                    final_acc.append(acc.unsqueeze(0))

                self.log(f"{stage}_loss", loss, on_epoch=True)
                self.log(f"{stage}_acc", acc, on_epoch=True)

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
                steps_per_epoch = 45000000000 // BATCH_SIZE
                scheduler_dict = {
                    "scheduler": OneCycleLR(
                        optimizer,
                        0.1,
                        epochs=self.trainer.max_epochs,
                        steps_per_epoch=steps_per_epoch,
                    ),
                    "interval": "step",
                }
                return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


        model = LitResnet(lr=learning_rate)

        from pytorch_lightning.callbacks import EarlyStopping

        trainer = Trainer(
            max_epochs=10000,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            logger= [
                CSVLogger(save_dir="logs/"),
                WandbLogger(
                    settings=wandb.Settings(start_method="fork"), reinit=True, config=hyp, 
                    project=f"{wdb_proj}", name=f'{wdb_name}', entity="unl4xai"
                    )
            ],
            callbacks=[
                LearningRateMonitor(logging_interval="step"), 
                TQDMProgressBar(refresh_rate=10),
                EarlyStopping(monitor='val_acc', patience=5, min_delta=.001, mode='max')
            ],
        )


        n=len(os.listdir(f'{root}/checkpoints'))
        PATH=f'{root}/checkpoints/{datetime.today().strftime("%Y-%m-%d")}_model-{model_name}'

        if len(class_to_delete) > 0:
            PATH+=f'_GOLDEN-{class_to_delete[0]}_{n}.pt'
        else:
            PATH += f'_FULL-TRAINING_{n}.pt'

        print(PATH)

        import yaml
        

        trainer.fit(model, cifar10_dm)
        test_acc = True
        trainer.test(model, datamodule=cifar10_dm)
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
            ymlpath = f'golden_checkpoints/{model_name}_CIFAR10_golden-{str(uc)}.txt'
            print(f'Saved at {PATH}')
            with open(ymlpath, 'w') as file:
                yaml.dump(data, file)
            print(f'Saved at {ymlpath}')
        except Exception as e:
            print(e)
        
        break