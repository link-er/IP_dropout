import time
import torch
import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc
import numpy as np
import os
import pickle

from dropout_netw import create_ResNet_model
import sys
sys.path.append("../utils")
from utils.collect_repr_callback import CollectRepresentationDistribution

# PyTorch Lightning
import pytorch_lightning as pl

pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)

# Weights & Biases
import wandb

wandb.login(key='...')

from pytorch_lightning.loggers import WandbLogger

LR = 0.05
BS = 64
EPOCHS = 200
#FILTER_PERC = 1.0
#BETA = 10.0
DRP_METHOD = 'gaussian'
P = 0.4


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.cifar10_train = tdatasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform)
            self.cifar10_val = tdatasets.CIFAR10(self.data_dir, train=False, transform=self.test_transform)
        if stage == 'test' or stage is None:
            self.cifar10_test = tdatasets.CIFAR10(self.data_dir, train=False, transform=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        cifar10_train = dt.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)
        return cifar10_train

    def train_saving_dataloader(self):
        mnist_saving_train = dt.DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=False)
        return mnist_saving_train

    def val_dataloader(self):
        cifar10_val = dt.DataLoader(self.cifar10_val, batch_size=self.batch_size)
        return cifar10_val

    def test_dataloader(self):
        cifar10_test = dt.DataLoader(self.cifar10_test, batch_size=10 * self.batch_size)
        return cifar10_test


class LitDropoutNet(pl.LightningModule):
    def __init__(self, dropout_method, beta=10):
        super().__init__()
        self.dropout_method = dropout_method
        # save the estimated mutual information for IP visualization
        self.train_mi_xz = {}
        self.train_mi_zy = {}
        self.val_mi_xz = {}
        self.val_mi_zy = {}

        # 3 channels in CIFAR10 images
        #self.core = DropoutNetw(dropout_method=self.dropout_method)
        #self.core = DropoutConvNetw(inputs=3, filter_perc=FILTER_PERC, dropout_method=self.dropout_method)
        self.core = create_ResNet_model(P)
        #self.beta = beta

        self.hparams["network"] = 'ResNet'
        self.hparams["dropout_method"] = self.dropout_method
        self.hparams["p"] = P
        #if self.dropout_method == 'information':
        #    self.hparams["beta"] = self.beta
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torch_acc.Accuracy()
        self.valid_acc = torch_acc.Accuracy()
        self.test_acc = torch_acc.Accuracy()

    def loss(self, xs, ys):
        outputs = self.core(xs)
        ys = ys.type(torch.LongTensor).cuda()
        # have to use functional loss in here!
        mi_xz = torch.Tensor([0])
        mi_zy = F.cross_entropy(outputs, ys)
        if self.dropout_method == 'information':
            mi_xz = self.core.kl()
            # multiply mi_xz by 1/N, where N is train data size: some compatibility with Kingma (VAE)
            loss = self.beta * (1.0/50000) * mi_xz + mi_zy
        else:
            loss = mi_zy
        return outputs, loss, mi_xz.item(), mi_zy.item()

    # lightning hook to add an optimizer
    def configure_optimizers(self):
        lr = LR
        self.logger.experiment.config.optimizer = 'SGD'
        self.logger.experiment.config.lr = lr
        self.logger.experiment.config.scheduler = 'one_cycle_lr'
        optimizer = torch.optim.SGD(
            self.core.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    # lightning hook to make a training step
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss, mi_xz, mi_zy = self.loss(xs, ys)
        if self.train_mi_xz.get(self.current_epoch) is None:
            self.train_mi_xz[self.current_epoch] = [mi_xz]
        else:
            self.train_mi_xz[self.current_epoch].append(mi_xz)
        if self.train_mi_zy.get(self.current_epoch) is None:
            self.train_mi_zy[self.current_epoch] = [mi_zy]
        else:
            self.train_mi_zy[self.current_epoch].append(mi_zy)
        preds = torch.argmax(outputs, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    # lightning hook to make a testing run
    def test_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss, _, _ = self.loss(xs, ys)
        preds = torch.argmax(outputs, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    # lightning hook to make a validation run
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        outputs, loss, _, _ = self.loss(xs, ys)
        preds = torch.argmax(outputs, 1)

        self.valid_acc(preds, ys)
        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)

        # get validation IP numbers
        self.core.train()
        # want to get random samples of noise applied
        outputs, loss, mi_xz, mi_zy = self.loss(xs, ys) # self.loss(xs.repeat(10,1,1,1), ys.repeat(10))
        if self.val_mi_xz.get(self.current_epoch) is None:
            self.val_mi_xz[self.current_epoch] = [mi_xz]
        else:
            self.val_mi_xz[self.current_epoch].append(mi_xz)
        if self.val_mi_zy.get(self.current_epoch) is None:
            self.val_mi_zy[self.current_epoch] = [mi_zy]
        else:
            self.val_mi_zy[self.current_epoch].append(mi_zy)
        self.core.eval()

        return outputs


wandb_logger = WandbLogger(project="cifar10_resnet")

wandb_logger.experiment.config.bs = BS
# setup data
cifar10 = CIFAR10DataModule(data_dir="datasets", batch_size=BS)
cifar10.prepare_data()
cifar10.setup()

# setup model
# dropout method one of: standard, gaussian, information
model = LitDropoutNet(dropout_method=DRP_METHOD)
# wandb_logger.watch(model, log="all") # logging all gradients, seen only for individual run

trainer = pl.Trainer(
    logger=wandb_logger,  # W&B integration
    log_every_n_steps=101,  # set the logging frequency
    gpus=1,  # use all GPUs
    max_epochs=EPOCHS,  # number of epochs
    deterministic=True,  # keep it deterministic
    callbacks=[CollectRepresentationDistribution(cifar10.train_saving_dataloader(), cifar10.test_dataloader(), "representations")]
)

# fit the model
trainer.fit(model, cifar10)

train_mi_xz = {}
train_mi_zy = {}
val_mi_xz = {}
val_mi_zy = {}
for i in range(EPOCHS):
    train_mi_xz[i] = np.array(model.train_mi_xz[i]).mean()
    train_mi_zy[i] = np.array(model.train_mi_zy[i]).mean()
    val_mi_xz[i] = np.array(model.val_mi_xz[i]).mean()
    val_mi_zy[i] = np.array(model.val_mi_zy[i]).mean()
pickle.dump(train_mi_xz, open('IP/train_mi_xz', "wb"))
pickle.dump(train_mi_zy, open('IP/train_mi_zy', "wb"))
pickle.dump(val_mi_xz, open('IP/val_mi_xz', "wb"))
pickle.dump(val_mi_zy, open('IP/val_mi_zy', "wb"))

# evaluate the model on a test set
trainer.test(datamodule=cifar10, ckpt_path=None)  # uses last-saved model

torch.save(model.core.state_dict(), 'models/'+model.core.saveName()+"_"+str(int(time.time())))


# wandb.finish()

