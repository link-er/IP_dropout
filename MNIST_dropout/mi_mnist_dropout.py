import time
import torch
import torch.utils.data as dt
import torchvision.transforms as transforms
import torchvision.datasets as tdatasets
import torch.nn.functional as F
import torchmetrics.classification.accuracy as torch_acc
import numpy as np
import pickle

from dropout_netw import DropoutNetw, DropoutFCNetw
import sys
sys.path.append("../utils")
from utils_local.collect_repr_callback import CollectRepresentationDistribution

# PyTorch Lightning
import pytorch_lightning as pl

pl.seed_everything(hash("setting random seeds") % 2 ** 32 - 1)

# Weights & Biases
import wandb

wandb.login(key='...')

from pytorch_lightning.loggers import WandbLogger

LR = 0.1
BS = 256
EPOCHS = 200
BETA = 3.0
DRP_METHOD = 'gaussian'
P = 0.2
dataset_size = 60000

class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            ])

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.mnist_train = tdatasets.MNIST(self.data_dir, train=True, download=True, transform=self.train_transform)
            self.mnist_val = tdatasets.MNIST(self.data_dir, train=False, download=True, transform=self.test_transform)
        if stage == 'test' or stage is None:
            self.mnist_test = tdatasets.MNIST(self.data_dir, train=False, download=True, transform=self.test_transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        mnist_train = dt.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
        return mnist_train

    def train_saving_dataloader(self):
        mnist_saving_train = dt.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=False)
        return mnist_saving_train

    def val_dataloader(self):
        mnist_val = dt.DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = dt.DataLoader(self.mnist_test, batch_size=10 * self.batch_size)
        return mnist_test


class LitDropoutNet(pl.LightningModule):
    def __init__(self, beta, p, dropout_method):
        super().__init__()
        self.dropout_method = dropout_method
        # save the estimated mutual information for IP visualization
        self.train_mi_xz = {}
        self.train_mi_zy = {}
        self.val_mi_xz = {}
        self.val_mi_zy = {}

        self.core = DropoutFCNetw(p=p, dropout_method=self.dropout_method)
        self.beta = beta

        self.hparams["network"] = self.core.saveName()
        self.hparams["dropout_method"] = self.dropout_method
        if self.dropout_method == 'information':
            self.hparams["beta"] = self.beta
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
            loss = self.beta * (1.0/dataset_size) * mi_xz + mi_zy
        else:
            loss = mi_zy
        return outputs, loss, mi_xz.item(), mi_zy.item()

    # lightning hook to add an optimizer
    def configure_optimizers(self):
        lr = LR
        self.logger.experiment.config.optimizer = 'SGD'
        self.logger.experiment.config.lr = lr
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.001)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0.001, momentum=0.9)
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
        outputs, loss, mi_xz, mi_zy = self.loss(xs.repeat(10,1), ys.repeat(10))
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


wandb_logger = WandbLogger(project="mnist_dropout")

wandb_logger.experiment.config.bs = BS
# setup data
mnist = MnistDataModule(data_dir="datasets", batch_size=BS)
mnist.prepare_data()
mnist.setup()

# setup model
# dropout method one of: standard, gaussian, information
model = LitDropoutNet(beta=BETA, p=P, dropout_method=DRP_METHOD)
# wandb_logger.watch(model, log="all") # logging all gradients, seen only for individual run

trainer = pl.Trainer(
    logger=wandb_logger,  # W&B integration
    log_every_n_steps=101,  # set the logging frequency
    gpus=1,  # use all GPUs
    max_epochs=EPOCHS,  # number of epochs
    deterministic=True,  # keep it deterministic
    callbacks=[CollectRepresentationDistribution(mnist.train_saving_dataloader(), mnist.test_dataloader(), "representations", freq=2)]
)

# fit the model
trainer.fit(model, mnist)

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
trainer.test(datamodule=mnist, ckpt_path=None)  # uses last-saved model

torch.save(model.core.state_dict(), 'models/'+model.core.saveName()+"_"+str(int(time.time())))


# wandb.finish()

