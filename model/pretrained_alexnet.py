import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
import time
from torchvision.models import alexnet, AlexNet_Weights
import dataset.dataset as dataset

class PretrainedAlexNet(pl.LightningModule):
    def __init__(self, class_weights, lr=0.01, num_classes=1000):
        super().__init__()
        # Save network parameters for logging and checkpointing
        self.save_hyperparameters()
        # Set the criterion to Cross Entropy Loss, as AlexNet is a classifier
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.lr = lr

        # Needed for logging
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.name = "Pre-Trained AlexNet"

        # Network model
        self.net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        # Remove last layer
        self.net.classifier[-1] = nn.Identity()
        # Add a fine tuning layer as last layer
        self.fine_tuning_layer = nn.Linear(in_features=4096,
                                           out_features=num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.fine_tuning_layer(x)
        return x

    def configure_optimizers(self):
        # Configure the stochastic gradient descent with the original AlexNet
        # parameters: 0.9 momentum and 0.0005 weight decay
        optimizer = torch.optim.SGD([
                                    {'params': self.net.parameters(),                'lr': self.lr * 0.1},  #Lower lr for pretrained parameters
                                    {'params': self.fine_tuning_layer.parameters(),  'lr': self.lr      }   #Higher lr for new parameters
                                    ],
                                    momentum=0.9,
                                    weight_decay=0.0005)

        return optimizer

    # Logging validation step
    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage=f"test")

    # Logging training step
    def training_step(self, batch, batch_idx):
        return self.step(batch, stage=f"train")

    # Aux function
    def step(self, batch, stage):
        X, y = batch
        assert torch.isfinite(X).all(), "X contains NaNs or infs"
        assert torch.isfinite(y).all(), "y contains NaNs or infs"

        logits = self(X)
        assert torch.isfinite(logits).all(), "Logits contain NaNs or infs"

        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=-1)
        acc = self.accuracy(preds, y)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        duration = time.time() - self.epoch_start_time
        self.log('epoch_duration',
                 duration,
                 on_step=False,
                 on_epoch=True)

def get_cifar10_model():
    cw = dataset.get_cifar10_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                             lr=0.01,
                             num_classes=10)

def get_flowers102_model():
    cw = dataset.get_flowers102_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                             lr=0.01,
                             num_classes=102)

def get_cifar100_model():
    cw = dataset.get_cifar100_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                             lr=0.01,
                             num_classes=100)

def get_CatBreeds67_model():
    cw = dataset.get_CatBreeds67_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                             lr=0.01,
                             num_classes=67)

def get_oxfordIIITpet_model():
    cw = dataset.get_oxfordIIITpet_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                             lr=0.01,
                             num_classes=37)

def get_caltech101_model():
    cw = dataset.get_caltech101_class_weights()
    return PretrainedAlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=102)