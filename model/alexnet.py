import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
import time
from dataset import dataset
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AlexNet(pl.LightningModule):
    def __init__(self, class_weights, lr=0.01, num_classes=1000):
        super().__init__()
        # Save network parameters for logging and checkpointing
        self.save_hyperparameters()
        # Set the criterion to Cross Entropy Loss, as AlexNet is a classifier
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.lr = lr

        # Needed for logging
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.name = "AlexNet"

        # Network model
        self.net = nn.Sequential(
            # ---------------------------------------------------------
            # Start of convolutional layers
            # ---------------------------------------------------------
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=11,
                      stride=4,
                      padding=2),
            # 224x224 --> 55x55

            # inplace = True allows reusing the memory of the input, without
            # allocating extra memory for the output.
            nn.ReLU(inplace=True),

            # LocalResponseNorm to help generalization of the model
            nn.LocalResponseNorm(size=5,
                                 alpha=1e-4,
                                 beta=0.75,
                                 k=2),

            nn.MaxPool2d(kernel_size=3,
                         stride=2),
            # 55x55 --> 27x27

            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      padding=2),
            # 27x27 --> 27x27

            nn.ReLU(inplace=True),

            # LocalResponseNorm to help generalization of the model
            nn.LocalResponseNorm(size=5,
                                 alpha=1e-4,
                                 beta=0.75,
                                 k=2),

            nn.MaxPool2d(kernel_size=3,
                         stride=2),
            # 27x27 --> 13x13

            # Triple convolutional layer
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3,
                         stride=2),
            # 13x13 --> 6x6

            # ---------------------------------------------------------
            # End of convolutional layers, starting with the classifier
            # ---------------------------------------------------------
            nn.Flatten(),
            # From 256x6x6 to 9216

            nn.Linear(in_features=9216,
                      out_features=4096),
            nn.ReLU(inplace=True),
            # Dropout layers help to prevent overfitting by randomly
            # dropping half od the neurons in each dense layer during
            # training.
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096,
                      out_features=num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net.modules():
            if isinstance(layer, nn.Conv2d):
                # Initialize weights with Gaussian distribution(0, 0.1)
                init.normal_(layer.weight, mean=0.0, std=0.01)
                # Initialize only if layer has bias
                if layer.bias is not None:
                   init.constant_(layer.bias, 0.0)
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight, mean=0.0, std=0.01)
                # Initialize only if layer has bias
                if layer.bias is not None:
                   init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # Configure the stochastic gradient descent with the original AlexNet
        # parameters: 0.9 momentum and 0.0005 weight decay
        optim = torch.optim.SGD(self.parameters(),
                                lr = self.lr,
                                momentum=0.9,
                                weight_decay=0.0005)

        scheduler = ReduceLROnPlateau(
            optimizer=optim,
            mode='min',             # Because we want to minimize validation loss
            factor=0.1,             # Reduce LR by 10x
            patience=10,            # How many epochs to wait before reducing LR
            min_lr=10e-5            # Minimum lr = 10e-5
        )

        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'test_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # Logging validation step
    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage=f"test")

    # Logging training step
    def training_step(self, batch, batch_idx):
        return self._step(batch, stage=f"train")

    # Aux function
    def _step(self, batch, stage):
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
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=10)

def get_flowers102_model():
    cw = dataset.get_flowers102_class_weights()
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=102)

def get_cifar100_model():
    cw = dataset.get_cifar100_class_weights()
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=100)

def get_CatBreeds67_model():
    cw = dataset.get_CatBreeds67_class_weights()
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=67)

def get_oxfordIIITpet_model():
    cw = dataset.get_oxfordIIITpet_class_weights()
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=37)

def get_caltech101_model():
    cw = dataset.get_caltech101_class_weights()
    return AlexNet(class_weights=cw,
                   lr=0.01,
                   num_classes=102)
