import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
import time
import dataset.dataset as dataset
import torch.nn.init as init

# Depthwise Separate Convolution Block
# References: https://www.youtube.com/watch?v=vVaRhZXovbw
# From XCeption network (2018)
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        # Depthwise block: split in multiple blocks
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels,
                                   bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)

        # Normalize batch to mean 0 and std 1, usually used before using ReLU
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        # ReLU activation function
        self.act = nn.ReLU(inplace=True)
        # ResNet (2018), use residual if dimension are the same
        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        # Apply batch normalization before ReLU
        x = self.bn(x)
        # Apply ReLU
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x

class AlexNetV2(pl.LightningModule):
    def __init__(self, class_weights, lr=0.001, num_classes=1000):
        super().__init__()
        self.save_hyperparameters()
        # Set the criterion to Cross Entropy Loss, as AlexNet is a classifier
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.lr = lr

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.name = "AlexNetV2"

        # Network model
        self.net = nn.Sequential(
            # Here we write the layers that will be executed in the order
            # they are defined.

            # 2013, Matthew Zeiler. Reduce kernel size of filters and stride
            # in early layers.

            # Replace Conv2d with Depthwise and Pointwise convolutions
            # Use lower kernel size and lower stride
            DepthwiseSeparableBlock(in_channels=3,
                                    out_channels=96,
                                    kernel_size=5,
                                    stride=2,
                                    padding=2),
            # 224x224 --> 112x112

            nn.MaxPool2d(kernel_size=3, stride=2),
            # 112x112 --> 55x55

            DepthwiseSeparableBlock(in_channels=96,
                                    out_channels=256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0),
            # 55x55 --> 27x27

            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27 --> 13x13

            DepthwiseSeparableBlock(in_channels=256,
                                    out_channels=384,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
            DepthwiseSeparableBlock(in_channels=384,
                                    out_channels=384,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
            DepthwiseSeparableBlock(in_channels=384,
                                    out_channels=256,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),

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
            # Added batch normalization
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=4096,
                      out_features=num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # Kaiming normal init for ReLU activations
                # Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # Changed optimizer from SGD to Adam
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.lr)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage=f"test")

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage=f"train")

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
    return AlexNetV2(class_weights=cw,
                   lr=0.001,
                   num_classes=10)

def get_flowers102_model():
    cw = dataset.get_flowers102_class_weights()
    return AlexNetV2(class_weights=cw,
                   lr=0.001,
                   num_classes=102)

def get_cifar100_model():
    cw = dataset.get_cifar100_class_weights()
    return AlexNetV2(class_weights=cw,
                   lr=0.001,
                   num_classes=100)

def get_CatBreeds67_model():
    cw = dataset.get_CatBreeds67_class_weights()
    return AlexNetV2(class_weights=cw,
                     lr=0.001,
                     num_classes=67)

def get_oxfordIIITpet_model():
    cw = dataset.get_oxfordIIITpet_class_weights()
    return AlexNetV2(class_weights=cw,
                       lr=0.01,
                       num_classes=37)

def get_caltech101_model():
    cw = dataset.get_caltech101_class_weights()
    return AlexNetV2(class_weights=cw,
                   lr=0.01,
                   num_classes=102)
