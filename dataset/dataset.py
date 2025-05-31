import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, Flowers102, CIFAR100, OxfordIIITPet
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from dataset import cats_dataset, Caltech101Split

# Values from https://github.com/facebookarchive/fb.resnet.torch/issues/180
cifar10_mean = [0.4913997551666284 , 0.48215855929893703, 0.4465309133731618 ]
cifar10_std  = [0.24703225141799082, 0.24348516474564   , 0.26158783926049628]

# Values from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std  =  [0.2675, 0.2565, 0.2761]

# ImageNet mean and std (standard)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# --------------------------------------
# CIFAR-10 dataset
# --------------------------------------

def get_cifar10_class_weights():
    dataset = CIFAR10(root='./data', train=True, download=True)
    # Extract all labels in a numpy array
    all_labels = [label for _, label in dataset]
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_cifar10_class_names():
    dataset = CIFAR10(root='./data',
                      train=True,
                      download=True)
    return dataset.classes

def _aux_cifar10_loader(transform,
                       shuffle,
                       train,
                       batch_size,
                       num_workers):

    dataset = CIFAR10(root='./data',
                      train=train,
                      download=True,
                      transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_cifar10_train_loader(batch_size=64,
                             num_workers=6,
                             resize_w=224,
                             resize_h=224,
                             augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                    # Resize dataset image to fit requested dimension
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),                                      # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_cifar10_loader(transform=train_transform,
                               shuffle=True,
                               train=True,
                               batch_size=batch_size,
                               num_workers=num_workers)

def get_cifar10_test_loader(batch_size=64,
                            num_workers=6,
                            resize_w=224,
                            resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                    # Resize dataset image to fit requested dimension
        transforms.ToTensor(),                                      # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_cifar10_loader(transform=test_transform,
                               shuffle=False,
                               train=False,
                               batch_size=batch_size,
                               num_workers=num_workers)

# --------------------------------------
# CIFAR-100 dataset
# --------------------------------------

def get_cifar100_class_weights():
    dataset = CIFAR100(root='./data', train=True, download=True)
    # Extract all labels in a numpy array
    all_labels = [label for _, label in dataset]
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_cifar100_class_names():
    dataset = CIFAR100(root='./data',
                      train=True,
                      download=True)
    return dataset.classes

def _aux_cifar100_loader(transform,
                        shuffle,
                        train,
                        batch_size,
                        num_workers):

    dataset = CIFAR100(root='./data',
                       train=train,
                       download=True,
                       transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_cifar100_train_loader(batch_size=64,
                              num_workers=6,
                              resize_w=224,
                              resize_h=224,
                              augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                     # Resize dataset image to fit requested dimension
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.RandomRotation(20) if augment else transforms.Lambda(lambda x:x),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),                                        # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_cifar100_loader(transform=train_transform,
                                shuffle=True,
                                train=True,
                                batch_size=batch_size,
                                num_workers=num_workers)

def get_cifar100_test_loader(batch_size=64,
                             num_workers=6,
                             resize_w=224,
                             resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                      # Resize dataset image to fit requested dimension
        transforms.ToTensor(),                                        # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_cifar100_loader(transform=test_transform,
                                shuffle=False,
                                train=False,
                                batch_size=batch_size,
                                num_workers=num_workers)

# --------------------------------------
# Flowers102 dataset
# --------------------------------------

def get_flowers102_class_weights():
    dataset = Flowers102(root='./data', split='train', download=True)
    # Extract all labels in a numpy array
    all_labels = [label for _, label in dataset]
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_flowers102_class_names():
    dataset = Flowers102(root='./data',
                         split="train",
                         download=True)
    return dataset.classes

def _aux_flowers102_loader(transform,
                           shuffle,
                           split,
                           batch_size,
                           num_workers):

    dataset = Flowers102(root='./data',
                         split=split,
                         download=True,
                         transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_flowers102_train_loader(batch_size=64,
                                num_workers=6,
                                resize_w=224,
                                resize_h=224,
                                augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.RandomRotation(20) if augment else transforms.Lambda(lambda x:x),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_std, std=imagenet_std),
    ])

    return _aux_flowers102_loader(train_transform,
                                  shuffle=True,
                                  split="train",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

def get_flowers102_test_loader(batch_size=64,
                               num_workers=6,
                               resize_w=224,
                               resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_flowers102_loader(test_transform,
                                  shuffle=False,
                                  split="test",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

def get_flowers102_val_loader(batch_size=64,
                               num_workers=6,
                               resize_w=224,
                               resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_flowers102_loader(test_transform,
                                  shuffle=False,
                                  split="val",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

# --------------------------------------
# CatBreeds67 dataset
# --------------------------------------

def get_CatBreeds67_class_weights():
    # dataset = CatBreeds67(root='./data/cat-breeds-dataset/images/')
    # all_labels = [label for _, label in dataset]
    # all_labels = np.array(all_labels)
    # classes = np.unique(all_labels)

    # Compute class weights
    class_weights = [7.4104e+00, 1.9068e+00, 1.2682e+01, 3.5687e-01, 9.9455e+01, 1.3997e+01,
                     7.4395e+00, 7.6288e-01, 8.3244e+00, 1.0298e+00, 3.3327e+00, 5.4931e+00,
                     2.3621e+02, 5.4488e-01, 4.7241e+02, 2.2496e+01, 7.8735e+01, 6.2988e+02,
                     1.1051e+01, 1.1116e+02, 1.5879e+01, 5.8503e-01, 5.9951e-01, 4.2001e-01,
                     3.4470e-01, 3.5635e-02, 6.1955e+00, 4.0120e+00, 1.5960e+00, 1.0270e+01,
                     1.4525e+00, 1.4879e+01, 7.5586e+01, 2.7789e+01, 1.1116e+02, 1.3326e+00,
                     9.1775e-01, 1.0440e+01, 1.2768e+01, 3.2580e+00, 1.6014e+01, 5.1071e+01,
                     3.8486e+00, 1.8709e+01, 4.7029e-01, 1.7024e+01, 1.4102e+01, 7.0800e-01,
                     1.0105e+00, 4.9727e+00, 2.4541e+01, 6.5431e-01, 9.9981e+00, 2.0103e+01,
                     8.5893e+01, 1.1629e+00, 5.3990e+01, 9.0413e+00, 6.2737e-01, 8.3761e-01,
                     7.2679e+00, 5.5643e-01, 4.7682e-01, 2.5195e+00, 2.3186e+00, 5.9404e-01,
                     1.8896e+03]
    return torch.tensor(class_weights, dtype=torch.float)

def get_CatBreeds67_class_names():
    dataset = cats_dataset.CatBreeds67(root='./data/cat-breeds-dataset/images/')
    return dataset.classes

def _aux_CatBreeds67_loader(transform,
                            shuffle,
                            batch_size,
                            num_workers):

    dataset = cats_dataset.CatBreeds67(root='./data/cat-breeds-dataset/images/',
                          transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_CatBreeds67_train_loader(batch_size=64,
                                 num_workers=6,
                                 resize_w=224,
                                 resize_h=224,
                                 augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                     # Resize dataset image to fit requested dimension
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.RandomRotation(20) if augment else transforms.Lambda(lambda x:x),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),                                        # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_CatBreeds67_loader(transform=train_transform,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   num_workers=num_workers)

def get_CatBreeds67_test_loader(batch_size=64,
                                num_workers=6,
                                resize_w=224,
                                resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),                      # Resize dataset image to fit requested dimension
        transforms.ToTensor(),                                        # Transform RGB in the range [0, 1)
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)    # Normalize channels with mean ~= 0.5 and std ~= 0.5, mapping into ~[-1, 1]
    ])

    return _aux_CatBreeds67_loader(transform=test_transform,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

# --------------------------------------
# Oxford-IIIT Pet
# --------------------------------------

def get_oxfordIIITpet_class_weights():
    dataset = OxfordIIITPet(root='./data', download=True)
    # Extract all labels in a numpy array
    all_labels = [label for _, label in dataset]
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_oxfordIIITpet_class_names():
    dataset = OxfordIIITPet(root='./data',
                            split="trainval",
                            download=True)
    return dataset.classes

def _aux_oxfordIIITpet_loader(transform,
                             shuffle,
                             split,
                             batch_size,
                             num_workers):

    dataset = OxfordIIITPet(root='./data',
                            split=split,
                            download=True,
                            transform=transform,
                            target_types='category')

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_oxfordIIITpet_train_loader(batch_size=64,
                                  num_workers=6,
                                  resize_w=224,
                                  resize_h=224,
                                  augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_std, std=imagenet_std),
    ])

    return _aux_oxfordIIITpet_loader(train_transform,
                                     shuffle=True,
                                     split="trainval",
                                     batch_size=batch_size,
                                     num_workers=num_workers)

def get_oxfordIIITpet_test_loader(batch_size=64,
                                 num_workers=6,
                                 resize_w=224,
                                 resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_oxfordIIITpet_loader(test_transform,
                                  shuffle=False,
                                  split="test",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

def get_oxfordIIITpet_val_loader(batch_size=64,
                               num_workers=6,
                               resize_w=224,
                               resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_oxfordIIITpet_loader(test_transform,
                                  shuffle=False,
                                  split="trainval",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

# --------------------------------------
# Caltech101
# --------------------------------------

def get_caltech101_class_weights():
    dataset = Caltech101Split.Caltech101(split="train")
    # Extract all labels in a numpy array
    all_labels = [label for _, label in dataset]
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

def get_caltech101_class_names():
    dataset = Caltech101Split.Caltech101(split="train")
    return dataset.classes

def _aux_caltech101_loader(transform,
                             shuffle,
                             split,
                             batch_size,
                             num_workers):

    dataset = Caltech101Split.Caltech101(split=split,transform=transform)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

def get_caltech101_train_loader(batch_size=64,
                                  num_workers=6,
                                  resize_w=224,
                                  resize_h=224,
                                  augment=True):

    train_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x:x),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_std, std=imagenet_std),
    ])

    return _aux_caltech101_loader(train_transform,
                                     shuffle=True,
                                     split="train",
                                     batch_size=batch_size,
                                     num_workers=num_workers)

def get_caltech101_test_loader(batch_size=64,
                                 num_workers=6,
                                 resize_w=224,
                                 resize_h=224):

    test_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_caltech101_loader(test_transform,
                                  shuffle=False,
                                  split="test",
                                  batch_size=batch_size,
                                  num_workers=num_workers)

def get_caltech101_val_loader(batch_size=64,
                                 num_workers=6,
                                 resize_w=224,
                                 resize_h=224):

    val_transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return _aux_caltech101_loader(val_transform,
                                  shuffle=False,
                                  split="val",
                                  batch_size=batch_size,
                                  num_workers=num_workers)


