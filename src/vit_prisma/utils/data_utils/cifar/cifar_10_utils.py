from typing import Tuple

import torch
from torch.utils.data import random_split, Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import RandAugment


def get_cifar_transforms(augmentation: bool, image_size: int = 128):
    cifar_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((image_size, image_size)),
    ]

    if augmentation:
        cifar_transforms.insert(
            0,
            [
                RandAugment(2, 14),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    train_transform = transforms.Compose(cifar_transforms)
    test_transform = transforms.Compose(cifar_transforms)

    return train_transform, test_transform


def load_cifar_10(
    dataset_path: str,
    split_size: float = 0.8,
    augmentation: bool = False,
    image_size: int = 128,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load CIFAR-10 from Torchvision. It will cache the data locally."""

    train_transform, test_transform = get_cifar_transforms(augmentation, image_size)

    cifar10_trainset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=True,
        transform=train_transform,
    )
    cifar10_test_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = int(len(cifar10_trainset) * split_size)
    val_size = len(cifar10_trainset) - train_size

    cifar10_train_dataset, cifar10_val_dataset = random_split(
        cifar10_trainset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"There are {len(cifar10_train_dataset)} samples in the train dataset")
    print(f"There are {len(cifar10_val_dataset)} samples in the validation dataset")
    print(f"There are {len(cifar10_test_dataset)} samples in the test dataset")

    return cifar10_train_dataset, cifar10_val_dataset, cifar10_test_dataset
