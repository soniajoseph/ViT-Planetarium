import random

import torchvision
from torch.utils.data import Dataset

from vit_prisma.dataloaders.imagenet_dataset import (
    ImageNetValidationDataset,
)
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_cifar_10
from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths


class SubsetDataset(Dataset):
    """For testing purposes - reduce a Dataset to N of its samples."""

    def __init__(self, dataset, n, random_subset=False):
        self.dataset = dataset
        self.n = min(n, len(dataset))
        if random_subset:
            self.indices = random.sample(range(len(dataset)), self.n)
        else:
            self.indices = list(range(self.n))

        self.data = []
        for idx in self.indices:
            self.data.append(self.dataset[idx])

        del self.dataset

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.n


def load_dataset(cfg, model_type="clip", visualize=False):
    """Load and prepare datasets based on configuration and model type. Currently the
    function loads either ImageNet1K or CIFAR-10 datasets.
    """
    print(f"Dataset type: {cfg.dataset_name}") if cfg.verbose else None

    if cfg.dataset_name == "imagenet1k":
        if model_type == "clip":
            data_transforms = get_clip_val_transforms()
        else:
            raise ValueError("Invalid model type")

        imagenet_paths = setup_imagenet_paths(cfg.dataset_path)
        train_data = torchvision.datasets.ImageFolder(
            cfg.dataset_train_path, transform=data_transforms
        )
        val_data = ImageNetValidationDataset(
            cfg.dataset_val_path,
            imagenet_paths["label_strings"],
            imagenet_paths["val_labels"],
            data_transforms,
            return_index=True,
        )
        if visualize:
            val_data_visualize = ImageNetValidationDataset(
                cfg.dataset_val_path,
                imagenet_paths["label_strings"],
                imagenet_paths["val_labels"],
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((224, 224)),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
                return_index=True,
            )
        else:
            val_data_visualize = None

        print(f"Train data length: {len(train_data)}") if cfg.verbose else None
        print(f"Validation data length: {len(val_data)}") if cfg.verbose else None
        return train_data, val_data, val_data_visualize
    elif cfg.dataset_name == "cifar10":
        train_data, val_data, _ = load_cifar_10(cfg.dataset_path)
        return train_data, val_data, None
    else:
        # raise error
        raise ValueError("Invalid dataset name")
