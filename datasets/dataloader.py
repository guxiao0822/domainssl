import os
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from einops import rearrange

from datasets.transforms import (
    temporal_jitter,
    temporal_scaling,
    magnitude_warping,
    add_gaussian_noise,
    time_series_cutout,
    time_series_normalization,
)
from datasets.dataset import TS_Dataset

# Define default augmentations
data_augmentation = transforms.Compose([
    transforms.Lambda(lambda x: time_series_normalization(x)),
    transforms.Lambda(lambda x: temporal_jitter(x)),
    transforms.Lambda(lambda x: temporal_scaling(x)),
    transforms.Lambda(lambda x: magnitude_warping(x)),
    transforms.Lambda(lambda x: add_gaussian_noise(x, 0.2)),
    transforms.Lambda(lambda x: time_series_cutout(x)),
])

data_augmentation_wonorm = transforms.Compose([
    transforms.Lambda(lambda x: temporal_jitter(x)),
    transforms.Lambda(lambda x: temporal_scaling(x)),
    transforms.Lambda(lambda x: magnitude_warping(x)),
    transforms.Lambda(lambda x: add_gaussian_noise(x, 0.2)),
    transforms.Lambda(lambda x: time_series_cutout(x)),
])


class ContrastiveViewGenerator:
    """Generate multiple augmented views for contrastive learning."""
    def __init__(self, base_transform=data_augmentation, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class BaseViewGenerator:
    """Generate a single augmented view."""
    def __init__(self, base_transform=data_augmentation):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x)


def SimCLR_Dataloader(
    data_path: str,
    input_length: int = 1250,
    fold_num: int = 5,
    test_fold_index: int = 4,
    val_fold_index: int = 3,
    batch_size: int = 512,
):
    """
    DataLoader for SimCLR training and evaluation.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Determine folds
    val_test = {test_fold_index, val_fold_index}
    train_folds = [i for i in range(fold_num) if i not in val_test]

    # Select augmentation
    transform = data_augmentation
    view_transform = ContrastiveViewGenerator(base_transform=transform, n_views=2)

    # Datasets
    train_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=view_transform,
        folds=train_folds
    )
    val_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=view_transform,
        folds=[val_fold_index]
    )
    test_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=view_transform,
        folds=[test_fold_index]
    )


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=False)

    return train_loader, val_loader, test_loader


def Downstream_Dataloader(
    data_path: str,
    input_length: int = 1250,
    fold_num: int = 5,
    test_fold_index: int = 4,
    val_fold_index: int = 3,
    batch_size: int = 512,
    train_proportion: float = 1.0,
    feature_name: str = 'feature'
):
    """
    DataLoader for supervised downstream tasks.

    Returns:
        train_loader, val_loader, test_loader
    """
    val_test = {test_fold_index, val_fold_index}
    train_folds = [i for i in range(fold_num) if i not in val_test]

    # Augmentation for base views
    if 'imu' in data_path.lower() or 'capture' in data_path.lower():
        base_transform = data_augmentation_wonorm
    else:
        base_transform = data_augmentation

    train_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=BaseViewGenerator(base_transform=base_transform),
        folds=train_folds,
        proportion=train_proportion,
    )
    val_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=None,
        folds=[val_fold_index],
    )
    test_ds = TS_Dataset(
        data_path,
        input_length=input_length,
        transform=None,
        folds=[test_fold_index],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = SimCLR_Dataloader(data_path='../data/')
    train_loader, val_loader, test_loader = Downstream_Dataloader(data_path='../data/')
