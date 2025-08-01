import os
import pickle as pk
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

class Normalize:
    """Z-normalize each channel independently."""
    def __call__(self, time_series: np.ndarray) -> np.ndarray:
        for i in range(time_series.shape[0]):
            channel = time_series[i, :]
            mean = np.mean(channel)
            std = np.std(channel)
            time_series[i, :] = (channel - mean) / (std + 1e-8)
        return time_series


class TS_Dataset(Dataset):
    """
    Loads:
      - ecg.pkl (time series)
      - label.pkl (class labels)
      - group.pkl (fold indices)
      - feature.pkl (per-sample features)
      - optional weight.mat, kmeans.mat

    Args:
        data_path: Path with data files.
        folds: List of fold indices for this split.
        input_length: Desired signal length (# samples).
        transform: Callable for augmentations.
        proportion: Fraction of data to keep.
    """
    def __init__(
        self,
        data_path: str,
        folds: list = None,
        input_length: int = 1250,
        transform=None,
        proportion: float = 1.0,
        n_clusters: int = 128,
    ):
        self.data_path = data_path
        self.input_length = input_length
        self.transform = transform
        self.normalize = Normalize()

        # Load pickled arrays
        with open(os.path.join(data_path, 'ecg.pkl'), 'rb') as f:
            ecg_all = pk.load(f)
        with open(os.path.join(data_path, 'label.pkl'), 'rb') as f:
            labels_all = np.array(pk.load(f))
        with open(os.path.join(data_path, 'fold.pkl'), 'rb') as f:
            groups_all = np.array(pk.load(f))
        with open(os.path.join(data_path, 'feat.pkl'), 'rb') as f:
            features_all = np.array(pk.load(f))

        # Filter by selected folds first
        idx = np.arange(len(labels_all))
        if folds is not None:
            mask = np.isin(groups_all, folds)
            idx = idx[mask]

        self.ecg = ecg_all[idx]
        self.labels = labels_all[idx]
        features = features_all[idx]

        # Impute missing values in features
        imputer = SimpleImputer(strategy="median")
        features_imputed = imputer.fit_transform(features)

        # Normalize features across samples (zero mean, unit variance)
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features_imputed)

        # Run KMeans clustering on normalized features of selected folds only
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_norm)

        # Store processed features and clusters
        self.features = features_norm.astype(np.float32)
        self.kmeans = cluster_ids.astype(np.int64)

        # Downsample if needed
        if proportion < 1.0:
            step = int(1 / proportion)
            self.ecg = self.ecg[::step]
            self.labels = self.labels[::step]
            self.features = self.features[::step]
            self.kmeans = self.kmeans[::step]

        # Show label distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"Label counts for folds {folds}: {counts}")

    def __len__(self) -> int:
        return len(self.ecg)

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        # Ensure 2D [channels, length]
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
        if signal.shape[1] < self.input_length:
            pad = self.input_length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)), mode='constant')
        else:
            signal = signal[:, :self.input_length]

        if self.transform:
            signal = self.transform(signal)

        signal = np.array(signal)
        signal = self.normalize(signal)

        return signal.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        signal = self.process_signal(self.ecg[idx].copy())
        return {
            'x': torch.from_numpy(signal),
            'y': torch.tensor(self.labels[idx], dtype=torch.long),
            'f': torch.from_numpy(self.features[idx].astype(np.float32)),
            'c': torch.tensor(self.kmeans[idx], dtype=torch.long)
        }


def get_dataloader(
    data_path: str,
    folds: list = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    proportion: float = 1.0
) -> DataLoader:
    """
    Args:
        data_path: directory with preprocessed data files.
        folds: fold indices to include (e.g. [0] for fold 0).
        batch_size: number of samples per batch.
        shuffle: whether to shuffle each epoch.
        num_workers: subprocesses for data loading.
        transform: per-sample augmentation.
        proportion: fraction of dataset to keep.
    """
    dataset = TS_Dataset(
        data_path=data_path,
        folds=folds,
        input_length=1250,
        transform=transform,
        proportion=proportion
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle
    )


def plot_tsne(features: np.ndarray,
              labels: np.ndarray,
              clusters: np.ndarray) -> None:
    """
    Compute and plot t-SNE embeddings colorized by class labels and cluster ids using seaborn.

    Args:
        features: 2D array of shape (N_samples, N_features)
        labels: 1D array of class labels of length N_samples
        clusters: 1D array of cluster ids of length N_samples
    """
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(features)

    df_vis = pd.DataFrame({
        'TSNE1': emb[:, 0],
        'TSNE2': emb[:, 1],
        'Class': labels,
        'Cluster': clusters
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Class',
        palette='tab10',
        data=df_vis,
        legend='full',
        ax=axes[0],
        s=20
    )
    axes[0].set_title('t-SNE colored by Class Labels')

    sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Cluster',
        palette='tab10',
        data=df_vis,
        legend='full',
        ax=axes[1],
        s=20
    )
    axes[1].set_title('t-SNE colored by Cluster IDs')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    loader = get_dataloader(
        data_path='../data/',
        folds=[0,1,2],
        batch_size=64,
        num_workers=8,
        shuffle=True
    )
    batch = next(iter(loader))
    print({k: v.shape for k, v in batch.items()})

    # Plot t-SNE using the entire dataset
    dataset = loader.dataset
    plot_tsne(dataset.features, dataset.labels, dataset.kmeans)