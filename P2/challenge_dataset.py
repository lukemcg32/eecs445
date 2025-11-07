"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Dogs Dataset
    Class wrapper for interfacing with the dataset of dog images
    Usage: python dataset.py
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import numpy.typing as npt
from imageio.v2 import imread
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import config, set_random_seed


__all__ = [
    "get_train_val_test_loaders",
    "get_challenge",
    "get_train_val_test_datasets",
    "resize",
    "ImageStandardizer",
    "DogsDataset",
    "get_pretrain_loaders",
]

# make sure we give each dog its own label
CLASS_NAMES = [
    "chihuahua",
    "collie",
    "dalmatian",
    "golden_retriever",
    "great_dane",
    "miniature_poodle",
    "saint_bernard",
    "samoyed",
    "siberian_husky",
    "yorkshire_terrier",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_train_val_test_loaders(
    task: str,
    batch_size: int,
    **kwargs,
) -> tuple[DataLoader, DataLoader, DataLoader, str]:
    """Return DataLoaders for train, val and test splits.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr, va, te, _ = get_train_val_test_datasets(task, **kwargs)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader, tr.get_semantic_label


class ImageStandardizer:
    """Standardize a batch of images to mean 0 and variance 1.

    The standardization should be applied separately to each channel.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """

    def __init__(self) -> None:
        """Initialize mean and standard deviations to None."""
        self.image_mean = None
        self.image_std = None

    def fit(self, X: npt.NDArray) -> None:
        """Calculate per-channel mean and standard deviation from dataset X."""
        X = np.asarray(X)

        self.image_mean = X.mean(axis=(0, 1, 2))
        self.image_std = X.std(axis=(0, 1, 2))

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """Return standardized dataset given dataset X."""

        # won't work if fit() isn't called first
        if self.image_mean is None or self.image_std is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        return (X - self.image_mean) / self.image_std


class DogsDataset(Dataset):
    """Dataset class for dog images."""

    def __init__(self, partition: str, task: str = "target", **kwargs) -> None:
        """Read in the necessary data from disk.

        For parts 2 and 3, `task` should be "target".
        For source task of part 4, `task` should be "source".
        """
        super().__init__()

        if partition not in ["train", "val", "test", "challenge"]:
            raise ValueError(f"Partition {partition} does not exist")

        set_random_seed()
        self.partition = partition
        self.task = task
        # Load in all the data we need from disk
        if task == "target" or task == "source":
            self.metadata = pd.read_csv(config("csv_file"))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(
            zip(
                self.metadata[self.metadata.task == self.task]["numeric_label"],
                self.metadata[self.metadata.task == self.task]["semantic_label"],
            )
        )

    def __len__(self) -> int:
        """Return size of dataset."""
        return len(self.X)

    # added in some augmentation for training set only
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (image, label) pair at index `idx` of dataset.

        Augmentation is applied only when self.partition == "train".
        Note that by the time we actually use this for training, X
        has already been standardized and transposed to (N, C, H, W).
        """
        x = self.X[idx]  
        y = self.y[idx]

        # Work on a copy so we don't modify the stored data
        x = np.array(x, copy=True)

        if self.partition == "train":
            # 90 degree rotation a quarter of the inputs
            if np.random.rand() < 0.25:
                x = np.rot90(x, k=1, axes=(1, 2))   # still (C, H, W)

            # 20% brightness jitter 85% of the time
            if np.random.rand() < 0.85:
                brightness = 1.0 + np.random.uniform(-0.2, 0.2)
                x = x * brightness

            # some 20% jitter on contrast 70% of the time
            if np.random.rand() < 0.7:
                contrast = 1.0 + np.random.uniform(-0.2, 0.2)
                mean_per_channel = x.mean(axis=(1, 2), keepdims=True)
                x = (x - mean_per_channel) * contrast + mean_per_channel

            # some 20% jitter on saturation 70% of the time
            if np.random.rand() < 0.6:
                saturation = 1.0 + np.random.uniform(-0.2, 0.2)
                gray = x.mean(axis=0, keepdims=True)
                x = gray + (x - gray) * saturation
        
        x = np.ascontiguousarray(x)

        return torch.from_numpy(x).float(), torch.tensor(y).long()

    def _load_data(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Load a single data partition from file."""
        print(f"loading {self.partition}...")

        df = self.metadata[
            (self.metadata.task == self.task)
            & (self.metadata.partition == self.partition)
        ]

        path = config("image_path")

        X, y = [], []
        for _, row in df.iterrows():
            image = imread(os.path.join(path, row["filename"]))
            X.append(image)
            y.append(row["numeric_label"])
        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label: int) -> str:
        """Return the string representation of the numeric class label.

        (e.g., the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]


def get_challenge(task: str, batch_size: int, **kwargs) -> tuple[DataLoader, str]:
    """Return DataLoader for challenge dataset.

    Any keyword arguments are forwarded to the DogsDataset constructor.
    """
    tr = DogsDataset("train", task, **kwargs)
    ch = DogsDataset("challenge", task, **kwargs)

    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    ch.X = standardizer.transform(ch.X)

    tr.X = tr.X.transpose(0, 3, 1, 2)
    ch.X = ch.X.transpose(0, 3, 1, 2)

    ch_loader = DataLoader(ch, batch_size=batch_size, shuffle=False)
    return ch_loader, tr.get_semantic_label


def get_train_val_test_datasets(task: str = "default", **kwargs,) -> tuple[DogsDataset, DogsDataset, DogsDataset, ImageStandardizer]:
    """Return DogsDatasets and image standardizer.

    Image standardizer should be fit to train data and applied to all splits.
    """
    tr = DogsDataset("train", task, **kwargs)
    va = DogsDataset("val", task, **kwargs)
    te = DogsDataset("test", task, **kwargs)

    # Resize
    # We don't resize images, but you may want to experiment with resizing
    # images to be smaller for the challenge portion. How might this affect
    # your training?
    # tr.X = resize(tr.X)
    # va.X = resize(va.X)
    # te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0, 3, 1, 2)
    va.X = va.X.transpose(0, 3, 1, 2)
    te.X = te.X.transpose(0, 3, 1, 2)

    return tr, va, te, standardizer


def resize(X: npt.NDArray) -> npt.NDArray:
    """Resize the data partition X to the size specified in the config file.

    Use bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config("image_dim")
    image_size = (image_dim, image_dim)
    resized = []
    for i in range(X.shape[0]):
        xi = Image.fromarray(X[i]).resize(image_size, resample=2)
        resized.append(xi)
    resized = [np.asarray(im) for im in resized]

    return resized


class NumpyDogsDataset(Dataset):
    """
    Dataset over numpy arrays (already standardized + transposed to (N, C, H, W)),
    with the same augmentation recipe as DogsDataset, controlled by is_train.
    """

    def __init__(self, X: npt.NDArray, y: npt.NDArray, is_train: bool = False):
        super().__init__()

        self.X = X
        self.y = y
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        x = np.array(x, copy=True)

        if self.is_train:
            # 90 degree rotation a quarter of the inputs
            if np.random.rand() < 0.25:
                x = np.rot90(x, k=1, axes=(1, 2))

            # 20% brightness jitter 85% of the time
            if np.random.rand() < 0.85:
                brightness = 1.0 + np.random.uniform(-0.2, 0.2)
                x = x * brightness

            # some 20% jitter on contrast 70% of the time
            if np.random.rand() < 0.7:
                contrast = 1.0 + np.random.uniform(-0.2, 0.2)
                mean_per_channel = x.mean(axis=(1, 2), keepdims=True)
                x = (x - mean_per_channel) * contrast + mean_per_channel

            # some 20% jitter on saturation 70% of the time
            if np.random.rand() < 0.6:
                saturation = 1.0 + np.random.uniform(-0.2, 0.2)
                gray = x.mean(axis=0, keepdims=True)
                x = gray + (x - gray) * saturation

        x = np.ascontiguousarray(x)
        return torch.from_numpy(x).float(), torch.tensor(y).long()


def get_pretrain_loaders(
    batch_size: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 445,
):
    """
    Build DataLoaders for 10-class pretraining on *all* source + target data,
    ignoring the original (train/val/test) partition.

    - Uses dogs.csv and config("image_path")
    - Renumbers labels so source + target share a consistent 10-class space
      (source: 0..7, target: 8..9)
    - Applies ImageStandardizer just like get_train_val_test_datasets
    """

    set_random_seed()

    csv_path = config("csv_file")
    img_root = config("image_path")

    meta = pd.read_csv(csv_path)

    # all non-challenge examples from source + target
    meta = meta[
        meta["task"].isin(["source", "target"])
        & (meta["partition"] != "challenge")
    ].copy()

    # build combined 10-class labels
    # source we keep numeric_label
    # target we shift numeric_label by +8
    labels = []
    images = []

    for _, row in meta.iterrows():
        task = row["task"]
        old_y = int(row["numeric_label"])

        if task == "source":
            new_y = old_y
        else:
            new_y = old_y + 8   # 0->8, 1->9

        labels.append(new_y)

        img_path = os.path.join(img_root, row["filename"])
        img = imread(img_path)
        images.append(img)

    X = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    # maybe look into resizing

    # 4. Train/val/test split using our splits, not csv's partition
    indices = np.arange(len(y))

    # split off val + test from train
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y,
    )

    # split temp into val and test
    val_frac_of_temp = val_size / (test_size + val_size)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=1.0 - val_frac_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Standardize using ImageStandardizer
    standardizer = ImageStandardizer()
    standardizer.fit(X_train)
    X_train = standardizer.transform(X_train)
    X_val   = standardizer.transform(X_val)
    X_test  = standardizer.transform(X_test)

    # Convert to (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val   = X_val.transpose(0, 3, 1, 2)
    X_test  = X_test.transpose(0, 3, 1, 2)

    
    tr_ds = NumpyDogsDataset(X_train, y_train, is_train=True)
    va_ds = NumpyDogsDataset(X_val,   y_val,   is_train=False)
    te_ds = NumpyDogsDataset(X_test,  y_test,  is_train=False)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader


# for final test on epoch 37
def get_full_target_loader(batch_size: int) -> DataLoader:
    """
    Build a DataLoader that uses *all* target data:
    train + val + test (non-challenge), with standardization and augmentation.
    Use this only for the final challenge training stage, after you've
    tuned hyperparameters using the normal train/val split.
    """

    tr, va, te, standardizer = get_train_val_test_datasets(task="target")

    X_all = np.concatenate([tr.X, va.X, te.X], axis=0)
    y_all = np.concatenate([tr.y, va.y, te.y], axis=0)

    # wrap in NumpyDogsDataset with augmentation turned on
    full_ds = NumpyDogsDataset(X_all, y_all, is_train=True)

    full_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    return full_loader


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_test_datasets(task="target")
    print(f"Train:\t{len(tr.X)}")
    print(f"Val:\t{len(va.X)}")
    print(f"Test:\t{len(te.X)}")
    print(f"Mean:\t{standardizer.image_mean}")
    print(f"Std:\t{standardizer.image_std}")
