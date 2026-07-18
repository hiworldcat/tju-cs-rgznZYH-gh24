import gzip
import os
import random
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "http://yann.lecun.com/exdb/mnist",
]


def download_mnist(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        target = root / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        last_error = None
        for base in MNIST_MIRRORS:
            url = f"{base}/{filename}"
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, target)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if target.exists():
                    target.unlink()
        if last_error is not None:
            raise RuntimeError(f"Failed to download {filename}: {last_error}")


def _read_idx_images(path, limit=None):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image file: {path}")
        count = min(count, limit or count)
        data = np.frombuffer(f.read(rows * cols * count), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path, limit=None):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label file: {path}")
        count = min(count, limit or count)
        data = np.frombuffer(f.read(count), dtype=np.uint8)
    return data.astype(np.int32)


class MnistArrayDataset:
    def __init__(self, root, split="train", limit=None):
        root = Path(root)
        prefix = "train" if split == "train" else "test"
        self.images = _read_idx_images(root / MNIST_FILES[f"{prefix}_images"], limit)
        self.labels = _read_idx_labels(root / MNIST_FILES[f"{prefix}_labels"], limit)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index], mode="L").resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.1307) / 0.3081
        return arr[None, :, :], np.int32(self.labels[index])


class ImageFolderDataset:
    def __init__(self, root, image_size=32, grayscale=True, augment=False, seed=42):
        self.root = Path(root)
        self.image_size = image_size
        self.grayscale = grayscale
        self.augment = augment
        self.rng = random.Random(seed)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        for name in self.classes:
            for path in sorted((self.root / name).glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((path, self.class_to_idx[name]))
        if not self.samples:
            raise RuntimeError(f"No image files found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        mode = "L" if self.grayscale else "RGB"
        with Image.open(path) as img:
            img = img.convert(mode)
            if self.augment and self.rng.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.grayscale:
            arr = (arr - 0.5) / 0.5
            arr = arr[None, :, :]
        else:
            arr = (arr - 0.5) / 0.5
            arr = arr.transpose(2, 0, 1)
        return arr, np.int32(label)


def find_garbage_root(data_root):
    root = Path(data_root)
    candidates = [
        root,
        root / "data_en",
        root / "data_en" / "data_en",
        root / "data" / "data_en" / "data_en",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath)
        if "train" in dirnames and "test" in dirnames:
            return path
    raise RuntimeError(f"Could not find train/test folders under {root}")
