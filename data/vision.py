from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.base import DatasetProvider
from data.partition import dirichlet_partition, iid_partition


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, -1


@dataclass
class VisionSplits:
    client_loaders: List[DataLoader]
    proxy_loader: DataLoader
    test_loader: DataLoader
    client_metadata: List[Dict[str, Any]]


class TorchVisionClassificationProvider(DatasetProvider):
    dataset_cls: Type
    dataset_name: str = "vision"
    default_channels: int = 1
    default_image_size: int = 28

    def __init__(self, cfg):
        self.cfg = cfg

    def transform(self):
        return transforms.Compose([transforms.ToTensor()])

    def build(self):
        tf = self.transform()
        train = self.dataset_cls(self.cfg.data_dir, train=True, download=self.cfg.download_data, transform=tf)
        test = self.dataset_cls(self.cfg.data_dir, train=False, download=self.cfg.download_data, transform=tf)
        rng = np.random.default_rng(self.cfg.seed)
        all_idx = np.arange(len(train))
        rng.shuffle(all_idx)
        if self.cfg.quick_data_limit:
            all_idx = all_idx[: self.cfg.quick_data_limit]
        proxy_idx = all_idx[: self.cfg.proxy_size].tolist()
        client_idx = all_idx[self.cfg.proxy_size :].tolist()
        labels = np.array(train.targets)
        if self.cfg.partition_type == "iid":
            parts = iid_partition(client_idx, self.cfg.num_clients, rng)
        else:
            parts = dirichlet_partition(client_idx, labels, self.cfg.num_clients, self.cfg.dirichlet_alpha, rng)
        loaders = []
        meta = []
        for cid, idxs in enumerate(parts):
            loaders.append(DataLoader(Subset(train, idxs), batch_size=self.cfg.batch_size, shuffle=True))
            counts = {int(k): int(v) for k, v in zip(*np.unique(labels[idxs], return_counts=True))} if idxs else {}
            meta.append(
                {
                    "client_id": cid,
                    "num_samples": len(idxs),
                    "label_distribution": counts,
                    "malicious": cid in self.cfg.malicious_client_ids,
                    "dataset": self.dataset_name,
                }
            )
        proxy = DataLoader(UnlabeledDataset(Subset(train, proxy_idx)), batch_size=self.cfg.batch_size, shuffle=False)
        test_indices = list(range(min(self.cfg.test_size, len(test))))
        test_loader = DataLoader(Subset(test, test_indices), batch_size=self.cfg.batch_size)
        return VisionSplits(loaders, proxy, test_loader, meta)


class MNISTProvider(TorchVisionClassificationProvider):
    dataset_cls = datasets.MNIST
    dataset_name = "mnist"
    default_channels = 1
    default_image_size = 28


class FashionMNISTProvider(TorchVisionClassificationProvider):
    dataset_cls = datasets.FashionMNIST
    dataset_name = "fashion_mnist"
    default_channels = 1
    default_image_size = 28


class CIFAR10Provider(TorchVisionClassificationProvider):
    dataset_cls = datasets.CIFAR10
    dataset_name = "cifar10"
    default_channels = 3
    default_image_size = 32
