from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np, torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from data.base import DatasetProvider
from data.partition import iid_partition, dirichlet_partition

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx): x,_=self.base[idx]; return x, -1

@dataclass
class MNISTSplits:
    client_loaders:List[DataLoader]; proxy_loader:DataLoader; test_loader:DataLoader; client_metadata:List[Dict[str,Any]]

class MNISTProvider(DatasetProvider):
    def __init__(self,cfg): self.cfg=cfg
    def build(self):
        tf=transforms.Compose([transforms.ToTensor()])
        train=datasets.MNIST(self.cfg.data_dir, train=True, download=self.cfg.download_data, transform=tf)
        test=datasets.MNIST(self.cfg.data_dir, train=False, download=self.cfg.download_data, transform=tf)
        rng=np.random.default_rng(self.cfg.seed); all_idx=np.arange(len(train)); rng.shuffle(all_idx)
        if self.cfg.quick_data_limit: all_idx=all_idx[:self.cfg.quick_data_limit]
        proxy_idx=all_idx[:self.cfg.proxy_size].tolist(); client_idx=all_idx[self.cfg.proxy_size:].tolist()
        labels=np.array(train.targets)
        parts=iid_partition(client_idx,self.cfg.num_clients,rng) if self.cfg.partition_type=='iid' else dirichlet_partition(client_idx,labels,self.cfg.num_clients,self.cfg.dirichlet_alpha,rng)
        loaders=[]; meta=[]
        for cid,idxs in enumerate(parts):
            loaders.append(DataLoader(Subset(train,idxs),batch_size=self.cfg.batch_size,shuffle=True))
            counts={int(k):int(v) for k,v in zip(*np.unique(labels[idxs],return_counts=True))} if idxs else {}
            meta.append({'client_id':cid,'num_samples':len(idxs),'label_distribution':counts,'malicious':cid in self.cfg.malicious_client_ids})
        proxy=DataLoader(UnlabeledDataset(Subset(train,proxy_idx)), batch_size=self.cfg.batch_size, shuffle=False)
        test_indices=list(range(min(self.cfg.test_size,len(test))))
        return MNISTSplits(loaders,proxy,DataLoader(Subset(test,test_indices),batch_size=self.cfg.batch_size),meta)
