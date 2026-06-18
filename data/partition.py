from typing import List
import numpy as np

def iid_partition(indices:List[int], num_clients:int, rng)->List[List[int]]:
    arr=np.array(indices); rng.shuffle(arr); return [x.tolist() for x in np.array_split(arr,num_clients)]

def dirichlet_partition(indices, labels, num_clients:int, alpha:float, rng)->List[List[int]]:
    labels=np.asarray(labels); buckets=[[] for _ in range(num_clients)]
    for k in np.unique(labels[indices]):
        cls=np.array(indices)[labels[indices]==k]; rng.shuffle(cls); props=rng.dirichlet([alpha]*num_clients); splits=(np.cumsum(props)*len(cls)).astype(int)[:-1]
        for i,part in enumerate(np.split(cls,splits)): buckets[i].extend(part.tolist())
    return buckets
