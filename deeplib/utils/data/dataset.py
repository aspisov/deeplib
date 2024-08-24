from typing import List, Any, Tuple, Iterable
import bisect

class Dataset(object):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        raise NotImplementedError
    
    def __add__(self, other):
        return ConcatDataset([self, other])
    

class ConcatDataset(Dataset):
    
    @staticmethod
    def cumsum(datasets):
        cum_sum, s = [], 0
        for dataset in datasets:
            length = len(dataset)
            s += length
            cum_sum.append(s)
        return cum_sum
    
    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__()
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
    
   
        