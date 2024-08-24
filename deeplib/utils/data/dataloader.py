import deeplib as dl
from .dataset import Dataset
import numpy as np

class DataLoader(object):
    def __init__(self, dataset: Dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices =self.indices[start_idx:start_idx + self.batch_size]
            yield self._get_batch(batch_indices)
            
    def _get_batch(self, indices):
        batch_data = dl.stack([self.dataset[idx][0] for idx in indices])
        batch_labels = dl.stack([self.dataset[idx][1] for idx in indices])
        return batch_data, batch_labels
        