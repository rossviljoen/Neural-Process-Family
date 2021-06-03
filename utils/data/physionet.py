import os
import torch
import pandas as pd

from torch.utils.data import Dataset

from npf.utils.helpers import rescale_range
from .helpers import DIR_DATA

__all__ = ["PhysioNet"]

class PhysioNet(Dataset):
    def __init__(self, root=DIR_DATA, target="GCS", split="train", **kwargs):
        """Data from the PhysioNet2012 challenge [1].

        Parameters
        ----------
        root : The root data directory.

        target : The target variable variable for prediction. Typically either
            "GCS" - Glasgow coma scale - or "HCT" - hematocrit value.

        split : Whether to return the test or train data. 

        [1] https://www.physionet.org/content/challenge-2012/1.0.0/
        """
        self.data_path = DIR_DATA + "/physionet/data.h5"
        self.target = target
        self.dtype = torch.float32
        self.split = split
        self.min_max = (0, 48*60) # possible timestamp range (up to 48 hrs)
        
        self.dfs = _load_data(self.data_path)

        self.data = []
        for df in self.dfs.values():
            df = df[df.Parameter==self.target]
            if len(df) > 4:
                inputs_tensor = torch.tensor(df.index.values, dtype=self.dtype).unsqueeze(1)
                inputs_tensor = rescale_range(inputs_tensor, self.min_max, (-1, 1))
                targets_tensor = torch.tensor(df.Value.values, dtype=self.dtype).unsqueeze(1)
                self.data.append((inputs_tensor, targets_tensor))

        n = len(self.data) 
        if split=="train":
            self.data = self.data[:200]
        elif split=="test":
            self.data = self.data[200:400]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y

dfs = {}
def _load_data(path):
    """Preloads the physionet data from the hdf file and caches the result."""
    global dfs
    if not dfs:
        with pd.HDFStore(path, mode='r') as hdf_file:
            patients = hdf_file.keys()
            for p in patients:
                dfs[p] = hdf_file[p]
    return dfs
