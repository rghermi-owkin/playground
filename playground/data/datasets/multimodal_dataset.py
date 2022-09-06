from typing import List
import numpy as np

import torch
from torch.utils.data import Dataset

from playground.utils import align_modalities
from playground.data import (
    load_tcga,
    preprocess_exp,
    preprocess_histo,
)


class MultimodalDataset(Dataset):
    
    def __init__(
        self,
        cohort: str = "COAD",
        modalities: List = ["HISTO", "EXP"],
        id: str = "patient_id",
    ):
        self.cohort = cohort
        self.modalities = modalities
        self.id = id
        
        # Load data
        data = load_tcga(
            cohort=self.cohort,
            modalities=self.modalities,
        )
        
        # Pre-process data
        if "HISTO" in self.modalities:
            data["HISTO"] = preprocess_histo(data["HISTO"])
            
        if "EXP" in self.modalities:
            data["EXP"] = preprocess_exp(data["EXP"])
        
        # Align modalities
        data, ids = align_modalities(data, self.id)
        
        self.ids = ids
        self.data = data
    
    def __getitem__(self, item: int):
        id = self.ids[item]
        
        sample = {}
        
        if "HISTO" in self.modalities:
            features = self.data["HISTO"].iloc[item]
            features = features.feature_path
            features = np.load(features)
            sample["HISTO"] = torch.from_numpy(features).float()
        
        if "EXP" in self.modalities:
            features = self.data["EXP"].iloc[item]
            features = features.drop(["patient_id", "sample_id"])
            features = features.values.astype(float)
            sample["EXP"] = torch.from_numpy(features).float()
        
        return id, sample
    
    def __len__(self):
        return len(self.ids)
