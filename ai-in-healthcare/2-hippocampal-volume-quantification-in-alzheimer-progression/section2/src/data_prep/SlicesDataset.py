"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        # i = image idx, d = image data, j = slice idx
        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx
        
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices.
        
        image_data = self.data[slc[0]]
        
        image_3dtt = image_data["image"]
        image_slc = image_3dtt[slc[1]]
        
        seg_3dtt = image_data["seg"]
        seg_slc = seg_3dtt[slc[1]]
        
        image_sample = torch.from_numpy(image_slc[None, :])
        image_label = torch.from_numpy(seg_slc[None, :])
            
        sample["image"] = image_sample.type(torch.cuda.FloatTensor)
        sample["seg"] = image_label.type(torch.cuda.LongTensor)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
