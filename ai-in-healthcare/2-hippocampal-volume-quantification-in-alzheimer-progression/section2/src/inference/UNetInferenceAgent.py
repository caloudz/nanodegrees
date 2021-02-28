"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        conformant_volume = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))
        raise self.single_volume_inference(conformant_volume)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Create mask for each slice across the X (0th) dimension.
        
        result = np.zeros(volume.shape)
        
        for s in range(volume.shape[0]):
            vol_slice = volume[s,:,:]
            
            # normalize the slice
            vol_slice = vol_slice.astype(np.single)            
            norm = vol_slice / np.max(vol_slice)
            
            # convert to tensor and reshape into conformant shape for the model
            tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
            prediction = self.model(tensor.to(self.device))
            
            prediction = np.squeeze(prediction.cpu().detach())
            prediction = torch.argmax(prediction, dim=0)
            result[s,:,:] = prediction

        return result
