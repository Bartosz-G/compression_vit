import torch
import numpy as np
from torchvision.transforms import functional

class CompressedToTensor:
    """Convert ndarrays of compressed or scaled ycbcr images to Tensors."""

    def __call__(self, image):
        """
        Convert a numpy.ndarray compressed image to a tensor.
        Args:
            image (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """


        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, np.ndarray):
            return torch.from_numpy(image.transpose((2, 0, 1)))
        else:
            return functional.to_tensor(image)