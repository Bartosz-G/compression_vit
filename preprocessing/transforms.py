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


class ZigZagOrder:
    zigzag_order_indices = torch.tensor([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])

    block = (8, 8)

    def __call__(self, image):

        batch_size = image.shape[:-3]
        channels = image.shape[-3]

        unfold = torch.nn.Unfold(kernel_size=self.block, stride=self.block)
        windows = unfold(image).transpose(-2, -1).view(*batch_size, -1, channels, self.block[0]*self.block[1]).transpose(-2, -3)
        return windows[..., self.zigzag_order_indices]



class ChooseAC:
    #TODO: Implement
    def __init__(self, ac: int):
        pass

class FlattenZigZag:
    #TODO: Implement
    pass
#%%
