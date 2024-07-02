import torch
import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image
from torchvision.transforms import functional
from .transform_colorspace import *
from .transform_frequency import *
from .transform_quantize import *
from .transform_zigzag import *


LUMINANCE_QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


CHROMINANCE_QUANTIZATION_MATRIX = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


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
            output = functional.to_tensor(image).clone() *255
            return output.to(dtype=torch.uint8)

class TensorToCompressed:
    """Convert Tensors to ndarrays of compressed or scaled ycbcr images."""

    def __call__(self, tensor):
        """
        Convert a tensor to a numpy.ndarray compressed image.
        Args:
            tensor (torch.Tensor): Tensor to be converted to image.
        Returns:
            numpy.ndarray: Converted image.
        """

        if isinstance(tensor, Image.Image):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            if tensor.ndimension() == 3:
                # Convert tensor to numpy array
                array = tensor.cpu().numpy().transpose((1, 2, 0))
                # Convert from CHW to HWC format
                array = np.clip(array, 0, 255).astype(np.uint8)
                return array
            else:
                raise TypeError('Tensor should be 3D (C, H, W).')
        else:
            raise TypeError('Input should be a tensor.')


class ScaledPixels:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image / 255


class ChooseAC:
    __slots__ = ['ac'] # Slots magic
    def __init__(self, ac: int):
        self.ac = ac
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image[..., :self.ac + 1] # + 1 to get DCs at the 0th position

