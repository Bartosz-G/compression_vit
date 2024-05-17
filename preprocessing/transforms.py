import torch
import numpy as np
from scipy.fft import dctn, idctn
from torchvision.transforms import functional

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


class ConvertToYcbcr:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(dtype=torch.float64)
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]

        y = torch.tensor([0.299], dtype=torch.float64) * r + torch.tensor([0.587], dtype=torch.float64) * g + torch.tensor([0.114], dtype=torch.float64) * b
        cb = torch.tensor([-0.168736], dtype=torch.float64) * r - torch.tensor([0.331264], dtype=torch.float64) * g + torch.tensor([0.5], dtype=torch.float64) * b + torch.tensor([128], dtype=torch.float64)
        cr = torch.tensor([0.5], dtype=torch.float64) * r - torch.tensor([0.418688], dtype=torch.float64) * g - torch.tensor([0.081312], dtype=torch.float64) * b + torch.tensor([128], dtype=torch.float64)
        return torch.stack((y, cb, cr), dim=0)



class ConvertToFrequencyDomain:
    def __init__(self, compression_algorithm = dctn, block_size: tuple[int, int] = (8, 8), *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.blockwise_dct = _BlockwiseDct(compression_algorithm=compression_algorithm, block_size=block_size, *args, **kwargs)

    def __call__(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(img)
        channels = img.shape[0]
        for i in range(channels):
            output[i, :, :] = torch.from_numpy(self.blockwise_dct(img[i, :, :].numpy(), *self.args, **self.kwargs))

        return output



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



class Quantize:
    def __init__(self,
                 quantization_matrices: list[np.ndarray | torch.Tensor],
                 block_size: tuple[int, int] = (8, 8),
                 alpha: int = 1,
                 floor: bool = True,
                 *args,
                 **kwargs) -> None:
        # quantization_matrices must be equal to the number of channels
        self.args = args
        self.kwargs = kwargs
        self.blockwise_quantize = [_BlockwiseQuantize(qm, block_size, alpha, floor) for qm in quantization_matrices]

    def __call__(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(img)
        channels = img.shape[0]
        assert channels == len(self.blockwise_quantize), f'got number of channels {channels} but expected {len(self.blockwise_quantize)}'
        for i, quantization in enumerate(self.blockwise_quantize):
            output[i, :, :] = quantization(img[i, :, :], *self.args, **self.kwargs)
        return output

class _BlockwiseQuantize:
    def __init__(self,
                 quantitization_matrix: np.ndarray | torch.Tensor = LUMINANCE_QUANTIZATION_MATRIX,
                 block_size: tuple[int, int] = (8, 8),
                 alpha: int = 1,
                 floor: bool = True) -> None:
        self.quantitization_matrix = quantitization_matrix if isinstance(quantitization_matrix, np.ndarray) else quantitization_matrix.numpy()
        self.block_size = block_size
        self.alpha = alpha
        self.floor = floor

    def __call__(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        dct = img.numpy()

        height, width = dct.shape
        block_height_num, block_width_num = height // self.block_size[0], width // self.block_size[1]

        quantization_matrix_tiled = np.tile(self.quantitization_matrix, (block_height_num, block_width_num))

        if self.floor:
            return torch.from_numpy(np.round(dct / quantization_matrix_tiled).astype(np.int8)).float()

        return torch.from_numpy(dct / quantization_matrix_tiled).float()




class ZigZagOrder:
    zigzag_order_indices = torch.tensor([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])

    block = (8, 8)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        batch_size = image.shape[:-3]
        channels = image.shape[-3]

        unfold = torch.nn.Unfold(kernel_size=self.block, stride=self.block)
        windows = unfold(image).transpose(-2, -1).view(*batch_size, -1, channels, self.block[0]*self.block[1]).transpose(-2, -3)
        return windows[..., self.zigzag_order_indices]



class ChooseAC:
    def __init__(self, ac: int):
        self.ac = ac
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image[..., :self.ac + 1] # + 1 to get DCs at the 0th position


class FlattenZigZag:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flatten(image, start_dim = -2, end_dim = -1)


class _BlockwiseDct:
    """
    A class to perform blockwise Discrete Cosine Transform (DCT) on an image.

    Parameters
    ----------
    compression_algorithm : callable, optional
        The compression algorithm to use. Default is `scipy.fft.dctn`.
    block_size : tuple[int, int], optional
        The size of the blocks to divide the image into. Default is (8, 8).
    *args : tuple
        Additional positional arguments to pass to the compression algorithm.
    **kwargs : dict
        Additional keyword arguments to pass to the compression algorithm.

    Methods
    -------
    __call__(image: np.ndarray, *args, **kwargs) -> np.ndarray
        Applies the blockwise DCT to the given image.

    Parameters for __call__ method
    ------------------------------
    image : np.ndarray
        The input image on which the blockwise DCT is to be performed.
    *args : tuple
        Additional positional arguments to pass to the compression algorithm.
    **kwargs : dict
        Additional keyword arguments to pass to the compression algorithm.
    """

    def __init__(self, compression_algorithm = dctn, block_size: tuple[int, int] = (8, 8), *args, **kwargs) -> None:
        self.compression = compression_algorithm
        self.block_size = block_size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray,  *args, **kwargs):
        height, width = image.shape
        block_height, block_width = self.block_size

        dct_blocks = np.zeros_like(image, dtype=np.float64)


        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                block = image[i:i+block_height, j:j+block_width]

                dct_block = self.compression(block, *self.args, **self.kwargs)

                dct_blocks[i:i+block_height, j:j+block_width] = dct_block

        return dct_blocks



class _DefaultBlockwiseQuantize:
    def __init__(self,
                 luminance_matrix: np.ndarray = LUMINANCE_QUANTIZATION_MATRIX,
                 chrominance_matrix: np.ndarray = CHROMINANCE_QUANTIZATION_MATRIX,
                 block_size: tuple[int, int] = (8, 8),
                 alpha: int = 1,) -> None:
        self.luminance_matrix = luminance_matrix
        self.chrominance_matrix = chrominance_matrix
        self.block_size = block_size
        self.alpha = alpha

    def __call__(self, dct, mode = 'l', *args, **kwargs) -> np.ndarray:
        quantization_matrix = self.luminance_matrix if mode == 'l' else self.chrominance_matrix

        quantization_matrix = quantization_matrix * self.alpha

        height, width = dct.shape
        block_height_num, block_width_num = height // self.block_size[0], width // self.block_size[1]

        quantization_matrix_tiled = np.tile(quantization_matrix, (block_height_num, block_width_num))

        return np.round(dct / quantization_matrix_tiled).astype(np.int8)



class _CompressQuantiseAcrossChannels:
    def __init__(self, blockwise_compression, blockwise_quantization):
        self.compression = blockwise_compression
        self.quantize = blockwise_quantization

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        channels, height, width = image.shape
        assert channels == 3, f'channels must be 3 YCbCr but got {channels} instead'


        y = self.compression(image[0, :, :], *args, **kwargs)
        cb = self.compression(image[1, :, :], *args, **kwargs)
        cr = self.compression(image[2, :, :], *args, **kwargs)

        y = self.quantize(y, 'l',  *args, **kwargs)
        cb = self.quantize(cb, 'c', *args, **kwargs)
        cr = self.quantize(cr, 'c',  *args, **kwargs)


        return np.stack((y, cb, cr), axis=-3)





#%%
