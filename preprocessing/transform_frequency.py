import torch
import numpy as np
from scipy.fft import dctn, idctn

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

                # Scaling to [-128, 127]
                block = block - 128
                dct_block = self.compression(block, *self.args, **self.kwargs)

                dct_blocks[i:i+block_height, j:j+block_width] = dct_block

        return dct_blocks

class ConvertToSpatialDomain:
    def __init__(self, decompression_algorithm = idctn, block_size: tuple[int, int] = (8, 8), *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.blockwise_idct = _BlockwiseIdct(decompression_algorithm=decompression_algorithm, block_size=block_size, *args, **kwargs)

    def __call__(self, freq_img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(freq_img)
        channels = freq_img.shape[0]
        for i in range(channels):
            output[i, :, :] = torch.from_numpy(self.blockwise_idct(freq_img[i, :, :].numpy(), *self.args, **self.kwargs))

        return output

class _BlockwiseIdct:
    """
    A class to perform blockwise Inverse Discrete Cosine Transform (IDCT) on an image.

    Parameters
    ----------
    decompression_algorithm : callable, optional
        The decompression algorithm to use. Default is `scipy.fft.idctn`.
    block_size : tuple[int, int], optional
        The size of the blocks to divide the image into. Default is (8, 8).
    *args : tuple
        Additional positional arguments to pass to the decompression algorithm.
    **kwargs : dict
        Additional keyword arguments to pass to the decompression algorithm.

    Methods
    -------
    __call__(image: np.ndarray, *args, **kwargs) -> np.ndarray
        Applies the blockwise IDCT to the given image.

    Parameters for __call__ method
    ------------------------------
    image : np.ndarray
        The input image on which the blockwise IDCT is to be performed.
    *args : tuple
        Additional positional arguments to pass to the decompression algorithm.
    **kwargs : dict
        Additional keyword arguments to pass to the decompression algorithm.
    """

    def __init__(self, decompression_algorithm = idctn, block_size: tuple[int, int] = (8, 8), *args, **kwargs) -> None:
        self.decompression = decompression_algorithm
        self.block_size = block_size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray, *args, **kwargs):
        height, width = image.shape
        block_height, block_width = self.block_size

        idct_blocks = np.zeros_like(image, dtype=np.float64)

        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                block = image[i:i+block_height, j:j+block_width]

                idct_block = self.decompression(block, *self.args, **self.kwargs)

                # Rescaling back from [-128, 127] to [0, 255]
                idct_block = idct_block + 128

                idct_blocks[i:i+block_height, j:j+block_width] = idct_block

        return idct_blocks

