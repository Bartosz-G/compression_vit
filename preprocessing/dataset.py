import numpy as np
import torchvision
from torchvision.datasets import CIFAR10
from scipy.fft import dctn, idctn
from typing import Optional, Tuple, Any, Union, Callable
from pathlib import Path
from copy import deepcopy



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

BLOCK_SIZE = (8, 8)
ALPHA = 1



class ApplyAcrossBatch:
    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, array: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Get the shape of the input array
        batch_size, channels, H, W = array.shape

        # Initialize an output array of the same shape as the input
        output = np.zeros_like(array)

        # Iterate over the batch and channel dimensions
        for i in range(batch_size):
            output[i, :, :, :] = self.func(array[i, :, :, :], *args, **kwargs)

        return output



class BlockwiseDct:
    def __init__(self, compression_algorithm = dctn, block_size: tuple[int, int] = (8, 8), *args, **kwargs) -> None:
        self.compression = compression_algorithm
        self.block_size = block_size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray,  *args, **kwargs):
        height, width = image.shape
        block_height, block_width = self.block_size

        dct_blocks = np.zeros_like(image, dtype=np.float32)


        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                block = image[i:i+block_height, j:j+block_width]

                dct_block = self.compression(block, *self.args, **self.kwargs)

                dct_blocks[i:i+block_height, j:j+block_width] = dct_block

        return dct_blocks



class BlockwiseQuantize:
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



class CompressQuantiseAcrossChannels:
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


blockwise_dct = BlockwiseDct(compression_algorithm=dctn, block_size=BLOCK_SIZE, norm='ortho')
blockwise_quantize = BlockwiseQuantize(luminance_matrix=LUMINANCE_QUANTIZATION_MATRIX,
                                       chrominance_matrix=CHROMINANCE_QUANTIZATION_MATRIX,
                                       block_size=BLOCK_SIZE,
                                       alpha=ALPHA)
compress_quantise_across_channels = CompressQuantiseAcrossChannels(blockwise_compression=blockwise_dct,
                                                                   blockwise_quantization=blockwise_quantize)



class CIFAR10_custom(CIFAR10):
    def __init__(self, root: Union[str, Path],
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 compression: Optional[Callable] = None) -> None:
        super(CIFAR10_custom, self).__init__(root=root,
                                             train=train,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.format = 'rgb'
        if compression is None:
            self.apply_compression = ApplyAcrossBatch(compress_quantise_across_channels)
        else:
            self.apply_compression = ApplyAcrossBatch(compression)


    def to_ycbcr(self, in_place = False) -> Optional[Tuple[np.ndarray, list]]:
        r"""Convert entire dataset from RGB to YCbCr.

             Args:
                in_place (bool, optional): Whether to modify the entire CIFAR10 dataset in memory.
                If set to False, returns the (images, targets) as tuples where images are of
                shape (B, C, H, W). Defaults to False.
        """
        assert self.format != 'compressed', f'cannot transform format {self.format} into ycbcr, have you ran to_ycbcr after compressed?'

        if self.format == 'ycbcr' and not in_place: # prevent applying transformation twice
            return self.data.transpose((0, 3, 1, 2)), self.targets

        if self.format == 'ycbcr' and in_place: # prevent applying transformation twice
            return None

        data = self.data.transpose((0, 3, 1, 2))  # convert to CHW


        r = data[..., 0, :, :]
        g = data[..., 1, :, :]
        b = data[..., 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128


        if in_place:
            self.format = 'ycbcr'
            self.data = np.stack((y, cb, cr), axis=-3).transpose((0, 2, 3, 1)) # convert to HWC
        else:
            return np.stack((y, cb, cr), axis=-3), self.targets


    def compress(self, in_place = False, *args, **kwargs) -> Optional[Tuple[np.ndarray, list]]:
        r"""Compresses the entire dataset from YCbCr format.

        Args:
            in_place (bool, optional): Whether to modify the entire CIFAR10 dataset in memory.
                If set to False, returns the (images, targets) as tuples where images are of
                shape (B, C, H, W). Defaults to False.
            block_size (tuple[int, int], optional): Size of the window to apply compression.
                Defaults to (8, 8).
            alpha (int, optional): Alpha parameter to control the magnitude of compression.
                Defaults to 1.
        """


        assert self.format != 'rgb', f'format should be ycbcr but is {self.format}, call .to_ycbcr(in_place = True) before compress'

        if self.format == 'compressed' and not in_place: # prevent applying transformation twice
            return self.data.transpose((0, 3, 1, 2)), self.targets

        if self.format == 'compressed' and in_place: # prevent applying transformation twice
            return None

        data = self.data.transpose((0, 3, 1, 2)) # convert to CHW

        data = self.apply_compression(data, compress_quantise_across_channels, *args, **kwargs)

        if in_place:
            self.format = 'compressed'
            self.data = data.transpose((0, 2, 3, 1)) # convert to HWC
        else:
            return data, self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.format == 'rgb':
            return super().__getitem__(index)

        if self.format in ('compressed', 'ycbcr'):

            img, target = self.data[index], self.targets[index]

            if self.transform is not None:
                self._check_transform()
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def _check_transform(self) -> Any:
        invalid_transform = torchvision.transforms.transforms.__name__

        transformations = deepcopy(self.transform.__getstate__()['transforms'])

        are_valid = list(map(lambda t: t.__class__.__module__ != invalid_transform, transformations))

        assert all(are_valid), f"base image transformations from torchvision are not supported when format is {self.format}, use custom ones from preprocessing.transforms"













def _apply_across_batch(array, func, *args, **kwargs):
    # Get the shape of the input array
    batch_size, channels, H, W = array.shape

    # Initialize an output array of the same shape as the input
    output = np.zeros_like(array)

    # Iterate over the batch and channel dimensions
    for i in range(batch_size):
        output[i, :, :, :] = func(array[i, :, :, :], *args, **kwargs)

    return output


def _blockwise_dct(image: np.ndarray, block_size: tuple[int, int] = (8, 8)):
    height, width = image.shape
    block_height, block_width = block_size

    dct_blocks = np.zeros_like(image, dtype=np.float32)


    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            block = image[i:i+block_height, j:j+block_width]

            dct_block = dctn(block, norm='ortho')

            dct_blocks[i:i+block_height, j:j+block_width] = dct_block

    return dct_blocks



def _blockwise_quantize(dct: np.ndarray, mode='l', block_size: tuple[int, int] = (8, 8), alpha: int = 1,) -> np.ndarray:
    luminance_quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    chrominance_quantization_matrix = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])


    quantization_matrix = luminance_quantization_matrix if mode == 'l' else chrominance_quantization_matrix

    quantization_matrix = quantization_matrix * alpha

    height, width = dct.shape
    block_height_num, block_width_num = height // block_size[0], width // block_size[1]

    quantization_matrix_tiled = np.tile(quantization_matrix, (block_height_num, block_width_num))

    return np.round(dct / quantization_matrix_tiled).astype(np.int8)





def _compress_quantise_across_channels(image: np.ndarray, block_size: tuple[int, int] = (8, 8), alpha: int = 1, *args, **kwargs):
    channels, height, width = image.shape

    assert channels == 3, f'channels must be 3 YCbCr but got {channels} instead'


    y = blockwise_dct(image[0, :, :], block_size, *args, **kwargs)
    cb = blockwise_dct(image[1, :, :], block_size, *args, **kwargs)
    cr = blockwise_dct(image[2, :, :],block_size, *args, **kwargs)

    y = blockwise_quantize(y, 'l', block_size, alpha, *args, **kwargs)
    cb = blockwise_quantize(cb, 'c', block_size, alpha, *args, **kwargs)
    cr = blockwise_quantize(cr, 'c', block_size, alpha, *args, **kwargs)


    return np.stack((y, cb, cr), axis=-3)


