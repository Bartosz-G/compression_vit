import torch
import numpy as np

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

class Quantize:
    def __init__(self,
                 quantization_matrices: list[np.ndarray | torch.Tensor],
                 block_size: tuple[int, int] = (8, 8),
                 alpha: float = 1.0,
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