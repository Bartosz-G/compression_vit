import torch

class FlattenZigZag:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        patch_num = image.shape[-2]
        return image.permute((1, 0, 2)).reshape(patch_num, -1).flatten(start_dim=1, end_dim=-1)

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