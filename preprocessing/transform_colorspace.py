import torch

class ConvertToYcbcr:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(dtype=torch.float64)
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
        return torch.stack((y, cb, cr), dim=0)


#inverse of ConvertToYcbcr
class ConvertToRgb:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(dtype=torch.float64)
        y = img[0, :, :]
        cb = img[1, :, :] - 128
        cr = img[2, :, :] - 128

        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb

        return torch.stack((r, g, b), dim=0)