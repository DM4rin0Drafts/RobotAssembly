import numpy as np
import torch
from torch import Tensor


def random_noise(log_std: float, tensor: Tensor, device: str) -> Tensor:
    log_std_tensor = torch.full(tensor.size(), log_std).to(device)
    noise = torch.exp(log_std_tensor) * torch.randn_like(tensor)
    return noise.to(device)


def rgb2grayscale(img_rgb: np.array) -> np.array:
    grayscale_multiplier = [0.2989, 0.5870, 0.1140]
    if len(img_rgb.shape) == 3:
        if img_rgb.shape[-1] == 4:  # rgb image has alpha values
            return img_rgb[..., :3] @ grayscale_multiplier
        else:
            img_rgb @ grayscale_multiplier
    return img_rgb


def rgb2binary(img_rgb: np.array, invert=True) -> np.array:
    img_gray = rgb2grayscale(img_rgb)
    img_binary = np.round(img_gray / 255).astype(np.bool)

    if invert:
        # if img_binary = 1 set 0; if img_binary = 0 set 1
        img_binary = np.invert(img_binary)
    return img_binary


def numpy2torch(array: np.array, device='cpu'):
    """
    Converts numpy array of shape (H, W) or (H, W, C) to torch tensor of shape (N, C, H, W).
    """
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    elif array.ndim == 2:
        array = np.expand_dims(array, axis=0)  # (H, W) -> (C, H, W)

    tensor = torch.from_numpy(array).unsqueeze(0)  # (C, H, W) -> (N, C, H, W)
    # TODO: Check astype(np.uint8). array before is bool?
    # TODO: Why .to(device)?
    return tensor.to(device)
    # return torch.from_numpy(array.astype(np.uint8)).unsqueeze(0).to(device)
