import torch
import os
from PIL import Image
import numpy as np

def grid_sampling_and_downscaling(image_tensor, grid_size):
    """
    Perform grid sampling and downscaling for tensor input without nested loops.

    Args:
        image_tensor (torch.Tensor): Input image as a tensor of shape (C, H, W).
        grid_size (int): Size of each grid (e.g., 2 for 2x2 grids).

    Returns:
        list of torch.Tensor: List of downscaled tensors.
    """
    if image_tensor.dim() == 3:
        C, H, W = image_tensor.shape
    elif image_tensor.dim() == 4:
        N, C, H, W = image_tensor.shape
        assert N == 1, "N must be 1."
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    assert H % grid_size == 0 and W % grid_size == 0, "Image dimensions must be divisible by grid size."

    # Reshape the tensor to split into grids
    reshaped_tensor = image_tensor.view(C, H // grid_size, grid_size, W // grid_size, grid_size)
    # Permute dimensions to bring grid positions to the front
    permuted_tensor = reshaped_tensor.permute(2, 4, 0, 1, 3).contiguous()
    # Flatten the grid dimensions
    sampled_tensors = permuted_tensor.view(grid_size * grid_size, C, H // grid_size, W // grid_size)

    return sampled_tensors


def batch_grid_sampling_and_downscaling(image_tensor, grid_size):
    """
    Perform grid sampling and downscaling for tensor input, supporting batch processing.

    Args:
        image_tensor (torch.Tensor): Input image as a tensor of shape (N, C, H, W) or (C, H, W).
        grid_size (int): Size of each grid (e.g., 2 for 2x2 grids).

    Returns:
        torch.Tensor: Downscaled tensors of shape (N * grid_size * grid_size, C, H', W').
    """
    if image_tensor.dim() == 3:
        # Add batch dimension if missing
        image_tensor = image_tensor.unsqueeze(0)  # Shape becomes (1, C, H, W)

    assert image_tensor.dim() == 4, "Input tensor must have 4 dimensions (N, C, H, W)."
    N, C, H, W = image_tensor.shape
    assert H % grid_size == 0 and W % grid_size == 0, "Image dimensions must be divisible by grid size."

    # Initialize list to store results for each batch
    batch_sampled_tensors = []

    for batch_idx in range(N):
        # Process each batch separately
        batch_tensor = image_tensor[batch_idx]  # Shape (C, H, W)
        reshaped_tensor = batch_tensor.view(C, H // grid_size, grid_size, W // grid_size, grid_size)
        permuted_tensor = reshaped_tensor.permute(2, 4, 0, 1, 3).contiguous()
        sampled_tensors = permuted_tensor.view(grid_size * grid_size, C, H // grid_size, W // grid_size)
        batch_sampled_tensors.append(sampled_tensors)

    # Concatenate all batches into a single tensor
    result_tensor = torch.cat(batch_sampled_tensors, dim=0)  # Shape (N * grid_size * grid_size, C, H', W')

    return result_tensor


def grid_merging_and_restoration(sampled_tensors, grid_size):
    """
    Restore the high-resolution image from sampled low-resolution tensors.

    Args:
        sampled_tensors (list of torch.Tensor): List of tensors of shape (C, H', W') from grid sampling.
        grid_size (int): Size of the grid (e.g., 2 for 2x2, 3 for 3x3, etc.).

    Returns:
        torch.Tensor: Restored high-resolution image tensor of shape (C, H, W).
    """
    # Check the number of sampled tensors matches the expected grid size
    num_tensors = len(sampled_tensors)
    if num_tensors != grid_size * grid_size:
        raise ValueError("Number of sampled tensors does not match the expected grid size.")

    # Get the shape of the low-resolution tensors
    C, grid_h, grid_w = sampled_tensors[0].shape

    # Calculate the high-resolution image shape
    high_h, high_w = grid_h * grid_size, grid_w * grid_size

    # Initialize the high-resolution tensor
    high_res_tensor = torch.zeros((C, high_h, high_w), dtype=sampled_tensors[0].dtype).to(sampled_tensors[0].device)

    # Iterate through the grid tensors and place them into the high-resolution tensor
    for idx, sampled_tensor in enumerate(sampled_tensors):
        row, col = divmod(idx, grid_size)
        high_res_tensor[:, row::grid_size, col::grid_size] = sampled_tensor
    if sampled_tensors.dim() == 4:
        high_res_tensor = high_res_tensor.unsqueeze(0)
    return high_res_tensor


def batch_grid_merging_and_restoration(sampled_tensors, grid_size, batch_size):
    """
    Restore the high-resolution image from sampled low-resolution tensors, supporting batch processing.

    Args:
        sampled_tensors (torch.Tensor): Tensor of shape (N * grid_size * grid_size, C, H', W') from grid sampling.
        grid_size (int): Size of the grid (e.g., 2 for 2x2, 3 for 3x3, etc.).
        batch_size (int): Number of batches in the input tensor.

    Returns:
        torch.Tensor: Restored high-resolution image tensor of shape (N, C, H, W).
    """
    # Calculate the number of tensors per batch
    tensors_per_batch = grid_size * grid_size

    # Initialize list to store restored images for each batch
    batch_restored_images = []

    for batch_idx in range(batch_size):
        # Extract tensors for the current batch
        batch_sampled_tensors = sampled_tensors[batch_idx * tensors_per_batch:(batch_idx + 1) * tensors_per_batch]

        # Check the number of sampled tensors matches the expected grid size
        num_tensors = batch_sampled_tensors.shape[0]
        if num_tensors != tensors_per_batch:
            raise ValueError("Number of sampled tensors does not match the expected grid size.")

        # Get the shape of the low-resolution tensors
        C, grid_h, grid_w = batch_sampled_tensors[0].shape

        # Calculate the high-resolution image shape
        high_h, high_w = grid_h * grid_size, grid_w * grid_size

        # Initialize the high-resolution tensor
        high_res_tensor = torch.zeros((C, high_h, high_w), dtype=batch_sampled_tensors[0].dtype).to(batch_sampled_tensors[0].device)

        # Iterate through the grid tensors and place them into the high-resolution tensor
        for idx, sampled_tensor in enumerate(batch_sampled_tensors):
            row, col = divmod(idx, grid_size)
            high_res_tensor[:, row::grid_size, col::grid_size] = sampled_tensor

        batch_restored_images.append(high_res_tensor)

    # Concatenate all restored images into a single tensor
    result_tensor = torch.stack(batch_restored_images, dim=0)  # Shape (N, C, H, W)

    return result_tensor


import torch.nn as nn
def batch_grid_merge_pixelshuffle(sampled_tensors, grid_size, batch_size):
    """
    Restore high-resolution image using PixelShuffle (equivalent to your grid merge).
    """
    N = batch_size
    C, H, W = sampled_tensors.shape[1:]  # (C, H', W')

    # reshape: (N, grid_size^2 * C, H', W')
    sampled_tensors = sampled_tensors.view(N, grid_size * grid_size * C, H, W)

    # pixelshuffle upscales spatial size by grid_size
    pixel_shuffle = nn.PixelShuffle(grid_size)
    high_res = pixel_shuffle(sampled_tensors)  # (N, C, H*grid_size, W*grid_size)

    return high_res

def batch_grid_split_pixelunshuffle(high_res, grid_size):
    """
    Inverse of batch_grid_merge_pixelshuffle.
    Args:
        high_res: (N, C, H*grid_size, W*grid_size)
    Returns:
        (N*grid_size^2, C, H, W)
    """
    N, C, H_mul, W_mul = high_res.shape
    H, W = H_mul // grid_size, W_mul // grid_size

    pixel_unshuffle = nn.PixelUnshuffle(grid_size)
    low_res = pixel_unshuffle(high_res)  # (N, C*grid_size^2, H, W)

    # reshape 回到 (N*grid_size^2, C, H, W)
    low_res = low_res.view(N * grid_size * grid_size, C, H, W)
    return low_res


def batch_grid_patch(image_tensor, grid_size):
    """
    Efficiently split an image tensor into grid patches without downscaling.
    Similar to batch_grid_merge_pixelshuffle style, but patches keep original resolution.

    Args:
        image_tensor (torch.Tensor): (N, C, H, W)
        grid_size (int): number of patches along H and W

    Returns:
        torch.Tensor: (N * grid_size^2, C, H_patch, W_patch)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    N, C, H, W = image_tensor.shape
    assert H % grid_size == 0 and W % grid_size == 0, "H and W must be divisible by grid_size"

    H_patch = H // grid_size
    W_patch = W // grid_size

    # reshape to (N, C, grid_size, H_patch, grid_size, W_patch)
    patches = image_tensor.view(N, C, grid_size, H_patch, grid_size, W_patch)
    # permute to bring grid indices to batch dimension
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    # merge batch and grid dims: (N * grid_size^2, C, H_patch, W_patch)
    patches = patches.view(N * grid_size * grid_size, C, H_patch, W_patch)

    return patches


def batch_grid_merge_patch(patches, grid_size, batch_size):
    """
    Merge grid patches back to original images (no PixelShuffle, patches keep original resolution)

    Args:
        patches: (N * grid_size^2, C, H_patch, W_patch)
        grid_size: int, number of patches along H and W
        batch_size: int, original batch size N

    Returns:
        torch.Tensor: (N, C, H, W)
    """
    N = batch_size
    C, H_patch, W_patch = patches.shape[1:]  # (C, H_patch, W_patch)

    # reshape to (N, grid_size, grid_size, C, H_patch, W_patch)
    patches = patches.view(N, grid_size, grid_size, C, H_patch, W_patch)
    # permute to (N, C, grid_size, H_patch, grid_size, W_patch)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    # merge grid_size and patch dimensions: (N, C, H, W)
    merged = patches.view(N, C, H_patch * grid_size, W_patch * grid_size)

    return merged
