import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import math
import torch.nn as nn


def batch_grid_split_pixelunshuffle(high_res: torch.Tensor, grid_size: int):
    """
    high_res: (N, C, H*r, W*r)
    returns: (N * r^2, C, H, W)
    This uses pixel_unshuffle then rearranges to (N*r^2, C, H, W) with the correct order.
    """
    assert high_res.dim() == 4

    N, C, H_mul, W_mul = high_res.shape
    r = grid_size
    assert H_mul % r == 0 and W_mul % r == 0
    H, W = H_mul // r, W_mul // r

    # (N, C*r^2, H, W)
    low = F.pixel_unshuffle(high_res, r)

    # reshape (N, C, r^2, H, W) then permute -> (N, r^2, C, H, W)
    low = low.view(N, C, r * r, H, W).permute(0, 2, 1, 3, 4).contiguous()
    # final flatten -> (N*r^2, C, H, W)
    low = low.view(N * r * r, C, H, W)
    return low


def batch_grid_merge_pixelshuffle(sampled_tensors: torch.Tensor, grid_size: int, batch_size: int):
    """
    sampled_tensors: (N * r^2, C, H, W)
    returns: (N, C, H*r, W*r)
    This is the inverse of batch_grid_split_pixelunshuffle above.
    """
    r = grid_size
    N = batch_size
    total = sampled_tensors.shape[0]
    per_image = total // N
    assert per_image == r * r, "expected r^2 tiles per image"

    C, H, W = sampled_tensors.shape[1:]
    # group back to (N, r^2, C, H, W)
    x = sampled_tensors.view(N, r * r, C, H, W)
    # permute to (N, C, r^2, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # flatten channels to (N, C*r^2, H, W)
    x = x.view(N, C * r * r, H, W)

    # pixel shuffle to upsample spatially
    out = F.pixel_shuffle(x, r)  # (N, C, H*r, W*r)
    return out


def batch_grid_split_pixelunshuffle_pad(high_res: torch.Tensor, target_size: int):
    N, C, H, W = high_res.shape
    r_h = math.ceil(H / target_size)
    r_w = math.ceil(W / target_size)
    r = max(r_h, r_w)
    H_pad = r * target_size - H
    W_pad = r * target_size - W
    device = high_res.device
    high_res = high_res.cpu()
    # Reflection padding requires each padding amount to be smaller than the
    # corresponding input dimension. Very elongated remote-sensing strips may
    # violate that constraint, so fall back to replication for those cases.
    pad_mode = 'reflect' if H_pad < H and W_pad < W else 'replicate'
    high_res = F.pad(high_res, (0, W_pad, 0, H_pad), mode=pad_mode)
    # high_res = F.pad(high_res, (0, W_pad, 0, H_pad), mode='reflect')
    low = F.pixel_unshuffle(high_res, r)
    H_low, W_low = target_size, target_size
    low = low.view(N, C, r*r, H_low, W_low).permute(0,2,1,3,4).contiguous()
    low = low.view(N*r*r, C, H_low, W_low)
    return low, r, (H, W), (H_pad, W_pad)  # 返回 padding

def batch_grid_merge_pixelshuffle_pad(low_res: torch.Tensor, r: tuple, batch_size: int, original_size: tuple, padding: tuple):
    N = batch_size
    H_pad, W_pad = padding
    C, H_low, W_low = low_res.shape[1:]
    x = low_res.view(N, r*r, C, H_low, W_low).permute(0,2,1,3,4).contiguous()
    x = x.view(N, C*r*r, H_low, W_low)
    high_res = F.pixel_shuffle(x, r)
    H_orig, W_orig = original_size
    high_res = high_res[:, :, :H_orig + H_pad, :W_orig + W_pad]  # 先裁剪掉多余 padding
    high_res = high_res[:, :, :H_orig, :W_orig]  # 最终裁剪回原始大小
    return high_res


def batch_patch_split_pad(high_res: torch.Tensor, patch_size: int):
    """
    将高分辨率图像拆分成固定大小 patch
    high_res: (N, C, H, W)
    patch_size: 每个 patch 的高宽
    returns: (N*num_patches, C, patch_size, patch_size), num_patches_h, num_patches_w
    """
    N, C, H, W = high_res.shape

    # 计算需要多少 patch
    num_patches_h = math.ceil(H / patch_size)
    num_patches_w = math.ceil(W / patch_size)

    # padding 保证能整除 patch_size
    H_pad = num_patches_h * patch_size - H
    W_pad = num_patches_w * patch_size - W
    if H_pad > 0 or W_pad > 0:
        high_res = F.pad(high_res, (0, W_pad, 0, H_pad), mode='reflect')

    # unfold 拆成 patch
    patches = high_res.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches: (N, C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0,2,3,1,4,5).contiguous()  # (N, num_h, num_w, C, patch_size, patch_size)
    patches = patches.view(-1, C, patch_size, patch_size)  # flatten -> (N*num_h*num_w, C, patch_size, patch_size)

    return patches, (num_patches_h, num_patches_w), (H, W)  # 返回原始尺寸用于 merge


def batch_patch_merge_pad(patches: torch.Tensor, num_patches: tuple, batch_size: int, original_size: tuple):
    """
    将 patch 拼回高分图像
    patches: (N*num_h*num_w, C, patch_size, patch_size)
    num_patches: (num_patches_h, num_patches_w)
    batch_size: 原始 batch size N
    original_size: 原始高分辨率大小 (H, W)
    returns: (N, C, H, W)
    """
    num_patches_h, num_patches_w = num_patches
    N = batch_size
    C, patch_size, _ = patches.shape[1:]

    # reshape -> (N, num_h, num_w, C, patch_size, patch_size)
    x = patches.view(N, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.permute(0,3,1,4,2,5).contiguous()  # (N, C, num_h, patch_h, num_w, patch_w)
    x = x.view(N, C, num_patches_h*patch_size, num_patches_w*patch_size)

    # 裁剪回原始大小
    H_orig, W_orig = original_size
    x = x[:, :, :H_orig, :W_orig]
    return x


def test_style_transform(h=0,w=0):
    transform_list = []
    if h != 0:
        transform_list.append(transforms.Resize((h, w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# ------------------------
# Round-trip 测试
# ------------------------
if __name__ == "__main__":

    N, C, H, W = 1, 3, 3000, 4000
    patch_size = 1024
    img = torch.randint(0, 256, (N, C, H, W), dtype=torch.uint8)
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    image_path = 'input/content_05/100_0043_0371.jpg'
    style_tf = test_style_transform()
    image = style_tf(Image.open(image_path).convert("RGB"))

    image = image.to(device).unsqueeze(0)
    image = torch.cat([image, image, image, image], dim=0)  # N=4 for test

    patches, num_patches, orig_size = batch_patch_split_pad(image, 1024)
    print(num_patches)
    restored = batch_patch_merge_pad(patches, num_patches, batch_size=4, original_size=orig_size)

    patches, r, orig_size, pad_size = batch_grid_split_pixelunshuffle_pad(image, 1024)
    restored = batch_grid_merge_pixelshuffle_pad(patches, r, batch_size=4, original_size=orig_size, padding=pad_size)

    print("原图大小:", image.shape)
    print("Patch 数:", patches.shape)
    print("恢复大小:", restored.shape)
    print("是否完全一致:", torch.equal(image, restored))
    save_image(restored, 'restored.png')
    save_image(image, 'original.png')
    save_image(patches, 'patches.png')
