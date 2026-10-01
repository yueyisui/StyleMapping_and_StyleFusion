import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------------
# Example lightweight fusion core (you can replace with your own)
# ------------------------------
class StructurePreserveFusion(nn.Module):
    """
    Stylized + Content fusion core focusing on structure preservation.
    If global_scalar is provided (0..1), it biases the local attention toward content (closer to 1).
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv_feat = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    @torch.no_grad()
    def forward(self, stylized: torch.Tensor, content: torch.Tensor, global_scalar: Optional[torch.Tensor] = None) -> torch.Tensor:
        # stylized/content: (B,3,H,W)
        F_style = self.conv_feat(stylized)
        F_content = self.conv_feat(content)
        diff = torch.abs(F_style - F_content)
        attn_local = self.attn_conv(diff)  # (B,1,H,W)

        if global_scalar is not None:
            # global_scalar: (B,1,1,1) in [0,1]; 1=prefer content, 0=prefer style
            # Blend local map toward its mean by global preference for stability
            attn = global_scalar * attn_local + (1.0 - global_scalar) * attn_local.mean(dim=(2, 3), keepdim=True)
        else:
            attn = attn_local

        fused = attn * F_content + (1.0 - attn) * F_style
        out = self.conv_out(fused)
        return out


# ------------------------------
# Patch tiling, blending, stitching utilities
# ------------------------------

def hann2d_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create a separable 2D Hann window (cosine) to feather patch borders and avoid seams."""
    h_win = torch.hann_window(h, periodic=False, device=device)
    w_win = torch.hann_window(w, periodic=False, device=device)
    win2d = torch.ger(h_win, w_win)  # (h, w)
    # Normalize peak to 1
    win2d = win2d / win2d.max().clamp(min=1e-6)
    # A pure Hann window is zero at its outer boundary. Border pixels do not
    # have a neighboring tile outside the image, so a small floor prevents a
    # black one-pixel frame after normalization.
    win2d = win2d.clamp_min(1e-3)
    return win2d


def compute_grid(H: int, W: int, patch: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (y0, y1, x0, x1) covering the full image with the given patch size and overlap.
    Coordinates are inclusive-exclusive: [y0:y1, x0:x1].
    """
    assert patch > overlap >= 0
    stride = patch - overlap

    def starts(length: int) -> List[int]:
        if length <= patch:
            return [0]
        values = list(range(0, length - patch + 1, stride))
        if values[-1] != length - patch:
            values.append(length - patch)
        return values

    ys = starts(H)
    xs = starts(W)

    boxes = []
    for y in ys:
        for x in xs:
            y0 = max(0, y)
            x0 = max(0, x)
            y1 = min(H, y0 + patch)
            x1 = min(W, x0 + patch)
            boxes.append((y0, y1, x0, x1))
    return boxes


# @torch.no_grad()
def stitch_patches(
    model: nn.Module,
    stylized: torch.Tensor,
    content: torch.Tensor,
    patch: int = 1024,
    overlap: int = 64,
    device: Optional[torch.device] = None,
    amp: bool = True,
    batch_size: int = 1,
    global_scalar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Patch-wise forward with overlap and Hann blending. Handles ultra-high resolutions >10k.

    Args:
        model: fusion core with forward(stylized, content, global_scalar=None)
        stylized, content: (1,3,H,W) tensors in range arbitrary (model decides); will be chunked in tiles
        patch: tile size (square)
        overlap: pixel overlap between neighboring tiles
        device: torch device
        amp: use autocast fp16/bf16 where available
        batch_size: number of tiles per mini-batch (trade memory vs speed)
        global_scalar: optional (1,1,1,1) tensor to bias content/style globally
    Returns:
        fused: (1,3,H,W)
    """
    assert stylized.shape == content.shape and stylized.dim() == 4 and stylized.shape[0] == 1
    _, C, H, W = stylized.shape

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = model.to(device).eval()

    boxes = compute_grid(H, W, patch, overlap)
    window = hann2d_window(patch, patch, device=device).view(1, 1, patch, patch)

    out_acc = torch.zeros((1, C, H, W), device=device, dtype=stylized.dtype)
    w_acc = torch.zeros((1, 1, H, W), device=device, dtype=stylized.dtype)

    # Optional global scalar broadcasting per-tile
    if global_scalar is not None:
        global_scalar = global_scalar.to(device)

    # Process tiles in mini-batches
    def to_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for batch_boxes in to_batches(boxes, batch_size):
        s_tiles, c_tiles, coords = [], [], []
        for (y0, y1, x0, x1) in batch_boxes:
            s_patch = stylized[:, :, y0:y1, x0:x1].to(device, non_blocking=True)
            c_patch = content[:, :, y0:y1, x0:x1].to(device, non_blocking=True)
            # c_patch = F.resize(c_patch, (patch, patch), mode='area')
            # c_patch = nn.functional.interpolate(content, size=(patch, patch), mode='bilinear', align_corners=False)

            # If boundary tiles are smaller than patch, pad to patch size for consistent windowing
            pad_h = patch - (y1 - y0)
            pad_w = patch - (x1 - x0)
            if pad_h > 0 or pad_w > 0:
                s_patch = F.pad(s_patch, (0, pad_w, 0, pad_h), mode='reflect')
                c_patch = F.pad(c_patch, (0, pad_w, 0, pad_h), mode='reflect')

            s_tiles.append(s_patch)
            c_tiles.append(c_patch)
            coords.append((y0, y1, x0, x1, pad_h, pad_w))

        s_batch = torch.cat(s_tiles, dim=0)
        c_batch = torch.cat(c_tiles, dim=0)

        # with torch.autocast(device_type=device.type, enabled=amp):
        # Expand per-batch global scalar if provided
        g = None
        if global_scalar is not None:
            g = global_scalar.expand(s_batch.size(0), -1, -1, -1)
        # fused_batch = model(s_batch, c_batch, global_scalar=g) # !
        # fused_batch = model.pred(s_batch, c_batch)
        fused_batch = model(s_batch, c_batch)
        # fused_batch = s_batch

        # Accumulate with window
        for i, (y0, y1, x0, x1, pad_h, pad_w) in enumerate(coords):
            fused = fused_batch[i:i+1]
            if pad_h > 0 or pad_w > 0:
                fused = fused[:, :, : (y1 - y0), : (x1 - x0)]
                win = window[:, :, : (y1 - y0), : (x1 - x0)]
            else:
                win = window

            out_acc[:, :, y0:y1, x0:x1] += fused * win
            w_acc[:, :, y0:y1, x0:x1] += win

        # Optional: free VRAM between minibatches
        del s_batch, c_batch, fused_batch
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    fused_full = out_acc / w_acc.clamp(min=1e-6)
    return fused_full


# ------------------------------
# Global guidance estimator (low-res pass)
# ------------------------------
@torch.no_grad()
def estimate_global_scalar(
    stylized: torch.Tensor,
    content: torch.Tensor,
    down: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Estimate a single scalar in [0,1] that indicates how strongly to trust CONTENT globally.
    Heuristic: larger global difference -> more trust in content.
    Returns a tensor of shape (1,1,1,1).
    """
    assert stylized.shape == content.shape and stylized.dim() == 4 and stylized.shape[0] == 1
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, H, W = stylized.shape
    scale = min(1.0, down / max(H, W))
    if scale < 1.0:
        s_low = F.interpolate(stylized, scale_factor=scale, mode='area')
        c_low = F.interpolate(content, scale_factor=scale, mode='area')
    else:
        s_low, c_low = stylized, content

    s_low = s_low.to(device)
    c_low = c_low.to(device)

    # Simple normalized L1 difference as structure-change proxy
    diff = torch.mean(torch.abs(s_low - c_low))  # scalar
    # Map diff to [0,1] via logistic; tune k and midpoint as needed
    k = 10.0
    midpoint = 0.1  # adjust for your data dynamic range
    prob_content = torch.sigmoid(k * (diff - midpoint)).view(1, 1, 1, 1)
    return prob_content


# ------------------------------
# Convenience wrapper
# ------------------------------
class UltraHighResFuser:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        patch: int = 1024,
        overlap: int = 64,
        amp: bool = True,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.model = model if model is not None else StructurePreserveFusion(channels=32)
        self.patch = patch
        self.overlap = overlap
        self.amp = amp
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # @torch.no_grad()
    def __call__(
        self,
        stylized: torch.Tensor,
        content: torch.Tensor,
        use_global_guidance: bool = True,
    ) -> torch.Tensor:
        # Compute optional global guidance scalar
        g = None
        if use_global_guidance:
            # g = estimate_global_scalar(stylized, content, device=self.device)
            g = estimate_global_scalar(stylized, content)

        fused = stitch_patches(
            self.model,
            stylized,
            content,
            patch=self.patch,
            overlap=self.overlap,
            device=self.device,
            amp=self.amp,
            batch_size=self.batch_size,
            global_scalar=g,
        )
        return fused


# ------------------------------
# Example usage (commented)
# ------------------------------
if __name__ == "__main__":
    # Suppose you already loaded ultra-high-res tensors stylized/content with shape (1,3,H,W) and dtype float32
    B, C, H, W = 1, 3, 10000, 10000
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    content = torch.randn(B, C, H, W).to(device)
    stylized = torch.randn(B, C, H, W).to(device)

    # Configure the fuser
    fuser = UltraHighResFuser(
        model=StructurePreserveFusion(channels=32),
        patch=1024,      # try 2048 if VRAM allows
        overlap=64,      # 32~128 usually works well
        amp=True,        # enable mixed precision
        batch_size=2,    # increase if VRAM allows
        device=device
    )

    # Run (produces (1,3,H,W))
    for i in tqdm(range(1000)):
        fused = fuser(stylized, content, use_global_guidance=True)
    print(fused.shape)

    # Tip: If inputs are uint8 images, normalize to [0,1] float before calling
