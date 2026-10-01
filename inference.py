#!/usr/bin/env python3
"""Unified inference entry point for StyleMapping + StyleFusion.

The original research scripts are kept in ``style_mapping`` and
``style_fusion``. This entry point removes machine-specific paths and combines
the paper's three inference stages:

1. pixel unshuffle;
2. StyleMapping;
3. pixel shuffle and optional StyleFusion.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
STYLE_FUSION_ROOT = PROJECT_ROOT / "style_fusion"

# The copied research module uses local imports such as ``import net`` and
# ``from utils...``. Adding only its own root keeps those scripts intact while
# allowing this portable top-level entry point to reuse the implementation.
sys.path.insert(0, str(STYLE_FUSION_ROOT))

import net as style_net  # noqa: E402
from fusion import DWTFusionNet  # noqa: E402
from fusion_v2 import stitch_patches  # noqa: E402
from large_image_process.image_sample_v2 import (  # noqa: E402
    batch_grid_merge_pixelshuffle_pad,
    batch_grid_split_pixelunshuffle_pad,
)


DEFAULT_VGG = STYLE_FUSION_ROOT / "models" / "vgg_normalised.pth"
DEFAULT_DECODER = (
    STYLE_FUSION_ROOT
    / "experiments"
    / "photorealistic_remote_sence"
    / "NDNSMDecoder_iter_160000.pth.tar"
)
DEFAULT_FUSION = (
    STYLE_FUSION_ROOT
    / "experiments"
    / "DWTFusionNet_content_size=1024_l1+all_loss"
    / "StyleFusion_iter_100000.pth.tar"
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reference-guided remote-sensing image color correction."
    )
    parser.add_argument(
        "--content",
        type=Path,
        required=True,
        help="Content image or directory of content images.",
    )
    parser.add_argument(
        "--style",
        type=Path,
        required=True,
        help="Reference style image or directory. Directories use Cartesian pairing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Output directory (default: ./outputs).",
    )
    parser.add_argument("--vgg", type=Path, default=DEFAULT_VGG)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--fusion-weight", type=Path, default=DEFAULT_FUSION)
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Skip StyleFusion and write only the StyleMapping result.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Maximum StyleMapping subimage size; 0 disables pixel unshuffle.",
    )
    parser.add_argument(
        "--style-size",
        type=int,
        default=512,
        help="Square reference-image size used by StyleMapping.",
    )
    parser.add_argument(
        "--mapping-batch-size",
        type=int,
        default=1,
        help="Number of pixel-unshuffled subimages processed per batch.",
    )
    parser.add_argument(
        "--fusion-patch",
        type=int,
        default=1024,
        help="StyleFusion tile size for high-resolution images.",
    )
    parser.add_argument(
        "--fusion-overlap",
        type=int,
        default=64,
        help="Overlap between neighboring StyleFusion tiles.",
    )
    parser.add_argument(
        "--fusion-batch-size",
        type=int,
        default=2,
        help="Number of StyleFusion tiles processed per batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an indexed device such as cuda:0.",
    )
    parser.add_argument(
        "--save-ext",
        choices=("png", "jpg"),
        default="png",
        help="Output image format.",
    )
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def collect_images(path: Path) -> List[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {path}")
    return images


def validate_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint structure: {path}")
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module.") :]: value for key, value in checkpoint.items()}
    return checkpoint


def load_models(
    vgg_path: Path,
    decoder_path: Path,
    fusion_path: Optional[Path],
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    vgg = copy.deepcopy(style_net.vgg)
    vgg.load_state_dict(load_state_dict(vgg_path))
    mapping = style_net.Net(vgg)
    mapping.NDNSMDecoder.load_state_dict(load_state_dict(decoder_path))
    mapping.eval().to(device)

    fusion = None
    if fusion_path is not None:
        fusion = DWTFusionNet(in_ch=3, base_ch=32)
        fusion.load_state_dict(load_state_dict(fusion_path))
        fusion.eval().to(device)
    return mapping, fusion


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB")).unsqueeze(0)


def load_style(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size > 0:
            image = image.resize((size, size), Image.Resampling.BICUBIC)
        return TF.to_tensor(image).unsqueeze(0).to(device)


@torch.inference_mode()
def run_mapping(
    model: torch.nn.Module,
    content_cpu: torch.Tensor,
    style: torch.Tensor,
    target_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    _, _, height, width = content_cpu.shape
    if target_size <= 0 or max(height, width) <= target_size:
        return model.pred(content_cpu.to(device), style).cpu()

    low_res, factor, original_size, padding = batch_grid_split_pixelunshuffle_pad(
        content_cpu, target_size
    )
    outputs: List[torch.Tensor] = []
    for start in range(0, low_res.shape[0], batch_size):
        content_batch = low_res[start : start + batch_size].to(device)
        outputs.append(model.pred(content_batch, style).cpu())
    stylized_low_res = torch.cat(outputs, dim=0)
    return batch_grid_merge_pixelshuffle_pad(
        stylized_low_res,
        factor,
        batch_size=1,
        original_size=original_size,
        padding=padding,
    )


@torch.inference_mode()
def run_fusion(
    model: torch.nn.Module,
    content_cpu: torch.Tensor,
    stylized_cpu: torch.Tensor,
    patch: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    _, _, height, width = content_cpu.shape
    pad_height = height % 2
    pad_width = width % 2
    if pad_height or pad_width:
        content_cpu = F.pad(content_cpu, (0, pad_width, 0, pad_height), mode="replicate")
        stylized_cpu = F.pad(stylized_cpu, (0, pad_width, 0, pad_height), mode="replicate")

    content = content_cpu.to(device)
    stylized = stylized_cpu.to(device)
    if content.shape[-2] <= patch and content.shape[-1] <= patch:
        fused = model(content, stylized)
    else:
        fused = stitch_patches(
            model,
            content,
            stylized,
            patch=patch,
            overlap=overlap,
            device=device,
            amp=False,
            batch_size=batch_size,
        )
    return fused[:, :, :height, :width].cpu()


def output_name(content: Path, style: Path, extension: str) -> str:
    return f"{content.stem}_stylized_{style.stem}.{extension}"


def save_outputs(
    output_root: Path,
    filename: str,
    stage1: torch.Tensor,
    stage2: Optional[torch.Tensor],
) -> None:
    stage1_dir = output_root / "stage1_style_mapping"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    save_image(stage1.clamp(0, 1), stage1_dir / filename, padding=0)
    if stage2 is not None:
        stage2_dir = output_root / "stage2_style_fusion"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        save_image(stage2.clamp(0, 1), stage2_dir / filename, padding=0)


def iter_pairs(contents: Iterable[Path], styles: Iterable[Path]):
    for content in contents:
        for style in styles:
            yield content, style


def main() -> None:
    args = build_parser().parse_args()
    if args.mapping_batch_size < 1 or args.fusion_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.fusion_patch <= args.fusion_overlap:
        raise ValueError("--fusion-patch must be larger than --fusion-overlap.")
    if args.fusion_patch % 2:
        raise ValueError("--fusion-patch must be even because StyleFusion uses DWT.")

    device = resolve_device(args.device)
    contents = collect_images(args.content)
    styles = collect_images(args.style)
    vgg_path = validate_file(args.vgg, "VGG weight")
    decoder_path = validate_file(args.decoder, "StyleMapping decoder")
    fusion_path = None
    if not args.mapping_only:
        fusion_path = validate_file(args.fusion_weight, "StyleFusion weight")

    mapping, fusion = load_models(vgg_path, decoder_path, fusion_path, device)
    args.output.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    pairs = list(iter_pairs(contents, styles))

    for content_path, style_path in tqdm(pairs, desc="Color correction", unit="pair"):
        content = load_rgb(content_path)
        style = load_style(style_path, args.style_size, device)
        stage1 = run_mapping(
            mapping,
            content,
            style,
            args.target_size,
            args.mapping_batch_size,
            device,
        )
        stage2 = None
        if fusion is not None:
            stage2 = run_fusion(
                fusion,
                content,
                stage1,
                args.fusion_patch,
                args.fusion_overlap,
                args.fusion_batch_size,
                device,
            )
        save_outputs(
            args.output.expanduser().resolve(),
            output_name(content_path, style_path, args.save_ext),
            stage1,
            stage2,
        )

    print(f"Processed {len(pairs)} image pair(s). Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
