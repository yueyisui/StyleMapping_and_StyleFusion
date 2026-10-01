# Pretrained models

Checkpoints are distributed separately from the source code. They are ignored
by Git and must be downloaded before running inference or training.

## Download

**Public download link: not published yet.**

The release archive is named `stylemapping-stylefusion-checkpoints.tar.gz`.
The maintainer will add its verified download link here when it is available.
No placeholder URL is presented as a working download.

The prepared archive has SHA-256 checksum:

```text
98122b57010619168dbc5eb32592356e2d650687bd5907ea35f52d127e69fa60
```

## Install and verify

Extract the archive in the repository root:

```bash
tar -xzf /path/to/stylemapping-stylefusion-checkpoints.tar.gz -C .
python scripts/check_weights.py
```

The default command verifies the following three inference checkpoints:

| Model | Expected relative path |
|---|---|
| VGG-19 | `style_fusion/models/vgg_normalised.pth` |
| Remote-sensing StyleMapping | `style_fusion/experiments/photorealistic_remote_sence/NDNSMDecoder_iter_160000.pth.tar` |
| DWT StyleFusion | `style_fusion/experiments/DWTFusionNet_content_size=1024_l1+all_loss/StyleFusion_iter_100000.pth.tar` |

The complete archive also includes the original natural-image and artistic
models, the DIOR fine-tuned model, MLP checkpoints, a second VGG copy for stage-one
training, and StyleFusion ablation models. Verify all 18 checkpoint files with:

```bash
python scripts/check_weights.py --all
```

[weights_manifest.json](weights_manifest.json) records each original path and
SHA-256 checksum. The verification script requires only Python's standard
library and does not load or execute checkpoint contents.

## 中文说明

模型权重不包含在 Git 仓库中，需单独下载。当前公开下载地址尚未提供；获得上述压缩包
后，请在项目根目录解压，并运行校验脚本。默认检查完整推理所需的 3 个权重，
`--all` 检查训练与消融实验在内的全部 18 个权重。
