# StyleMapping and StyleFusion

这是论文 **A Universal Framework for Remote Sensing Image Color Correction via
Style Transfer**（IEEE TGRS 2026，DOI:
[10.1109/TGRS.2026.3702755](https://doi.org/10.1109/TGRS.2026.3702755)）对应的整理版工程。

项目将两个原始研究仓库合并为一条完整的遥感影像颜色校正流程：

1. **StyleMapping**：使用参考影像的颜色风格对内容影像进行光真实风格迁移；
2. **Pixel unshuffle / shuffle**：以全局重排代替常规切块，处理超高分辨率影像并减少接缝；
3. **StyleFusion**：通过 DWT 子带融合，将原影像结构细节与风格迁移结果自适应融合。

对于中等分辨率影像，可以仅运行 StyleMapping；对于高分辨率遥感影像，推荐运行完整流程。

## 项目结构

```text
.
├── inference.py                 # 推荐：统一推理入口
├── style_mapping/               # StyleMapping 训练、推理和评估代码
├── style_fusion/                # StyleFusion 训练、推理和评估代码
├── assets/                      # 从论文提取的流程图与定性结果
├── examples/                    # 最小内容图/参考图示例
├── requirements.txt
├── scripts/check_weights.py     # 下载权重完整性检查
└── README.md                    # GitHub 英文主页
```

两个原项目只作为复制源使用，源目录没有被修改。整合仓库保留论文复现所需的训练、
推理和评估代码，不包含模型权重、历史输出、训练数据及临时实验文件。所有本地权重
均已加入 Git 忽略规则，公开仓库通过独立下载链接提供模型。

## 环境安装

论文实验环境为 Ubuntu 22.04、单张 NVIDIA RTX 3090。原 StyleMapping
依赖记录为 PyTorch 1.13.1 / torchvision 0.14.1（CUDA 11.7）。建议使用 Python
3.8–3.10 创建独立环境（统一入口兼容 Python 3.8+）：

```bash
conda create -n stylemapping python=3.10 -y
conda activate stylemapping
pip install -r requirements.txt
```

如果需要运行 `style_fusion/test_w_fusion_DWT_gdal.py` 读取 GeoTIFF，还需按本机
GDAL 版本安装 `gdal`/`osgeo`。统一入口 `inference.py` 使用 Pillow 读取常见的
RGB、TIFF 影像，不依赖 GDAL，也不会保留 GeoTIFF 的地理参考元数据。

## 快速推理

完整的高分辨率颜色校正流程：

```bash
python inference.py \
  --content examples/content.jpg \
  --style examples/style.jpg \
  --output outputs/example \
  --device cuda:0
```

结果分为两级：

- `stage1_style_mapping/`：pixel shuffle 重建后的 StyleMapping 结果；
- `stage2_style_fusion/`：融合结构细节后的最终颜色校正结果。

CPU 也可运行，但速度较慢：

```bash
python inference.py \
  --content examples/content.jpg \
  --style examples/style.jpg \
  --output outputs/example_cpu \
  --device cpu
```

仅运行 StyleMapping：

```bash
python inference.py \
  --content /path/to/content \
  --style /path/to/reference.jpg \
  --output outputs/mapping_only \
  --mapping-only
```

`--content` 和 `--style` 都可接收单张影像或目录。若两者均为目录，程序使用笛卡尔积
组合。常用参数如下：

- `--target-size 1024`：pixel unshuffle 后每张子影像的目标尺寸；设为 `0` 可禁用该步骤；
- `--style-size 512`：参考影像送入 StyleMapping 前的尺寸；
- `--mapping-batch-size`：StyleMapping 子影像批大小；显存不足时设为 `1`；
- `--fusion-patch 1024`、`--fusion-overlap 64`：高分辨率 StyleFusion 的分块与重叠；
- `--decoder`、`--fusion-weight`：切换实验权重。

默认使用遥感微调后的 StyleMapping 解码器和论文对应的 1024 尺度 DWT
StyleFusion 权重。

## 权重

权重不随 Git 仓库上传。公开下载链接目前尚未提供，下载状态与安装方法见
[预训练权重说明](PRETRAINED_MODELS.md)。获取权重压缩包后，在项目根目录执行：

```bash
tar -xzf /path/to/stylemapping-stylefusion-checkpoints.tar.gz -C .
python scripts/check_weights.py
```

解压后，主要推理权重应位于以下路径：

| 模块 | 默认路径 | 用途 |
|---|---|---|
| VGG-19 | `style_fusion/models/vgg_normalised.pth` | 多尺度特征编码器 |
| StyleMapping | `style_fusion/experiments/photorealistic_remote_sence/NDNSMDecoder_iter_160000.pth.tar` | 遥感影像风格映射 |
| StyleFusion | `style_fusion/experiments/DWTFusionNet_content_size=1024_l1+all_loss/StyleFusion_iter_100000.pth.tar` | DWT 风格—结构融合 |

完整权重包还包含自然影像/COCO 阶段、遥感微调阶段和消融实验的最终 checkpoint。
运行 `python scripts/check_weights.py --all` 可校验全部 18 个权重文件。该仓库只提交
代码、文档及示例图片，不需要使用 Git LFS。

## 训练 StyleMapping

原论文使用 MS COCO 同时作为内容图与风格图来源，随机缩放裁剪至 256×256，
batch size 为 8，共训练 160,000 次。训练入口位于 `style_mapping/train.py`：

```bash
cd style_mapping
python train.py \
  --content_dir /path/to/coco/train2014 \
  --style_dir /path/to/coco/train2014 \
  --vgg models/vgg_normalised.pth \
  --NDNSMDecoder experiments/photorealistic/NDNSMDecoder.pth.tar \
  --training_mode photorealistic \
  --save_dir experiments/coco_finetune \
  --log_dir logs/coco_finetune \
  --lr 1e-4 \
  --lr_decay 6.25e-6 \
  --batch_size 8 \
  --max_iter 160000 \
  --gpu 0
```

原脚本的光真实模式默认学习率为 `5e-4`，与论文报告的 `1e-4` 不同；上面的命令已
显式设置 `1e-4`，并按脚本的反比例调度形式令第 160,000 次附近约为 `5e-5`。

遥感数据微调时，将 `--content_dir`、`--style_dir` 改为遥感数据目录，并将
`--NDNSMDecoder` 指向 COCO 阶段权重。训练脚本会在 `--save_dir/<training_mode>/`
下保存 `NDNSMDecoder` 和用于 CCPL 的 MLP 权重。

## 训练 StyleFusion

论文使用 DIOR 数据集，StyleFusion 输入尺寸为 1024×1024，batch size 为 4，
学习率为 1e-4，共训练 100,000 次。先在 `style_fusion/` 中生成以下配对目录：

```text
fusion_dataset/
├── content/  # 原始高分辨率内容图
├── style/    # 参考图
├── stage0/   # 直接在高分辨率上运行 StyleMapping 的监督目标
├── stage1/   # pixel-unshuffle 子图的 StyleMapping 结果
└── stage2/   # pixel-shuffle 重建结果
```

`style_fusion/test_get_remote_sence_datasets.py` 是原始数据生成脚本；其中保留了旧机器的
默认路径，使用时务必通过命令行显式传入数据路径。随后执行：

```bash
cd style_fusion
python train_all_loss.py \
  --data_root /path/to/fusion_dataset \
  --save_dir experiments/DWTFusionNet_content_size=1024_l1+all_loss_new \
  --vgg models/vgg_normalised.pth \
  --content_size 1024 \
  --loss l1 \
  --batch_size 4 \
  --max_iter 100000 \
  --save_model_interval 5000 \
  --gpu 0
```

## 原始研究脚本

两个子目录保留了各阶段的基础 `test.py`、训练脚本和评价指标（SSIM、LPIPS、SIFID）。
`style_fusion/test_w_fusion_DWT.py` 是原始两阶段完整流程脚本；同时保留了 GDAL 和
仅运行 StyleFusion 的专项测试，便于复现实验和追溯实现。部分脚本仍保留原作者机器上
的默认数据路径；推荐优先使用根目录 `inference.py`，运行研究脚本时显式覆盖所有
输入输出参数。

## 论文引用

```bibtex
@article{yue2026universal,
  title={A Universal Framework for Remote Sensing Image Color Correction via Style Transfer},
  author={Yue, Dongdong and Liu, Xinyi and Fan, Weiwei and Zhong, Jiachen and Liu, Xiaoan and Zhang, Yongjun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  year={2026},
  doi={10.1109/TGRS.2026.3702755}
}
```

## 许可证与合规说明

本仓库源代码采用 [MIT License](../LICENSE)。第三方数据集、预训练权重、示例影像和
论文插图不自动适用 MIT License，仍须遵循各自来源的许可证及使用条款。仓库不附带
论文 PDF，论文全文请通过 README 中的 DOI 获取。
