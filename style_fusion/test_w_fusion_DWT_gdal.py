import argparse
from pathlib import Path
import os
import torch
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
import time
from collections import OrderedDict
from tqdm import tqdm
from large_image_process.image_sample import grid_sampling_and_downscaling, batch_grid_sampling_and_downscaling, batch_grid_merging_and_restoration, batch_grid_merge_patch
from fusion import PatchCrossAttentionFusion, FusionNet, FFTFusionNet
from fusion_v2 import UltraHighResFuser, StructurePreserveFusion
from large_image_process.image_sample_v2 import batch_patch_split_pad, batch_patch_merge_pad, batch_grid_split_pixelunshuffle_pad, batch_grid_merge_pixelshuffle_pad
from fusion import DWTFusionNet
from osgeo import gdal


def test_style_transform(h=0,w=0):
    transform_list = []
    if h != 0:
        transform_list.append(transforms.Resize((h, w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='',
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='input/haikou_BJ3',
# parser.add_argument('--content_dir', type=str, default='input/content_05',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default="./input/style_05",
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='./output/haikou_BJ3_targetsize=1024',
                    help='Directory to save the output image(s)')

parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--NDNSMDecoder', type=str, default='experiments/photorealistic/NDNSMDecoder_iter_160000.pth.tar') # artistic or photorealistic
parser.add_argument('--StyleFusion', type=str, default='experiments/DWTFusionNet_content_size=1024_l1+all_loss/StyleFusion_iter_100000.pth.tar') # artistic or photorealistic

parser.add_argument('--content_size', type=int, default=512)
parser.add_argument('--style_size', type=int, default=512)
parser.add_argument('--save_ext', type=str, default='.jpg') # .jpg .JPG .png
parser.add_argument('--max_size', type=int, default=1280) # 1280

parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--keep_size', action='store_true', help='Whether to keep the size of the output the same as the input content.')


parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--gpu2', type=int, default=0, help='GPU id to use')
parser.add_argument('--grid_size', type=int, default=8, help='size of the grid for sampling')
parser.add_argument('--target_size', type=int, default=1024, help='size of the grid for sampling')
parser.add_argument('--b', type=int, default=1, help='size of the batchsize in sampling')

args = parser.parse_args()
output_path=args.output

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:"+str(args.gpu2) if torch.cuda.is_available() else "cpu")
# device = 'cpu'
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
content_paths = sorted(content_paths)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
style_paths = sorted(style_paths)

if not os.path.exists(output_path):
    os.makedirs(output_path)
stage1_output_path = os.path.join(output_path, 'stage1')
if not os.path.exists(stage1_output_path):
    os.makedirs(stage1_output_path)
stage2_output_path = os.path.join(output_path, 'stage2')
if not os.path.exists(stage2_output_path):
    os.makedirs(stage2_output_path)

# load network
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg.eval()
StyleMapping = net.Net(vgg)
StyleMapping.eval()
new_state_dict = OrderedDict()
state_dict = torch.load(args.NDNSMDecoder)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
StyleMapping.NDNSMDecoder.load_state_dict(new_state_dict)
StyleMapping.to(device)

# StyleFusion = FusionNet()
# StyleFusion = net.ImageAttentionFusion(channels=64)
StyleFusion = DWTFusionNet(in_ch=3, base_ch=32)

StyleFusion.eval()
new_state_dict = OrderedDict()
state_dict = torch.load(args.StyleFusion)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
StyleFusion.load_state_dict(new_state_dict)
StyleFusion.to(device)

fuser = UltraHighResFuser(
        model=StyleFusion,
        patch=1024,      # try 2048 if VRAM allows
        overlap=64,      # 32~128 usually works well
        amp=True,        # enable mixed precision
        batch_size=2,    # increase if VRAM allows
        device=device
    )

# load data
content_tf = test_style_transform()
# style_tf = test_style_transform(args.style_size, args.style_size)
style_tf = test_style_transform()

def img_resize(img, max_size=None, down_scale=None):
    w, h = img.size
    if down_scale!=None:
        max_size=None
    if max_size != None:
        if max(w, h) > max_size:
            w = int(1.0 * img.size[0] / max(img.size) * max_size)
            h = int(1.0 * img.size[1] / max(img.size) * max_size)
            img = img.resize((w, h), Image.BICUBIC)
    if down_scale is not None:
        w = w // down_scale * down_scale
        h = h // down_scale * down_scale
        img = img.resize((w, h), Image.BICUBIC)
    return img

# pred
for content_path in tqdm(content_paths):
    for style_path in style_paths:
        # content = Image.open(content_path).convert("RGB")
        content = gdal.Open(content_path)
        content = content.ReadAsArray()
        content = torch.from_numpy(content).float()/ 255.0
        C, W, H = content.shape
        # content = img_resize(content, 2048)
        # content = content_tf(content)
        style = (Image.open(style_path).convert("RGB"))
        style = style_tf(img_resize(style, args.style_size))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        # content_images_grid = batch_grid_sampling_and_downscaling(content, args.grid_size)
        content_images_grid, grid_size, o_size, pad_size = batch_grid_split_pixelunshuffle_pad(content, args.target_size)

        B, C, H, W = content_images_grid.size()
        b = args.b
        content_images_grid = content_images_grid.view(B // b, b, C, H, W)
        output_grid_sample = []
        with torch.no_grad():
            for content_grid in tqdm(content_images_grid):
                content_grid = content_grid.to(device)
                output  = StyleMapping.pred(content_grid, style).cpu()
                # content_grid = content_grid.cpu()
                output_grid_sample.append(output)

            output_grid_sample = torch.cat(output_grid_sample, dim=0)
            output_grid_sample = batch_grid_merge_pixelshuffle_pad(output_grid_sample, grid_size, 1, o_size, pad_size)

            output_grid_sample = output_grid_sample.to(device)
            content = content.to(device)
            # output_grid_sample2 = fuser(output_grid_sample, content)
            output_grid_sample2 = fuser(content, output_grid_sample)
            output_grid_sample2 = output_grid_sample2.squeeze(0)
            output_grid_sample = output_grid_sample.squeeze(0)

        output_name1 = '{:s}/{:s}{:s}'.format(
            stage1_output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content
        save_image(output_grid_sample, output_name1)

        output_name2 = '{:s}/{:s}{:s}'.format(
            stage2_output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content
        save_image(output_grid_sample2, output_name2)
