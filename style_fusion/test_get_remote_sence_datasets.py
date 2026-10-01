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
from fusion import PatchCrossAttentionFusion, FusionNet, GlobalLocalFusion
from large_image_process.image_sample_v2 import batch_patch_split_pad, batch_patch_merge_pad, batch_grid_split_pixelunshuffle_pad, batch_grid_merge_pixelshuffle_pad
import random
import shutil

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
parser.add_argument('--content_dir', type=str, default='/data/yueyisui/datasets/DIOR/JPEGImages-trainval/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default="/data/yueyisui/datasets/DIOR/JPEGImages-trainval/",
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='/data/yueyisui/datasets/DIOR/JPEGImages-2048-256_fusion',
                    help='Directory to save the output image(s)')

parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--NDNSMDecoder', type=str, default='experiments/photorealistic_remote_sence/NDNSMDecoder_iter_160000.pth.tar') # artistic or photorealistic
parser.add_argument('--StyleFusion', type=str, default='experiments/photorealistic_remote_sence_fusion_targetsize=256_l1_2/StyleFusion_iter_160000.pth.tar') # artistic or photorealistic

parser.add_argument('--content_size', type=int, default=2048)
parser.add_argument('--style_size', type=int, default=256)
parser.add_argument('--save_ext', type=str, default='.jpg') # .jpg .JPG .png
parser.add_argument('--max_size', type=int, default=1280) # 1280

parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--keep_size', action='store_true', help='Whether to keep the size of the output the same as the input content.')


parser.add_argument('--gpu', type=int, default=5, help='GPU id to use')
parser.add_argument('--grid_size', type=int, default=4, help='size of the grid for sampling')
parser.add_argument('--target_size', type=int, default=256, help='size of the grid for sampling')
parser.add_argument('--b', type=int, default=4, help='size of the batchsize in sampling')

args = parser.parse_args()
output_path=args.output

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

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

stage0_output_path = os.path.join(output_path, 'stage0')
if not os.path.exists(stage0_output_path):
    os.makedirs(stage0_output_path)

stage1_output_path = os.path.join(output_path, 'stage1')
if not os.path.exists(stage1_output_path):
    os.makedirs(stage1_output_path)

stage2_output_path = os.path.join(output_path, 'stage2')
if not os.path.exists(stage2_output_path):
    os.makedirs(stage2_output_path)

content_output_path = os.path.join(output_path, 'content')
if not os.path.exists(content_output_path):
    os.makedirs(content_output_path)

style_output_path = os.path.join(output_path, 'style')
if not os.path.exists(style_output_path):
    os.makedirs(style_output_path)

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

if StyleMapping.NDNSMDecoder.load_state_dict(new_state_dict):
    print("load NDNSMDecoder successfully!")
StyleMapping.to(device)

# load data
content_tf = test_style_transform(args.content_size, args.content_size)
style_tf = test_style_transform(args.style_size, args.style_size)

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
random.seed(42)
for content_path in tqdm(content_paths):
        style_path = random.choice(style_paths)
        content_filename = os.path.basename(content_path)
        style_output = os.path.join(style_output_path, content_filename)
        shutil.copy2(content_path, content_output_path)
        shutil.copy2(style_path, style_output)

        content = Image.open(content_path).convert("RGB")
        W, H = content.size
        # content = img_resize(content, 2048)
        content = content_tf(content)
        style = style_tf(Image.open(style_path).convert("RGB"))
        with torch.no_grad():
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            output_stage_o = StyleMapping.pred(content, style).detach() # warm up
            content_images_grid, grid_size, o_size, pad_size = batch_grid_split_pixelunshuffle_pad(content, args.target_size)
        # with torch.no_grad():
            output_grid_sample  = StyleMapping.pred(content_images_grid, style).detach()
            output = batch_grid_merge_pixelshuffle_pad(output_grid_sample, grid_size, 1, o_size, pad_size)
            output = output.squeeze(0)

        output_name0 = '{:s}/{:s}{:s}'.format(
            stage0_output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content
        save_image(output_stage_o, output_name0, padding=0)

        output_name1 = '{:s}/{:s}{:s}'.format(
            stage1_output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content
        save_image(output_grid_sample, output_name1, padding=0, nrow=args.grid_size)

        output_name2 = '{:s}/{:s}{:s}'.format(
            stage2_output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content
        save_image(output, output_name2, padding=0)
