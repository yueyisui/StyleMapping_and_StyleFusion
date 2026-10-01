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
parser.add_argument('--content_dir', type=str, default='/media/yueyisui/file/Ubuntu/style_transfer/code/yueyisui/StyleMapping_fusion/input/DIOR_s_all',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default="/media/yueyisui/file/Ubuntu/style_transfer/code/yueyisui/StyleMapping_fusion/input/DIOR_s_all",
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output/RS_DIOR_s/DIOR_s_all',
                    help='Directory to save the output image(s)')

parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--NDNSMDecoder', type=str, default='./experiments/photorealistic_remote_sence/NDNSMDecoder_iter_160000.pth.tar') # artistic or photorealistic

parser.add_argument('--content_size', type=int, default=512)
parser.add_argument('--style_size', type=int, default=512)
parser.add_argument('--save_ext', type=str, default='.jpg') # .jpg .JPG .png
parser.add_argument('--max_size', type=int, default=1280) # 1280

parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--keep_size', action='store_true', help='Whether to keep the size of the output the same as the input content.')

args = parser.parse_args()
output_path=args.output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.makedirs(output_path)

# load network
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg.eval()
network = net.Net(vgg)
network.eval()
new_state_dict = OrderedDict()
state_dict = torch.load(args.NDNSMDecoder)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
network.NDNSMDecoder.load_state_dict(new_state_dict)
network.to(device)

# load data
content_tf = test_style_transform()
style_tf = test_style_transform(args.style_size, args.style_size)
# style_tf = test_style_transform()

def img_resize(img, max_size=None, down_scale=None):
    w, h = img.size
    if down_scale!=None:
        max_size=None
    if max_size != None:
        if max(w, h) > max_size:
            w = int(1.0 * img.size[0] / max(img.size) * max_size)
            h = int(1.0 * img.size[1] / max(img.size) * max_size)
            # w=3840
            # h=2160
            img = img.resize((w, h), Image.BICUBIC)
    if down_scale is not None:
        w = w // down_scale * down_scale
        h = h // down_scale * down_scale
        img = img.resize((w, h), Image.BICUBIC)
    return img

import time
mean_time = 0
count = 0
# pred
for content_path in tqdm(content_paths):
    for style_path in style_paths:
        content = Image.open(content_path).convert("RGB")
        W, H = content.size
        # content = img_resize(content, 1280)
        content = content_tf(content)

        style = style_tf(Image.open(style_path).convert("RGB"))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            start_time = time.time()
            output  = network.pred(content, style)
            end_time = time.time()
            execution_time = end_time - start_time
            # print(execution_time)
            mean_time = mean_time + execution_time
            count = count + 1
        output = output[0].cpu()
        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], args.save_ext
        )
        # output_name = '{:s}/{:s}{:s}'.format(
        #     output_path, splitext(basename(content_path))[0], args.save_ext) # want name to be the same as content

        save_image(output, output_name)
mean_time = mean_time/count
print('mean_time:', mean_time)
