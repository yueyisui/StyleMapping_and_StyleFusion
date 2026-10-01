import torch
from PIL import Image
from torchvision import transforms
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math
from tqdm import tqdm

def test_transform(size1=0, size2=0):
    transform_list = []

    if size1 != 0:
        transform_list.append(transforms.Resize((size1, size2)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def calculate_psnr(img1_array, img2_array):
    mse = np.mean((img1_array - img2_array) ** 2)
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

content_tf = test_transform() # 1216
style_tf = test_transform() # 1216
vgg_path = './models/vgg_normalised.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_resize(img, max_size, o_w=None, o_h=None):
    w, h = img.size

    if max(w, h) > max_size:
        w = int(1.0 * img.size[0] / max(img.size) * max_size)
        h = int(1.0 * img.size[1] / max(img.size) * max_size)
        img = img.resize((w, h), Image.BICUBIC)
    if o_w is not None:
        img = img.resize((o_w, o_h), Image.BICUBIC)
    return img, w, h

content_dir = 'input/content_05'
output_dir = 'output/DWTFusionNet_content_size=1024_l1+all_loss_targetsize=1024_/stage2'
# output_dir = 'output/StyleID_05/stage2'

content_images = os.listdir(content_dir)
output_images = os.listdir(output_dir)
ssim_ = 0
psnr_ = 0
count = 0
for output_image in tqdm(output_images):

    # if "_stylized_" in output_image: # _stylized_
            # content_name = output_image.split("crossfuse_")[1] #  + ".jpg"
            content_name = output_image
        # if content_name in content_images:
            # content_name = output_image
            content_path = os.path.join(content_dir, content_name)
            output_path = os.path.join(output_dir, output_image)
            content = Image.open(content_path).convert("L")
            # content, w, h = img_resize(content, 1280)
            content = style_tf(content).squeeze().numpy()
            output = Image.open(output_path).convert("L")
            # output = output.resize((800, 800), Image.NEAREST)
            # output, w, h = img_resize(output, 1280, w, h)

            output = style_tf(output).squeeze().numpy()
            ssim_index, ssim_image = ssim(content, output, data_range=1.0, full=True)
            psnr = calculate_psnr(content, output)
            psnr_ = psnr_+psnr
            count = count+1
            ssim_ = ssim_+ssim_index
ssim_ = ssim_/count
psnr_ = psnr_/count
print("ssim_:", ssim_)
print("psnr_:", psnr_)
print("count:", count)
