import torch
from PIL import Image
import net
from torchvision import transforms
import os
import torch.nn as nn


def test_transform(size=0):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize((size, size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

style_tf = test_transform()
vgg_path = './models/vgg_normalised.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_dir = 'input/content'
style_dir = 'input/style'
output_dir = 'output/'
content_images = os.listdir(content_dir)
style_images = os.listdir(style_dir)
output_images = os.listdir(output_dir)

loss_c_all = 0
loss_s_all = 0
loss_g_all = 0
count = 0
for output_image in output_images:
    if "_stylized_" in output_image:
        content_name = output_image.split("_stylized_")[0] + ".jpg"
        style_name = output_image.split("_stylized_")[1]
        if content_name in content_images and style_name in style_images:
            content_path = os.path.join(content_dir, content_name)
            style_path = os.path.join(style_dir, style_name)
            output_path = os.path.join(output_dir, output_image)
            content = Image.open(content_path).convert("RGB")
            style = Image.open(style_path).convert("RGB")
            output = Image.open(output_path).convert("RGB")
            content = style_tf(content)
            style = style_tf(style)
            output = style_tf(output)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            output = output.to(device).unsqueeze(0)
            vgg = net.vgg
            vgg.load_state_dict(torch.load(vgg_path))
            vgg = nn.Sequential(*list(vgg.children())[:31])
            vgg.eval()
            network = net.Net(vgg)
            network.eval()
            network.to(device)
            loss_c, loss_s, loss_g = network.get_image_loss(content, style, output)
            count = count+1
            loss_c_all = loss_c_all+loss_c
            loss_s_all = loss_s_all+loss_s
            loss_g_all = loss_g_all+loss_g
loss_c_all = loss_c_all/count
loss_s_all = loss_s_all/count
loss_g_all = loss_g_all/count
print("loss_c_all:", loss_c_all)
print("loss_s_all:", loss_s_all)
print("loss_g_all: {:.5f}".format(loss_g_all))
print("count:", count)
