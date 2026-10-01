import torch
import lpips
import torchvision.transforms as transforms
from PIL import Image
import os
import pathlib
from tqdm import tqdm


# Load pre-trained LPIPS model
loss_fn = lpips.LPIPS(net='alex')

# Load and preprocess the images
transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
])


def cal_lpips(path1, path2, suffix):
    path1 = pathlib.Path(path1)
    files1 = sorted(list(path1.glob('*.%s' %suffix)))

    path2 = pathlib.Path(path2)
    files2 = sorted(list(path2.glob('*.%s' %suffix)))

    lpips_values = 0
    count = 0
    size1 = len(files1)
    size2 = len(files2)
    temp = size2//size1

    for i in tqdm(range(size1)):
        image_x = Image.open(files1[i]).convert('RGB')
        image_x = transform(image_x).unsqueeze(0)  # Assuming image_x is the first image
        for j in range(temp):
            image_y = Image.open(files2[j+i*temp]).convert('RGB')
            image_y = transform(image_y).unsqueeze(0)  # Assuming image_y is the second image
            # Compute LPIPS
            with torch.no_grad():
                lpips_score = loss_fn(image_x, image_y).item()
            count += 1
            lpips_values += lpips_score
    lpips_values = lpips_values/count
    return lpips_values


if __name__ == "__main__":
    def test_transform(size=0):
        transform_list = []

        if size != 0:
            transform_list.append(transforms.Resize((size, size)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    style_tf = test_transform(512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)

    content_dir = 'input/content_04_10'
    style_dir = 'input/style_04_10'
    output_dir = 'output/NDNSM22_04_10'
    content_images = os.listdir(content_dir)
    style_images = os.listdir(style_dir)
    output_images = os.listdir(output_dir)

    lpips_c_score_all = 0
    lpips_s_score_all = 0
    count = 0
    for output_image in tqdm(output_images[:100]):
        if "_stylized_" in output_image:
            content_name = output_image.split("_stylized_")[0] + ".jpg"
            style_name = output_image.split("_stylized_")[1].split(".")[0] + ".jpg"
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
            with torch.no_grad():
                lpips_c_score = loss_fn(content, output).item()
                lpips_s_score = loss_fn(style, output).item()
            lpips_c_score_all = lpips_c_score_all+lpips_c_score
            lpips_s_score_all = lpips_s_score_all+lpips_s_score
            count = count+1
    lpips_c_score_all = lpips_c_score_all/count
    lpips_s_score_all = lpips_s_score_all/count
    print("lpips_c_score_all:", lpips_c_score_all)
    print("lpips_s_score_all:", lpips_s_score_all)
    print("count:", count)
