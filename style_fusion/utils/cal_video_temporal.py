import cv2
import numpy as np
import os

import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import glob
import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()


def warp(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img.astype(np.float32), flow.astype(np.float32), None, cv2.INTER_LINEAR)
    return res


def crop_to_multiple_of_eight(image):
    height, width = image.shape[:2]
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    cropped_image = image[:new_height, :new_width]
    return cropped_image


def heatmap(path1, path2, forward_flow, mask=None):
    FirstFrame = cv2.imread(path1).astype(np.float32) / 255.0
    SecondFrame = cv2.imread(path2).astype(np.float32) / 255.0
    FirstFrame = crop_to_multiple_of_eight(FirstFrame)
    SecondFrame = crop_to_multiple_of_eight(SecondFrame)
    if forward_flow is not None:
        forward_flow = cv2.readOpticalFlow(forward_flow)
    else:
        flow_imgs, predicted_flows = raft_flows(FirstFrame, SecondFrame)
        forward_flow = predicted_flows.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    warpped_pre_frame = warp(SecondFrame, -forward_flow)

    if mask is not None:
        mask = cv2.imread(mask)
        mask = 1-mask / 255.
        H, W = FirstFrame.shape[0], FirstFrame.shape[1]
        mask *= warp(np.ones([H, W, 3]), -forward_flow)
        warpped_pre_frame = warpped_pre_frame*mask
        FirstFrame = FirstFrame*mask

    # calculate temporal error
    error = abs(FirstFrame - warpped_pre_frame)
    error = np.mean(error, axis=2, keepdims=True)
    temporal_loss = np.mean(error)
    print(f'Temporal Loss: {temporal_loss}')

    # set figure
    figsize = (int(error.shape[1]//100)+1, int(error.shape[0]//100)+1)
    fig = plt.figure(figsize=figsize)   # figsize=[width, height] in inches. num_pixel=figsize*100
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # draw heatmap
    sns.heatmap(error[:, :, 0], cmap="hot", cbar=False)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

    w, h = fig.canvas.get_width_height()
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = image.resize((error.shape[1], error.shape[0]), Image.NEAREST)

    image = np.asarray(image)[:, :, :3]
    image = image[:, :, ::-1]
    plt.close()
    return image, temporal_loss


def raft_flows(frame1, frame2):
    frame1 = torch.tensor(frame1).unsqueeze(0)
    frame2 = torch.tensor(frame2).unsqueeze(0)
    frame1 = frame1.permute(0, 3, 1, 2)
    frame2 = frame2.permute(0, 3, 1, 2)
    img1_batch = torch.stack([frame1])
    img2_batch = torch.stack([frame2])
    img1_batch, img2_batch = preprocess(frame1, frame2)

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = list_of_flows[-1]
    flow_imgs = flow_to_image(predicted_flows)
    return flow_imgs, predicted_flows

def preprocess(img1_batch, img2_batch):
    # img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    # img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path=''
    output_path = 'output/temporal'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_list = glob.glob(os.path.join(path, '*'))
    path_list.sort()
    iamge_num = len(path_list)
    Temporal_loss_all = 0
    temp_i = 10
    for i in range(iamge_num-temp_i):
        first_frame_dir = path_list[i]
        second_frame_dir = path_list[i+temp_i]
        output_name = os.path.basename(path_list[i])[:-4] + '_and_' + os.path.basename(path_list[i+temp_i])[:-4] + '.png'
        output_dir = os.path.join(output_path, output_name)
        image, image_temporal_loss = heatmap(first_frame_dir, second_frame_dir, forward_flow=None, mask=None)
        cv2.imwrite(output_dir, image)
        Temporal_loss_all = Temporal_loss_all + image_temporal_loss

    Temporal_loss_mean = Temporal_loss_all/(iamge_num-temp_i)
    print('Temporal_loss_mean', Temporal_loss_mean)
