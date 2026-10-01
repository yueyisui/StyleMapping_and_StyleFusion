import argparse
from pathlib import Path
import os
import torch
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import time
from collections import OrderedDict
from tqdm import tqdm
from fusion import DWTFusionNet
from dataset_load.dataload import FusionDataset, FusionDataset_v2
import itertools
from sampler import InfiniteSamplerWrapper
from tensorboardX import SummaryWriter
import torch.nn.functional as F


def test_style_transform(h=0,w=0):
    transform_list = []
    if h != 0:
        transform_list.append(transforms.Resize((h, w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def get_parsser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--data_root', type=str, default='/data/yueyisui/datasets/DIOR/JPEGImages-1024-256_fusion/',
                        help='Directory path to a batch of data')
    parser.add_argument('--content_size', type=int, default=1024)
    parser.add_argument('--style_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='./experiments/DWTFusionNet_content_size=1024_mseloss',
                        help='Directory to save the models and log.')
    parser.add_argument('--max_iter', type=int, default=160000, help='maximum number of iterations')
    parser.add_argument('--save_ext', type=str, default='.jpg') # .jpg .JPG .png
    parser.add_argument('--save_model_interval', type=int, default=2000, help='interval between saving models')

    parser.add_argument('--gpu', type=int, default=7, help='GPU id to use')
    parser.add_argument('--batch_size', type=int, default=4, help='size of the batchsize in sampling')
    parser.add_argument('--n_threads', type=int, default=16, help='number of threads for data loading')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--load_iter', type=int, default=0, help='load the model from the specified iteration')
    parser.add_argument('--loss', type=str, default='mse')
    return parser

def train(args):
    print("-----------------------------------------------------")
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("-----------------------------------------------------")

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cs_transform = test_style_transform(args.content_size, args.content_size)
    other_transform = test_style_transform()
    dataset = FusionDataset_v2(root_dir=args.data_root, transform=other_transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    dataloader_iter = iter(torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(dataset),
        num_workers=args.n_threads))

    StyleFusion = DWTFusionNet(in_ch=3, base_ch=32)
    StyleFusion.to(device)
    StyleFusion.train()

    optimizer = torch.optim.Adam(StyleFusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    if args.loss == "mse":
        loss_f = torch.nn.MSELoss()
    elif args.loss == "l1":
        loss_f = torch.nn.L1Loss()

    print("loss is: ", args.loss)

    log_dir = os.path.join(save_dir, 'logs')
    save_image_dir = os.path.join(save_dir, 'images')
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=str(log_dir))

    # train
    progress_bar = tqdm(range(args.load_iter, args.max_iter), desc="Training", unit="iter")
    for i in progress_bar:
        data = next(dataloader_iter)
        content = data['content'].to(device)
        style = data['style'].to(device)
        content = F.interpolate(content, size=(args.content_size, args.content_size), mode='bilinear', align_corners=False)
        style = F.interpolate(style, size=(args.content_size, args.content_size), mode='bilinear', align_corners=False)
        styled_content = data['stage0'].to(device)
        grid_styled = data['stage1'].to(device) # no used
        grid_styled_merge = data['stage2'].to(device)
        output = StyleFusion(content, grid_styled_merge)

        loss = loss_f(output, styled_content)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        writer.add_scalar('loss', loss.item(), i + 1)

        progress_bar.set_postfix({
            "loss": loss.item(),
            })

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = StyleFusion.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'StyleFusion_iter_{:d}.pth.tar'.format(i + 1))

        if (i) % 100 == 0:
            output_name = str(save_image_dir)+'/{:s}{:s}'.format(str(i),".jpg")
            # style_images = torch.nn.functional.interpolate(style_images, size=(args.content_size, args.content_size), mode='bilinear', align_corners=False)
            out = torch.cat((content, style, styled_content, grid_styled_merge, output), 0)
            save_image(out, output_name, nrow=args.batch_size, normalize=True)
    writer.close()




if __name__ == '__main__':
    parser = get_parsser()
    args = parser.parse_args()
    train(args)
