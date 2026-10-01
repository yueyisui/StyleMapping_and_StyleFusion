import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import itertools
import net
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():
    transform_list = [
        transforms.Resize(512),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/home/yueyisui/YDD/dataset/COCO2014/train2014/',
                    help='Directory path to COCO2014 data-set')
# parser.add_argument('--style_dir', type=str, default='/WIKIART_dataset/', # artistic
parser.add_argument('--style_dir', type=str, default='/home/yueyisui/YDD/dataset/COCO2014/train2014/', # photorealistic
                    help='Directory path to COCO2014 data-set or Wikiart data-set')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--NDNSMDecoder', type=str, default='experiments/photorealistic/NDNSMDecoder.pth.tar')
# training options
parser.add_argument('--training_mode', default='photorealistic', # artistic or photorealistic
                    help='artistic or photorealistic')
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4) # photorealistic=5e-4 or artistic=1e-4
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1.0) # photorealistic=1.0 or artistic=10
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--ccp_weight', type=float, default=1.0) # photorealistic=1.0 or artistic=5
parser.add_argument('--l1_weight', type=float, default=20.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--num_s', type=int, default=8, help='number of sampled anchor vectors')
parser.add_argument('--num_l', type=int, default=3, help='number of layers to calculate CCPL')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
args = parser.parse_args()
# device
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
# make dir
save_dir = Path(args.save_dir + '/' + args.training_mode)
save_dir.mkdir(parents=True, exist_ok=True)
save_image_dir = Path(args.save_dir + '/' + args.training_mode + '/image')
save_image_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# load network
vgg = net.vgg
if vgg.load_state_dict(torch.load(args.vgg)):
    print('VGG model loaded successfully')
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg)
if network.NDNSMDecoder.load_state_dict(torch.load(args.NDNSMDecoder)):
    print('NDNSMDecoder model loaded successfully')
network.train()
network.to(device)
# load data
content_tf = train_transform()
style_tf = train_transform()
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
# optimizer
optimizer = torch.optim.Adam(itertools.chain(network.NDNSMDecoder.parameters(), network.mlp.parameters()), lr=args.lr)
# train
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    if args.training_mode == 'photorealistic':
        loss_c, loss_s, loss_ccp, loss_l1, gimage_01, gimage_02 = network(content_images, style_images, args.num_s, args.num_l, args.training_mode)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_ccp = args.ccp_weight * loss_ccp
        loss_l1 = args.l1_weight * loss_l1
        loss = loss_c + loss_s + loss_l1 + loss_ccp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_ccp', loss_ccp.item(), i + 1)
        writer.add_scalar('loss_l1', loss_l1.item(), i + 1)
    else:
        loss_c, loss_s, loss_ccp, gimage_01 = network(content_images, style_images, args.num_s, args.num_l, args.training_mode)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_ccp = args.ccp_weight * loss_ccp
        loss = loss_c + loss_s + loss_ccp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_ccp', loss_ccp.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.NDNSMDecoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'NDNSMDecoder_iter_{:d}.pth.tar'.format(i + 1))
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.mlp.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'mlp_iter_{:d}.pth.tar'.format(i + 1))

    if (i) % 100 == 0:
        output_name = str(save_image_dir)+'/{:s}{:s}'.format(str(i),".jpg")
        out = torch.cat((content_images,style_images,gimage_01),0)
        save_image(out, output_name)
writer.close()
