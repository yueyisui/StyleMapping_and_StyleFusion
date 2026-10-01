import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.function import calc_mean_std, nor_mean_std, nor_mean, gram_matrix, calc_mask_mean_std
import random
from large_image_process.image_sample import batch_grid_merging_and_restoration, grid_merging_and_restoration, batch_grid_sampling_and_downscaling
from large_image_process.image_sample import batch_grid_merge_pixelshuffle, batch_grid_split_pixelunshuffle, batch_grid_patch, batch_grid_merge_patch
l1_loss_fn = nn.L1Loss()

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)])


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d =self.mlp[3*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    ## PatchNCELoss code
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp

def get_style_image(image_tensor, grid_size):
    num_patches_per_image = grid_size * grid_size
    # repeat the style image tensor to match the number of patches
    style_tensor_result = image_tensor.repeat_interleave(num_patches_per_image, dim=0)
    return style_tensor_result


class ImageAttentionFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv_feat = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.attn_conv = nn.Conv2d(2*channels, 1, kernel_size=1)  # 生成注意力权重
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, image1, image2):
        # 提取特征
        F1 = self.conv_feat(image1)
        F2 = self.conv_feat(image2)

        # 生成注意力权重
        attn = torch.sigmoid(self.attn_conv(torch.cat([F1, F2], dim=1)))

        # 融合
        F_fused = attn * F1 + (1 - attn) * F2

        # 输出融合图像
        fused_image = self.conv_out(F_fused)
        return fused_image


class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        self.NDNSMDecoder = NDNSMDecoder([64,128,256,512], [64,128,256,512])
        # self.NDNSMDecoder = NDNSMDecoder2()

        self.mlp = mlp
        self.CCPL = CCPL(self.mlp)
        self.mse_loss = nn.MSELoss()
        self.end_layer = 4

        self.fusion = ImageAttentionFusion(channels=64)

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            feature = func(results[-1])
            results.append(feature)
        return results[1:]


    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input


    def feature_compress(self, feat):
        feat = feat.flatten(2,3)
        feat = self.mlp(feat)
        feat = feat.flatten(1,2)
        feat = Normalize(2)(feat)
        return feat


    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)


    def calc_mask_content_loss(self, input, target, mask=None):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input = input*mask
        target = target*mask
        loss = self.mse_loss(input, target)
        return loss


    def calc_style_loss(self, input, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_gram_loss(self, input, target):
        input_gram = gram_matrix(input)
        target_gram = gram_matrix(target)
        loss = F.mse_loss(input_gram, target_gram)
        return loss

    def calc_mask_style_loss(self, input, target, input_mask=None, target_mask=None):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        if input_mask == None:
            input_mean, input_std = calc_mean_std(input)
        else:
            input_mean, input_std = calc_mask_mean_std(input, input_mask)
        if target_mask == None:
            target_mean, target_std = calc_mean_std(target)
        else:
            target_mean, target_std = calc_mask_mean_std(target, target_mask)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    def encode_with_intermediate_mask(self, input, mask):
        results = [input]
        mask = mask.float()
        mask = mask.unsqueeze(1)
        masks = [mask]
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            feature = func(results[-1])
            mask = F.interpolate(mask, size=(feature.shape[-2], feature.shape[-1]), mode='nearest')
            feature = feature*mask
            masks.append(mask)
            mask = pool(mask)

            results.append(feature)
        return results[1:], masks[1:]

    def cal_all_loss(self, g_t_feats, content_feats, style_feats, end_layer, num_s, num_layer, content_mask=None, style_mask=None):
        if (content_mask==None and style_mask==None):
            loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1])
            loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, end_layer):
                loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        elif (content_mask!=None and style_mask==None):
            loss_c = self.calc_mask_content_loss(g_t_feats[-1], content_feats[-1], content_mask[-1])
            loss_s = self.calc_mask_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, end_layer):
                loss_s += self.calc_mask_style_loss(g_t_feats[i], style_feats[i])
        elif (content_mask==None and style_mask!=None):
            loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1])
            loss_s = self.calc_mask_style_loss(g_t_feats[0], style_feats[0], target_mask=style_mask[0])
            for i in range(1, end_layer):
                loss_s += self.calc_mask_style_loss(g_t_feats[i], style_feats[i], target_mask=style_mask[i])
        else:
            loss_c = self.calc_mask_content_loss(g_t_feats[-1], content_feats[-1], content_mask[-1])
            loss_s = self.calc_mask_style_loss(g_t_feats[0], style_feats[0], content_mask[0], style_mask[0])
            for i in range(1, end_layer):
                loss_s += self.calc_mask_style_loss(g_t_feats[i], style_feats[i], content_mask[i], style_mask[i])
        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)
        return loss_c, loss_s, loss_ccp


    def cal_content_and_style_loss(self, g_t_feats, content_feats, style_feats, end_layer=4):
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1])
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s,

    def forward_11(self, content, style, content_grid, syle_grid, num_s, num_layer, train_mode, grid_size, batch_size):
        syle_grid_feats = self.encode_with_intermediate(syle_grid)
        content_grid_feats = self.encode_with_intermediate(content_grid)
        style_feats = []
        for f in syle_grid_feats:
            style_feats.append(get_style_image(f, grid_size))
        content_pixelshuffle_feats = []
        content_patch_feats = []
        for f in content_grid_feats:
            content_pixelshuffle_feats.append(batch_grid_split_pixelunshuffle(f, grid_size))
            content_patch_feats.append(batch_grid_patch(f, grid_size))


        if train_mode == 'photorealistic':
            gimage_pixelshuffle_01 = self.NDNSMDecoder(content_pixelshuffle_feats, style_feats)
            gimage_pixelshuffle_02 = self.NDNSMDecoder(content_pixelshuffle_feats, content_pixelshuffle_feats)
            gimage_patch_01 = self.NDNSMDecoder(content_patch_feats, style_feats)
            gimage_patch_02 = self.NDNSMDecoder(content_patch_feats, content_patch_feats)

            gimage_pixelshuffle_01_all = batch_grid_merge_pixelshuffle(gimage_pixelshuffle_01, grid_size, batch_size)
            gimage_pixelshuffle_02_all = batch_grid_merge_pixelshuffle(gimage_pixelshuffle_02, grid_size, batch_size)
            gimage_patch_01_all = batch_grid_merge_patch(gimage_patch_01, grid_size, batch_size)
            gimage_patch_02_all = batch_grid_merge_patch(gimage_patch_02, grid_size, batch_size)

            gimage_01_all = self.fusion(gimage_pixelshuffle_01_all, gimage_patch_01_all)
            gimage_02_all = self.fusion(gimage_pixelshuffle_02_all, gimage_patch_02_all)
            g_t_01_all_feats = self.encode_with_intermediate(gimage_01_all)
            g_t_02_all_feats = self.encode_with_intermediate(gimage_02_all)

            loss_c_01_all, loss_s_01_all, loss_ccp_01_all = self.cal_all_loss(g_t_01_all_feats, content_grid_feats, syle_grid_feats, self.end_layer, num_s, num_layer)
            loss_c_02_all, loss_s_02_all, loss_ccp_02_all = self.cal_all_loss(g_t_02_all_feats, content_grid_feats, content_grid_feats, self.end_layer, num_s, num_layer)
            loss_l1_all = l1_loss_fn(gimage_02_all, content_grid)

            loss_c = loss_c_01_all + loss_c_02_all
            loss_s = loss_s_01_all + loss_s_02_all
            loss_ccp = loss_ccp_01_all + loss_ccp_02_all
            loss_l1 = loss_l1_all
            return loss_c, loss_s, loss_ccp, loss_l1, gimage_01_all, gimage_02_all

    def forward12(self, content, style, content_grid, syle_grid, num_s, num_layer, train_mode, grid_size, batch_size):
        # syle_grid_feats = self.encode_with_intermediate(syle_grid)
        # content_grid_feats = self.encode_with_intermediate(content_grid)
        # style_feats = []
        # for f in syle_grid_feats:
        #     style_feats.append(get_style_image(f, grid_size))
        # content_pixelshuffle_feats = []
        # content_patch_feats = []

        content_patch = batch_grid_patch(content_grid, grid_size)
        content_patch_feats = self.encode_with_intermediate(content_patch)

        content_pixelshuffle_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)

        content_grid_feats = self.encode_with_intermediate(content_grid)
        syle_grid_feats = self.encode_with_intermediate(syle_grid)


        if train_mode == 'photorealistic':
            gimage_pixelshuffle_01 = self.NDNSMDecoder(content_pixelshuffle_feats, style_feats)
            gimage_pixelshuffle_02 = self.NDNSMDecoder(content_pixelshuffle_feats, content_pixelshuffle_feats)
            gimage_patch_01 = self.NDNSMDecoder(content_patch_feats, style_feats)
            gimage_patch_02 = self.NDNSMDecoder(content_patch_feats, content_patch_feats)

            gimage_pixelshuffle_01_all = batch_grid_merging_and_restoration(gimage_pixelshuffle_01, grid_size, batch_size)
            gimage_pixelshuffle_02_all = batch_grid_merging_and_restoration(gimage_pixelshuffle_02, grid_size, batch_size)
            gimage_patch_01_all = batch_grid_merge_patch(gimage_patch_01, grid_size, batch_size)
            gimage_patch_02_all = batch_grid_merge_patch(gimage_patch_02, grid_size, batch_size)

            gimage_01_all = self.fusion(gimage_pixelshuffle_01_all, gimage_patch_01_all)
            gimage_02_all = self.fusion(gimage_pixelshuffle_02_all, gimage_patch_02_all)
            g_t_01_all_feats = self.encode_with_intermediate(gimage_01_all)
            g_t_02_all_feats = self.encode_with_intermediate(gimage_02_all)

            loss_c_01_all, loss_s_01_all, loss_ccp_01_all = self.cal_all_loss(g_t_01_all_feats, content_grid_feats, syle_grid_feats, self.end_layer, num_s, num_layer)
            loss_c_02_all, loss_s_02_all, loss_ccp_02_all = self.cal_all_loss(g_t_02_all_feats, content_grid_feats, content_grid_feats, self.end_layer, num_s, num_layer)
            loss_l1_all = l1_loss_fn(gimage_02_all, content_grid)

            loss_c = loss_c_01_all + loss_c_02_all
            loss_s = loss_s_01_all + loss_s_02_all
            loss_ccp = loss_ccp_01_all + loss_ccp_02_all
            loss_l1 = loss_l1_all
            return loss_c, loss_s, loss_ccp, loss_l1, gimage_01_all, gimage_02_all



    def forward(self, content, style, content_grid, syle_grid, num_s, num_layer, train_mode, grid_size, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        if train_mode == 'photorealistic':
            gimage_01 = self.NDNSMDecoder(content_feats, style_feats)
            gimage_02 = self.NDNSMDecoder(content_feats, content_feats)
            g_t_01_feats = self.encode_with_intermediate(gimage_01)
            g_t_02_feats = self.encode_with_intermediate(gimage_02)
            loss_c_01, loss_s_01, loss_ccp_01 = self.cal_all_loss(g_t_01_feats, content_feats, style_feats, self.end_layer, num_s, num_layer)
            loss_c_02, loss_s_02, loss_ccp_02 = self.cal_all_loss(g_t_02_feats, content_feats, content_feats, self.end_layer, num_s, num_layer)
            loss_l1_ = l1_loss_fn(gimage_02, content)

            gimage_01_all = batch_grid_merging_and_restoration(gimage_01, grid_size, batch_size)
            gimage_02_all = batch_grid_merging_and_restoration(gimage_02, grid_size, batch_size)

            content_grid_feats = self.encode_with_intermediate(content_grid)
            syle_grid_feats = self.encode_with_intermediate(syle_grid)

            gimage_01_all = self.fusion(gimage_01_all, content_grid)
            gimage_02_all = self.fusion(gimage_02_all, content_grid)

            g_t_01_all_feats = self.encode_with_intermediate(gimage_01_all)
            g_t_02_all_feats = self.encode_with_intermediate(gimage_02_all)

            loss_c_01_all, loss_s_01_all, loss_ccp_01_all = self.cal_all_loss(g_t_01_all_feats, content_grid_feats, syle_grid_feats, self.end_layer, num_s, num_layer)
            loss_c_02_all, loss_s_02_all, loss_ccp_02_all = self.cal_all_loss(g_t_02_all_feats, content_grid_feats, content_grid_feats, self.end_layer, num_s, num_layer)
            loss_l1_all = l1_loss_fn(gimage_02_all, content_grid)
            loss_c = loss_c_01 + loss_c_02 + loss_c_01_all + loss_c_02_all
            loss_s = loss_s_01 + loss_s_02 + loss_s_01_all + loss_s_02_all
            loss_ccp = loss_ccp_01 + loss_ccp_02 + loss_ccp_01_all + loss_ccp_02_all
            loss_l1 = loss_l1_ + loss_l1_all

            return loss_c, loss_s, loss_ccp, loss_l1, gimage_01_all, gimage_02_all
        else:
            gimage_01 = self.NDNSMDecoder(content_feats, style_feats)
            g_t_01_feats = self.encode_with_intermediate(gimage_01)
            loss_c, loss_s, loss_ccp = self.cal_all_loss(g_t_01_feats, content_feats, style_feats, self.end_layer, num_s, num_layer)
            return loss_c, loss_s, loss_ccp, gimage_01


    def forward_10(self, content, style, content_grid, syle_grid, num_s, num_layer, train_mode, grid_size, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        if train_mode == 'photorealistic':
            gimage_01 = self.NDNSMDecoder(content_feats, style_feats)
            gimage_02 = self.NDNSMDecoder(content_feats, content_feats)

            # g_t_01_feats = self.encode_with_intermediate(gimage_01)
            # g_t_02_feats = self.encode_with_intermediate(gimage_02)
            # loss_c_01, loss_s_01, loss_ccp_01 = self.cal_all_loss(g_t_01_feats, content_feats, style_feats, self.end_layer, num_s, num_layer)
            # loss_c_02, loss_s_02, loss_ccp_02 = self.cal_all_loss(g_t_02_feats, content_feats, content_feats, self.end_layer, num_s, num_layer)
            # loss_l1 = l1_loss_fn(gimage_02, content)
            # loss_c = loss_c_01 + loss_c_02
            # loss_s = loss_s_01 + loss_s_02
            # loss_ccp = loss_ccp_01 + loss_ccp_02

            gimage_01_all = batch_grid_merging_and_restoration(gimage_01, grid_size, batch_size)
            gimage_02_all = batch_grid_merging_and_restoration(gimage_02, grid_size, batch_size)

            content_grid_feats = self.encode_with_intermediate(content_grid)
            syle_grid_feats = self.encode_with_intermediate(syle_grid)

            gimage_01_all = self.fusion(gimage_01_all, content_grid)
            gimage_02_all = self.fusion(gimage_02_all, content_grid)

            g_t_01_all_feats = self.encode_with_intermediate(gimage_01_all)
            g_t_02_all_feats = self.encode_with_intermediate(gimage_02_all)
            loss_c_01_all, loss_s_01_all, loss_ccp_01_all = self.cal_all_loss(g_t_01_all_feats, content_grid_feats, syle_grid_feats, self.end_layer, num_s, num_layer)
            loss_c_02_all, loss_s_02_all, loss_ccp_02_all = self.cal_all_loss(g_t_02_all_feats, content_grid_feats, content_grid_feats, self.end_layer, num_s, num_layer)
            loss_l1_all = l1_loss_fn(gimage_02_all, content_grid)

            loss_c = loss_c_01_all + loss_c_02_all
            loss_s = loss_s_01_all + loss_s_02_all
            loss_ccp = loss_ccp_01_all + loss_ccp_02_all
            loss_l1 = loss_l1_all
            return loss_c, loss_s, loss_ccp, loss_l1, gimage_01_all, gimage_02_all
        else:
            gimage_01 = self.NDNSMDecoder(content_feats, style_feats)
            g_t_01_feats = self.encode_with_intermediate(gimage_01)
            loss_c, loss_s, loss_ccp = self.cal_all_loss(g_t_01_feats, content_feats, style_feats, self.end_layer, num_s, num_layer)
            return loss_c, loss_s, loss_ccp, gimage_01


    def pred(self, content, style):
        style_features = self.encode_with_intermediate(style)
        content_features = self.encode_with_intermediate(content)
        gimage = self.NDNSMDecoder(content_features, style_features)
        return gimage

    def style_mask_pred(self, content, style, style_mask):
        if style_mask.dim() == 4:
            style_mask = style_mask.squeeze(0)
        style_features = self.encode_with_intermediate_mask(style, style_mask)
        content_features = self.encode_with_intermediate(content)
        gimage = self.NDNSMDecoder(content_features, style_features)
        return gimage

    def style_and_contnet_mask_pred(self, content, style, content_mask, style_mask):
        if content_mask.dim() == 4:
            content_mask = content_mask.squeeze(0)
        if style_mask.dim() == 4:
            style_mask = style_mask.squeeze(0)
        style_features, style_masks = self.encode_with_intermediate_mask(style, style_mask)
        content_features, content_masks = self.encode_with_intermediate_mask(content, content_mask)
        back_style_mask = -style_mask+1
        back_content_mask = -content_mask+1
        back_style_features, back_style_masks = self.encode_with_intermediate_mask(style, back_style_mask)
        back_content_features, back_content_masks = self.encode_with_intermediate_mask(content, back_content_mask)

        forward_gimage = self.NDNSMDecoder(content_features, style_features, style_masks)
        back_gimage = self.NDNSMDecoder(back_content_features, back_style_features, back_style_masks)
        gimage = self.fuse_images(forward_gimage, back_gimage, content_mask)
        return gimage


    def single_mask_pred(self, content, style, content_mask, style_mask):
        if content_mask.dim() == 4:
            content_mask = content_mask.squeeze(0)
        if style_mask.dim() == 4:
            style_mask = style_mask.squeeze(0)
        style_features, style_masks = self.encode_with_intermediate_mask(style, style_mask)
        content_features, content_masks = self.encode_with_intermediate_mask(content, content_mask)
        gimage = self.NDNSMDecoder(content_features, style_features, content_mask_list=content_masks, style_mask_list=style_masks)#*content_mask
        return gimage, content_masks

    def single_image_mask_pred(self, image, mask_1, mask_2):
        if image.dim() == 4:
            image = image.squeeze(0)
        if mask_1.dim() == 4:
            mask_1 = mask_1.squeeze(0)
        if mask_2.dim() == 4:
            mask_2 = mask_2.squeeze(0)
        content_features, content_masks = self.encode_with_intermediate_mask(image, mask_2)
        style_features, style_masks = self.encode_with_intermediate_mask(image, mask_1)
        gimage = self.NDNSMDecoder(content_features, style_features, content_mask_list=content_masks, style_mask_list=style_masks)
        return gimage

    def mutil_masks_pred(self, content, style, content_masks, style_masks):
        mask_number = len(content_masks)
        content_masks_list = []
        style_masks_list = []
        content_features_list = []
        style_features_list = []
        content_features_all = self.encode_with_intermediate(content)
        style_features_all = self.encode_with_intermediate(style)
        for i in range(mask_number):
            content_mask = content_masks[i]
            style_mask = style_masks[i]
            if content_mask.dim() == 4:
                content_mask = content_mask.squeeze(0)
            if style_mask.dim() == 4:
                style_mask = style_mask.squeeze(0)
            style_features, style_m_s = self.encode_with_intermediate_mask(style, style_mask)
            content_features, content_m_s = self.encode_with_intermediate_mask(content, content_mask)
            content_features_list.append(content_features)
            style_features_list.append(style_features)
            content_masks_list.append(content_m_s)
            style_masks_list.append(style_m_s)
        # content_features = self.encode_with_intermediate(content)
        content_features_list.append(style_features)
        style_features_list.append(style_features)
        content_masks_list.append(style_m_s)
        style_masks_list.append(style_m_s)
        gimage = self.NDNSMDecoder(content_features_list, style_features_list, content_masks_list, style_masks_list)

        return gimage, content_masks

    def get_image_loss(self, content, style, output):
        style_features = self.encode_with_intermediate(style)
        content_features = self.encode_with_intermediate(content)
        output_features = self.encode_with_intermediate(output)
        loss_c = self.calc_content_loss(output_features[-1], content_features[-1])
        loss_s = self.calc_style_loss(output_features[0], style_features[0])
        loss_g = self.calc_gram_loss(output_features[0], style_features[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(output_features[i], style_features[i])
            loss_g = loss_g+self.calc_gram_loss(output_features[i], style_features[i])
        return loss_c, loss_s, loss_g


class DNSM(nn.Module):
    'DNSM: Deterministic Neural Style Mapping'
    def __init__(self, k, size=3):
        super(DNSM, self).__init__()
        self.size=size
        self.P = nn.Parameter(torch.randn(self.size, k))
        self.Q = nn.Parameter(torch.randn(k, self.size))
        self.activation = nn.ReLU()

    def forward(self, x, T):
        n, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(n, -1, c)
        x = torch.matmul(x, self.P)
        x = self.activation(x)
        x = torch.matmul(x, T)
        x = self.activation(x)
        x = torch.matmul(x, self.Q)
        x = self.activation(x)
        x = x.view(-1, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class NDNSMDecoder(nn.Module):
    def __init__(self, k_list, input_dim_list):
        super(NDNSMDecoder, self).__init__()
        self.k_list = k_list
        self.input_dim_list = input_dim_list
        self.DNSM_Gram1 = DNSM(self.k_list[0], self.input_dim_list[0])
        self.DNSM_Gram2 = DNSM(self.k_list[1], self.input_dim_list[1])
        self.DNSM_Gram3 = DNSM(self.k_list[2], self.input_dim_list[2])
        self.DNSM_Gram4 = DNSM(self.k_list[3], self.input_dim_list[3])

        # self.decoder = decoder
        self.decoder_0 = self.get_decoder(512)
        self.decoder_1 = self.get_decoder(256)
        self.decoder_2 = self.get_decoder(128)
        self.decoder_3 = self.get_decoder(64)

        self.cat_c_1 = self.get_cat_c(512)
        self.cat_c_2 = self.get_cat_c(256)
        self.cat_c_3 = self.get_cat_c(128)

        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(),

            nn.Conv2d(64, 64, (3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(),

            nn.Conv2d(64, 3, (3, 3), padding=1, padding_mode='replicate'),
            )


    def get_decoder(self, dim):
        out_dim = dim//2
        return nn.Sequential(
            nn.Conv2d(dim, out_dim, (3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(out_dim, out_dim, (3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU()
            )

    def get_cat_c(self, dim):
        out_dim = dim//2
        return nn.Sequential(nn.Conv2d(dim, out_dim, kernel_size=(3, 3), padding=1, padding_mode='replicate'),nn.ReLU())

    def decode(self, NDNSM_out_list, content_mask_list=None):
        x = self.decoder_0(NDNSM_out_list[3])
        # if content_mask_list != None:
        #     x = x*content_mask_list[2]
        N,C,W,H = NDNSM_out_list[2].shape
        x = torch.cat((NDNSM_out_list[2], x[:,:,:W,:H]), dim=1)
        x = self.cat_c_1(x)
        x = self.decoder_1(x)
        # if content_mask_list != None:
        #     x = x*content_mask_list[1]
        N,C,W,H = NDNSM_out_list[1].shape
        x = torch.cat((NDNSM_out_list[1], x[:,:,:W,:H]), dim=1)
        x = self.cat_c_2(x)
        x = self.decoder_2(x)
        # if content_mask_list != None:
        #     x = x*content_mask_list[0]
        N,C,W,H = NDNSM_out_list[0].shape
        x = torch.cat((NDNSM_out_list[0], x[:,:,:W,:H]), dim=1)
        x = self.cat_c_3(x)
        x = self.conv(x)
        # if content_mask_list != None:
        #     x = x*content_mask_list[0]
        return x


    def get_nor_mean_NDNSM_out(self, content_f, style_f, DNSM_Gram, content_mask=None, style_mask=None):
        cF_nor, cmean, c_std = nor_mean_std(content_f)
        sF_nor, smean = nor_mean(style_f, style_mask) #!
        gram = gram_matrix(sF_nor, style_mask)
        NDNSM_out = DNSM_Gram(cF_nor, gram)
        NDNSM_out = NDNSM_out + smean.expand(cF_nor.size())
        if content_mask != None:
            NDNSM_out = NDNSM_out*content_mask
        return NDNSM_out


    def forward(self, content_f_list, style_f_list, content_mask_list=None, style_mask_list=None):
        NDNSM_out_list = []
        if content_mask_list==None and style_mask_list==None:
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[0], style_f_list[0], self.DNSM_Gram1))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[1], style_f_list[1], self.DNSM_Gram2))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[2], style_f_list[2], self.DNSM_Gram3))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[3], style_f_list[3], self.DNSM_Gram4))

        elif content_mask_list!=None and style_mask_list==None:
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[0], style_f_list[0], self.DNSM_Gram1, content_mask_list[0]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[1], style_f_list[1], self.DNSM_Gram2, content_mask_list[1]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[2], style_f_list[2], self.DNSM_Gram3, content_mask_list[2]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[3], style_f_list[3], self.DNSM_Gram4, content_mask_list[3]))
        elif content_mask_list==None and style_mask_list!=None:
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[0], style_f_list[0], self.DNSM_Gram1, style_mask=style_mask_list[0]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[1], style_f_list[1], self.DNSM_Gram2, style_mask=style_mask_list[1]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[2], style_f_list[2], self.DNSM_Gram3, style_mask=style_mask_list[2]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[3], style_f_list[3], self.DNSM_Gram4, style_mask=style_mask_list[3]))
        else:
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[0], style_f_list[0], self.DNSM_Gram1, content_mask_list[0], style_mask_list[0]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[1], style_f_list[1], self.DNSM_Gram2, content_mask_list[1], style_mask_list[1]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[2], style_f_list[2], self.DNSM_Gram3, content_mask_list[2], style_mask_list[2]))
            NDNSM_out_list.append(self.get_nor_mean_NDNSM_out(content_f_list[3], style_f_list[3], self.DNSM_Gram4, content_mask_list[3], style_mask_list[3]))
        NDNSM_out = self.decode(NDNSM_out_list, content_mask_list)
        return NDNSM_out


   # for mask
    # def forward(self, content_f_list, style_f_list, mutil_content_mask_list=None, mutil_style_mask_list=None):
    #     NDNSM_out_list = []
    #     mask_number = len(mutil_content_mask_list)
    #     NDNSM_out_0 = self.get_nor_mean_NDNSM_out(content_f_list[0][0], style_f_list[0][0], self.DNSM_Gram1, mutil_content_mask_list[0][0], mutil_style_mask_list[0][0])
    #     NDNSM_out_1 = self.get_nor_mean_NDNSM_out(content_f_list[0][1], style_f_list[0][1], self.DNSM_Gram2, mutil_content_mask_list[0][1], mutil_style_mask_list[0][1])
    #     NDNSM_out_2 = self.get_nor_mean_NDNSM_out(content_f_list[0][2], style_f_list[0][2], self.DNSM_Gram3, mutil_content_mask_list[0][2], mutil_style_mask_list[0][2])
    #     NDNSM_out_3 = self.get_nor_mean_NDNSM_out(content_f_list[0][3], style_f_list[0][3], self.DNSM_Gram4, mutil_content_mask_list[0][3], mutil_style_mask_list[0][3])
    #     for i in range(1, mask_number):
    #         content_mask_list = mutil_content_mask_list[i]
    #         style_mask_list = mutil_style_mask_list[i]
    #         NDNSM_out_0 += self.get_nor_mean_NDNSM_out(content_f_list[i][0], style_f_list[i][0], self.DNSM_Gram1, content_mask_list[0], style_mask_list[0])
    #         NDNSM_out_1 += self.get_nor_mean_NDNSM_out(content_f_list[i][1], style_f_list[i][1], self.DNSM_Gram2, content_mask_list[1], style_mask_list[1])
    #         NDNSM_out_2 += self.get_nor_mean_NDNSM_out(content_f_list[i][2], style_f_list[i][2], self.DNSM_Gram3, content_mask_list[2], style_mask_list[2])
    #         NDNSM_out_3 += self.get_nor_mean_NDNSM_out(content_f_list[i][3], style_f_list[i][3], self.DNSM_Gram4, content_mask_list[3], style_mask_list[3])

    #     NDNSM_out_list.append(NDNSM_out_0)
    #     NDNSM_out_list.append(NDNSM_out_1)
    #     NDNSM_out_list.append(NDNSM_out_2)
    #     NDNSM_out_list.append(NDNSM_out_3)

    #     NDNSM_out = self.decode(NDNSM_out_list)

    #     return NDNSM_out
