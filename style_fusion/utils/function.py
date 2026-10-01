import torch

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mask_mean_std(feat, mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # feat = feat*mask
    feat_reshape = feat.view(N, C, -1)
    mask = mask.view(N, 1, -1)
    mean = torch.sum(feat_reshape * mask, dim=2) / (torch.sum(mask, dim=2) + eps)
    feat_var = torch.sum((feat_reshape - mean.unsqueeze(2))**2, dim=2)/ (torch.sum(mask, dim=2) + eps) # TODO
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = mean.view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean(feat, mask=None, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    if mask == None:
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    else:
        valid_mask = mask > 0.5
        valid_count = torch.sum(valid_mask, dim=(2, 3), keepdim=True).view(size[0], 1, -1)
        feat_sum = feat.view(N, C, -1).sum(dim=2, keepdim=True)
        feat_mean = feat_sum/(valid_count+eps)
        feat_mean = feat_mean.view(N, C, 1, 1)
    return feat_mean


def nor_mean_std(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat, mean, std


def nor_mean(feat, mask=None):
    size = feat.size()
    mean = calc_mean(feat, mask)
    nor_feat = feat - mean.expand(size)
    if mask != None:
        nor_feat = nor_feat*mask

    return nor_feat, mean


def calc_cov(feat):
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0,2,1)).div(feat.size(2))
    return f_cov


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def gram_matrix(features, mask=None):
    """
    args:
        features: (N, C, H, W)
        mask: (N, 1, H, W)
    return:
        gram: (N, C, C)
    """
    N, C, H, W = features.size()
    if mask==None:
        flattened_features = features.view(N, C, -1)
        gram = torch.bmm(flattened_features, flattened_features.transpose(1, 2))
        gram = gram / (C * H * W)
        # gram = gram/ (H * W)

    else:
        # Expand the mask to the same shape as features for element-wise multiplication
        mask = mask.expand(N, C, H, W)

        # Apply the mask to features
        features = features * mask

        # Calculate the number of valid features
        valid_mask = mask > 0.5
        valid_count = torch.sum(valid_mask, dim=(2, 3), keepdim=True).view(N, C, -1)

        # Select only the valid features
        # flattened_features = masked_features.masked_select(valid_mask)
        flattened_features = features.view(N, C, -1)

        gram = torch.bmm(flattened_features, flattened_features.transpose(1, 2))

        gram = gram / (C * valid_count + 1e-5)
        # gram = gram / valid_count
    return gram
