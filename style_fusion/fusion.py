import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PatchCrossAttentionFusion(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embed_dim=64, num_heads=4):
        super(PatchCrossAttentionFusion, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # embedding 将 patch 拉到一个 feature 维度
        self.proj_content = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.proj_stylized = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # 输出还原
        self.proj_out = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    def forward(self, content, stylized):
        """
        content: (B, C, H, W) 结构信息
        stylized: (B, C, H, W) 风格化结果
        """
        B, C, H, W = content.shape
        P = self.patch_size

        # unfold 分块 (B, C*P*P, N), N = patch数
        content_patches = F.unfold(content, kernel_size=P, stride=P).transpose(1, 2)  # (B, N, C*P*P)
        stylized_patches = F.unfold(stylized, kernel_size=P, stride=P).transpose(1, 2)

        # 投影到 embed_dim
        Q = self.proj_content(content_patches)    # (B, N, D)
        K = self.proj_stylized(stylized_patches)  # (B, N, D)
        V = K  # value 一般和 key 同源

        # CrossAttention: content query, stylized key-value
        fused, _ = self.cross_attn(Q, K, V)  # (B, N, D)

        # 投影回 patch 空间
        fused_patches = self.proj_out(fused)  # (B, N, C*P*P)

        # fold 回原图
        fused_patches = fused_patches.transpose(1, 2)  # (B, C*P*P, N)
        out = F.fold(fused_patches, output_size=(H, W), kernel_size=P, stride=P)  # (B, C, H, W)

        return out


class FusionNet(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, patch_size=16):
        super(FusionNet, self).__init__()
        self.patch_size = patch_size

        # patch embedding
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # cross-attention: stylized作为query，content作为key/value
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # feed-forward层
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 残差 & norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 反patch重建
        self.unproj = nn.ConvTranspose2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, stylized1, content1):
        """
        stylized1: (B, 3, H, W) 风格化结果
        content1: (B, 3, H, W) 原始内容图
        """
        B, C, H, W = stylized1.shape

        # patch embedding
        q = self.proj(stylized1).flatten(2).transpose(1, 2)  # (B, N, D)
        kv = self.proj(content1).flatten(2).transpose(1, 2)  # (B, N, D)

        # cross-attention
        attn_out, _ = self.cross_attn(q, kv, kv)  # stylized query, content key/value
        x = self.norm1(q + attn_out)

        # feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # reshape back
        x = x.transpose(1, 2).view(B, -1, H // self.patch_size, W // self.patch_size)
        out = self.unproj(x)  # (B, 3, H, W)

        return out


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


class FFTImageAttentionFusion(nn.Module):
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

class ChannelSpatialAttention(nn.Module):
    """结合通道注意力与空间注意力的简单模块（输入 concat pair）"""
    def __init__(self, in_ch, mid=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # channel attention
        self.ca_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid//2, mid, 1),
            nn.Sigmoid()
        )
        # spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(mid, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, in_ch, H, W]
        f = self.conv(x)  # [B,mid,H,W]
        ca = self.ca_fc(f)  # [B,mid,1,1]
        f = f * ca
        sa = self.sa(f)  # [B,1,H,W]
        # final attention map combine channel-aggregated map (reduce channel)
        return sa  # [B,1,H,W]


class FFTFusionNet(nn.Module):
    """
    使用 FFT 分解，低频/高频在空间域上用 attention 权重融合，然后回到频域重构。
    """
    def __init__(self, keep_ratio=0.12, attn_mid=32, use_stronger_attn=False):
        super().__init__()
        self.keep_ratio = keep_ratio
        if use_stronger_attn:
            self.low_attn = ChannelSpatialAttention(in_ch=6, mid=attn_mid)
            self.high_attn = ChannelSpatialAttention(in_ch=6, mid=attn_mid)
        else:
            self.low_attn = nn.Sequential(
                nn.Conv2d(6, attn_mid, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(attn_mid, 1, 1), nn.Sigmoid()
            )
            self.high_attn = nn.Sequential(
                nn.Conv2d(6, attn_mid, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(attn_mid, 1, 1), nn.Sigmoid()
            )

        self.refine = nn.Sequential(
            nn.Conv2d(3, attn_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_mid, 3, 3, padding=1)
        )

    def forward(self, stylized, content):
        B,C,H,W = stylized.shape
        low_sty, high_sty, _, _ = fft_decompose_image(stylized, self.keep_ratio)
        low_con, high_con, _, _ = fft_decompose_image(content, self.keep_ratio)

        low_pair = torch.cat([low_sty, low_con], dim=1)  # [B,6,H,W]
        high_pair = torch.cat([high_sty, high_con], dim=1)

        a_low = self.low_attn(low_pair)  # [B,1,H,W]
        a_high = self.high_attn(high_pair)

        low_fused_img = a_low * low_sty + (1 - a_low) * low_con
        high_fused_img = a_high * high_sty + (1 - a_high) * high_con

        # 转到频域并合并（更准确）
        F_low_fused = fft_shift(torch.fft.fft2(low_fused_img, dim=(-2,-1)))
        F_high_fused = fft_shift(torch.fft.fft2(high_fused_img, dim=(-2,-1)))
        fused = fft_reconstruct_from_freq(F_low_fused, F_high_fused)
        fused = fused + self.refine(fused)  # small residual refine
        # return fused.clamp(0,1), {'a_low': a_low, 'a_high': a_high,
        #                           'low_fused': low_fused_img, 'high_fused': high_fused_img}
        return fused.clamp(0,1)


import torch.fft as fft

class FFTFusionNet_v2(nn.Module):
    def __init__(self, keep_ratio=0.12, attn_mid=32, use_stronger_attn=False):
        super().__init__()
        self.keep_ratio = keep_ratio
        if use_stronger_attn:
            self.stylized_attn = ChannelSpatialAttention(in_ch=3, mid=attn_mid)
            self.content_attn = ChannelSpatialAttention(in_ch=3, mid=attn_mid)
        else:
            self.stylized_attn = nn.Sequential(
                nn.Conv2d(3, attn_mid, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(attn_mid, 1, 1), nn.Sigmoid()
            )
            self.content_attn = nn.Sequential(
                nn.Conv2d(3, attn_mid, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(attn_mid, 1, 1), nn.Sigmoid()
            )
        self.refine = nn.Sequential(
                nn.Conv2d(3, attn_mid, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(attn_mid, 3, 3, padding=1)
            )

    def forward(self, stylized, content):
        B, C, H, W = stylized.shape

        # --- 1. 获取注意力权重 ---
        stylized_attn = self.stylized_attn(stylized)  # [B,1,H,W]
        content_attn = self.content_attn(content)     # [B,1,H,W]

        # --- 2. FFT 分离高低频 ---
        def fft_split(x, keep_ratio):
            x_freq = fft.fftn(x, dim=(-2, -1))
            x_freq = fft.fftshift(x_freq, dim=(-2, -1))

            crow, ccol = H // 2, W // 2
            threshold_h = int(H * keep_ratio / 2)
            threshold_w = int(W * keep_ratio / 2)

            mask_low = torch.zeros_like(x_freq)
            mask_low[..., crow-threshold_h:crow+threshold_h, ccol-threshold_w:ccol+threshold_w] = 1
            mask_high = 1 - mask_low

            x_low = x_freq * mask_low
            x_high = x_freq * mask_high
            return x_low, x_high

        s_low, _ = fft_split(stylized, self.keep_ratio)   # stylized 的低频
        _, c_high = fft_split(content, self.keep_ratio)   # content 的高频

        # --- 3. 空间域注意力加权融合 ---
        # IFFT 回空间域
        s_low_spatial = fft.ifftn(fft.ifftshift(s_low, dim=(-2,-1)), dim=(-2,-1)).real
        c_high_spatial = fft.ifftn(fft.ifftshift(c_high, dim=(-2,-1)), dim=(-2,-1)).real

        # 注意力加权融合
        fused = s_low_spatial * stylized_attn + c_high_spatial * content_attn

        # --- 4. refine ---
        out = self.refine(fused)

        return out


def fft_reconstruct_from_freq(F_low, F_high):
    F_fused = F_low + F_high
    img = torch.fft.ifft2(ifft_shift(F_fused), dim=(-2,-1)).real
    return img

def fft_decompose_image(img, keep_ratio):
    # img: [B,C,H,W]
    B,C,H,W = img.shape
    device = img.device
    F_img = torch.fft.fft2(img, dim=(-2,-1))
    F_img = fft_shift(F_img)
    mask = make_lowpass_mask(H, W, keep_ratio, device).view(1,1,H,W)
    F_low = F_img * mask
    F_high = F_img * (1 - mask)
    low_img = torch.fft.ifft2(ifft_shift(F_low), dim=(-2,-1)).real
    high_img = torch.fft.ifft2(ifft_shift(F_high), dim=(-2,-1)).real
    return low_img, high_img, F_low, F_high

def fft_shift(x):
    return torch.fft.fftshift(x, dim=(-2, -1))

def ifft_shift(x):
    return torch.fft.ifftshift(x, dim=(-2, -1))

def make_lowpass_mask(H, W, keep_ratio, device):
    crow, ccol = H // 2, W // 2
    rh, rw = max(1, int(H * keep_ratio / 2)), max(1, int(W * keep_ratio / 2))
    mask = torch.zeros((H, W), device=device)
    mask[crow-rh:crow+rh, ccol-rw:ccol+rw] = 1.0
    return mask  # (H,W)


class GlobalLocalFusion(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv_feat = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # 局部结构注意力
        self.local_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 全局通道注意力（控制整体风格/内容比例）
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # 全局池化
            nn.Conv2d(channels, channels // 4, 1),    # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),           # 输出一个全局权重
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, stylized, content):
        F_style = self.conv_feat(stylized)
        F_content = self.conv_feat(content)

        # 差异驱动的局部注意力
        diff = torch.abs(F_style - F_content)
        attn_local = self.local_attn(diff)  # H×W 权重

        # 全局通道注意力（只输出一个数）
        attn_global = self.global_attn(diff)  # shape: (B,1,1,1)

        # 结合局部 & 全局
        attn = attn_global * attn_local + (1 - attn_global) * attn_local.mean()

        # 融合
        F_fused = attn * F_content + (1 - attn) * F_style
        return self.conv_out(F_fused)

class DualBranchFusion(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv_feat = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # 结构注意力 (diff-based)
        self.attn_struct = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 风格注意力 (sum-based)
        self.attn_style = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 融合两路的加权参数 (可学习)
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, stylized, content):
        F_style = self.conv_feat(stylized)
        F_content = self.conv_feat(content)

        # 结构分支
        diff = torch.abs(F_style - F_content)
        attn_struct = self.attn_struct(diff)
        F_struct = attn_struct * F_content + (1 - attn_struct) * F_style

        # 风格分支
        summed = F_style + F_content
        attn_style = self.attn_style(summed)
        F_style_enhanced = attn_style * F_style + (1 - attn_style) * F_content

        # 双分支融合
        F_fused = self.alpha * F_struct + (1 - self.alpha) * F_style_enhanced

        return self.conv_out(F_fused)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from glob import glob

# -----------------------
# Utilities: Haar DWT / IDWT (per-channel, separable)
# -----------------------
# Haar filters (un-normalized convenient forms)
# Low-pass 1D: [1, 1] ; High-pass 1D: [1, -1]
# For simplicity we will not normalize by sqrt(2); network can learn scaling.

def dwt2(x):
    """
    x: (B, C, H, W)
    returns tuple of 4 tensors each (B, C, H/2, W/2): (LL, LH, HL, HH)
    Implementation: separable filtering + subsample by 2 (stride slicing).
    """
    # assume H,W even
    # compute pairwise sums/diffs: we can compute on pixels:
    # block: [[a, b],[c,d]] -> LL=(a+b+c+d)/2 ; LH=(a-b+c-d)/2 ; HL=(a+b-c-d)/2 ; HH=(a-b-c+d)/2
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]

    LL = (a + b + c + d) * 0.5
    LH = (a - b + c - d) * 0.5
    HL = (a + b - c - d) * 0.5
    HH = (a - b - c + d) * 0.5
    return LL, LH, HL, HH

def idwt2(LL, LH, HL, HH):
    """
    Inverse Haar DWT. Inputs each (B,C,H, W) where H,W are half of target.
    returns (B,C, 2H, 2W)
    Reconstruct block a,b,c,d from subbands:
    a = LL + LH + HL + HH
    b = LL - LH + HL - HH
    c = LL + LH - HL - HH
    d = LL - LH - HL + HH
    """
    a = LL + LH + HL + HH
    b = LL - LH + HL - HH
    c = LL + LH - HL - HH
    d = LL - LH - HL + HH
    # interleave to form upsampled image
    B, C, h, w = a.shape
    out = torch.zeros((B, C, h*2, w*2), device=a.device, dtype=a.dtype)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out

# -----------------------
# Subband Fusion Module
# -----------------------
class SubbandFusionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # small conv net to fuse corresponding subbands from content & styled
        self.net = nn.Sequential(
            nn.Conv2d(ch*2, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
        )
        # gating to allow weighted blending (learnable)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch*2, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, c, s):
        # c,s: (B,C,H,W) channels equal
        cat = torch.cat([c, s], dim=1)
        fused = self.net(cat)
        g = self.gate(cat)  # (B,ch,1,1)
        return fused * g + c * (1 - g)  # prefer content by default (learnable)

# -----------------------
# Full DWT Fusion Network
# -----------------------
class DWTFusionNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, use_high_freq_from_content=True):
        """
        Strategy:
          - Apply one-level DWT to both content & styled
          - Fuse each subband with SubbandFusionBlock
          - IDWT to reconstruct
          - Then light refinement convs (optional)
        """
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        # project input channels to base channels per-subband
        self.proj = nn.Conv2d(in_ch, base_ch, 1)
        # subband fusion blocks (same channels)
        self.f_ll = SubbandFusionBlock(base_ch)
        self.f_lh = SubbandFusionBlock(base_ch)
        self.f_hl = SubbandFusionBlock(base_ch)
        self.f_hh = SubbandFusionBlock(base_ch)
        # after IDWT refine
        self.refine = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def forward(self, content, styled):
        """
        content, styled: (B,3,H,W), assumed normalized [0,1]
        """
        # project
        pc = self.proj(content)   # (B,base,H,W)
        ps = self.proj(styled)

        # DWT per-channel on projected features
        LLc, LHc, HLc, HHc = dwt2(pc)
        LLs, LHs, HLs, HHs = dwt2(ps)

        # Fuse each subband
        LLf = self.f_ll(LLc, LLs)
        LHf = self.f_lh(LHc, LHs)   # prefer content in gate init
        HLf = self.f_hl(HLc, HLs)
        HHf = self.f_hh(HHc, HHs)

        # IDWT to reconstruct features in base_ch channels
        recon_feat = idwt2(LLf, LHf, HLf, HHf)  # (B,base, H, W)
        # refine and project back to RGB
        out = self.refine(recon_feat)
        # out = torch.sigmoid(out)  # assume we want in [0,1]
        return out


class FusionNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, use_high_freq_from_content=True):
        """
        Strategy:
          - Apply one-level DWT to both content & styled
          - Fuse each subband with SubbandFusionBlock
          - IDWT to reconstruct
          - Then light refinement convs (optional)
        """
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        # project input channels to base channels per-subband
        self.proj = nn.Conv2d(in_ch, base_ch, 1)
        self.fuse = SubbandFusionBlock(base_ch)
        # after IDWT refine
        self.refine = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def forward(self, content, styled):
        """
        content, styled: (B,3,H,W), assumed normalized [0,1]
        """
        # project
        pc = self.proj(content)   # (B,base,H,W)
        ps = self.proj(styled)

        # Fuse each subband
        f = self.fuse(pc, ps)

        # refine and project back to RGB
        out = self.refine(f)
        # out = torch.sigmoid(out)  # assume we want in [0,1]
        return out


# 示例调用
if __name__ == "__main__":
    B, C, H, W = 1, 3, 4000, 4000
    # 11725
    devicre = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    content = torch.randn(B, C, H, W).to(devicre)
    stylized = torch.randn(B, C, H, W).to(devicre)

    model = ImageAttentionFusion(32).to(devicre)
    # model = GlobalLocalFusion(channels=32).to(devicre)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in tqdm(range(1000)):
        out = model(content, stylized)
        # optimizer.zero_grad()
        # optimizer.step()
    print(out.shape)  # (B, C, H, W)

    # model2 = FusionNet(embed_dim=64, num_heads=4, patch_size=16)
    # out2 = model2(stylized, content)
    # print(out2.shape)  # (B, C, H, W)

    # model3 = GlobalLocalFusion(channels=32).to(devicre)
    # out3 = model3(stylized, content)
    # print(out3.shape)  # (B, C, H, W)
