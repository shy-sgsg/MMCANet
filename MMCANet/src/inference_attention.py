import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = max(in_channels // 8, 1)
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat, opt_feat):
        b, c, h, w = sar_feat.size()
        n = h * w

        proj_query = self.query_conv(sar_feat).view(b, -1, n).permute(0, 2, 1)
        proj_key = self.key_conv(opt_feat).view(b, -1, n)
        proj_value = self.value_conv(opt_feat).view(b, -1, n)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return sar_feat + self.gamma * out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))
        return [x1, x2, x3, x4], xb


class TwoStreamCrossAttentionUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.sar_encoder = Encoder(in_channels=1, base_channels=base_channels)
        self.opt_encoder = Encoder(in_channels=3, base_channels=base_channels)
        self.cross_attn = SpatialCrossAttention(in_channels=base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, sar_input, opt_input):
        sar_skips, sar_bottleneck = self.sar_encoder(sar_input)
        _, opt_bottleneck = self.opt_encoder(opt_input)

        fused = self.cross_attn(sar_bottleneck, opt_bottleneck)

        d4 = self.up4(fused)
        d4 = self.dec4(torch.cat([d4, sar_skips[3]], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, sar_skips[2]], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, sar_skips[1]], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, sar_skips[0]], dim=1))

        return self.out_conv(d1)

def load_image_and_preprocess(sar_path, opt_path):
    """读取图像并进行与训练时完全一致的预处理"""
    # 1. 读取图像
    sar_img = Image.open(sar_path).convert('L')
    opt_img = Image.open(opt_path).convert('RGB')
    
    # 2. 转为 Tensor (归一化到 0-1)
    to_tensor = transforms.ToTensor()
    sar_hr = to_tensor(sar_img)
    opt_hr = to_tensor(opt_img)
    
    # 3. 模拟退化：制造低分辨率模糊 SAR
    sar_lr = F.interpolate(sar_hr.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
    sar_lr = F.interpolate(sar_lr, size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
    
    return sar_lr, opt_hr, sar_hr


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stream attention inference for SAR reconstruction")
    parser.add_argument("--weight", default="twostream_crossattn_ratio_tv_epoch_50.pth", help="模型权重路径")
    parser.add_argument("--sar", default="../data/sar/8078.png", help="输入 SAR 图像路径")
    parser.add_argument("--opt", default="../data/opt/8078.png", help="输入光学图像路径")
    parser.add_argument("--out", default="./recon_attention.png", help="输出重建 SAR 图像路径")
    parser.add_argument("--figure", default="./compare_attention.png", help="对比图保存路径")
    parser.add_argument("--base-channels", type=int, default=32, help="与训练时一致的 base_channels")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示对比图")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)

    if not os.path.isfile(args.weight):
        raise FileNotFoundError(f"未找到权重文件: {args.weight}")
    if not os.path.isfile(args.sar):
        raise FileNotFoundError(f"未找到 SAR 图像: {args.sar}")
    if not os.path.isfile(args.opt):
        raise FileNotFoundError(f"未找到光学图像: {args.opt}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 2. 加载模型与权重
    # ==========================================
    print("正在加载 Attention 模型...")
    model = TwoStreamCrossAttentionUNet(base_channels=args.base_channels)
    
    state_dict = torch.load(args.weight, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # ==========================================
    # 3. 数据预处理与前向推理
    # ==========================================
    sar_lr, opt_hr, sar_hr = load_image_and_preprocess(args.sar, args.opt)
    sar_input = sar_lr.unsqueeze(0).to(device)
    opt_input = opt_hr.unsqueeze(0).to(device)
    
    print("正在进行网络推理...")
    with torch.no_grad():
        output_tensor = model(sar_input, opt_input)
        
    # ==========================================
    # 4. 后处理与 matplotlib 可视化
    # ==========================================
    out_img = np.clip(output_tensor.squeeze().cpu().numpy(), 0, 1)
    out_img_uint8 = (out_img * 255.0).round().astype(np.uint8)
    Image.fromarray(out_img_uint8, mode="L").save(args.out)
    print(f"重建图已保存: {args.out}")
    
    # 准备其他对比图的 numpy 数组
    sar_lr_img = sar_lr.squeeze().numpy()
    sar_hr_img = sar_hr.squeeze().numpy()
    # 光学图是 3 通道 (C, H, W)，matplotlib 需要 (H, W, C)，所以用 permute 转换维度
    opt_hr_img = opt_hr.permute(1, 2, 0).numpy()
    
    # --- 画图展示 ---
    print("展示对比结果...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 图1：低分辨率带噪 SAR (网络的输入之一)
    axes[0].imshow(sar_lr_img, cmap='gray')
    axes[0].set_title("Input: LR SAR (Blurred)")
    axes[0].axis('off')
    
    # 图2：高分辨率光学 (网络的引导输入)
    axes[1].imshow(opt_hr_img)
    axes[1].set_title("Input: HR Optical (Guide)")
    axes[1].axis('off')
    
    # 图3：网络输出的高分辨率 SAR
    axes[2].imshow(out_img, cmap='gray')
    axes[2].set_title("Output: Super-Resolved SAR")
    axes[2].axis('off')
    
    # 图4：原始高分辨率 SAR (作为真值参考)
    axes[3].imshow(sar_hr_img, cmap='gray')
    axes[3].set_title("Target: Ground Truth HR SAR")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.figure, dpi=150)
    print(f"对比图已保存: {args.figure}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    main()