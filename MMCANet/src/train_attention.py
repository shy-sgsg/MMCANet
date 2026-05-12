import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm # 用于显示进度条

# ==========================================
# 1. 定义数据集 (Dataset)与预处理
# ==========================================
class SAROptDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        """
        sar_dir: 存放 256x256 SAR 图像的文件夹路径
        opt_dir: 存放 256x256 光学图像的文件夹路径（确保文件名一一对应）
        """
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.image_names = sorted(os.listdir(sar_dir)) # 假设两边文件名相同
        
        # 基础的张量转换（会自动将 0-255 的像素值归一化到 0.0-1.0）
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # 1. 读取图像 (SAR为灰度L，光学为RGB)
        sar_path = os.path.join(self.sar_dir, img_name)
        opt_path = os.path.join(self.opt_dir, img_name)
        
        sar_img = Image.open(sar_path).convert('L')   # 1通道
        opt_img = Image.open(opt_path).convert('RGB') # 3通道
        
        # 2. 转换为 Tensor, 此时 shape 分别为 (1, 256, 256) 和 (3, 256, 256)
        sar_hr = self.to_tensor(sar_img)
        opt_hr = self.to_tensor(opt_img)
        
        # 3. 构造自监督任务的输入：将 SAR 退化为低分辨率 (模拟模糊)
        # 先下采样到 64x64，再用双线性插值拉回 256x256，这样图像就变糊了
        sar_lr = F.interpolate(
            sar_hr.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False
        )
        sar_lr = F.interpolate(
            sar_lr, size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0) # 恢复形状 (1, 256, 256)
        
        # 4. 目标真值是原始的清晰 SAR
        target_tensor = sar_hr 
        
        # 返回双流输入，避免暴力通道拼接
        return sar_lr, opt_hr, target_tensor


class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = max(in_channels // 8, 1)
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 初始为0，先让模型以SAR分支为主，再逐步学习光学引导
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
        out = sar_feat + self.gamma * out
        return out


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


def soft_ratio_loss(sar_out, sar_noisy, epsilon=1e-4, log_clip=3.0):
    """
    稳定版软比值约束。
    关键点：
    1) sar_out 先做正值映射，避免分母接近0导致比值爆炸；
    2) 在 log-ratio 空间约束均值和梯度，抑制异常值。
    """
    # 保证分母为正，且在训练初期不会因为接近0而数值爆炸
    out_pos = F.softplus(sar_out) + epsilon
    noisy_pos = torch.clamp(sar_noisy, min=epsilon)

    # R = noisy / out，在 log 空间中 logR = log(noisy) - log(out)
    log_ratio = torch.log(noisy_pos) - torch.log(out_pos)
    log_ratio = torch.clamp(log_ratio, min=-log_clip, max=log_clip)

    # 均值约束：R 接近 1 等价于 logR 接近 0
    loss_mean = torch.abs(torch.mean(log_ratio))

    # 梯度约束：惩罚比值图中的异常突变（目标残留）
    grad_x = log_ratio[:, :, :, :-1] - log_ratio[:, :, :, 1:]
    grad_y = log_ratio[:, :, :-1, :] - log_ratio[:, :, 1:, :]

    zero_x = torch.zeros_like(grad_x)
    zero_y = torch.zeros_like(grad_y)
    loss_residual = F.smooth_l1_loss(grad_x, zero_x) + F.smooth_l1_loss(grad_y, zero_y)

    return loss_mean + 0.5 * loss_residual


def adaptive_tv_loss(sar_out, opt_hr, alpha=5.0):
    """光学引导的自适应 TV：平坦区域更强平滑，边缘区域弱平滑。"""
    opt_gray = opt_hr.mean(dim=1, keepdim=True)
    opt_grad_x = torch.abs(opt_gray[:, :, :, :-1] - opt_gray[:, :, :, 1:])
    opt_grad_y = torch.abs(opt_gray[:, :, :-1, :] - opt_gray[:, :, 1:, :])

    weight_x = torch.exp(-alpha * opt_grad_x)
    weight_y = torch.exp(-alpha * opt_grad_y)

    sar_grad_x = torch.abs(sar_out[:, :, :, :-1] - sar_out[:, :, :, 1:])
    sar_grad_y = torch.abs(sar_out[:, :, :-1, :] - sar_out[:, :, 1:, :])

    return torch.mean(weight_x * sar_grad_x) + torch.mean(weight_y * sar_grad_y)

# ==========================================
# 2. 训练主函数
# ==========================================
def main():
    # --- 超参数设置 ---
    batch_size = 8       # 如果显存不够（比如报 OOM），改成 4 或 2
    num_epochs = 50      # 跑 50 轮看看 Loss 下降情况
    learning_rate = 1e-4 # 初始学习率
    lambda_ratio = 0.1   # 软比值损失权重
    lambda_tv = 0.5     # 自适应TV损失权重
    warmup_epochs = 5    # 前几轮线性升权，避免初期训练不稳定
    
    # 请把这里的路径替换成你电脑上的真实路径
    SAR_DIR = "../data/sar_mini" 
    OPT_DIR = "../data/opt_mini"
    
    # --- 准备数据 ---
    dataset = SAROptDataset(sar_dir=SAR_DIR, opt_dir=OPT_DIR)
    # 划分一下训练集，这里简单起见全放进 dataloader，实际写论文需要分 train/val
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"成功加载数据集，共 {len(dataset)} 张图片！")
    
    # --- 定义模型 ---
    # 双流网络：SAR分支 + 光学分支，在瓶颈层使用 Cross-Attention 融合
    model = TwoStreamCrossAttentionUNet(base_channels=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已加载至设备: {device}")
    
    # --- 定义损失函数与优化器 ---
    criterion = nn.L1Loss() # 最基础的 L1 Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 开始训练循环 ---
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_ratio_loss = 0.0
        epoch_tv_loss = 0.0

        warmup_scale = min(1.0, float(epoch + 1) / float(max(warmup_epochs, 1)))
        ratio_weight = lambda_ratio * warmup_scale
        tv_weight = lambda_tv * warmup_scale
        
        # 使用 tqdm 增加高大上的进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sar_inputs, opt_inputs, targets in pbar:
            sar_inputs = sar_inputs.to(device)
            opt_inputs = opt_inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(sar_inputs, opt_inputs)
            
            # 组合损失：L1 + 软比值约束 + 光学引导TV
            l1_loss = criterion(outputs, targets)
            ratio_loss = soft_ratio_loss(sar_out=outputs, sar_noisy=targets)
            tv_loss = adaptive_tv_loss(sar_out=outputs, opt_hr=opt_inputs)
            ratio_term = ratio_weight * ratio_loss
            tv_term = tv_weight * tv_loss
            total_loss = l1_loss + ratio_term + tv_term
            
            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_ratio_loss += ratio_loss.item()
            epoch_tv_loss += tv_loss.item()
            pbar.set_postfix({
                "Total": f"{total_loss.item():.4f}",
                "L1": f"{l1_loss.item():.4f}",
                "RatioRaw": f"{ratio_loss.item():.4f}",
                # "RatioW": f"{ratio_term.item():.4f}",
                "TVRaw": f"{tv_loss.item():.4f}",
                # "TVW": f"{tv_term.item():.4f}",
            })
            
        avg_total = epoch_total_loss / len(dataloader)
        avg_l1 = epoch_l1_loss / len(dataloader)
        avg_ratio = epoch_ratio_loss / len(dataloader)
        avg_tv = epoch_tv_loss / len(dataloader)
        gamma_value = model.cross_attn.gamma.detach().item()
        print(
            f"👉 Epoch [{epoch+1}/{num_epochs}] 结束 | "
            f"Total: {avg_total:.5f}, L1: {avg_l1:.5f}, "
            f"Ratio: {avg_ratio:.5f}, TV: {avg_tv:.5f}, "
            # f"w_ratio: {ratio_weight:.3f}, w_tv: {tv_weight:.3f}, "
            f"gamma: {gamma_value:.5f}"
        )
        
        # 每 10 轮保存一次权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"twostream_crossattn_ratio_tv_epoch_{epoch+1}.pth")
            print("💾 模型权重已保存！")

if __name__ == "__main__":
    main()