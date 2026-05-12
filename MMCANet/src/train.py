import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
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
        sar_lr = torch.nn.functional.interpolate(
            sar_hr.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False
        )
        sar_lr = torch.nn.functional.interpolate(
            sar_lr, size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0) # 恢复形状 (1, 256, 256)
        
        # 4. 在通道维度拼接输入：[模糊SAR(1通道), 高清光学(3通道)] -> 4通道
        input_tensor = torch.cat([sar_lr, opt_hr], dim=0) 
        
        # 5. 目标真值是原始的清晰 SAR
        target_tensor = sar_hr 
        
        return input_tensor, target_tensor

# ==========================================
# 2. 训练主函数
# ==========================================
def main():
    # --- 超参数设置 ---
    batch_size = 8       # 如果显存不够（比如报 OOM），改成 4 或 2
    num_epochs = 50      # 跑 50 轮看看 Loss 下降情况
    learning_rate = 1e-4 # 初始学习率
    
    # 请把这里的路径替换成你电脑上的真实路径
    SAR_DIR = "../data/sar" 
    OPT_DIR = "../data/opt"
    
    # --- 准备数据 ---
    dataset = SAROptDataset(sar_dir=SAR_DIR, opt_dir=OPT_DIR)
    # 划分一下训练集，这里简单起见全放进 dataloader，实际写论文需要分 train/val
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"成功加载数据集，共 {len(dataset)} 张图片！")
    
    # --- 定义模型 ---
    # 使用现成的 ResNet34 作为 Encoder 的 U-Net
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", # 使用预训练权重加速收敛
        in_channels=4,              # 关键！适应我们的 4 通道输入
        classes=1,                  # 输出 1 通道的清晰 SAR
        activation=None             # 回归任务不需要激活函数
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已加载至设备: {device}")
    
    # --- 定义损失函数与优化器 ---
    criterion = nn.L1Loss() # 最基础的 L1 Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 开始训练循环 ---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 增加高大上的进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"👉 Epoch [{epoch+1}/{num_epochs}] 结束, 平均 Loss: {avg_loss:.5f}")
        
        # 每 10 轮保存一次权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet_phase1_epoch_{epoch+1}.pth")
            print("💾 模型权重已保存！")

if __name__ == "__main__":
    main()