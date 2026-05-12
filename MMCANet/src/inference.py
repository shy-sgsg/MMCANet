import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np

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
    
    # 4. 拼接通道 [1通道模糊SAR, 3通道光学] -> 4通道
    input_tensor = torch.cat([sar_lr, opt_hr], dim=0).unsqueeze(0) # unsqueeze(0) 是为了增加 Batch 维度 (1, 4, 256, 256)
    
    return input_tensor, sar_lr, opt_hr, sar_hr

def main():
    # ==========================================
    # 1. 设置路径与参数 (请替换为你的真实路径)
    # ==========================================
    weight_path = "twostream_crossattn_epoch_50.pth"  # 你的训练权重文件路径
    test_sar_path = "../data/sar/14056.png" # 测试用 SAR 图像
    test_opt_path = "../data/opt/14056.png" # 测试用 光学 图像
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 2. 加载模型与权重
    # ==========================================
    print("正在加载模型...")
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None,       # 推理时不需要下载预训练权重了，用我们自己训练的
        in_channels=4, 
        classes=1, 
        activation=None
    )
    
    # 加载你训练好的权重字典
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval() # 极其关键！将模型设置为评估/推理模式，关闭 Dropout 和 BatchNorm 的动态更新
    
    # ==========================================
    # 3. 数据预处理与前向推理
    # ==========================================
    input_tensor, sar_lr, opt_hr, sar_hr = load_image_and_preprocess(test_sar_path, test_opt_path)
    input_tensor = input_tensor.to(device)
    
    print("正在进行网络推理...")
    with torch.no_grad(): # 推理时不计算梯度，极其省显存，速度极快
        output_tensor = model(input_tensor)
        
    # ==========================================
    # 4. 后处理与 matplotlib 可视化
    # ==========================================
    # 将 Tensor 从 GPU 挪回 CPU，去掉 Batch 维度，并转为 numpy 数组方便画图
    # np.clip 用于防止回归任务输出偶尔越界(小于0或大于1)，保证显示正常
    out_img = np.clip(output_tensor.squeeze().cpu().numpy(), 0, 1) 
    
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
    plt.show()

if __name__ == "__main__":
    main()