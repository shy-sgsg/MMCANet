# EDSR SAR 单通道任务适配基线冻结协议

**状态：执行前冻结**  
**日期：2026-11-24**  
**方法身份：`TASK_ADAPTED_PUBLISHED`**  
**上游架构：** `BasicSR/basicsr/archs/edsr_arch.py`  
**实现入口：** `experiments/phase_d1_edsr_20261124_v1/edsr_sar.py`、`train_edsr_sar.py`

## 研究目的与解释边界

此基线补齐一个可运行的第二种已发表恢复架构，以便将 SAR-only SwinIR 与 EDSR 任务适配版放在同一数据、目标和评价器下比较。它不是 SAR 专用论文方法、作者 SAR 权重或 EDSR 原论文完整训练复现。目标仍是归一化 8-bit SAR 表示域 noisy pseudo-target；PSNR/SSIM 仅表示对该代理目标的一致程度。

development 已反复用于历史训练和模型选择。本 run 只提供探索性对照，不构成独立泛化证据，也不用于进入 C3-B 或访问封存集合。

## 数据与隔离

- 固定输入为 C1 v2 `x2_light`：train 1,600 张/92 个来源组，development 200 张/19 个来源组。
- 只读取 SAR `lr` 与 `target`，训练只使用 train；development 不做随机增强，并用于每轮 checkpoint 选择及一次最终报告。
- 不读取 optical、confirmation 或 test 图像、成员清单、测量值；不生成新的退化数据。
- 使用现有固定输入 loader 校验每个 LR/target 文件 SHA-256、数组类型、数值范围、倍率、样本身份与来源组。

## 架构与输入输出

- 采用 vendored BasicSR EDSR 架构源码；单通道输入/输出，倍率 ×2。
- `num_feat=64`、`num_block=16`、`res_scale=1`、无 BatchNorm、双卷积残差块、残差主干长跳连、PixelShuffle 上采样。
- 输入保持固定 SAR LR 数值，不逐图拉伸或裁剪；`img_range=1`、单通道均值 `0.5`。输出使用 Sigmoid 映射到 `[0,1]`，与当前 SAR 目标范围一致；该输出适配作为方法身份的一部分披露。
- 随机初始化，不使用外部权重。记录参数量、源码哈希、配置与 checkpoint 哈希。

## 固定训练配置

- 随机种子：`20261125`；确定性算法开启；DataLoader 不使用异步 worker。
- 训练上限：30 epochs；每轮对固定 1,600 张训练影像各取一个随机对齐 patch。
- batch 4；LR patch `64×64`、HR target patch `128×128`；仅训练应用同步水平翻转、垂直翻转和转置。
- Adam，学习率 `1e-4`，weight decay `0`；逐像素 L1；不使用混合精度、预训练权重或学习率调度。
- 每轮完整计算 development 组等权 PSNR；以其最高值选择 best，平分保留较早 epoch。保存 best/last 模型、optimizer 与 RNG 状态、每轮训练日志和逐图 checkpoint 选择记录。
- 不搜索超参数，不根据本 run 结果修改预算、模型或门槛。

## 评价与公平性

- 全 200 张 development 上报告 PSNR、SSIM、EPI correlation、edge-strength ratio、bright-return mean/P95 ratio 与 background mean ratio。
- 与固定 SwinIR SAR 任务适配版按相同 `sample_id`、来源组和 target 配对；先按来源组求均值，再做 10,000 次来源组 bootstrap（seed `20261125`）。
- 预测文件集合必须与 200 张 development 完全一致；不 resize、不逐图拉伸、不额外后处理。
- 报告参数量、训练更新步数/耗时、推理总耗时/单张中位数、峰值显存和运行后磁盘余量。
- 既有 SwinIR 的 checkpoint 选择与 development 重用历史沿用并明确披露；不据两者开发集差异作独立验证或投稿主张。

## 资源与失败策略

- 输出使用新目录 `experiments/runs/phase_d1_edsr_sar_20261124_v1/`；目录存在即失败，禁止覆盖。
- 记录执行命令、Python/PyTorch/CUDA/GPU、git 状态、代码/协议/数据清单哈希、训练与推理日志、checkpoint、逐图和来源组统计、预测、开始结束空间。
- 不检查或终止其他 CUDA 进程，不设空闲显存门槛；以真实 CUDA OOM 为 GPU 资源停止条件。OOM 或其他实际错误均保存失败状态和诊断，不回退 CPU、不覆盖失败记录。
- 不接触或修改论文文件。
