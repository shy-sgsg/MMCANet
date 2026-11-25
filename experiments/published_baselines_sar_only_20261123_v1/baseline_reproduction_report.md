# 已发表架构 SAR 任务适配基线复现记录

## 方法身份

本仓库运行的是 SwinIR 与 EDSR 的 SAR 单通道 x2 `TASK_ADAPTED_PUBLISHED` 版本：保留公开发表架构的核心主干，按本项目归一化 SAR 表示域改为单通道输入/输出、从随机初始化训练，并使用 Sigmoid 输出。没有使用作者 SAR 权重，也没有复现原论文的完整数据、预训练权重或训练 recipe，因此报告的是任务适配实验，不是 SAR 论文结果或原方法精确复现。

DSRKD 为直接 SAR 联合去斑与超分方法，但现阶段缺可核验作者实现、权重与完整协议，列为“不可执行/不报告本地数值”。不会创建自研网络并以 DSRKD 名义登记。

## 可复现运行

| 方法 | 冻结配置 | 运行目录 | 关键结果 |
|---|---|---|---|
| SwinIR | [`swinir_sar_task_adaptation.json`](baseline_configs/swinir_sar_task_adaptation.json)；30 轮、seed 20261123、随机初始化 | [`运行_swinir_20261123_v3`](运行_swinir_20261123_v3) | best epoch 30；组均衡 PSNR 21.6352 dB；参数量 252,784 |
| EDSR | [`edsr_sar_task_adaptation.json`](baseline_configs/edsr_sar_task_adaptation.json)；30 轮、seed 20261125、随机初始化 | `experiments/runs/phase_d1_edsr_sar_20261124_v1` | best epoch 29；组均衡 PSNR 21.7316 dB；参数量 1,367,553 |

两者使用 C1 v2 固定 train 1,600 张/92 组与 development 200 张/19 组，只载入 SAR LR 和 noisy pseudo-target。Optical 未载入，confirmation/test 未访问或用于选模。所有 run 均记录协议、代码哈希、配置、命令、环境、checkpoint、训练日志、逐图指标、组统计和资源；大 checkpoint 与 NPY 预测留在本地唯一 run 目录，不提交 Git。

## 复现注意事项

**运行计时口径：** `runtime.json` 中的 `training_seconds` 覆盖训练、逐轮 development checkpoint 选择评价及最终 development 全量评价；SwinIR 计时还包含内部 C2 对照推理。比较表以完整运行累计耗时呈现，不能当作纯训练时间。

1. 使用 AISAR 环境及实际 run 中记录的 PyTorch/CUDA 版本；当前运行环境为 PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop GPU。
2. 根据各 run 的 `command.txt` 使用固定 train/development 输入根目录。输入必须是冻结 C1 v2 x2_light manifest 对应的同一数据，不得按文件名重新划分。
3. SwinIR 和 EDSR 运行使用单 seed，仅可作为探索复现；development 已重复用于训练和模型选择。不要把 bootstrap 区间解释成对独立场景的外推。
4. EDSR 运行清单未在使用前绑定 SwinIR 预测文件 SHA-256。本轮已用 SwinIR 来源运行、基线 checkpoint、相同 split hashes，以及 D2 清单的逐文件哈希做训练后交叉核验；限制见[来源核验报告](../../docs/research/PHASE_D1_PAIRED_BASELINE_PROVENANCE_AUDIT_20261124.md)。新的正式运行须在开始时绑定这些来源。
5. DSRKD、SAR-BM3D + EDSR 等仍未满足可核验实现/参数/数据域条件；不要把未执行的方法补写为结果。

## 核心限制

这两种网络是通用发表架构的本地 SAR 任务适配；C1 v2 noisy pseudo-target 是归一化 8-bit 表示域目标。当前 PSNR/SSIM 不能说明真实 clean SAR 去斑或辐射物理一致性。已有两种基线也没有建立 matched Optical 的增量价值。
