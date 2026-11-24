# SAR-only 基线谱系审计（2026-11-23）

## 审计范围

核对当前 C2 SAR-only 确定性 checkpoint、C3-A Flow 实现，以及本轮正在训练的 SwinIR 任务适配版。审计只读取代码、run manifest、checkpoint 元数据和哈希；用于新实验的影像载荷仅限冻结的 train/development SAR LR 与 HR noisy pseudo-target。confirmation/test 影像及测量值未访问。

## 谱系与性质

| 对象 | 谱系类别 | 核验事实 | 可支持的定位 |
|---|---|---|---|
| `C2MatchedJoint` SAR-only | `INTERNAL_BASELINE` | 定义见 `MMCANetV2/phase_c2_matched.py`，调用项目内部 `JointJDSR/model.py::ResidualBlock`。结构为 32 宽度、4 个残差块、x2 PixelShuffle、16 宽 HR head；SAR-only 路径向兼容输出头提供全零 optical 张量。 | 项目内部的匹配确定性 joint 控制，不是论文发表方法，也不是强外部 baseline。 |
| C2 x2-light `best.pt` | `INTERNAL_BASELINE_CHECKPOINT` | 路径为 `experiments/phase_c2_c1_v2_20261031_v1/matrix/runs/x2_light__sar_only/checkpoints/best.pt`；SHA-256 `8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165`。结构参数 32/4/16；seed 20261029；Adam、学习率 `1e-4`、L1、30 轮、batch 8、无增强/随机裁剪；按 group-balanced development PSNR 选择。 | 历史项目内确定性参考。权重和开发集已用于选择，结果属于探索，不是独立泛化证据。 |
| C3-A SAR-only Flow | `INTERNAL_PROPOSED_METHOD` | 代码和冻结配置位于 `MMCANetV2/phase_c3a_flow.py` 与 `protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json`。已按其冻结 v1 完成 5 轮及 v3 评价；Flow 相对上述内部确定性 checkpoint 的来源组配对 ΔPSNR 为 −3.0724 dB，ΔSSIM 为 −0.17818。 | 只说明该像素域 Flow 配置在这次探索 run 未达到预设的追加 seed 条件；不否定其他扩散/生成路线，不改变 C2 gate。 |
| 仓库既有 `SwinIRJoint` | `TASK_ADAPTED_IMPLEMENTATION` | `JointJDSR/model.py` 使用 vendored `BasicSR/basicsr/archs/swinir_arch.py`，SAR 单通道、x2、Sigmoid、随机初始化；本地封装既有配置为 embed 60、depths `(2,2,2,2)`、heads `(4,4,4,4)`。没有外部权重。 | 采用 SwinIR 组件的本地 SAR 任务封装，不是原论文预训练权重复现，也不是 SAR 专用发表方法。 |
| 本轮 SwinIR-Light SAR 单通道 run | `TASK_ADAPTED_IMPLEMENTATION` | 使用 vendored SwinIR 上游实现，embed 60、depths `(6,)`、heads `(6,)`、window 8、MLP ratio 2、pixelshuffledirect x2；单 SAR 通道和 Sigmoid，随机初始化。训练只读 C1 v2 train 1,600/92 组及 development 200/19 组，不加载 optical 或 confirmation/test。run 目录见 `experiments/published_baselines_sar_only_20261123_v1/运行_swinir_20261123_v1/`。 | 这是已发表通用恢复架构的 SAR 任务适配探索。即使训练与评价完成，也不能称为 SwinIR SAR 论文复现或独立泛化结果。 |

## SwinIR 导入问题及处理

直接经 `basicsr` 顶层包导入失败，因为本仓库 vendored 子树缺少 `basicsr.data`，而 `BasicSR/basicsr/__init__.py` 会无条件导入该子包。未修改 BasicSR 文件；训练入口以 namespace package 只加载仓库内 `swinir_arch.py`、registry 和 SwinIR 所需的 `to_2tuple`/截断正态初始化辅助函数。失败日志和随后通过的入口合同检查均保存在本轮实验目录。一次全图 CUDA 反向传播首批探测通过，峰值 PyTorch 分配显存 581.4 MiB、耗时 1.622 秒；该探测未触及 OOM。

## 比较边界

- C2 `best.pt` 来自相同的冻结 x2-light train/development 身份集合，但其训练发生在历史 run；本轮使用其保存模型在当前同一 development 清单上重算指标。
- 因开发集已重复参与历史选择，SwinIR run 的 best checkpoint 选择及所有配对差只能视为探索观察。
- noisy pseudo-target 是归一化 8-bit 表示域目标；PSNR/SSIM 仅描述对该目标的一致性，不等于 clean SAR 恢复质量。
- 本轮 SwinIR 未使用作者发布的 RGB/Natural-image 权重，训练参数也不是原论文完整训练 recipe；相对原论文的结构及数据适配差异必须在报告中保留。
- 生成式比较须把单 draw、样本像素均值和样本分布分开。Flow 的历史负结果不能替代 SAR-only conditional diffusion 或 SwinIR + residual diffusion 实验。

## 可追溯源

- SwinIR 论文与官方代码：[论文](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)、[作者仓库](https://github.com/JingyunLiang/SwinIR)。
- 项目内部 checkpoint：上列 C2 matrix run manifest、checkpoint context/model config 和哈希；当前主状态汇总见 [CURRENT_STATE.md](CURRENT_STATE.md)。
- Flow Research Decision Review：[C3-A 复核](C3-A%20SAR-only%20Flow%20决策复核_20261123.md)。
