# SAR-only 条件扩散探索协议（2026-11-23）

**状态：** 已预登记，等待 SwinIR SAR 任务适配 run 结束并产生 `best.pt` 后执行。此协议是新的探索性方案，不修改 C3-A v1/v3、C2 gate 或研究总纲。

## 研究问题

在固定 C1 v2 x2-light 表示域 train/development 上，像素域条件扩散能否比确定性 SAR-only 基线更好地拟合 noisy HR pseudo-target？在同一扩散网络和预算下，以 SAR 上采样图像为条件的直接生成，与以 SAR-only SwinIR 确定性预测为中心的残差生成，是否呈现不同的结构、亮点、单 draw 和样本集合行为？

这不是 clean-SAR 恢复检验，也不检验 Optical 增量、物理辐射一致性或经过校准的不确定性。

## 数据与权限

- 唯一输入清单：C1 v2 `x2_light` train 1,600 样本/92 来源组，development 200 样本/19 来源组；只读取 SAR `lr` 与 noisy HR `target` 数组。
- 每张影像按 `files.csv` 绑定的 SHA-256 校验；train/development 来源组互斥。
- confirmation/test 载荷、标识和指标均不访问；optical payload 不读取。
- development 曾用于历史训练/选择，只用于探索性模型选择和诊断，不作为独立泛化证据。
- HR target 是归一化 8-bit 表示域 noisy pseudo-target，不称为 clean SAR 或物理定标量。

## 两个预登记条件

1. **SAR-only Conditional Diffusion**：目标 `x0 = 2 × target − 1`；条件为冻结 SAR LR 以双三次插值放大至 HR 的单通道图，`mode=bicubic, align_corners=false`；去噪器输入 `[x_t, condition]`。
2. **SwinIR + Residual Diffusion**：加载同一 C1 v2 SAR-only SwinIR run 的开发组均衡 PSNR 最佳权重，先从 SAR LR 产生确定性 base，再预测 residual `target − base`；条件为映射到 `[-1,1]` 的 base，即 `2 × base − 1`。residual scale 仅由 train 的逐图 residual RMS 等权汇总得到，公式固定为 `scale = 1 / sqrt(mean_i(mean(residual_i²)))`；development 不参与归一化。reverse process 不裁剪归一化 residual；最终输出为 `clip(base + residual_sample / scale, 0, 1)`，并逐图记录裁剪比例。

两种扩散模型使用相同 U-Net、训练样本身份、batch、优化器、噪声日程、epoch 数、验证选择规则、DDIM 步数、draw 数和评价器。两者各自复位至相同 seed，使初始化、训练噪声和开发采样初始噪声可配对。

## 固定模型与训练设置

- 条件 U-Net：两个输入通道（`x_t` 与条件图），base width 32、两次下采样、通道 32/64/128、GroupNorm+SiLU、正弦时间嵌入 128 维、残差块注入时间嵌入、单通道噪声预测头；不加 attention。
- 像素域 DDPM：1,000 个离散扩散步；线性 `beta` 从 `1e-4` 到 `0.02`；预测 `epsilon`，训练损失为均方误差。
- EMA：衰减 `0.999`；validation 与采样使用 EMA 权重。
- seed：两条件均为 `20261124`；Python、NumPy、PyTorch CPU/CUDA 固定种子；确定性算法开启。
- 训练：30 epochs、batch 4、SAR LR patch 64×64（HR 128×128）、同步随机裁剪与水平/垂直翻转/转置、Adam `1e-4`、weight decay 0、无 AMP、DataLoader workers=0。
- 开发选择：每个 development sample 使用固定中心 64×64 SAR LR crop、预登记固定 timestep/noise；按来源组等权的 epsilon MSE 最小值选 EMA checkpoint，同分取较早 epoch。每个 epoch 评估同一组开发随机量。
- 失败恢复：发生真实 CUDA OOM 时保留日志和最后 checkpoint；在全新 run 目录用 batch 2 + 两步梯度累积重启，以维持 effective batch 4。若仍 OOM，写入失败记录并将 patch 改为 48×48 的后续修订协议；不覆盖已有 run。

## 生成与评价

- 采样：DDIM 20 步、`eta=0`、共 4 draws/sample。每个 sample/draw 的初始 Gaussian noise seed 由固定 run seed、sample ID 与 draw 索引确定；两种生成条件使用相同噪声种子。
- 输出：保存每个 development sample 的 4 个完整 float32 `[4,256,256]` draw；同时保存每 draw 和逐像素样本均值的指标。两者不能相互替代。
- 图像指标：PSNR（target data range 1）、SSIM（项目冻结实现）、Sobel EPI/edge strength ratio、由 LR 99.5 百分位生成的亮点区域均值和 95 百分位保持比、由 LR 20 百分位生成的背景均值比。
- 分布补充量：复用 `scripts/distribution_metrics.py` 的逐像素 CRPS、按 `sqrt(H×W)` 归一化整图 energy score、相邻 variogram score；按每张图计算。K=4 只作低样本数探索分数，不称为完整或校准的条件分布评价。
- 比较：将每个方法的 4 个 draw 分开保存；同一幅图上先对 4 个 draw 的指标取均值（这不是像素均值图），再按 sample ID 与 SwinIR/C2 配对并按来源组 bootstrap 10,000 次，报告 95% 区间。另对像素均值图独立计算同一组配对区间；直接 Conditional-vs-Residual 也分别对两种评价对象计算区间。PSNR/SSIM 不单独等同于 SAR 去斑质量。
- 成本：记录训练墙钟、采样总耗时/每图耗时、参数量、峰值 allocated/reserved GPU 显存、磁盘空间与所有失败。Residual Diffusion 额外报告 SwinIR base 计算时间。

## 预先声明的 Research Decision Review 规则

该规则只决定是否继续研究本配置，不构成论文阶段 gate 或独立确认：

- **CONTINUE（保留候选继续做有界复核）：** Residual Diffusion 在预先指定的 primary metrics（PSNR、SSIM、EPI correlation）中，至少有一个评价对象（draw 指标逐图平均，或像素均值图）相对 SwinIR 的来源组配对 95% 区间下界大于零；同时没有亮点保持或 edge strength 的明确严重退化，且 clipping 未显示大量饱和。需把结论标为单 seed、重复使用 development 上的探索信号。
- **PIVOT（改进生成参数化/约束后重新预登记）：** 结果混合、区间跨零、收益只出现在 pixel mean 或只出现在样本评分中、或者出现 PSNR/SSIM 与结构/亮点之间的 trade-off。不得据此直接扩大训练预算。
- **STOP（停止当前两种配置）：** Conditional 与 Residual 两种配置在两个评价对象的所有 primary metrics 上均无正向支持（对应区间上界不大于零），且亮点/结构没有可见补偿优势。该决策只停止这两种具体单 seed 配置，不否定其他扩散方法。

样本分布评分、跨 draw 标准差、draw-pair 差异和 clipping 作为辅助证据，不能替代 primary metrics 或把 K=4 解释为校准不确定性。即使满足 CONTINUE，也须另建合格独立 development 协议并重复验证后，才考虑晋级或光学条件实验。

## CUDA 执行规则

用户已明确要求本轮 CUDA 实验不因并行评价脚本或 3 GiB 空闲显存门槛而拦截，允许以真实 CUDA OOM 作为资源停止边界。训练前记录设备和显存快照，但不将进程名/PID 或空闲阈值作为启动门槛，不终止外部进程。该授权仅适用于当前 SAR-only 探索任务，不自动改变其他冻结协议或未来实验的资源规则。

## 解释规则

- 单 seed 与历史重复使用的 development 结果仅支持受限探索结论。
- diffusion 样本之间的差异是生成随机性，不等于校准不确定性。
- pixel mean 是点估计，不代表任何单次 draw 的生成质量，也不等于真实样本分布。
- C3-A Flow 的失败不预判本协议中的两种 DDPM 条件；任何一个条件失败都按该配置和 seed 的观察记录，不推广为“扩散无效”。
