# SAR-only Published Baseline 与 Diffusion 决策复核（2026-11-24）

## 决策摘要

- **C3-A 像素域 Flow Matching：STOP 当前配置。** 冻结的第二 seed 继续条件未达到；已在[此前复核](C3-A%20SAR-only%20Flow%20决策复核_20261123.md)记录，不追加相同配置的 seed。
- **SAR-only Conditional DDPM：STOP 当前配置。** 四个生成 draw 均出现明显亮区和背景幅度压低，PSNR、SSIM、EPI 均远低于确定性方法。
- **SwinIR-centered Residual DDPM：STOP 当前质量优先配置。** 残差条件比直接 SAR 条件明显改善，但其 pixel-mean 仍在主要结构与一致性指标上显著落后于 SwinIR 任务适配基线，四 draw 推理也约慢 111 倍。
- **不据此否定所有生成模型。** 当前结果没有验证经校准的 SAR 后验、真实散射分布或生成样本对下游任务的独立价值。若以后提出不同的、可证伪的分布目标和合格的新 development 协议，再另行评审；本轮不再追加同类 GPU 训练。
- **不进入 C3-B。** 本次未使用 optical，C2 v2 gate 仍未判定，C1 v2 development 已被重复使用；C3-B 仍未获授权，须先解决新的独立 development 数据协议和阶段条件。

## 实验范围与可复现设置

使用 C1 v2 冻结 x2-light 派生输入：train 1,600 张、92 个来源组；development 200 张、19 个来源组。训练和比较只读取 SAR LR 与 noisy pseudo-target。未重新划分数据；development 已有历史训练/选模复用，因此本报告是单 seed、范围受限的探索观察，不是独立泛化证据。

两项扩散条件共用 `ConditionalUNet`、Adam `1e-4`、batch 4、LR patch 64、30 epochs、seed `20261124`、1,000 个训练扩散步、EMA 0.999、DDIM 20 步和每图 4 draws。Conditional DDPM 直接以 SAR x2 插值图为条件；Residual DDPM 以 SAR-only SwinIR 输出为中心，残差尺度只由 train 估计（RMS `0.0952794`，倒数尺度 `10.49545`）。无 AMP。基线及扩散方法使用同一 development 身份、固定目标、掩膜和评价实现。

GPU 为 NVIDIA GeForce RTX 3050 Laptop，PyTorch `2.8.0+cu128`、CUDA `12.8`、Python `3.13.7`，总显存 `3,942,383,616` 字节。依用户明确指示，没有拦截并行 CUDA 评价任务，也没有使用 3 GiB 空闲门槛；按实际 OOM 作为资源停止条件。续训、采样和汇总日志中没有 CUDA OOM 记录；Residual 原训练进程在 epoch 24 后停止，具体退出原因未查明，内核日志也没有 OOM 证据，随后由 checkpoint 成功续跑。扩散协议 SHA-256 为 `b0d54871d7ec7cf4f248dd8ed271619a459e11ba2fe8a8d4c6af1a88e3c4f3b7`，配置 SHA-256 为 `4e92974e0ff1f810340a0ae6005576e2179f18e3915a9d3d7f53e25c9a638c43`。

## 确定性参照

以下为 200 张 development 上的来源组等权均值。`SwinIR` 是用上游 SwinIR 架构源码构建的 SAR 单通道任务适配版，随机初始化且未使用作者 SAR 权重；不是作者原 recipe 的复现。`C2MatchedJoint` 是项目内部 SAR-only checkpoint，不是已发表强基线。比值指标以接近 1 为更接近 pseudo-target 的均值/幅度参照；所有像素指标均在归一化 8-bit 表示域 noisy pseudo-target 上计算。

| 方法 | PSNR（dB） | SSIM | EPI corr | 边缘幅值比 | 亮区均值比 | 亮区 P95 比 | 背景均值比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 项目内部 C2 SAR-only | 21.6082 | 0.3627 | 0.3557 | 0.2551 | 1.0077 | 0.9503 | 1.0139 |
| SwinIR SAR 任务适配 | 21.6352 | 0.3597 | 0.3587 | 0.2379 | 0.9998 | 0.9415 | 1.0184 |

## 生成结果：单次 draw 与 pixel-mean 分开报告

像素均值图是四个样本的点估计，不能代表单次生成的质量，也不能代表生成分布。下表均为来源组等权值。

| 方法与评价对象 | PSNR（dB） | SSIM | EPI corr |
|---|---:|---:|---:|
| Conditional DDPM，单次 draw | 11.5083 | 0.2131 | 0.2076 |
| Conditional DDPM，四 draw pixel-mean | 11.7661 | 0.2889 | 0.2827 |
| Residual DDPM，单次 draw | 20.0948 | 0.3207 | 0.3176 |
| Residual DDPM，四 draw pixel-mean | 21.0971 | 0.3320 | 0.3427 |

| 方法与评价对象 | 边缘幅值比 | 亮区均值比 | 亮区 P95 比 | 背景均值比 |
|---|---:|---:|---:|---:|
| Conditional DDPM，单次 draw | 0.5321 | 0.7426 | 0.7573 | 0.6810 |
| Conditional DDPM，四 draw pixel-mean | 0.3800 | 0.7426 | 0.7289 | 0.6810 |
| Residual DDPM，单次 draw | 0.2106 | 0.9777 | 0.9394 | 1.0328 |
| Residual DDPM，四 draw pixel-mean | 0.1888 | 0.9777 | 0.9322 | 1.0328 |

Conditional DDPM 的 PSNR 比 SwinIR 低 `9.8691 dB`（来源组 bootstrap 95% CI `[−11.3081, −8.5247]`），亮区均值比和背景均值比分别低 `0.2572`、`0.3374`；该条件配置明显压暗输出。Residual DDPM 相比 Conditional DDPM 的 pixel-mean PSNR 提高 `9.3310 dB`（95% CI `[8.0080, 10.7223]`）、SSIM 提高 `0.04309`、EPI corr 提高 `0.05994`。这说明 SwinIR 中心残差条件比直接 SAR 条件有效，但不代表达到确定性基线。

Residual DDPM 的 pixel-mean 相对 SwinIR 与内部 C2 的来源组配对差异如下。区间按 19 个来源组进行 10,000 次 bootstrap；它们只描述本次历史 development 上的差异。

| 指标 | Residual − SwinIR | Residual − 内部 C2 |
|---|---:|---:|
| PSNR（dB） | `−0.5381` `[−0.6072, −0.4713]` | `−0.5111` `[−0.5763, −0.4475]` |
| SSIM | `−0.02773` `[−0.03269, −0.02238]` | `−0.03070` `[−0.03613, −0.02453]` |
| EPI corr | `−0.01598` `[−0.01886, −0.01305]` | `−0.01304` `[−0.01684, −0.00883]` |
| 边缘幅值比 | `−0.04909` `[−0.05394, −0.04411]` | `−0.06636` `[−0.07157, −0.06068]` |
| 亮区均值比 | `−0.02219` `[−0.02698, −0.01760]` | `−0.03005` `[−0.03506, −0.02523]` |
| 亮区 P95 比 | `−0.00930` `[−0.01339, −0.00532]` | `−0.01813` `[−0.02221, −0.01438]` |
| 背景均值比 | `+0.01441` `[+0.00789, +0.02051]` | `+0.01892` `[+0.01204, +0.02559]` |

这些区间支持“当前 Residual DDPM 在本 development 上逊于这两个确定性参照”的观察；结构、亮区和背景比值存在不同方向的偏差，不应将单一 PSNR 作为全部结论。SAR-only SwinIR 仍是任务适配的确定性参照，不能冒称已发表 SAR 专用基线。

## 生成变化、稳定性与成本

| 方法 | pixelwise sample std 均值 | draw 两两绝对差均值 | pixelwise CRPS | energy score | adjacent variogram score | 同噪声重算最大差 |
|---|---:|---:|---:|---:|---:|---:|
| Conditional DDPM | 0.06760 | 0.09531 | 0.21605 | 0.22815 | 0.004041 | 0 |
| Residual DDPM | 0.04509 | 0.06387 | 0.05529 | 0.07614 | 0.004685 | 0 |

每种方法均保存 200 个 `[4,256,256]` 预测阵列、800 行逐 draw 指标、200 行 pixel-mean 指标、200 行分布评分和 19 个来源组统计。相同噪声重算误差为 0，说明实现对固定噪声可重复；不同 draw 的离散程度只说明样本变化。4 draws 没有足够的分布级参考，CRPS、energy 与 variogram 分数是在 noisy pseudo-target 上的有限样本经验评分，不能称作后验校准、不确定性校准或真实散斑分布验证。

| 方法 | 训练时间 | 200 张评价/采样时间 | 参数量 | 训练峰值已分配显存 |
|---|---:|---:|---:|---:|
| SwinIR SAR 任务适配 | 7,406.7 秒 | 18.58 秒 | 252,784 | 494.7 MiB |
| Conditional DDPM | 2,099.0 秒 | 2,737.4 秒（四 draw） | 958,401 | 521.9 MiB |
| Residual DDPM | 4,458.5 秒 | 2,059.1 秒（四 draw） | 958,401 | 549.9 MiB |

Residual DDPM 四 draw 评价耗时约为 SwinIR 单次确定性推理的 `110.8×`；固定四 draw 预算下的 diffusion 全运行训练与采样合计 `11,354.1` 秒。扩散 run 峰值已分配/保留显存分别为 `549.9/634.0 MiB`。确定性 SwinIR base 对 train/development 的总推理时间未保存在初始 run 中，故不估算并在 runtime 中标为缺失。

## 运行连续性与记录限制

Residual 训练进程在第 24 轮 checkpoint 后停止，日志和 kernel 记录没有给出 CUDA OOM 证据。续跑从 epoch 24 的 `last.pt` 恢复模型、EMA、optimizer 与 CUDA RNG，完成 epoch 25–30；`train_log.csv` 的 epoch 1–24 由原控制台日志重建，全部 30 轮与 best/last checkpoint 均已核对。初始进程退出前未写出 epoch 1–23 的逐图选模行，无法从已覆盖的 checkpoint 还原；现存 `development_selection_per_image.csv` 保留 epoch 24–30 共 1,400 行，运行记录明确列出不可恢复轮次。最终 development 图像级指标仍完整覆盖 200 张。

两次 finalizer 字段/数值类型错误已以定向回归测试修复并成功重新汇总；失败信息保存在恢复记录中，不影响训练、checkpoint、采样文件或成功汇总。完整运行 manifest 标记 `complete`；`test_set_accessed=false`、`test_set_used=false`、`confirmation_payload_accessed=false`、`optical_payload_loaded=false`。

## Research Decision Review

| 研究项 | 决定 | 范围 |
|---|---|---|
| 当前 C3-A Flow Matching | **STOP 当前像素域配置** | 冻结第二 seed 条件未达标；不追加相同配置训练。 |
| SAR-only Conditional DDPM | **STOP 当前配置** | 表示域输出幅度严重压低，且结构指标显著低于确定性参照。 |
| SwinIR-centered Residual DDPM | **STOP 当前质量优先配置** | 比 Conditional DDPM 好，但仍系统性落后于 SwinIR/C2，并承担高采样成本。 |
| 其他生成路线 | **开放，需重新立题** | 若研究目标改为分布效用，必须有独立 development、合格参照/评价和预注册继续门槛；当前 4-draw spread 不足以支持该价值。 |
| Optical 条件 C3-B | **不启动** | 本轮没有 Optical；C2 gate 未判定、新 development 不合格，且没有本次授权。 |

**事实：** protocol/config、固定 train/development 输入、checkpoint、预测数、指标文件和资源可由 run 文件核验。**观察：** 在 seed `20261124` 与重复使用的 19 组 development 上，两个 DDPM 配置均低于确定性参照。**假设：** 其他参数化可能在新的、合格的分布任务上有用，但本轮没有检验。**开放问题：** Optical 增量、独立泛化、SAR 测量/物理一致性与样本分布真实性仍未回答。**无支持主张：** “所有 Diffusion 都无效”“当前输出具物理一致性”或“C3-B 已解锁”均不受这些结果支持。

## 产物

- [实验协议](SAR_ONLY_DIFFUSION_EXPERIMENT_PROTOCOL_20261123.md)、[发表方法注册表](PUBLISHED_METHOD_REGISTRY_20261123.md)、[基线谱系审计](BASELINE_LINEAGE_AUDIT_20261123.md)。
- [完整扩散 run manifest](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/manifest.json)、[汇总](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/summary.json)、[运行资源](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/runtime.json)。
- [逐 draw 指标](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/all_methods_per_draw_metrics.csv)、[pixel-mean 指标](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/all_methods_pixel_mean_metrics.csv)、[来源组统计](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/source_group_stats.csv)、[配对来源组 bootstrap](../../experiments/published_baselines_sar_only_20261123_v1/运行_sar_only_diffusion_20261123_v1/paired_group_bootstrap_vs_swinir.csv)。
- [SwinIR 任务适配结果](../../experiments/published_baselines_sar_only_20261123_v1/运行_swinir_20261123_v3/summary.json)、[运行注册](../EXPERIMENT_REGISTRY.md)和[研究决策记录](DECISION_LOG.md)。
- checkpoint 与 `[4,256,256]` 生成阵列保留在本机唯一 run 目录，不进入 Git；小型配置、manifest、CSV、日志、代码和本报告进入版本控制。
