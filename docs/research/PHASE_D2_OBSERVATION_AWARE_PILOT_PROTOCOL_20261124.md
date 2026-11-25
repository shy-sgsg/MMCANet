# Phase D2 观测约束扩散先验 Pilot 冻结协议

**状态：执行前冻结**  
**日期：2026-11-24**  
**方法选择：** [Phase D2 方法选择](PHASE_D2_METHOD_SELECTION.md)  
**代码入口：** `experiments/phase_d2_observation_aware_20261124_v1/train_prior.py`、`sample_posterior.py`

## 研究假设

已训练的 HR SAR noisy pseudo-target 扩散先验，若在生成每一步使用已核验的 C2 `x2_light` 表示域算子和 Gamma 观测项引导，应减少缺少观测依据的高频改动，并改善至少一项观测一致性量；若点估计/结构质量明显差于 SwinIR 或观测项无改进，则停止当前配置。

该实验只研究 SAR 单模态表示域，target 是 noisy pseudo-target；不检验 Optical 增益，不作 clean-SAR、辐射标定或独立泛化主张。

## 数据与隔离

- 输入固定为 C1 v2 `x2_light`：train 1,600/92 来源组，development 200/19 来源组。
- 可读影像仅限 train/development 的 SAR `lr` 和 `target`；训练仅使用 train；development 只按固定中心 crop 计算 epsilon MSE，并在 Pilot 结束时评价模型。
- 禁止读取 optical、confirmation/test 影像/清单成员/指标，禁止新增或重采样退化数据。
- train/development 来源组不重叠；样本和 LR/target 文件哈希由固定 manifest 及加载器校验。

## 模型与训练配置

- 无条件 prior 使用已有两输入通道时序 U-Net：输入为加噪 HR 与全零 null-condition；`base_channels=32`、`time_dim=128`。
- target `x∈[0,1]` 映射为 `x0=2x−1`；epsilon prediction，1000 个 diffusion step，线性 beta 从 `1e-4` 到 `0.02`。
- 固定 `seed=20261124`；batch 4；64×64 LR 坐标 crop 对应 128×128 HR crop；随机翻转和转置只用于训练。
- 最多训练 5 epochs；AdamW `lr=1e-4`、weight decay 0；EMA decay `0.995`。每轮用固定中心 HR crop 与按 sample ID 导出的固定噪声，记录来源组等权 development epsilon MSE；以其最小值选 best，另存 last 和完整优化器/RNG 状态。
- 不做新超参数搜索。任何阈值/权重不得在看到本次结果后调整。

## 观测能量和采样

- 固定算子：`degradation.py` SHA-256 `b7c0e9208a132792824bdde4296d7d3650587a144bf7ab4ec34bb161e0cb7bd1`；129×129 中心频域截断、双线性 128×128 下采样（`align_corners=false`）、非负 clamp、Gamma(shape=8, rate=8) 表示域乘法、无上界 clamp。前一轮 1,800 张 train/development LR 已逐位复现，最大误差 0。
- 正观测项：`8 * mean(log(max(a,1e-8)) + y/max(a,1e-8))`，省略只依赖观测的常数；`a` 是 clamp 后前向均值。
- 零观测项：对 `y=0` 使用未 clamp 值 `z` 的 `mean(relu(z)^2)`；它是单侧可微约束近似，不宣称精确 Gamma 后验。组合项固定 `positive_nll + zero_constraint`，同时报告零支持违规率。
- 用 20 个线性间隔反向时刻执行 deterministic DDIM；每步对 denoised `x0` 的观测能量反传并按固定权重 1 引导 epsilon。每次样本输出 clamp 到 `[0,1]`。采样 seeds 固定为 `20261125`、`20261126`、`20261127`、`20261128`。

## Pilot 评价与决策规则

- 对全部 200 张 development 生成四个 draw；分别评价每个 draw、四 draw pixel-mean 与 draw 集合统计。
- 同一固定输入下与已有 SAR-only SwinIR 配对报告 PSNR、SSIM、RMSE、EPI、边缘强度误差、亮值均值/P95、背景、正值观测负对数似然、零支持违规率、四 draw 方差/两两差、参数量、运行时间和峰值显存。
- 以来源组等权配对差及 10,000 次来源组 bootstrap（seed `20261124`）报告区间。旧 development 已反复使用，任何区间仅为描述性探索，不是独立验证、晋级或泛化证据。
- 若损失/梯度非有限、checkpoint 写入失败或实际 CUDA OOM，保存失败日志并停止；不检查/终止其他进程，不设置 3 GiB 启动门槛。
- 5 epoch 完成后进行 Research Decision Review：质量与观测量均无有意义改善则 STOP；出现一致的质量/观测改善则只提出下一阶段 Candidate 预算，不自动改写总纲或运行其他旧配置。

## 资源与输出

- 固定新目录：`experiments/runs/phase_d2_observation_aware_pilot_20261124_v1/`；存在时失败，不覆盖。若在首个 optimizer step 前发生实现/环境异常，保留该失败目录并在唯一新目录 `_retryNN` 中按相同配置恢复；不能覆盖或清理失败证据。
- 记录 Python/PyTorch/CUDA/GPU、代码与协议哈希、配置、manifest、命令、训练日志、best/last checkpoint、推理输出、逐图和来源组结果、耗时及显存。
- 启动前实测持久化分区约 47 GiB 可用（约 50 GB）；训练和评价后复核余量。只清理本轮创建且确认可重建的临时文件；不清理数据、checkpoint 或历史实验。
