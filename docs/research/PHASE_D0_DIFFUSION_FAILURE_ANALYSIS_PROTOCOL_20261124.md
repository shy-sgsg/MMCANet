# Phase D0 扩散失败机理诊断方案

**状态：执行前固定**  
**日期：2026-11-24**  
**目的：** 解释现有 SwinIR-centered Residual DDPM 为何未超过确定性 SwinIR；不训练、不调参、不选择新 checkpoint。

## 1. 证据边界

- 只使用 C1 v2 的 train（1,600 张、92 个来源组）与 development（200 张、19 个来源组）中的 SAR `lr`、noisy pseudo-target `target`、已有 SwinIR 预测和已有 Residual DDPM 四个 development draw。
- 不打开 optical 数组；不访问 confirmation/test 影像、指标或载荷。train/development 角色清单只读取各自目录的样本身份、来源组和 patch 元数据。
- development 已反复用于历史训练/检查点选择，本诊断仅是探索性机理分析，不构成独立泛化证据、正式模型选择、晋级或新超参数依据。
- 已公布的聚合指标和“停止当前 Residual DDPM 配置”的决策在本方案形成前已知。本方案只约束新增的逐图残差诊断，不能被描述为对旧结果的前瞻性注册。
- 全程使用 CPU 后处理，不启动训练、推理或 GPU 任务；不更改历史 checkpoint、预测或指标。

## 2. 输入和对齐规则

以冻结 SAR 输入 manifest 中的 `sample_id` 为唯一键，分别校验 train/development 与既有 SwinIR base prediction 的文件集合完全相等。Residual DDPM 只在 development 有四个已保存 draw。每张影像按单样本读取，检查 shape、有限值及范围，不把数组写回原目录。

对每个样本定义：

\[
r_t = x_{target}-x_b,\qquad r_d^{(k)}=x_d^{(k)}-x_b,
\]

其中 `x_b` 为 SwinIR 输出，`x_d^(k)` 为第 k 个已保存 draw。四 draw 像素均值为 \(\bar x_d\)，对应 \(\bar r_d=\bar x_d-x_b\)。

## 3. 预先规定的逐图量

### 3.1 SwinIR 目标残差分布

对 train 和 development 的 `r_t` 逐图计算均值、标准差、RMS、绝对值中位数、90/95/99 分位数及频谱能量比例。高频/中频/低频按径向频率（cycles/pixel）固定为：低频 `f < 0.125`、中频 `0.125 ≤ f < 0.25`、高频 `f ≥ 0.25`；报告每个频带能量除以总频谱能量后的比例。

train 的 SwinIR 预测来自模型已见样本，development 预测来自未用于 SwinIR 梯度训练的样本。因此两者差异仅称为“训练内与开发集残差差异”，不将其单独归因于来源分布偏移。

### 3.2 残差方向和点估计效果

对每个 development draw 及 draw 像素均值，计算：

- 残差余弦相似度 \(\langle r_d,r_t\rangle/(\|r_d\|_2\|r_t\|_2)\)；
- 去均值 Pearson 相关；
- 残差 RMS 比值；
- 像素符号一致率；
- `r_d` 对 `r_t` 的内积及其符号；
- 基础输出与生成输出各自对 pseudo-target 的 MSE 及差值；
- 无约束 oracle 系数 \(\alpha^*=\langle r_d,r_t\rangle/(\|r_d\|_2^2+\epsilon)\)，以及 \(x_b+\alpha^*r_d\) 的 oracle MSE。

oracle 使用目标残差，只作为“已知目标后的理论缩放诊断”；绝不用于选系数、修改预测、选择 checkpoint 或主张可部署效果。不得将其称为模型增益。

### 3.3 结构和高值区域

高值、背景、边缘 mask 仅复用现有正式评价器 `build_fixed_masks(lr, target, scale=2)` 的定义，不另行拟合阈值。分别在整体、high-return、background、edge mask 内计算目标/预测残差 RMS、方向余弦和 MSE 变化。该区域分析仍依赖 noisy pseudo-target mask，解释限于当前表示域。

### 3.4 四次生成的随机性

逐样本报告四 draw 之间两两平均绝对差、像素级样本标准差均值与 RMS，以及像素均值相对单次 draw 和 SwinIR 的 MSE 差。四次均值只作为点估计诊断，不能代表单次生成质量或分布真实性。

### 3.5 来源组汇总和不确定性

先按来源组对样本量取均值，再对来源组等权平均。development 的主要残差方向量另以固定 seed `20261124` 做 10,000 次来源组 bootstrap，报告百分位 95% 区间。该区间只描述此次已复用 development 的组间波动，不转化为独立泛化置信区间。

## 4. 训练目标与代码审查

- 从现有逐轮日志逐字核对 train/development epsilon MSE 与 checkpoint 选择记录。
- 只有当同一 epoch 同时存在已保存的完整采样预测和像素级质量指标时，才计算 epsilon MSE 与采样指标的跨 epoch 关联；缺数据则明确不可估，不补跑旧 checkpoint。
- 静态核对 beta/alpha-bar、前向加噪、epsilon 反推 `x0`、DDIM 时间索引、条件张量、residual scale、EMA、恢复逻辑及裁剪位置。任何疑点先与 run 中记录的代码哈希和实际来源版本核对；仅凭工作树当前源码不能推断旧 run 执行过何种代码。
- 只有可复现的源码与运行证据共同证明具体缺陷时才报告实现错误；不通过事后改算法解释或挽救旧 run。

## 5. 产物

在新唯一目录 `experiments/diagnostics/phase_d0_residual_failure_20261124_v1/` 写入逐图诊断、来源组摘要、统计 JSON、输入来源说明、命令和环境记录。核心解释写入 `docs/research/PHASE_D0_DIFFUSION_FAILURE_ANALYSIS.md`。所有输出只引用 train/development SAR 载荷，不复制原始图像或 checkpoint。

