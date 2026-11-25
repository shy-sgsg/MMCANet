# Phase D2 生成式方法选择与探索性 pilot 设计

**状态：主候选已选择；pilot 协议在运行前冻结**  
**日期：2026-11-24**  
**依据：** [D0 失败诊断](PHASE_D0_DIFFUSION_FAILURE_ANALYSIS.md)、[算子往返协议](PHASE_D2_OPERATOR_ROUNDTRIP_PROTOCOL_20261124.md)、[算子往返结果](../../experiments/diagnostics/phase_d2_operator_roundtrip_20261124_v1/summary.json)、[零值支持域统计](../../experiments/diagnostics/phase_d2_zero_support_20261124_v1/summary.json)

## 选择

主候选选 **B：SAR 表示域观测约束的扩散后验采样**。备选只保留 **C：条件扩散桥/输运**，且仅在 B 被明确的似然/梯度问题否决后重新论证。A“确定性输出上生成残差”本轮不再进入主候选：现有残差余弦接近零、符号一致约 50%，oracle 缩放接近零，生成残差在所有 19 个 development 来源组上均增加 MSE。

### 三个方向的证据比较

| 方向 | 机制与相对现状的差别 | 当前证据与代价 | 决策 |
|---|---|---|---|
| A：确定性—生成式协同 | 确定性输出作中心，生成分支只估计可预测残差 | 当前 Residual DDPM 的单次残差余弦 0.0121、像素均值 0.0185；MSE 差分别 +0.003576、+0.001011，19/19 组均为正。继续堆叠残差分支没有数据依据 | 停止本轮 A；不能推广为所有残差生成无效 |
| B：观测约束扩散 | 学习 HR SAR pseudo-target 的无条件先验；每个采样步通过已核验退化算子的观测代价引导先验样本 | 固定 train/development 的 1,800 个 LR 全部逐位复现，最大误差 0；Gamma 似然可把“生成看起来更锐”与“是否解释观测”分开。代价是逐步反传且零值需按 clamp 支持处理 | **主候选**；先做短 Pilot，失败则停止 |
| C：扩散桥/条件输运 | 从粗恢复或条件状态出发，在数据和观测间学习桥接轨迹 | 可能减少从高斯噪声开始生成的困难，但目前没有证据显示单改起点会修复 D0 的残差方向问题；仍须增加训练目标和采样器 | 仅保留备选，不与 B 同时开发 |

一般图像逆问题中的 DAPS 通过解耦噪声退火改进复杂非线性逆问题的采样修正；A-DPS 研究由受损数据训练的扩散先验用于另一线性观测算子。它们说明应显式建模观测与采样误差，但并不证明其机制可直接迁移到本项目的 8-bit SAR 表示域。扩散后验采样还存在理论上的普遍计算困难结果，故本方案把单次/多次推理成本列为硬报告项，而不宣称获得精确校准后验。[DAPS（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.html)、[A-DPS（ICLR 2025）](https://openreview.net/pdf/75c2ef5bcb7048eca74d59b808dec52fb190df40.pdf)、[扩散后验采样计算复杂性分析（ICML 2024）](https://proceedings.mlr.press/v235/gupta24a.html)。

本方案属于已知生成式逆问题路线的 SAR 任务适配和可证伪实验，不把“使用 Gamma 似然”或“使用扩散后验采样”本身写成新颖性贡献。项目已发表方法登记仍将 DSRKD 等直接 SAR 联合恢复作为必须对照的先前工作；可执行作者实现/权重缺失时须标为不可复现，不得以任务适配版冒充作者复现。

## 数学对象与边界

令 (x\in[0,1]^{256\times256}) 表示由 8-bit SAR 输入经 `log1p(20x)/log1p(20)` 得到的 noisy pseudo-target。已冻结的受控观测链为：

\[
z(x)=\mathcal D_{\mathrm{bilinear},128}\bigl(\mathcal F_{129}(x)\bigr),\qquad
a(x)=\max(z(x),0),\qquad
y=a(x)\,\eta,\quad \eta\sim\mathrm{Gamma}(L=8,\mathrm{rate}=8).
\]

其中频域窗口、双线性 `align_corners=false`、负值 clamp、表示域 Gamma 和无上界 clamp 均来自固定 `x2_light` manifest；`degradation.py` SHA-256 为 `b7c0e920…e0cb7bd1`。此处的 `a(x)` 是表示域前向量，不是经辐射定标的传感器观测模型；(x) 本身已是 noisy pseudo-target，故 Gamma 项是受控复合噪声假设。

对于 (y_i>0)，正值 Gamma 似然的、去掉只依赖 (y_i) 的常数项为：

\[
\ell_+(a_i;y_i)=L\left(\log a_i+\frac{y_i}{a_i}\right),\quad L=8.
\]

精确 (y_i=0) 来自 clamp 后的零支持点，不能当作普通正值 Gamma 观测，也不能只把 (y_i) 替换成任意 epsilon。实现采用分开的正值 Gamma 项与零值单侧约束项；零值项是可微的似然近似/约束能量，须同时报告零支持违规率，不宣称精确后验。正值稳定项下限固定为 `1e-8`，低于 train 最小正 LR `2.08e-7`；零值项固定为 `relu(z)^2`。两项分别按各自有效像素数求均值，组合权重在 Pilot 前固定为 1，不利用 development 调权。

## 可证伪假设与模型框架

**H-D2-B：** 若扩散先验负责表示 HR pseudo-target 分布，而逐步观测引导使用同一固定 LR 的 Gamma 正值似然与 clamp 零支持约束，则一次生成相对既有 SAR-only SwinIR 应减少无观测依据的高频变化，并至少不显著恶化 SAR pseudo-target 的 PSNR/SSIM/EPI；若数据项不能改善观测一致性或图像质量仍明显落后，则停止该后验采样配置。

1. **先验模型：** 复用已存在的小型 U-Net/扩散代码结构，但以固定零条件训练 HR target 的无条件 epsilon-prediction 先验；不复制条件 DDPM 或 Residual DDPM 的训练结论。
2. **训练目标：** (x_0=2x-1)，1000-step 线性 beta schedule，标准 epsilon MSE；固定 SAR LR 不作为网络条件，只在采样阶段进入观测似然。
3. **采样：** 20-step deterministic DDIM；每一步从 denoised (\hat x_0) 计算经 FFT 低通、双线性下采样和 clamp 的表示域观测代价，以 autograd 梯度修正 epsilon。最终输出映射至 `[0,1]`。记录 clipped 比例、正值负对数似然、零支持违规率与梯度范数。
4. **对照：** 已完成的 SwinIR SAR 任务适配版；另建立 EDSR SAR 单通道任务适配对照。两者均标为 `TASK_ADAPTED_PUBLISHED`，不写成作者权重/原方案复现。若 EDSR 尚未完成训练，则本次只能给 SwinIR 单基线探索结果，不能宣称满足完整强基线比较。

## 本轮冻结的 Pilot 预算与评价

- 数据：C1 v2 固定 train 1,600/92 与 development 200/19 的 SAR `lr`、`target`；不读 optical、confirmation 或 test。
- 随机种子：`20261124`；target HR patch 128×128（由 64×64 LR crop 对齐）；batch 4；最多 5 epochs；AdamW `lr=1e-4`、weight decay 0；EMA `0.995`。
- 先验网络：现有 U-Net `base_channels=32`；1000 diffusion steps；beta 从 `1e-4` 到 `0.02`；每 epoch 固定记录 train 与 group-balanced development epsilon MSE，保存 `last` 与验证 epsilon MSE 最低的 `best`。
- 每个 epoch 必须记录 finite loss、梯度范数、输出范围、训练时间、峰值 CUDA allocated/reserved 和剩余磁盘；遇到非有限值、checkpoint 写入失败或**实际 CUDA OOM**即保留日志并停止。未知 CUDA 进程和 3 GiB 空闲显存不作为启动/停止门槛。
- Pilot 完成后在全 200 张 development 上做一次 20-step 单次 draw 和四次 draw；分别报告单次、pixel-mean 和集合稳定性。与 SwinIR 配对报告 PSNR、SSIM、RMSE、EPI、边缘强度、高值均值/P95、背景、正值观测 NLL、零支持违规率、参数量、推理时间和显存。
- 解释范围仅为探索性：development 已反复使用；不因这次结果晋级 C3-B，不读 confirmation/test，不作独立泛化结论。
- 若 Pilot 失败或相对 SwinIR 明显损失结构/高值保持，STOP 当前 B 设置；若出现一致的观测项改善且点估计不显著差于 SwinIR，再依据完整证据制定下一阶段 Candidate，不自动追加旧配置训练。

## 风险和停止条件

- 训练目标拟合 noisy pseudo-target，不是 clean SAR；不能把像素指标解释为去斑物理真值。
- 短 Pilot 可能不足以学到有效先验；只有明确的训练链错误才修复后重跑，不能用“训练不够”无限续训。
- Gamma 零值项是可微近似，需查看零支持违规率；若近似显著影响结论，暂停采样器而非事后调参数。
- 20-step autograd guidance 的质量/成本取舍必须实测；若接近旧扩散约 110.8×成本且收益不足，停止当前采样设置。
- 新来源组独立开发/confirmation 协议仍未解决；本 Pilot 不满足独立确认资格。
