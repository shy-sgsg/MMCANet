# Phase D2 观测约束扩散先验 Research Decision Review

**日期：2026-11-24**  
**决策：STOP 当前 B 配置；不追加同配置训练或随机种子**  
**证据范围：** C1 v2 固定 `x2_light` train/development；仅 SAR LR 与 noisy pseudo-target。development 已被多次用于模型选择，以下均为探索性观察。

## 输入、操作与可追溯产物

- Pilot 按[冻结协议](PHASE_D2_OBSERVATION_AWARE_PILOT_PROTOCOL_20261124.md)训练 5 轮：train 1,600 张/92 组，development 200 张/19 组；seed `20261124`，batch 4，128×128 HR patch，U-Net `base_channels=32`，1000 步 epsilon-prediction，AdamW `1e-4`，EMA `0.995`。
- 先验选择轮次为第 5 轮，development 来源组等权 epsilon MSE `0.0518022`；最后一轮 train epsilon MSE `0.0531069`。训练 333.96 秒，峰值 allocated/reserved 显存为 426.72/510 MiB，CUDA OOM 为否。
- 之后对全 200 张 development 做 4 draws × 20 个确定性 DDIM 步，共 800 个样本生成、16,000 次迭代观测梯度。800 次模型采样调用累计耗时 1,604.45 秒；单样本单次 draw 中位数为 2.0018 秒。开发集生成与评价全流程累计耗时 1,612.76 秒（包含结果保存、指标计算与汇总）；峰值 allocated/reserved 为 405.4/476 MiB，无 OOM。
- 关键资产位于 `experiments/runs/phase_d2_observation_aware_pilot_20261124_v1_retry01/` 与 `experiments/runs/phase_d2_observation_aware_eval_20261124_v1/`。checkpoint SHA-256：`af66546e4e7ef52624a569df4ca56f4182eed6605b13194f09b6d28a044a472e`；训练协议 SHA-256：`2d470e2814477cf0b43470ddf24a1b8d6d936a314899b88be5e6fda982b07d10`；评价汇总 SHA-256：`360931a07c822044b7ed3e3160d313f6914cc11d68bbbb7fb3d21295794e831c`。
- 固定退化算子 `degradation.py` SHA-256 为 `b7c0e9208a132792824bdde4296d7d3650587a144bf7ab4ec34bb161e0cb7bd1`；1800 个 train/development LR 输入均逐位复现，最大绝对误差 0。评价 manifest 标记 `test_set_accessed=false`、`test_set_used=false`、`confirmation_payload_accessed=false`、`optical_payload_loaded=false`。

## 主要结果

四个单 draw 的来源组等权 PSNR 分别为 10.7301、10.7847、10.7103、10.7818 dB；四 draw 平均后的单次生成表现为 PSNR 10.7517 dB、SSIM 0.1813、RMSE 0.3017、EPI correlation −0.0038。SwinIR SAR 任务适配版对应为 21.6352 dB、0.3597、0.0867、0.3587。

四 draw **pixel-mean** 为 PSNR 10.9763 dB、SSIM 0.1877、RMSE 0.2890、EPI correlation −0.0029。相对 SwinIR 的来源组配对 ΔPSNR 为 −10.6589 dB（10,000 次来源组 bootstrap 95% CI [−11.9876, −9.3764]），ΔSSIM 为 −0.1721（95% CI [−0.1828, −0.1619]），ΔEPI 为 −0.3616（95% CI [−0.4183, −0.3043]）。四 draw 均明显低于 SwinIR，故负结果并非单个随机种子造成。

亮值与结构也不支持继续：单 draw 的亮区均值比为 0.6216、边缘强度比为 0.1784；pixel-mean 的边缘强度比降至 0.1301。pixel-mean 正值 Gamma 项从 SwinIR 的 1079.8 降至 6.78，但零值 hinge 约束由 0.00089 升至 0.14170，零观测位置违规率由 36.53% 升至 48.85%；零约束配对差为 +0.1408，95% CI [0.0982, 0.1824]。这表明正值似然改善不能单独代表整个观测模型相容，且像素均值也不能替代单次 draw 的图像质量。

4 draws 的平均绝对两两差与像素方差已在 `像素均值指标.csv` 按图记录。完整逐 draw、pixel-mean、SwinIR 和来源组结果见该 run 目录的 `逐draw指标.csv`、`像素均值指标.csv`、`SwinIR基线指标.csv`、`来源组统计.csv` 与 `汇总.json`。

## 失败解释与证据级别

### 已证实事实

1. 固定算子实现与固定 LR 的往返一致；训练、checkpoint 加载、20-step 采样和评价链路均完成，日志无 CUDA OOM。
2. 当前方法在全部四个 draw 上都严重劣于固定 SwinIR；pixel-mean 只略微提高本方法自身 PSNR，未改变比较结论。
3. 正值 Gamma 代价和 clamp 零支持 hinge 的行为不一致：pixel-mean 的正值项很低，零值违规却显著增加。
4. 4-draw 模型采样调用累计耗时 1,604.45 秒，约为 SwinIR 200 张单次推理 18.58 秒的 86.4 倍；单样本单次 draw 中位数为 2.0018 秒。

### 高可信观察与未证实假设

- **高可信观察：** 当前无条件先验加近似观测引导没有形成可用的 SAR 重建 posterior；输出具有明显的结构和亮值衰减。负结果来自这份固定 checkpoint、协议和开发集合，不推广到全部 posterior diffusion。
- **合理但未证实的解释：** 单侧 hinge 只是 clamp 零值支持的可微代理，不是精确零原子似然；固定权重 1 下，正值 Gamma 项、零支持项和 prior score 可能存在尺度/优化冲突。像素均值正值项改善与零值约束恶化是该冲突的证据，但不能证明唯一根因。
- **未证实假设：** 五轮训练可能不足以学到高质量 HR 分布。epsilon MSE 持续下降至第 5 轮并不能证明完整采样质量会继续提高；协议不允许以“训练不够”为理由追加同配置训练。
- 对 DDIM 反向公式、观测梯度符号和张量合同的首轮静态审查遗漏了一个重要边界。后续只读复核发现：当频域振铃产生 `z<0` 且观测 `y>0` 时，正值项先经 `clamp_min(0)` 再经 `clamp_min(epsilon)`，对 `z` 的梯度为零；posterior guidance 无法从该区域把预测拉回正观测。新增定向测试复现并固定记录了该零梯度行为。
- 该缺陷说明当前实现的正值观测引导并未覆盖完整数值域，但不能据此断言它是整体质量崩塌的唯一原因。既有训练、采样和指标均未修改或重算；不得用事后修补把冻结 Pilot 变成通过结果。后续若研究此路线，须先定义支持负振铃状态的观测梯度/似然，再用正负预测及零/正观测边界测试验证。

## 决策

- **STOP：** 当前 B 方案的 5-epoch prior + 20-step 后验引导配置，不增加 epoch、不改引导权重、不追加 seed、不在同一 development 上调参。
- **不晋级：** 不将它称为具有质量优势的生成方法，不据此推进 C3-B，也不访问 confirmation/test。
- **不启动 C：** diffusion bridge 改变初始化轨迹，但本次突出风险是表示域观测模型的零值支持近似及质量崩塌；仅换 bridge 起点不能由现有证据推出会解决该问题。若未来重开，必须先提出新的、针对零值统计机制的独立协议并取得可用的新 development 资格。
- **下一步优先级：** EDSR 第二个已发表通用架构的 SAR 任务适配基线已完成，并保留 SwinIR 为同协议对照；当前最高优先级是解决新独立 development 数据获取与 SAR 观测/零值统计协议。只有新的独立证据足以支持假设时，再研究生成式方法。此审查不修改总纲 v1.8、C2 gate 或历史 C3-A/Conditional DDPM/Residual DDPM 决策。

## 决策证据类别

| 类别 | 本轮结论 |
|---|---|
| 事实 | 训练与 800 次 draw 完成；无 OOM；固定算子往返误差为 0；开发集和来源组数与协议一致；预测/指标逐文件保存。 |
| 观察 | 四个 draw 和 pixel-mean 均显著落后 SwinIR；单 draw 输出结构、亮值不足；零支持违规率更高；推理成本高。 |
| 假设 | 先验不足、似然尺度和零原子代理的冲突可能贡献于失败，但本轮无法分离各自因果。 |
| 开放问题 | 新来源组上结论是否相同；正确建模 clamp 零原子后能否降低违规；是否存在真正可预测且有成本优势的条件生成路线。 |
| 无支持主张 | “所有扩散/后验采样均无效”“该模型恢复了 clean SAR”“已达到辐射物理一致”“开发集结果证明独立泛化”。 |
