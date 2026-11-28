# Phase E Task B 表示域观测似然引导 DDIM 零训练候选协议（2026-11-28 v1）

## 状态与目的

**状态：冻结，待实现与开发集评价。** 本协议在任何候选 B 代码实现和推理前冻结。它响应候选 A 的预登记失败；只验证一个具有明确观测似然机制的备选，不启动新训练、不搜索 guidance 权重。

候选 A 的试验结果显示：以均值校正点估计为中心的目标残差标准差均值为 0.0631，而扩散样本残差标准差均值为 0.5081；残差与开发目标残差的平均相关为 0.0017，四 draw 的像素标准差均值为 0.4003。方向 B 的问题是检验：由固定合成观测模型导出的似然梯度能否在不改变权重的情况下抑制不相容重建。

本候选是**推理期、零训练**方法：复用候选 A 的第 3 轮 EMA checkpoint、同一个均值校正器、同一固定输入和四组逐样本生成 seed。它与候选 A 的唯一区别是每个反向 DDIM 步都加入下述观测似然 score；不重训、不改 checkpoint、不挑选 seed、不改评价阈值。

## 固定数据与权限

- 数据、来源分组、512×512 GT25 VV 伪参考、256×256 固定 LR、样本顺序、输入 manifest 和 train/development/confirmation 成员均沿用[Task B v3 冻结协议](PHASE_E_TASK_B_GLOBAL_SENTINEL1_PILOT_PROTOCOL_20261126_V3.md)。
- 只加载 train/development 权限 manifest 中的 development 参考、固定 LR 和已生成的候选 A 产物；训练清单只用于来源/哈希合同核验。confirmation、历史 test 与 Optical 载荷及其派生影像均不读取。
- 比较使用同一 70 张 development 样本、70 个来源组、相同参考与 evaluator。结果仍是单 seed、已用于候选 A 的探索性 development 结果，不作独立泛化证据。

## 固定观测模型与似然近似

冻结协议规定每个 2×2 HR 单元的四个独立乘性因子 `n_j ~ Gamma(k=1, theta=1)`，令 `y = (1/4) Σ_j x_j n_j`。这里 `x` 是 `[0,1]` 归一化 GT25 栅格表示，`y` 是固定 LR 表示；不作真实 SAR 强度或物理似然主张。

由 `E[n_j]=1`、`Var(n_j)=1` 可得：

\[
\mu(x)=\operatorname{AvgPool}_{2\times2}(x),\qquad
v(x)=\frac{1}{16}\sum_{j=1}^{4}x_j^2
=\frac{1}{4}\operatorname{AvgPool}_{2\times2}(x^2).
\]

四个加权 Gamma 项的和一般不服从单一 Gamma 分布。为构造可微、数值可控的似然，只采用匹配前两矩的异方差高斯近似：

\[
\ell(x;y)=\frac12\sum_p\left[
\frac{(y_p-\mu_p(x))^2}{v_p(x)+10^{-6}}+
\log(v_p(x)+10^{-6})
\right].
\]

这只是已知合成噪声下的**表示域矩匹配近似**，不是精确概率密度或经校准的 SAR 物理似然。

## 采样规则

- 使用候选 A 的 best EMA 网络、均值校正输出 `x_m`、1000 步既定线性 beta、相同 20 步均匀 DDIM 时间索引、`eta=0` 和逐样本四个 seed。
- 在每步先以当前 residual 状态 `r_t` 预测 `r_0`，组成 `x̂_0=clip(x_m+r̂_0,10^{-4},1-10^{-4})`，由上式计算 `ell`，通过自动微分得到 `g_t=∇_{r_t} ell(x̂_0;y)`。
- 依 posterior score 恒等式作**单位系数**修正：

\[
\epsilon_{\mathrm{guided}}
=\epsilon_\theta+\sqrt{1-\bar\alpha_t}\,g_t,
\]

随后使用冻结的 `eta=0` DDIM 更新。不得增加经验 guidance scale、阈值或针对 development 的二次修正。
- 若任一步产生 NaN/Inf，或超过 4 小时单候选/总 12 小时预算，保留日志并停止该候选；不自动改系数重跑。

## 评价与预登记决策

保存每张 development 的四个引导 draw、四 draw 像素均值、逐图指标、逐来源组指标、相对无引导 DDIM 与最强学习基线的配对组 bootstrap（10,000 次、seed `20261127`）、采样时间、峰值显存和四 draw spread。

沿用 v3 的继续门槛，不作任何调整：相对最强学习基线 PSNR 至少 `+0.05 dB`、来源组 bootstrap 95% CI 下界大于 0、SSIM/EPI 不降、亮回波与背景均值比绝对偏差不增、重投影 MAE 不增。任何 EPI 缺失均使门槛失败；其它可计算指标照常报告。

候选 B 未通过即停止本 Task B 生成式候选路线；不得据此宣称所有生成方法无效。结果范围限于该 GT25 多时相伪参考上的固定表示域合成代理任务。
