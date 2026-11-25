# Phase D0 Residual DDPM 失败机理诊断

**状态：已完成；当前 Residual DDPM 配置维持 STOP**  
**日期：2026-11-24**  
**证据目录：** [`phase_d0_residual_failure_20261124_v1`](../../experiments/diagnostics/phase_d0_residual_failure_20261124_v1/)  
**执行方案：** [诊断前冻结方案](PHASE_D0_DIFFUSION_FAILURE_ANALYSIS_PROTOCOL_20261124.md)

## 结论

现有 SwinIR-centered Residual DDPM 的首要可观测失败机制是：生成残差与目标残差几乎没有方向关系，且其加入稳定地恶化当前 noisy pseudo-target 的像素误差。四次采样的像素均值降低了随机残差幅度，却仍未恢复有用的方向信息。证据支持停止当前残差生成配置，并将“确定性输出上叠加普通随机残差”从本轮主候选中移除；这不等于否定所有扩散或生成式方法。

## 输入、边界与可复现性

- 仅使用 C1 v2 `x2_light` train 1,600 张/92 个来源组与 development 200 张/19 个来源组中的 SAR LR、noisy pseudo-target、既有 SwinIR 预测和已保存的四个 Residual DDPM development draw。
- 不读取 optical 载荷、confirmation/test 影像、成员清单或指标；没有训练、推理、GPU 调用或 checkpoint 选择。
- 新分析脚本 SHA-256 为 `59d0318ecb6914de94c3a659b5f710d8a48d6c6ccc71db26a6d1c1e723044e9b`；原始生成 run manifest SHA-256 为 `ffbf520c95f74fd66aff60bdaa2b14e03c8d89f6fafd2d17aa6a34b99b3f648b`。输入 manifest 与 `files.csv` 哈希均写在结果 JSON 中。
- development 已反复用于训练和检查点选择；这里的来源组区间只描述该历史复用 development，不是独立泛化置信区间，也不构成晋级或新的参数选择证据。

## 主要观察

按来源组等权汇总 200 张 development：

| 诊断量 | 单次 draw | 四次像素均值 |
|---|---:|---:|
| 残差余弦相似度 | 0.0121 | 0.0185 |
| 残差 Pearson 相关 | 0.0162 | 0.0223 |
| 符号一致率 | 50.44% | 50.65% |
| 预测残差 / 目标残差 RMS | 0.6446 | 0.3653 |
| 相对 SwinIR 的 MSE 差 | +0.003576 | +0.001011 |
| 目标辅助 oracle 缩放系数 | 0.0162 | 0.0354 |

来源组 bootstrap（19 组、10,000 次、seed `20261124`）给出：单次残差余弦 95% 区间 `[-0.00162, 0.02669]`；单次 MSE 差 `[+0.003157, +0.004013]`；像素均值残差余弦 `[-0.00359, 0.04177]`；像素均值 MSE 差 `[+0.000830, +0.001203]`。所有 200 张的单次 draw 平均 MSE 均变差；像素均值仅 9/200 张改善，且 19/19 个来源组平均 MSE 差仍为正。oracle 系数使用目标信息，只是不可部署的诊断上限，不能用作后处理。

频带/区域结果没有显示生成残差专门修复结构区域：单次 draw 的 edge cosine 为 −0.0117、edge MSE 差 +0.005414；high-return cosine 为 +0.0228、MSE 差 +0.003040；background cosine 为 +0.0048、MSE 差 +0.004873。四次均值仍分别使 edge/high-return/background 的 MSE 增加 +0.001990/+0.001017/+0.001615。四 draw 的像素级样本标准差均值为 0.0451、RMS 为 0.0477；随机变化被均值部分抵消，但没有转成目标方向上的修正。

SwinIR 目标残差的来源组等权 RMS：train 为 0.09077（组中位数 0.09386，P10–P90 为 0.05758–0.11626），development 为 0.08668（组中位数 0.08974，P10–P90 为 0.05129–0.11298）。高频残差能量占比（`f≥0.25` cycles/pixel）分别为 0.2140 与 0.2026。该差异同时混有 SwinIR 对 train 样本的拟合效应，不能单独解释为来源分布偏移；当前数值没有提供“开发残差整体显著失配”的直接证据。

## 训练目标和代码核查

- Residual DDPM `train_epsilon_mse` 从 epoch 1 的 0.226626 降至 epoch 30 的 0.148793；来源组等权 development epsilon MSE 从 0.454444 降至 0.144605，最低值在 epoch 30。
- 逐 epoch 没有完整采样预测与图像质量指标，无法估计 epsilon MSE 与 sample PSNR/结构质量的跨 epoch 关系。当前 checkpoint 的生成质量差，不能据此事后断言“训练目标导致失败”，也不能以“可能训练不够”作为续训理由。
- 对实际 run 代码哈希核对了 beta/alpha-bar、前向加噪、epsilon 到 `x0` 还原、20-step DDIM eta=0 时间索引、条件张量、残差尺度、EMA、checkpoint 状态及最终裁剪。未发现有运行证据支持的实现错误；当前源码检查不能替代历史 run 代码哈希。

## 解释与路线决策

1. **方向不可预测是最直接的证据。** 余弦和相关接近零、符号一致率接近随机，oracle 缩放也接近零；现有预测残差不携带稳定的目标修正方向。
2. **随机残差方差不是主要收益来源。** 像素均值明显缩小残差，但 MSE 仍变差，说明简单增大 draw 数或把像素均值当单次质量没有依据。
3. **数据域限制继续有效。** 目标是带噪表示域 pseudo-target。这里的 MSE、PSNR、SSIM 不能称为 clean-SAR 恢复质量，也不能支撑辐射物理保真结论。
4. **D2 主线改选观测约束生成（方向 B）。** 下一步将把经逐位核验的 x2-light 合成算子及其表示域 Gamma 观测项纳入后验采样；重点检验输出是否同时减少目标误差、遵守已知退化观测，避免无依据的高频增添。扩散先验本身并不保证收益，必须由冻结 pilot 直接证伪。

当前停止决策只针对已停止的 Residual DDPM 配置和把它当作主候选的简单残差协同方案；不追溯修改 C3-A、历史 DDPM 决策或任何研究总纲 gate。

## 运行异常记录

诊断分析器早期两次尝试分别错误假设 LR 值位于 `[0,1]`、预测数组统一为 `[1,H,W]`；实际固定 LR 可超过 1，历史 SwinIR 输出保存为 `[H,W]`，Residual draw 保存为 `[4,H,W]`。修正为读取真实 manifest/写入形状后，逐样本哈希和张量契约核验通过。没有覆盖或更改任何历史预测、输入或 checkpoint。逐行结果、命令、环境及摘要均保存在证据目录。
