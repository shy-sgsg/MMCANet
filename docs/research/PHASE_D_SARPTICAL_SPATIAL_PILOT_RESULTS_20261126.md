# SARptical 空间分组 SAR-only Pilot 结果与研究决策复核

**运行编号：** `phase_d_sarptical_spatial_pilot_20261126_retry01`

**执行日期：** 2026-11-26
**结果目录：** `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/phase_d_sarptical_spatial_pilot_20261126_retry01`

## 1. 研究问题与结论

本次有限 pilot 检验：双三次锚定条件 Flow Bridge 能否在 SARptical 柏林城区的空间分组开发集上，超过任务适配 EDSR/SwinIR，并保持局部亮回波、边缘结构和背景稳定。

**决策：停止当前 Flow Bridge 配置。** 它的单次生成在 PSNR、SSIM、EPI 上均明显落后于最强确定性参考 SwinIR；背景偏差在 8/8 来源组恶化。冻结协议的亮回波 mask 在全部开发图上都不足 16 像素，因此该护栏不可估，按预先规则判为未通过。没有事后放宽阈值，也不追加相同配置训练。

该结果只回答本次单城区、单 seed、单一表示域退化 pilot 是否值得按原配置继续。它不否定所有 Flow/Diffusion，也不证明真实 SAR 去斑、跨城泛化、SAR 辐射物理一致性或 Optical 增量价值。

## 2. 数据与冻结协议执行

- 数据来自 SARptical 官方仓库所链接归档。旧归档在读取训练影像时发现损坏，未开始训练；该归档与失败 run 均保留。按同一官方链接重新获取的归档大小为 `1,115,515,719` 字节，SHA-256 为 `044cfa600e301f3b1f3d72d38f2182c4e74453d9403a0aab66076d6308c567f9`。
- 只解码 train/development 的 1,222 个 SAR MAT 载荷；ZIP CRC、`ampCrop`、112×112 尺寸和有限值检查全部通过。train 为 916 张/12 组，development 为 306 张/8 组。confirmation 为 306 张/8 组，但其影像载荷未读取；历史 test 未使用；Optical 载荷未加载。
- 空间划分使用 250 m UTM 网格、75 m 内缩缓冲、seed `187762`。只凭样本坐标和角色清单分组，不按模型结果挑组。结果仍局限于柏林单城区；八个开发空间组不等于八个独立城市或独立传感器来源。
- 任务为 112×112 SAR dB 幅度 noisy target 的 ×2 表示域重建，固定 bicubic 下采样；不额外合成 speckle，不称 target 为 clean SAR。训练集统计固定标准化与评价范围。
- 预先固定方法为 bicubic、任务适配 EDSR、任务适配 SwinIR、双三次锚定 Flow Bridge。神经网络均最多 30 轮、batch size 4、seed `20261125`、每方法 6,870 次更新；Flow 使用 `sigma=0.5`、8 步 Heun、4 个预设 draw。每轮按完整 development 来源组等权 PSNR 选 checkpoint。

数据与方法的完整预登记见[冻结设计稿](PHASE_D_SARPTICAL_SPATIAL_PILOT_DESIGN_20261125.md)。

## 3. Development 结果

以下为八个空间组等权均值。所有指标都在固定表示域评价；PSNR/SSIM 参考是 SAR dB noisy pseudo-target，不代表 clean-reference 去斑效果。

| 方法 | 身份 | PSNR (dB) | SSIM | EPI corr | RMSE |
|---|---|---:|---:|---:|---:|
| Bicubic ×2 | 固定插值基线 | 25.2596 | 0.72469 | 0.56042 | 0.29513 |
| EDSR | 任务适配发表架构 | 26.3034 | 0.80280 | 0.78122 | 0.26152 |
| SwinIR | 任务适配发表架构 | **26.3322** | **0.80315** | **0.78657** | **0.26065** |
| Flow Bridge 单次 draw | 候选生成方法 | 23.6911 | 0.69033 | 0.65049 | 0.35312 |
| Flow Bridge 四 draw 像素均值 | 点估计补充 | 25.4980 | 0.76864 | 0.73969 | 0.28688 |

EDSR 与 SwinIR 是沿用发表架构并适配当前单通道、×2 SAR 表示域任务，不是作者权重或原论文训练方案的精确复现。SwinIR 相对 EDSR 的 PSNR 差仅 `+0.0288 dB`；本次不据此声称二者具有实质性能差异。

相对 SwinIR，Flow 单次 draw 的来源组等权差为：

- PSNR：`−2.6412 dB`，95% 来源组 bootstrap CI `[−2.7286, −2.5647]`；306 张配对图、8 组。
- SSIM：`−0.11282`，95% CI `[−0.12158, −0.10499]`。
- EPI corr：`−0.13608`，95% CI `[−0.14877, −0.12356]`。

四 draw 像素均值是单独报告的点估计，不是一次生成样本或经校准的分布；它的 PSNR 仍比 SwinIR 低约 `0.8342 dB`。

## 4. 结构、亮回波与稳定性护栏

- **高回波 ROI 不可用：** 训练集固定阈值为 `3.34245`；306 张 development 图的 mask 像素数为 0–6，均低于冻结的 16 像素下限；有效图像为 0/306，有效来源组为 0/8。亮回波均值误差和 P95 不能估计。按照冻结的 8 组规则，护栏失败；没有调整阈值或掩膜。
- **边缘强度误差：** Flow 相对 SwinIR 改善 `0.03866`，95% CI `[-0.05249, -0.02615]`，8/8 组未恶化；该单项通过。
- **背景绝对偏差：** Flow 增加 `0.01078`，95% CI `[0.00949, 0.01192]`，8/8 组恶化；该护栏失败。
- **生成稳定性：** 单 draw 指标明显低于 pixel mean；因此均值点估计的改善不能说明随机样本质量或分布真实性。四个 draw 和逐图指标均已保存，未把像素均值当生成样本。

综合门槛中 PSNR、SSIM、EPI、亮回波可用性与背景稳定性均未通过，故按预登记规则停止本配置。

## 5. 计算成本与资源

| 方法 | 最佳轮次 | 参数量 | 更新数 | 训练秒数 | 峰值 CUDA allocated / reserved |
|---|---:|---:|---:|---:|---:|
| EDSR | 5 | 1,367,553 | 6,870 | 2,685.96 | 119.93 / 164 MiB |
| SwinIR | 25 | 252,784 | 6,870 | 3,866.71 | 323.04 / 380 MiB |
| Flow Bridge | 19 | 323,617 | 6,870 | 3,679.29 | 160.06 / 386 MiB |

同一 checkpoint 和固定 development 输入的推理重放中，SwinIR 单次为 11.56 秒，Flow 单次 draw 为 54.68 秒（4.73 倍），四 draw 共 215.59 秒（18.65 倍）。重放逐图指标与归档结果一致。训练与推理在 RTX 3050 Laptop GPU 4 GiB 上完成；有其他 GPU 工作并行，未发生 CUDA OOM。训练/后处理结束后外部 run 卷约剩 87.77 GiB，根卷约剩 6.0 GiB。

## 6. 运行异常与恢复记录

首次官方归档副本的 ZIP local header 损坏，读取 train 样本时 fail-closed；未训练。重新获取的归档通过获准 SAR 载荷检查。第二次启动发现脚本入口早于训练函数定义，在第一次优化器调用前以 `NameError` 失败；该失败 run 保留，无权重更新。先添加入口顺序回归测试，再用新目录完成正式运行。完整训练后最初汇总器因所有高回波 ROI 都是 NA 而中止；修复汇总逻辑为显式记录不可用原因并按冻结 gate 失败后，使用保存的 checkpoint 和缓存重放推理并完成报告。重放指标匹配，没有重训、重新选模或访问新数据。

上述修复使结果可完整记录，不改变任何科学参数、门槛、checkpoint 或已计算指标。原失败日志、训练日志、checkpoint、预测和复核记录均保留。

## 7. Research Decision Review 与后续

**Research Decision：`STOP_CURRENT_FLOW_BRIDGE_CONFIGURATION`。** 不对相同双三次锚定 Flow 配置加 seed、延长训练或调亮回波阈值。当前 pilot 不解锁 C3-B；不修改论文，不把结果解释为 Optical 效用、跨城区泛化或物理一致性。

后续优先事项：

1. 保留本次负结果，并复核亮回波指标设计是否能在预登记且足量的真实空间组上获得有效支持；任何新阈值必须用于新协议，不能回看本开发集调参。
2. 继续寻找覆盖多个独立采集区域、具有可核验来源与空间足迹的 SAR development 数据。BigEarthNet 原官方 train/validation 的场景代理重叠与空间近邻问题尚未解决，不能直接作为独立 split。
3. 生成式下一路线需要不同于本配置的可证伪机制和前瞻性协议；优先先验证确定性/可预测残差诊断，不立即投入新大型 GPU 训练。
4. Optical 研究仍要求匹配 Optical 相对 SAR-only 的可重复效用及新开发数据资格；本 run 没有加载 Optical。

详细结果、逐图指标、来源组统计、预测、checkpoint、训练日志、资源记录和 manifest 均在上方 run 目录；本仓库只保存研究报告、协议与必要代码，不提交大型数据、checkpoint 或预测文件。关键溯源见 run `manifest.json`、`config.json`、`split_manifest.json`、`input_manifest.json`、`runtime.json` 和 `finalization_manifest.json`。

## 8. 实现验证与产物核对

- SARptical pilot 定向测试：22 项通过；`tests/` 全部项目测试：210 项通过（1 项 PyTorch `meshgrid` API 提示）。4 个 Python 文件 `py_compile` 通过，`git diff --check` 通过。
- run 产物复核：配置、split manifest、input manifest 的 SHA-256 均与主 manifest 相符；6 个 best/last checkpoint 的 SHA-256 均匹配；实际预测文件数 2,448，与 manifest 一致；逐图 CSV 1,530 行、配对组比较 CSV 44 行；最终化记录确认重放指标与保存结果一致。
- 仓库根目录无筛选的 `pytest` 另会收集 vendored `BasicSR/` 自带测试，分别在未设置/设置 `PYTHONPATH=BasicSR` 时因裁剪版 BasicSR 缺少 `basicsr.data` 而出现 17/12 项收集错误。项目 `tests/` 目录完整测试已单独通过；未为运行不相关的 vendored 测试补装依赖或改动其源码。
