# C3-A SAR-only Flow Matching 研究决策复核（2026-11-23）

## 决策

停止当前像素域 SAR-only Flow Matching 配置，不追加第二个训练 seed。冻结协议规定的第二 seed 继续条件是：开发集来源组等权配对 ΔPSNR ≥ +0.05 dB、来源组 bootstrap 95% 区间下界 > 0，且 ΔSSIM ≥ 0。本次观察值分别为 −3.0724 dB（95% 区间 [−3.4720, −2.6922]）和 −0.1782，三项均未达到。多样本 v3 复核方向相同。

这项决定只针对当前数据表示、模型、训练预算和采样器组合；它不证明所有 Flow/Diffusion 方法均无效，也不回答匹配 Optical 是否有增量价值。C3-A 不解锁 C3-B；C2 v2 gate 仍未判定。本轮没有运行 C3-B 或读取 Optical、confirmation、test 载荷。

## 实验执行与边界

- 使用 C1 v2 固定身份：训练 1,600 个样本/92 个来源组，development 200 个样本/19 个来源组。输入为 x2_light SAR LR 与 noisy pseudo-target；仅使用 SAR LR 和 target。
- 按 C3-A v1 冻结设置训练一个 seed（20261105）共 5 轮，batch size 4、Adam 学习率 1e-4、无增强、无混合精度、确定性算法；8 步 Heun 采样。按来源组等权 development PSNR 选出第 2 轮为 best。
- 与 C2 x2_light SAR-only 确定性联合重建基线同身份比较。基线 checkpoint SHA-256 为 8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165，基线选择轮次为 30。两种方法使用同一固定输入、pseudo-target、评价身份、边缘 mask 和指标实现；训练轮数和参数量不是计算预算匹配对照，因此将质量比较解释为冻结筛选，而不是等算力结论。
- 原训练协议与多样本 v3 JSON 均未改动，分别保持 SHA-256 147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7 和 ee83e2cb4dadea7d4fca6576e479d2922d70df44935f1be51116add26e06016c。用户另行明确授权本次运行不因并行 CUDA 进程或 3 GiB 空闲显存门槛而等待，以 OOM 为停止边界；相关偏离写入 run manifest。全程未发生 OOM。
- 独立复核后，删除了预检代码中按桌面进程名和 Python 脚本名无条件放行的例外。当前默认路径会把所有其他 CUDA 计算进程视为并发占用，并保留冻结的 3 GiB 门槛；显式授权旗标仍用于记录/执行本次资源豁免。修订只改变预检过滤，不涉及模型与指标代码。本次运行的代码哈希由 manifest 保留，未因该预检修订重跑已满足停止条件的负结果实验。
- 训练环境为 Python 3.13.7、PyTorch 2.8.0+cu128、CUDA 12.8、NVIDIA GeForce RTX 3050 Laptop GPU，总显存 3,942,383,616 字节。训练峰值已分配显存 189,086,208 字节；v3 Flow 峰值 189,401,088 字节，基线峰值 85,587,456 字节。
- C3-A 训练累计运行时间记录为 1,663.71 秒。对 200 个 development 身份进行 4 draws 的 Flow 推理耗时 458.28 秒，确定性基线耗时 4.60 秒；按相同 200 个身份的 4-draw 总成本计算约慢 99.6 倍。折算单 draw 后约为基线的 24.9 倍。Flow 与基线参数量分别为 323,617 和 96,209。
- 训练首次启动因缺少 CUBLAS_WORKSPACE_CONFIG 在训练前退出；第二次训练完成 5 轮后，checkpoint 的安全加载遇到 TorchVersion 兼容问题。后续通过仅对白名单 TorchVersion 元数据生效的局部 sitecustomize 恢复检查并完成 run。各次失败与成功终端记录均保留在对应中文日志目录；冻结 seed、轮数、checkpoint 权重与指标规则没有改变。

## 质量、结构与高值区域结果

所有指标针对已重复用于训练选择的 development noisy pseudo-target；19 个来源组等权。它们是范围受限的表示域观察，不是独立泛化或 clean-SAR 质量证据。

| 方法/统计对象 | PSNR | SSIM | EPI | 边缘强度比 |
|---|---:|---:|---:|---:|
| C2 SAR-only 确定性基线 | 21.6082 | 0.36271 | 0.35573 | 0.25512 |
| Flow 单次 draw（4 draw 的组均值） | 18.5388 | 0.18452 | 0.18702 | 0.54079 |
| Flow 四个 draw 的逐像素均值点估计 | 20.4675 | 0.27609 | 0.25729 | 0.34779 |

多样本 v3 的 Flow 减基线来源组等权配对差为：ΔPSNR −3.0694 dB（95% 区间 [−3.4646, −2.6974]）、ΔSSIM −0.17819（[−0.20714, −0.15086]）、ΔEPI −0.16871（[−0.18814, −0.14697]）。四 draw 像素均值相对基线的 PSNR、SSIM、EPI 差分别为 −1.1407、−0.08662、−0.09843；该补充没有计算区间。

Flow 的边缘强度比高于基线，但 EPI 明显更低，说明梯度幅值较大并未转化为更好的目标边缘对应关系；不能单独把边缘强度解释为结构保持。基于每个身份 noisy pseudo-target 第 99 百分位（包括并列值）的归一化表示域高值代理中，Flow 减基线的区域绝对误差差为 +0.01691（95% 区间 [+0.01427, +0.01980]），区域均值比差为 −0.01644（[−0.01917, −0.01389]），峰值比差为 +0.06462（[+0.05899, +0.07021]），饱和比例差为 +0.03667（[+0.02753, +0.04642]）。表现为高值区域误差和饱和增加、均值偏低而峰值偏高。该代理只涉及归一化 8-bit 表示域，不能解释为物理散射体保持或辐射保真。

## 生成稳定性与分布评分

每身份 4 draws 的来源组等权平均身份内标准差为：PSNR 0.04507 dB、SSIM 0.00472、EPI 0.01885；平均逐像素样本标准差为 0.07305。四个 draw 的来源组均值均约 18.54 dB，均低于确定性基线约 3.07 dB；配对 bootstrap 区间整体低于 0。有限 draw 的离散度只描述本次样本，不能称为校准不确定性，也没有证明分布真实性。

样本集合补充评价的 Energy Score 插件值/IID-fair 值为 0.07792/0.06357，Variogram Score 插件值/IID-fair 值为 0.01752/0.01483。由于这里只有 noisy pseudo-target 和四个 draw，没有已知生成真分布或经校准的参照后验，这些值仅为描述性评分，不能单独支持 Flow 分布优于基线的结论。Flow 原始输出低于 0 和高于 1 的比例分别为 0.00149 和 0.01563；指标计算按冻结规则裁剪后进行。

## 证据类别

| 类别 | 本轮结论 |
|---|---|
| 事实 | 冻结 v1/v3 协议、C1 v2 train/development 身份数、run 状态、checkpoint 哈希、资源环境和载荷访问 flags 可由协议与 manifest 核验。
| 观察 | 在这一个 seed、五轮预算和已复用 development 上，当前 Flow 配置在 PSNR、SSIM、EPI、高值代理和推理成本方面落后于确定性基线。
| 假设 | 不同生成参数化或新的 SAR 观测模型也许能改善结果；本次没有检验。
| 开放问题 | 新独立 development 数据、匹配 Optical 增益、生成样本分布质量和 SAR 物理/观测一致性仍未解决。
| 无支持主张 | “所有 Flow/Diffusion 都无效”“Optical 没有价值”“输出具物理一致性”均不受本轮证据支持。

## Research Decision Review

1. **是否继续当前 SAR-only Flow？** 否。冻结第二 seed 条件三项均未满足；不追加 seed，也不通过调参、改 gate 或重复同类实验挽救当前结果。
2. **Flow 是否仍有进一步研究价值？** 当前像素域、x2_light、5 轮训练与 Heun8 组合没有证据值得继续投入。不同生成参数化或任务假设若未来要研究，需另立可证伪假设、使用新的合格 development，并设置与确定性方法公平的训练/推理预算和停止标准；不能把本次负结果当作普遍否定。
3. **是否进入 C3-B？** 否。本结果没有 Optical，也不评估跨模态信息。当前 C2 v2 gate 仍未判定；本次 C3-A 也没有晋级权限。
4. **数据前置条件：** C1 v2 的 131 个来源组已分配到 train/development/confirmation；development 已反复使用，confirmation 曾被访问后永久封存。仓库数字命名的 2,000 对 PNG 尚未映射到 C1 身份。因此，当前证据不能构造新的独立 development manifest。
5. **C3-B 后续实施方案（仅计划，未授权执行）：**
   - **数据阶段：** 现有 C1 v2 的 131 个来源组都已分配，当前没有可直接冻结的新独立 development manifest；本地 2,000 对数字 PNG 尚无 C1 身份映射。最低修复是取得额外、正确配对且来源组未参与既往训练/选模的数据；初始目标至少达到 200 对、20 个独立来源组（约等于当前 development 规模），最终样本量需根据来源组异质性和所需区间精度复核，不能在未知来源清单下冻结。先核对文件身份、配准、表示域、场景/空间/时间重叠和来源组，再版本化 train/new-development 清单及哈希；confirmation/test 保持独立并封存。
   - **先决筛查：** 在该新 development 上重新运行任务匹配的 SAR-only 确定性联合重建与 matched-Optical 对照，并加入组内置换、missing/zero Optical 和预声明几何错位控制；对各臂使用同一新训练身份、输入清单、选模规则与统计单位。先解决 C2 v2 目标背景为零导致 gate 未判定的问题，并在运行前固定指标定义、门槛、重复 seed 与停止规则。没有可重复 Optical 增量时，不开始 C3-B 生成式训练。
   - **生成阶段：** 若 Optical 筛查通过，再单独预注册 SAR-only 与 Optical-conditioned 生成臂，使用同一架构、初始化策略、训练预算、种子、采样器和新 development 身份，匹配/置换/缺失/错位控制按同一随机种子配对。当前 C3-A Flow 的负结果意味着它不应直接作为首选骨干；新生成候选先要在新数据上证明 SAR-only 可行，再测 Optical 的条件增量。
   - **评价与晋级：** 报告逐图和来源组等权的 PSNR/SSIM、EPI/边缘幅值、高值代理、raw 越界、单次 draw、像素均值点估计、分布评分、运行时间与峰值显存；多 seed 使用预登记的来源组配对 bootstrap。以新 protocol 预先登记 Optical 相对 SAR-only 的继续/停止门槛，且 matched 优势须超过置换/缺失控制。满足后先做 Research Decision Review，再另行申请 C3-B 正式训练。本轮不执行上述任何 C3-B 实验。
6. **目标边界：** SAR-only Flow 的负结果不解决“正确配对 Optical 是否能提升 SAR 联合去斑与超分”，也不提供 SAR 物理一致性证据。项目核心问题仍开放。

## 运行产物

- [C3-A 训练 run 清单](../../experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123/manifest.json)、[逐轮日志](../../experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123/train_log.csv)、[逐图 development 指标](../../experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123/development_metrics.csv)、[SAR-only 基线指标](../../experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123/sar_only_baseline_development.csv)和[配对比较](../../experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123/paired_development_comparison.csv)。
- [C3-A v3 多样本运行清单](../../experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3_attempt2_20261123/manifest.json)、[逐 draw 指标](../../experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3_attempt2_20261123/flow_per_draw_metrics.csv)、[身份汇总](../../experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3_attempt2_20261123/identity_summary.csv)及[来源组 bootstrap](../../experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3_attempt2_20261123/group_bootstrap_summary.json)。
- [样本集合评价摘要](../../experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3/summary.json)、[来源组评分](../../experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3/source_group_metrics.csv)与[逐图指标](../../experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3/single_draw_metrics.csv)。
- [四 draw 生成样本归档清单](../../experiments/diagnostics/c3a_sample_archive_c1_v2_20261123_v1/manifest.json)、[高值代理逐 draw 指标](../../experiments/diagnostics/c3a_sample_archive_c1_v2_20261123_v1/bright_proxy_per_draw.csv)和[来源组 bootstrap](../../experiments/diagnostics/c3a_sample_archive_c1_v2_20261123_v1/bright_proxy_group_bootstrap.json)。800 个生成样本 NPZ（439,898,403 字节）及训练 checkpoint 保留在本地 run 目录，未纳入 Git。
- [GPU 预检及后续授权执行说明](../../experiments/diagnostics/c3a_gpu_preflight_20261123_v1/报告.md)。
- [冻结训练协议](../../protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json)与[v3 多样本协议](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v3.json)保持原哈希。

提交包含上述小型 run manifest、逐图/来源组结果、汇总与中文终端日志，因此主要数值与决策可从干净检出审阅，但不能仅凭该提交独立重跑实验。C1 train/development 固定输入载荷及 SAR-only baseline checkpoint 未纳入 Git；重跑还需要获准访问同一固定输入与权重。包含 confirmation 身份的完整 C1 来源清单也未提交；仅在 run/input manifest 保留其 SHA-256（`6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`）。本轮未读取 confirmation/test 影像载荷。Flow checkpoint 与生成样本保留在本机运行目录，验证其哈希见归档清单。
