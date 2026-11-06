# JDSR / MMCANet：SAR 联合去斑与超分辨率

**正式研究总纲：** [v1.7 研究总纲](docs/research/RESEARCH_CHARTER.md)，于 2026-11-06 获用户明确批准。正式题目为“光学先验引导与 SAR 物理一致性约束的生成式联合去斑和超分辨率重建方法研究”。v1.7 明确沿用冻结 C1 v2、confirmation 已封存，并补充条件流/后验覆盖文献边界；不改变历史 gate。对当前 8-bit PNG 表示域实验，不作未经验证的辐射物理解释。

## 项目目标

研究匹配 optical 是否能在任务匹配的 SAR-only 联合恢复基线上提供可重复增益；识别光学线索的有效范围、负迁移情形、观测一致性与安全回退边界。生成模型、物理约束、错配抑制和局部回退均为待检验方向，不预设为已验证贡献。

## 当前阶段与运行状态

- **C1 v2 划分：** train 1,600 对、development 200 对、confirmation 200 对，按来源场景文件夹划分；没有重新切分。官方 train/development 来源载荷已按冻结清单核验并暂存，见[暂存 manifest](experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json)。confirmation 已于 2026-11-05 用于一次访问后修订的描述性复核，结果 `inconclusive`，现已封存，不用于训练、调参或选模；详见[当前状态](docs/research/CURRENT_STATE.md)。
- **C2 v2 方法筛查：** 沿用冻结 C1 v2 来源场景划分完成四种固定退化条件、八组 ×30 epoch 训练。原 paired screen 因 2 个 development 样本的 target 背景中位数为零而 fail-closed，故 gate 未判定；另完成全 200 个 development 样本的非门槛诊断。matched optical 相对 SAR-only 的 PSNR 增益点估计为 −0.0055 至 +0.0061 dB，四个区间均跨零且低于预设 +0.05 dB 继续阈值。见[训练矩阵](experiments/phase_c2_c1_v2_20261031_v1/matrix/manifest.json)、[失败记录](experiments/phase_c2_c1_v2_20261031_v1/screen/execution_manifest.json)和[补充诊断报告](experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。
- **C2 指标代码诊断：** 新增零背景下仍有定义的表示域均值/中位数目标—背景对比差函数；独立于 C2 v2 evaluator 和 gate。4 个合成边界案例及既有 C2/evaluator 回归共 43 项通过；尚未用于真实模型输出。[设计与验证记录](docs/research/C2_ZERO_BACKGROUND_CONTRAST_DIAGNOSTIC.md)。
- **C3-A SAR-only Flow 筛选：** `x2_light` 冻结 train/development 协议、模型与入口已准备，网络有 323,617 个可训练参数；真实训练与 development 评价按用户要求暂停，GPU 等待器未运行。张量合同、Heun 数值诊断、输入清单保护和显式 epoch 续训已完成 CPU 验证；checkpoint v2 严格绑定协议、输入、代码、环境及 RNG/generator 状态，并修复 best/last 写入间中断。C3-A 续训、Flow、评价器及等待器定向回归 67 项通过，相关 `py_compile` 通过。另已在 CPU 上核验 train/development 共 3,600 个 SAR LR/target 数组文件（1,600/200 个样本、92/19 个来源组），未读取 optical/confirmation/test 载荷。以上均不代表模型效果，不改变冻结 C3-A 协议或任何研究 gate；见[当前状态](docs/research/CURRENT_STATE.md)、[实验登记](docs/research/EXPERIMENT_REGISTRY.md)和[数据预检](experiments/diagnostics/c3a_cpu_data_preflight_20261106_v1/report.md)。
- **阶段门槛：** B1 保持 not_promoted / stop_at_b1；B2/B3 继续关闭。C2 结果只按冻结规则决定继续、停止或不确定，不自动晋级历史门槛。
- **状态入口：** [当前状态](docs/research/CURRENT_STATE.md)、[下一步行动](docs/research/NEXT_ACTIONS.md)、[实验登记](docs/research/EXPERIMENT_REGISTRY.md)、[决策记录](docs/research/DECISION_LOG.md)和[研究风险](docs/research/RESEARCH_RISKS.md)。

## 核心科学问题

1. 匹配 optical 是否相对匹配的 SAR-only joint baseline 提供可重复的增量价值？
2. 光学几何、边缘、纹理、低频和语义信息分别何时可迁移、何时造成负迁移？
3. 生成模型是否提供超出确定性方法及 SAR-only 生成控制的任务收益？
4. 如何在已核验的数据表示与观测模型下评价输出一致性？
5. optical 缺失、错位、过时或语义冲突时，方法如何限制损害并回退到 SAR 证据？
6. 联合恢复相对级联方法的质量、鲁棒性和计算成本如何？

完整假设、阶段条件和决策规则见[正式总纲](docs/research/RESEARCH_CHARTER.md)与[研究假设表](docs/research/research_hypotheses.md)。

## 总体研究思路与技术路线

研究按已批准的总纲 v1.7 分阶段推进：

1. **C1 数据与观测审计：** 核实样本来源、数据表示、目标和可支持的观测/退化模型。
2. **C2 光学效用：** 在匹配的 SAR-only 对照上测量光学信息的增量价值和失效条件。
3. **C3 生成式模型：** 先建立 SAR-only Flow/Diffusion 基线，再在满足阶段条件后比较光学条件生成；不预设生成模型更优。
4. **C4 观测一致性：** 只在观测模型有证据支持且生成基线具有研究价值后，验证数据一致性约束。
5. **C5 光学可靠性：** 研究错配/缺失时的抑制与 SAR 回退，须先有光学收益或可复现负迁移证据。
6. **C6 效率：** 仅在质量、结构和观测证据成立后优化少步推断、时间和显存。

这些阶段是可检验路线，不是自动晋级，也不要求把所有候选模块组合进单一网络。

## 研究主张状态

- **已有证据支持：** 用户已批准研究总纲 v1.7；C1 v2 的来源场景文件夹划分和 train/development/confirmation 身份数已有清单证据。地理与采集独立性仍未证实。
- **已实现：** C2 固定筛查代码、C3-A SAR-only Flow 模型/入口，以及零背景对比差诊断代码已实现；代码测试不等同于方法效果证据。
- **评估中：** C3-A 训练仍等待 GPU 资源；多次采样补充评价代码和协议已实现并通过合成验证，尚无真实训练或 development 结果。
- **当前证据不支持：** 现有证据不支持“稳定 optical 增益”主张；C2 v2 gate 仍未判定，不能解释为 optical 无效。
- **已停止：** 历史 B1 仍为 `not_promoted / stop_at_b1`，B2/B3 按现行门槛关闭。
- **待验证候选：** Optical-conditioned Flow、Conditional Diffusion、Residual Flow、后验/观测一致性约束和可靠性自适应引导均是待验证方向，不是已完成贡献。

## 数据与证据边界

- C1 v2 训练/开发来源组为 92/19，组身份互斥；目前没有足够证据证明地理位置、采集时间或精细空间范围独立。当前结果只能描述为 source-folder-disjoint development screen。
- 当前 C2 使用的 SAR 数据是 8-bit PNG 表示域值，目标是已含噪的 noisy pseudo-target；没有原始定标幅度/功率和完整测量映射。合成 Gamma 退化属于表示域受控实验，不等于真实传感器散斑模型。
- 仓库 data/ 中数字命名的 2,000 对 PNG 尚未映射到 C1 v2 身份；C2 和 C3-A 均使用冻结清单绑定的派生固定输入，没有使用该子集。
- C1 v2 有 train、development、confirmation 三种角色，没有独立 test 角色。development 用于选 epoch 和方法比较，不能称独立 holdout；confirmation 已访问一次并封存，不能继续用于方法选择。
- 旧本地中断训练曾使用旧 test_100 名单中的 13 个样本优化。该旧集合不再是未触碰测试集；它与本轮 C1 v2 confirmation 不同。

## C2 v2 冻结筛查

四种退化条件为 x2_light、x2_severe、x4_light、x4_severe。每种条件各训练 sar_only 与 optical_guided 两个 arm，使用相同 C1 v2 train/development 身份、固定输入和预声明 checkpoint 规则。训练配置、样本 seed、指标、mask、来源组统计和继续/停止门槛均以[冻结协议](protocols/mmcanet_phase_c2/screen_protocol_20261029_v2.json)为准，不在运行中调整。

冻结训练矩阵已完成。正式 paired screen 入口为 [evaluate_phase_c2_screen.py](scripts/evaluate_phase_c2_screen.py)；执行因必需 median TBR 未定义而依协议失败关闭，不能报告 gate 通过或停止。另用 [evaluate_phase_c2_non_gate_diagnostic.py](scripts/evaluate_phase_c2_non_gate_diagnostic.py) 完成只用于探索的补充分析。该诊断不是协议晋级结果；单 seed、19 个来源组的结果仅为描述性探索，不支撑独立泛化结论。C2 v2 原 gate 保持不变；下一项独立 SAR-only 生成方法探索及其资源阻塞见 [C3-A 预登记记录](docs/research/EXPERIMENT_REGISTRY.md)。

## 重要研究记录

- [总研究计划](docs/research/MASTER_RESEARCH_PLAN.md)
- [v1.7 修订差异与同步记录](docs/research/RESEARCH_CHARTER_CHANGELOG_20261105.md)
- [v1.7 支持审查](docs/research/RESEARCH_CHARTER_SUPPORTING_AUDIT_20261105.md)
- [文献综述](docs/research/literature_review.md)与[相关工作矩阵](docs/research/related_work_matrix.csv)
- [研究缺口](docs/research/research_gaps.md)、[扩散可行性](docs/research/diffusion_feasibility.md)和[光学信息分析](docs/research/optical_information_analysis.md)
- [数据表示审计](docs/research/DATA_REPRESENTATION_AUDIT.md)、[目标审计](docs/research/REFERENCE_TARGET_AUDIT.md)和[退化审计](docs/research/DEGRADATION_MODEL_AUDIT.md)
- [C1 v2 冻结身份清单](protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/manifest.json)
- 官方数据来源：[TUM SEN1-2 数据集记录](https://mediatum.ub.tum.de/1436631)；原始论文报告 282,384 对样本（[论文 DOI](https://doi.org/10.5194/isprs-annals-IV-1-141-2018)）。当前样本集合规模不等于官方数据集全量规模。

## 主要代码入口

- 主要模型代码：MMCANet/
- 条件筛查模型与训练模块：MMCANetV2/
- 级联与评价参考实现：Early-FusionCascadeJDSR/
- 固定退化输入生成：scripts/prepare_phase_c2_fixed_inputs.py
- 八组冻结训练入口：scripts/run_phase_c2.py
- development 配对评价入口：scripts/evaluate_phase_c2_screen.py
- SAR-only Flow 筛选入口：scripts/run_phase_c3a_sar_flow.py
- C3-A 多次采样评价与安全等待入口：scripts/evaluate_c3a_multisample.py、scripts/wait_for_c3a_gpu.py
- 运行数据、checkpoint、逐图指标与配置记录：experiments/phase_c2_c1_v2_20261031_v1/

## 复现说明

固定输入生成和训练仅在冻结 C1 v2 manifest、C2 v2 protocol、AISAR CUDA 环境及全新输出目录上运行。训练前检查 GPU/磁盘资源；遇失败保留 run manifest 和 checkpoint，并按同配置恢复。不得覆盖历史输出、改变冻结 seed/batch/epochs、用 confirmation 调参或把单 seed screen 写成正式泛化结论。

## 论文状态

paper/MMCANet/ 中的 2026 年 7 月草稿是内部工程试验稿，不是当前投稿稿。后续论文数字须由符合阶段资格的最终 run manifest、逐图结果和预声明统计自动生成；历史 pilot 数字不得作为本轮实验结论。

## 仓库入口

- 研究总纲：docs/research/RESEARCH_CHARTER.md
- 当前状态：docs/research/CURRENT_STATE.md
- 训练矩阵：experiments/phase_c2_c1_v2_20261031_v1/matrix/
- C1 train/development 暂存：data/derived/c1_v2_train_development_20261031_v1/
- 固定输入：data/derived/c2_fixed_inputs_c1_v2_20261031_v1/
