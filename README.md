# JDSR / MMCANet：SAR 联合去斑与超分辨率

> **当前研究状态（2026-11-24）：** C3-A 当前像素域 Flow 配置已按冻结条件停止；SAR-only SwinIR 任务适配、Conditional DDPM 和 SwinIR-centered Residual DDPM 均完成训练、development 评价与 Research Decision Review。两种 DDPM 当前设置均未达到确定性参照，停止同配置扩展。C2 v2 gate 仍未判定，C3-B 未获授权，须先解决新的独立 development 数据协议。详见[2026-11-24 决策复核](docs/research/SAR-only%20Published%20Baseline%20与%20Diffusion%20决策复核_20261124.md)、[Flow 决策复核](docs/research/C3-A%20SAR-only%20Flow%20决策复核_20261123.md)和[当前状态](docs/research/CURRENT_STATE.md)。

> **本轮实验已完成：** 使用 C1 v2 固定 train 1,600/92 来源组、development 200/19 来源组，实际训练并评价 SwinIR SAR 适配、Conditional DDPM 与 Residual DDPM。未访问 Optical、confirmation 或 test 载荷；续训/采样/汇总日志未见 CUDA OOM，Residual 原训练进程在 epoch 24 后停止的原因未查明并已从 checkpoint 恢复。所有数字限于历史复用 development 上的探索观察，不是独立泛化或物理一致性证据。运行产物与方法身份见[基线谱系审计](docs/research/BASELINE_LINEAGE_AUDIT_20261123.md)、[发表方法注册表](docs/research/PUBLISHED_METHOD_REGISTRY_20261123.md)、[冻结扩散协议](docs/research/SAR_ONLY_DIFFUSION_EXPERIMENT_PROTOCOL_20261123.md)、[实验登记](docs/research/EXPERIMENT_REGISTRY.md)和[Research Decision Review](docs/research/SAR-only%20Published%20Baseline%20与%20Diffusion%20决策复核_20261124.md)。


**正式研究总纲：** [v1.8 研究总纲](docs/research/RESEARCH_CHARTER.md)，于 2026-11-09 获用户明确批准。正式题目为“光学先验引导与 SAR 物理一致性约束的生成式联合去斑和超分辨率重建方法研究”。v1.8 增补 SAR 两阶段潜扩散先例，并明确单次生成、像素均值点估计和样本分布是不同评价对象；沿用冻结 C1 v2，不改变历史 gate、数据权限或训练状态。对当前 8-bit PNG 表示域实验，不作未经验证的辐射物理解释。

## 项目目标

研究匹配 optical 是否能在任务匹配的 SAR-only 联合恢复基线上提供可重复增益；识别光学线索的有效范围、负迁移情形、观测一致性与安全回退边界。生成模型、物理约束、错配抑制和局部回退均为待检验方向，不预设为已验证贡献。

## 当前阶段与运行状态

- **C1 v2 划分：** train 1,600 对、development 200 对、confirmation 200 对，按来源场景文件夹划分；没有重新切分。官方 train/development 来源载荷已按冻结清单核验并暂存，见[暂存 manifest](experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json)。confirmation 已于 2026-11-05 用于一次访问后修订的描述性复核，结果 `inconclusive`，现已封存，不用于训练、调参或选模；详见[当前状态](docs/research/CURRENT_STATE.md)。
- **数据来源限制：** 冻结 C1 v2 train/development 的 3,600 个文件已由暂存记录连接到 TUM MediaTUM 条目 1436631 和官方 SHA-512 文件索引；旧 `fixed_train_dev2k` / `fixed_val_dev2k` 哈希审计的零精确匹配只适用于那批历史文件。`./data/` 数字命名 2,000 对仍未映射到 C1 v2。坐标、采集日期、轨道与 patch footprint 交叉表仍缺，故只支持来源场景文件夹级隔离，不声称地理/采集独立。见[来源链核查](docs/research/C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)。
- **C2 v2 方法筛查：** 沿用冻结 C1 v2 来源场景划分完成四种固定退化条件、八组 ×30 epoch 训练。原 paired screen 因 2 个 development 样本的 target 背景中位数为零而 fail-closed，故 gate 未判定；另完成全 200 个 development 样本的非门槛诊断。matched optical 相对 SAR-only 的 PSNR 增益点估计为 −0.0055 至 +0.0061 dB，四个区间均跨零且低于预设 +0.05 dB 继续阈值。见[训练矩阵](experiments/phase_c2_c1_v2_20261031_v1/matrix/manifest.json)、[失败记录](experiments/phase_c2_c1_v2_20261031_v1/screen/execution_manifest.json)和[补充诊断报告](experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。
- **C2 指标诊断：** 新增零背景下仍有定义的表示域均值/中位数目标—背景对比差函数，独立于 C2 v2 evaluator 和 gate。合成边界及既有 evaluator 回归 43 项通过；随后按单独冻结的 CPU v4 协议，在 6 条既有 development 零中位数失败记录上完成 24 次前向。结果仅作失败样本描述，不更新 optical 效用判断或 C2 gate；详见[设计与结果记录](docs/research/C2_ZERO_BACKGROUND_CONTRAST_DIAGNOSTIC.md)及[运行报告](experiments/diagnostics/c2_zero_background_contrast_cpu_20261107_v4/report.md)。
- **C3-A SAR-only Flow 筛选：** 按冻结 x2_light 协议完成单 seed、5 轮训练；best epoch=2。Flow 相对 C2 SAR-only 确定性基线的 development 来源组配对 ΔPSNR 为 −3.0724 dB、ΔSSIM 为 −0.17818，未达到追加第二 seed 条件；当前像素域配置停止。四 draw v3、样本集合评分、逐图指标、checkpoint 和样本归档均已保存；用户明确授权本次忽略并行 CUDA 与 3 GiB 预检门槛，实际无 OOM。只用 SAR LR/noisy pseudo-target，未访问 optical、confirmation、test。C3-A 结果只属复用 development 的探索性观察；C2 gate 仍未判定，不晋级 C3-B。详见[决策复核](docs/research/C3-A%20SAR-only%20Flow%20决策复核_20261123.md)与[实验登记](docs/research/EXPERIMENT_REGISTRY.md)。
- **C3-A 表示域范围与裁剪诊断：** x2_light 固定输入中 SAR 条件可超过 1；本轮 Flow 原始输出低于 0/高于 1 的像素比例为 0.00149/0.01563。冻结评价在最终裁剪至 [0,1] 后计算指标，同时保存 raw/clipped draws 与亮值代理。上述仅是归一化 PNG 表示域统计，不代表物理散射体或辐射一致性；详见[决策复核](docs/research/C3-A%20SAR-only%20Flow%20决策复核_20261123.md)。
- **C3-A 数值域合同：** 本轮按冻结实现保留 SAR 条件、速度场与 Heun 积分数值范围，仅在评价末端裁剪并统计越界；没有据表示域不对称擅改协议。实测数值与限制见[决策复核](docs/research/C3-A%20SAR-only%20Flow%20决策复核_20261123.md)。
- **阶段门槛：** B1 保持 not_promoted / stop_at_b1；B2/B3 继续关闭。C2 结果只按冻结规则决定继续、停止或不确定，不自动晋级历史门槛。
- **v1.8 文献与生成评价边界：** MONTI 2026 已有 SAR 结构超分后潜扩散生成散斑外观的两阶段先例。未来随机生成结果须分别报告单次 draw、多个 draw 的像素均值点估计和生成分布；均值图不能替代单次生成质量或分布真实性。v1.8 不追溯修改 C3-A v3 或历史 gate，详见[逐节差异与采纳记录](docs/research/RESEARCH_CHARTER_CHANGELOG_20261107.md)。
- **旧论文稿状态：** paper/MMCANet/ 中的英文稿是 2026 年 7 月旧 pilot（1,900/100 文件级协议），不是当前 C1 v2 或模型效果证据；仅作历史材料保留，主张边界见[当前状态](docs/research/CURRENT_STATE.md)。
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

研究按已批准的总纲 v1.8 分阶段推进：

1. **C1 数据与观测审计：** 核实样本来源、数据表示、目标和可支持的观测/退化模型。
2. **C2 光学效用：** 在匹配的 SAR-only 对照上测量光学信息的增量价值和失效条件。
3. **C3 生成式模型：** 先建立 SAR-only Flow/Diffusion 基线，再在满足阶段条件后比较光学条件生成；不预设生成模型更优。
4. **C4 观测一致性：** 只在观测模型有证据支持且生成基线具有研究价值后，验证数据一致性约束。
5. **C5 光学可靠性：** 研究错配/缺失时的抑制与 SAR 回退，须先有光学收益或可复现负迁移证据。
6. **C6 效率：** 仅在质量、结构和观测证据成立后优化少步推断、时间和显存。

这些阶段是可检验路线，不是自动晋级，也不要求把所有候选模块组合进单一网络。

## 研究主张状态

- **已有证据支持：** 用户已批准研究总纲 v1.8；C1 v2 的来源场景文件夹划分和 train/development/confirmation 身份数已有清单证据。地理与采集独立性仍未证实。
- **已实现：** C2 固定筛查代码、C3-A SAR-only Flow 模型/入口，以及零背景对比差诊断代码已实现；代码测试不等同于方法效果证据。
- **已停止当前 C3-A Flow 配置：** 冻结单 seed 实验已完成，但未达到追加第二 seed 条件；C3-A 不回答 Optical 效用，C2 v2 gate 仍未判定。详见[决策复核](docs/research/C3-A%20SAR-only%20Flow%20决策复核_20261123.md)。
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
- [v1.8 逐节差异与采纳记录](docs/research/RESEARCH_CHARTER_CHANGELOG_20261107.md)
- [v1.8 候选文本快照](docs/research/RESEARCH_CHARTER_REVISION_PROPOSAL_20261107.md)
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
