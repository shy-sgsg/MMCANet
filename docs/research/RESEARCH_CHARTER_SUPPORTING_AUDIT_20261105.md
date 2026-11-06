# 研究总纲修订支持审查

**审查日期：** 2026-11-05
**审查目的：** 为 v1.7-proposed 提供当前研究状态、历史门槛、数据资格、论文谱系与一手文献依据。
**审查日状态（2026-11-05）：** v1.6 为当时生效版本，v1.7-proposed 尚未获批。**后续更新（2026-11-06）：** 用户已批准 v1.7，批准与同步情况见本文末尾。

## 1. 研究方向审查

附件提出的题目与研究主线，已由 2026-10-30 批准的 v1.6 覆盖：SAR 为主要证据、光学为可靠性受限先验、Conditional Flow/Diffusion/后验采样为候选，并按 C1 数据和观测审计、C2 光学效用、C3 生成模型、C4 观测一致性、C5 可靠性、C6 效率推进。因而本次候选仅更新当前事实和方法边界，不重置研究问题或历史 gate。

## 2. 当前数据协议与实验状态

C1 v2 是项目已有的冻结划分，train/development/confirmation 为 1,600/200/200 对、92/19/20 个来源场景文件夹。C1 v2 资格审计允许有限的来源文件夹互斥开发探索，但不证明地理、采集或细配准独立。用户已明确要求直接沿用该划分；本次没有重新随机分组，也没有访问 confirmation/test 影像或测量文件。

confirmation 已于 2026-11-05 用于一次固定 checkpoint 的描述性复核，复核发生在载荷访问后的执行器字段修订之后，结果为 inconclusive。它已封存，不能继续作为 untouched confirmatory set，也不能用于调参/选模或重复确认。C1 v2 本身没有独立 test 角色。train/development 仍能服务于当前预登记的有限探索，但 development 已反复用于评价与选择。

C2 v2 八组训练已经完成。原 paired screen 因必需的 target median TBR 在两个 development 样本上未定义，按 fail-closed 规则不能产生 gate 判定；非门槛诊断没有达到预设匹配光学增益阈值。原 C2 gate 保持未判定，confirmation 描述性复核不替代它。

C3-A 的 SAR-only Flow 代码和协议在 v1.6 下已预登记。本次经 GPU 提权在 `AISAR` 环境重试；最新预检检测到 3 个其他 CUDA 计算进程，训练器在读取影像前退出，未创建运行目录、checkpoint 或指标。详见[最新预检记录](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。因此方法实验尚未完成。

本次候选审阅只读取了含历史 test 关联说明的项目文字记录，`test_set_accessed=true`（文字）、`test_set_used=false`；没有打开 test 图像、数组、指标文件或 checkpoint，也没有访问 confirmation 载荷。

## 3. 历史方法证据

- B1：在旧 fixed_val_dev2k 上失败并停止；后续审计发现 train/validation 有精确空间/目标重叠，不能当作独立泛化证据。
- B1R：表示/机制审计未通过，停于 B1R；不能解释为光学增益。
- B1F：inner-development gate 未通过；其 fit/inner-dev 后来发现明显精确重叠，不能作独立证据。
- Phase 1-R：目标辅助 oracle ceiling 不能转化为 learned suppressor，训练分支停止。
- Phase 1-S：SAR-derived proxy 与修正后的 suppressor screen 未晋级。
- Phase 1-T：patch utility routing 仅有 target-assisted oracle 审计，T1 未训练、分支停止。
- 当前来源组 B1 v2 revalidation：decision 为 not_promoted / stop_at_b1；15 个来源组的受限开发结果不能被改写为“Joint > Strong Cascade”。
- Phase A2 级联分析仅作历史参考，不能替代当前匹配协议比较。

以上历史结论按原报告保留；v1.7 候选不重跑、不改门槛、不把诊断数据当作晋级证据。详细来源见各阶段报告、实验 manifest 与 CURRENT_STATE.md。

## 4. 新增一手文献与方法影响

本次阅读全文并核验两篇 2026 arXiv 一手来源：

- Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems（arXiv:2609.12953）：算子感知条件速度场与数据一致性已是通用图像逆问题的先例；其理论建立在线性算子和加性高斯噪声假设上。
- Twist Flow for Inverse Problems（arXiv:2610.09281）：指出成对训练下的条件生成可能欠离散；低维 toy 问题使用 MCMC 参考后验，图像任务报告观测一致性与样本 spread，但没有真实图像后验样本作校准参照。

两篇均未测试 SAR speckle、光学条件或 SAR–光学 JDSR，不能证明或否定本地方法。它们只更新通用方法新颖性和候选评价维度。完整审计见[条件流逆问题文献报告](../../experiments/literature_audits/flow_operator_posterior_support_20261105_v1/report.md)；数据角色见[C1 v2 协议审查](C1_V2_DATA_PROTOCOL_REVIEW_20261105.md)。

## 5. 当前论文材料

现有论文目录审计显示：

- paper/MMCANet_formal/main.tex 属于旧的 2k formal preview；其目录 README 已明确其为历史稿，数据、split、checkpoint/evaluator lineage 与当前研究方向不构成完整投稿证据链。
- paper/MMCANet/mmcanet_jdsr_draft.tex 是 2026 年 7 月内部 pilot draft，处于归档范围。
- 当前没有单独的 MMCANetV2 研究稿件可支撑新主线的贡献或结果。

因此，本候选不改写这些历史论文；达到论文证据门槛前，不创建已验证贡献叙事。论文谱系依据为 docs/research/manuscript_lineage_audit.md 及各目录 README。

## 6. 决策边界

如果 v1.7-proposed 获批，应同步正式总纲、AGENTS.md 与 README.md 的研究方向入口；保留全部工程、安全、数据、测试、可复现性规则。批准本身不会解锁 B2/B3、确认集/测试集、改动 C2 v2 gate 或声称 Flow/optical/physics-consistency 有效。方法探索仍按阶段资格和单独冻结的实验协议执行。

## 7. 本次审阅的访问披露

本次读取了项目状态、README、论文目录说明和论文谱系审计。部分历史项目文字包含 test-associated 结果叙述；本次没有打开 test 图像、测量/逐图 CSV、固定输入数组或 checkpoint，也没有使用历史 test 数值作文献、方法或总纲判断。将当前任务的 test 文字暴露记为 test_set_accessed=true、test_set_used=false；累计项目访问/使用标记继续以既有状态文件为准。本次也未打开 confirmation 影像或测量文件；仅核对既有 confirmation 访问状态及其描述性结果标签。

## 2026-11-06 版本题名与 C3-A 资源续核

arXiv:2609.12953 当前公开全文为 v2，题名是 FIRM: Flow-based Imaging via Regularized Minimization；v1 曾题为 Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems。方法边界未变：理论建立在已知线性观测算子和加性高斯噪声上。候选总纲、差异报告、文献综述及矩阵现已使用当前题名并保留 v1 检索名。该更正不改变 C1–C6、历史门槛或任何实验。

C3-A 的 v7 等待会话已无法继续读取，最后一次资源记录为 2026-11-06 09:32:03 UTC；当时训练启动字段为空。没有找到等待器、训练或评价进程，训练及评价目录不存在。因等待脚本此前固定写入已存在的 v7 目录，本轮先按测试驱动方式增加唯一等待编号，再以 `c3a_gpu_wait_20261106_v8` 重新等待；截至本次检查 GPU 仍未满足冻结门槛，尚未训练或读取载荷。C2 gate 仍未判定，C3-B 不因此解锁。

## 2026-11-06 内部稿的有限范围审阅与访问披露

核对 `paper/MMCANet/mmcanet_jdsr_draft.tex` 的题名、摘要开头、Introduction、Related Work 和 Method 至训练目标；在 `Audited Pilot Protocol` 及结果章节前停止。稿件自述为 2026 年 7 月工程 pilot，使用三轮、单 seed、无 validation，且未核实数据物理域；它不是当前新主线的投稿稿件。稿件叙述中出现历史 test 身份数量与定性失败总结，故记录 `test_set_accessed=true`（限历史文字）、`test_set_used=false`。本次未读取 test 图像、test split 清单、任何逐图/汇总 test 数值或 checkpoint，也未使用稿件内容选择模型或阈值。详细谱系状态见[论文谱系审计](manuscript_lineage_audit.md)。


## 2026-11-06 用户批准与同步结果

用户已明确批准 v1.7-proposed，并于本日同步至正式总纲、`AGENTS.md`、`README.md` 与当前状态文件。批准不改变该审查记录中的数据证据和阶段门槛：confirmation 仍为已访问后封存；C2 gate 未判定；B1 保持 `not_promoted / stop_at_b1`，B2/B3 继续关闭；C3-A 等待 GPU 条件，C3-B 仍 gated。磁盘当前约 15.06 GiB，来源波动未明；项目内没有本任务新增的大型清理对象。
