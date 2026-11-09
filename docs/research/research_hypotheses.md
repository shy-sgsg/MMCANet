# 研究假设

## 已知分布的评价量诊断不更新 H1–H6 — 2026-11-09

二维双峰和归一化图像域 CPU 实验确认，均值误差、能量分数、逐像素 CRPS、模态统计及空间变差可揭示所设候选分布的不同失配；集合大小影响有限样本比较稳定性。它们未使用本地数据、模型或 SAR 物理表示，也没有测试光学效用、训练泛化、观测一致性或回退。因此不支持或反证 H1–H6，不改变 C2/C3 gate、C3-A v3、训练暂停或阶段资格。真实 Flow/Diffusion 输出的样本分布及 draw 相关性仍开放。见[二维评价量诊断](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/report.md)、[样本数诊断](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/report.md)与[图像域诊断](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/report.md)。

## Amieva 论文全文不更新 H1–H6 — 2026-11-07

该文确认存在非生成式 SAR-only IW→SM 联合去斑/超分先例，并要求避免宽泛的“此前没有 SAR 联合恢复”表述。但它不使用 optical，也没有测试错配或 SAR 观测一致性，因此不支持或反证 optical 增量、生成模型、观测一致性或可靠性假设；H1–H6、C2 gate 与 C3-A 阶段状态不变。见 D-2026-11-07-055 与[全文审查](../../experiments/literature_audits/amieva_direct_fulltext_followup_20261107_v1/report.md)。

## IOP 生成式 SAR JDSR 二手线索不更新 H1–H6 — 2026-11-07

该线索当前只有摘要级二手条目，未核实一手正文；因此它只提示需要收窄宽泛新颖性措辞，不支持或反证光学增量效用、生成模型效果、观测一致性或可靠性假设。H1–H6、C2 gate 与 C3-A 资格状态不变。见 D-2026-11-07-054 与[来源报告](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## 相关 Gamma draw 均值诊断不更新 H1–H6 — 2026-11-07

45 个共同 Gamma 分量合成条件核验了 draw 相关性如何改变像素均值的误差与 ENL 代理；它未使用模型、项目载荷或 SAR 实测，也没有检验光学增量效用、训练泛化或真实生成分布。因此不支持或反证 H1–H6，不改变 C2 gate、C3-A 冻结协议或阶段资格。真实 C3-A draw 相关性和样本质量仍待训练/评价恢复后检查。见 D-2026-11-07-053 与[CPU 报告](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/report.md)。

对冻结 C3-A v3 protocol/evaluator 的静态对应核对也不更新 H1–H6：当前输出范围不足以评价像素均值或估计生成分布，且该范围与协议相符。若提出新的评价问题，需另作前瞻协议；不以现有 v3 汇总结果推断未保存图像的性质。见[审查记录](C3A_MULTI_SAMPLE_DEPENDENCE_AUDIT_20261107.md)。

## C2 全 development 对比差观察不更新 H1–H6 — 2026-11-07

零背景中位数诊断扩展至全部四条件 development，每条件 200 样本、19 个来源组。matched−SAR-only 的目标—背景对比误差在三种条件上轻微下降、x4_severe 上升；该指标是表示域局部描述，不是整体重建质量。development 已重复用于训练/选择，因此这些观察既不支持也不否定 H1/Q1，不改变 C2 gate 或 H1–H6 定义。见 D-2026-11-07-049 与[CPU 报告](../../experiments/diagnostics/c2_all_development_contrast_cpu_20261107_v5/report.md)。

## SAR 条件 Flow 的 SAR-to-optical 先例不更新 H1–H6 — 2026-11-07

新审阅的 Flow Matching 工作以 SAR 为条件生成 optical 输出，没有测试 optical 是否改善 SAR 输出恢复。因此它只收窄通用 Flow/SAR 条件生成的新颖性，不为本地 H1–H6 提供支持或反证；C3-A 质量仍未知，C2 gate 不变。见 D-2026-11-07-050 与[来源审查](../../experiments/literature_audits/sar_conditioned_flow_sar_to_optical_review_20261107_v1/report.md)。

## DiffusionSAR 全文只收窄新颖性，不更新 H1–H6 — 2026-11-07

一手全文确认已有条件扩散 SAR 去斑，但没有 SR 或推理时 optical。此文献事实要求避免宽泛的方法首创主张；它不测试本地 joint JDSR、optical 增量价值或失配回退，也不提供本地 C3-A 质量证据。H1–H6 与 Q1–Q6 均不更新，C2 gate 和 C3-A 状态不变。见 D-2026-11-07-026 与[全文审查](../../experiments/literature_audits/diffusionsar_conditional_sar_despeckling_fulltext_20261107_v1/report.md)。

## C2 零背景对比差失败案例不支持或否定 H1 — 2026-11-07

CPU 诊断只覆盖 6 条原 median-TBR 未定义的 development 记录；matched−SAR-only 对比误差方向随退化条件和统计量变化，且每条件仅 1–2 条、总计 2 个来源组。它既非完整 development 的预注册晋级评价，也非独立验证；H1/Q1 保持开放，不据此声称稳定 optical 增益或 optical 无效。C2 v2 gate 仍未判定。见 D-2026-11-07-005。

## C1 v2 划分元数据审计不更新 H1–H6 — 2026-11-07

角色/来源组及文件身份互斥只核验数据划分元数据；它不检验 SAR 重建质量、Flow 收益、生成模型优势、观测一致性或 optical 效用。因此 H1–H6 均无新增支持或反证，保持原定义和状态；见 D-2026-11-07-003。

## 新近流逆问题与 SAR 成像先例不改变 Q1–Q6 — 2026-11-07

MS-Flow 研究通用流先验逆问题求解，UAV SAR Imaging 研究仿真 MIMO-OFDM 回波上的扩散后验重建；两者都没有检验本项目的光学增益、表示域 pseudo-target 质量、光学可靠性或本地 SAR-only Flow 收益。因此 H1–H6 保持原定义和开放状态，不作支持或反证更新；C3-A 结果需待用户恢复训练后取得。详见 D-2026-11-07-002 与[文献审查](../../experiments/literature_audits/sar_echo_and_flow_trajectory_prior_art_20261107_v1/report.md)。

## C3-A 四次采样稳定性补充仍不更新 H3 — 2026-11-05 17:24（中国标准时间）

已预登记 C3-A best checkpoint 的 development 多次采样补充协议 v2：每个身份固定 4 个噪声 draw，与确定性 C2 SAR-only baseline 配对，并以来源组为 bootstrap 单位；新增的 EPI/边缘强度比使用固定 LR/noisy pseudo-target edge mask，只反映表示域结构一致性。该分析不估计训练 seed 波动、分布校准或 optical 效用。训练与评价尚未启动；GPU 等待会话 `77673` 最近检测到 4 个 CUDA 计算进程、1,186/4,096 MiB 空闲。H3、H4 及 Q1/Q2 均保持开放，历史 gate 不变。见[补充协议 v2](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v2.json)、[等待记录](../../experiments/diagnostics/c3a_gpu_wait_20261105_v3/manifest.json)和[实验登记](EXPERIMENT_REGISTRY.md)。
## C3-A 代码与单样本计算检查不更新生成假设 — 2026-11-05

预登记的 SAR-only Flow Matching 筛选已完成实现和定向验证，但完整 GPU 训练因显存不足未启动。真实 train/development 各一张的 CPU smoke 仅证明张量、反向传播及 Heun 推理链路可执行，没有用于比较或选模。因此 H3（生成模型是否带来任务收益）、H4（观测一致性）及 Q1 optical utility 均保持开放；C2 gate 和历史晋级门槛不变。见 D-2026-11-05-002、[协议](../../protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json)和[实验登记](EXPERIMENT_REGISTRY.md)。

## C1 v2 confirmation 描述性复核未确认 Q1 — 2026-11-05

沿用项目冻结的 C1 v2 来源场景文件夹划分，没有重新切分。confirmation 复核中，matched optical 相对 SAR-only 的来源组等权 PSNR 增益在四种退化条件下为 −0.0114、−0.0108、+0.0048、−0.0098 dB；均未达到预设的 +0.05 dB 且 95% bootstrap 下界大于零的条件。由于载荷访问后修订了执行器字段映射，并进行了同一冻结设置的技术恢复重跑，该结果仅作描述性观察，不是原 gate 的晋级证据。Q1 仍开放；不能据此确认稳定 optical 增益，也不能据此否定所有模型或指标下的效用。confirmation 后续封存，不再用于调参或模型选择。见 D-2026-11-05-001 和[中文实验报告](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/report.md)。

## C1 v2 开发集探索未确认 Q1 光学增益 — 2026-10-31

沿用冻结 C1 v2 来源场景文件夹划分完成 C2 v2 八组 ×30 epoch 训练。原 paired screen 因两条 development 身份的背景 median TBR 未定义而 fail-closed，正式 gate 未判定。非门槛诊断中 matched optical 相对 SAR-only 的四条件 PSNR 点估计很小（−0.0055 至 +0.0061 dB）、区间均跨零且低于预设 +0.05 dB 阈值；matched 相对 permuted optical 的差值为正。该单 seed、同一 development 集观察既不能确认 Q1 的稳定效用，也不能否定其他指标/模型下的可能效用。Q1–Q6 均保持开放，不据此阶段晋级。详见 D-2026-10-31-007 与[诊断报告](../../experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。

## C2 代码验收不支持或否定研究假设 — 2026-10-30

本轮只验证合成数据上的协议/来源绑定、指标 fail-closed 与掩膜记录合同，没有进行模型训练、推理或真实样本比较。Q1–Q6 均保持开放，未有假设因本轮测试被支持或拒绝。

更新日期：2026-10-30。以下假设均处于开放或部分观察状态，尚无任何一项可视为已确立贡献。

## 自适应可靠性先例收窄机制新颖性，但不提供 H1/H2 证据 — 2026-10-30

Ada-RefSR 为自然图像参考超分辨率中的隐式参考相关性门控，以及受控无关参考/模糊/仿射实验提供直接先例。STMP-DiT 提供阶段感知多模态引导的相关证据，并报告语义先验失效案例。两者都没有测试 光学 引导的 SAR 输出恢复，也未证明 光学 可靠性估计经过校准。文献不改变 H1/H2 或历史门槛；任何 C5 机制都必须以有效的 光学 效用问题和任务匹配的 SAR 保真度证据为前提。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。

## VFM 与 BFM 文献不能验证本地假设 — 2026-10-30

VFM 增加了与未来 H3/H4 设计相关的条件图像逆问题/超分辨率方法先例，但其 ImageNet 结果和加性高斯似然不能证明本地 SAR 恢复有效。已解析得到 C2 已含噪伪目标的候选 Gamma 密度，但完整后验目标和数值行为仍未验证，且不能支持干净 SAR 或物理主张。BFM 只在质量门槛通过后提供通用采样效率选项。两者都不提供 H1 光学效用、H2 信息迁移或错配安全回退的证据；本地假设和门槛均未改变。见 D-2026-10-30-001/002/003 和[来源审查](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md)。

## GLCdiffcr does not update SAR-output hypotheses — 2026-10-29

GLCdiffcr's optical-output cloud-removal objective does not test H1, SAR-output restoration, SAR observation consistency, or mismatch-safe fallback. No local hypothesis or gate changes. See D-2026-10-29-018 and the [abstract-level audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).

## DMDiff does not update SAR-output optical-utility hypotheses — 2026-10-29

DMDiff's output is cloud-free optical imagery conditioned on SAR and cloudy optical input. Its author-reported results do not test H1, optical use in SAR restoration, or SAR fallback under geometric mismatch. No local hypothesis or gate changes. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).

## CDD does not update H1 or mismatch-safety hypotheses — 2026-10-29

CDD's abstract describes SAR-conditioned optical generation for clustering, not optical-conditioned SAR restoration. Its reported clustering outcomes do not answer H1 or test controlled mismatch and fallback. No local hypothesis or gate changes. See D-2026-10-29-016 and the [abstract-level audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).

## SDIF does not update optical-utility or mismatch-safety hypotheses — 2026-10-29

SDIF is optical–SAR diffusion image fusion with semantic classification guidance, not optical-conditioned SAR-output JDSR. Its reported fusion/classification outcomes do not answer H1 or test controlled mismatch and fallback. The inference-time availability of labels in its reverse-sampling loss is unclear; this paper-level ambiguity does not change a local hypothesis or gate. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).

## SAR-to-EO flow-matching results do not update H1 — 2026-10-29

ContraFM-S2O and ReFlowSET are reverse-task SAR-to-EO generation papers. They do not evaluate optical-guided SAR-output restoration or the matched SAR-only comparison required for H1; ReFlowSET's alignment discussion is not a mismatch-robustness result. No hypothesis or gate changes. See D-2026-10-29-010 and the [audit](../../experiments/literature_audits/sar_to_eo_flow_matching_preprints_20261029_v1/report.md).

## γ-Bridge adds despeckling prior art, not hypothesis evidence — 2026-10-29

The arXiv preprint supplies a SAR-only Gamma-structured diffusion candidate for future comparison, but does not test optical guidance, spatial SR, or the local target/observation model. It changes the prior-art boundary only; H1 and the open C3/C4 hypotheses remain untested. See D-2026-10-29-008.

## Generative/observation-consistent direction remains a set of untested hypotheses — 2026-10-29

The review-only charter candidate frames six open questions: (1) when optical adds value, (2) which optical cues transfer, (3) whether generation adds task-relevant value, (4) how to constrain outputs to SAR observations, (5) how to reduce negative transfer, and (6) whether joint restoration is better suited than a cascade. None is established. Conditional Flow Matching, Rectified Flow, diffusion priors, posterior guidance, residual generation, and few-step flow maps have distinct roles and none implies physical correctness. The current noisy normalized PNG target cannot support clean-SAR or calibrated-likelihood claims. Keep the existing B1 stop, B2/B3 gate, and C2 exploratory scope unchanged while review is pending. See D-2026-10-29-006 and the [supporting audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md).

## CloudLULC does not update H1 or the restoration hypothesis — 2026-10-28

CloudLULC provides adjacent, author-described reliability-guided SAR–optical semantic mapping under cloud contamination. It does not test pixelwise SAR restoration or optical's marginal value over a matched SAR-only joint-restoration model, and it does not test geometric misregistration. It narrows generic degraded-optical fusion novelty but provides no evidence for or against H1. Keep B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-065 and the [audit](../../experiments/literature_audits/cloudlulc_cloud_degraded_semantic_fusion_20261028_v1/report.md).

## MTDN does not test H1 or establish optical reliability — 2026-10-28

MTDN uses SAR temporal-change cues to guide dual-polarization SAR despeckling. It neither takes optical imagery as an inference input nor evaluates joint spatial SR. It therefore adds no evidence for or against H1 and does not alter the B1 not_promoted / stop_at_b1 gate. Translating task-derived change cues into a cross-modal optical-reliability estimator remains a separate, unverified hypothesis. See D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).

## Cross-modal misalignment is established prior art; SAR utility remains open — 2026-10-28

The bibliography crosswalk restores general misalignment-aware filtering/reconstruction and adds a SAR–optical matching study. This rules out broad novelty claims for those mechanisms alone; it neither confirms nor refutes H1 because no cited study supplies eligible evidence for the local optical-guided joint SAR restoration question. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-28-043.

## DPS1-SAR is not H1 evidence — 2026-10-28

The conditional DPS1-SAR candidate concerns SAR-only despeckling against multitemporal pseudo-references. It has no optical-guided joint SR comparison and does not advance H1, change B1 not_promoted, or unlock B2/B3. See D-2026-10-28-042 and the [audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).

## Literature crosswalk does not advance H1 — 2026-10-28

The newly indexed Capella SR, 3MOS/SOMA-1M alignment, and conditional SAR400 records do not test optical-guided SAR joint restoration against a matched SAR-only model. They add no evidence for or against H1, do not change B1, and do not unlock B2/B3. See the [crosswalk audit](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).

## H10 — Temporal pseudo-reference feasibility on SEN12TS (candidate only; 2026-10-25)

**Hypothesis:** For a subset of Sumatra tiles with stable backscatter, a leave-input-date-out average of the other same-track Sentinel-1 dates, formed in verified linear γ⁰ power, may provide a lower-noise but still noisy reference for testing optical utility under a real Sentinel-1/2 representation.

**Status/evidence:** The 2,000-pair metadata cohort has matching S1/S2 grids and passes the conservative 400 m spatial buffer. Available filename metadata yields 43 linkage components and a 1,101/899 train/development candidate split, but these components are not verified parent-scene IDs. Official documentation supplied the channel map. A frozen 43-tile pilot and 38-tile second-rank supplement (26 train, 12 development; five components were singletons) passed decode, schema, value-domain, and `<1%` invalid-mask checks. Median 5×5 local-mean change was about 0.90–0.96 dB; normalized-residual correlations remained positive at each temporal lag, and component-paired differences varied. This supports only representation/transport feasibility for the selected tiles. It does not establish stationary reflectivity, independent speckle, a clean target, or optical utility. No local test payload/metric, model, or checkpoint was used.

**Next discriminating work:** Stop further sampling from this cohort for the current temporal-reference question. Continue only if materially new product/acquisition/source-group evidence or an independently supported reference protocol becomes available. Any future reference remains explicitly noisy and requires its own frozen group-disjoint protocol; do not use this cohort for restoration training or model selection.

**Stop/revision condition:** Reject any claim of an independent clean reference unless temporal dependence, radiometric meaning, and residual alignment are independently resolved. The current positive residual correlations and component-level variability leave those conditions open. Even a future passing feasibility screen would establish only conditional pseudo-reference utility, not clean SAR truth.

## H9 literature boundary — MuChaPro-based InSAR regularization (2026-10-25)

The 2026 JSTARS abstract adds spatial-regularization prior art for InSAR phase/coherence restoration after projection-wise despeckling. It does not validate an observation model for the local normalized PNG intensity data, whose source/product/domain are unresolved. H9 remains proposed and untestable until local source/product/signal-domain provenance is recovered and the forward operator is independently validated. See the [audit](../../experiments/literature_audits/muchapro_insar_regularized_inversion_abstract_20261025_v1/report.md) and [D-2026-10-25-07](DECISION_LOG.md#d-2026-10-25-07--bound-muchapro-insar-spatial-regularization-prior-art).

**Split-integrity constraint:** A later exact-overlap audit confirmed shared target pixels between fixed train/validation (580 geometries) and B1F fit/inner-dev (440 geometries). All existing hypothesis observations from those partitions remain historical and conditional; they are not independent generalization evidence or a basis for future selection/promotion. Source-group provenance or a new independently verified prospective split is required first. See the [audit report](sen1_2_train_val_spatial_overlap_audit.md).

| ID | Falsifiable hypothesis | Status / evidence | Next discriminating test | Stop or revision condition |
|---|---|---|---|---|
| H1. Matched optical utility | Under a frozen, matched protocol, correctly paired optical improves full-metric quality over SAR-only joint restoration across seeds, rather than only PSNR or a selected subset. | The current-source prospective B1 matrix was **not promoted** under its frozen gate. Across three seeds and 15 fall validation source folders in one ROI-season, mean paired ΔPSNR was `+0.001617 dB` (source-group bootstrap 95% CI `[+0.000075, +0.003180]`), below the `+0.05 dB` threshold; all three seed means were positive, but mean ΔSSIM was `−0.000892` and failed its zero threshold. A read-only recomputation of the 600 canonical rows found negative mean ΔSSIM in all 45 seed×source-group cells; positive source-group mean ΔPSNR counts were 7/15, 9/15, and 12/15 by seed. The other declared guardrails passed. The group pattern is descriptive, does not identify a cause, and remains relative to a normalized PNG-derived noisy pseudo-target; it is not clean-SAR utility evidence. The sealed C2 config uses only reconstruction and high-frequency residual losses, with no direct SSIM/edge/guard terms; all seeds selected epoch 1, while validation loss fell but SSIM and selection score declined through epoch 10. This suggests a possible objective/metric tension, not a proven cause. Inputs represent source-matched nominal georegistration; fine pixel co-registration remains unverified. This result neither supports a promoted local optical method nor shows optical is intrinsically useless. Earlier B1/B1F observations on the overlap-contaminated partitions remain historical and conditional. Prior work (OMA-SSR, HSFMamba) reports gains under different SR/joint protocols; the 2015 optical-driven nonlocal filter and 2019 GNLM establish optical-guided SAR despeckling and predictor rejection, not this matched joint-SR control. MONet adds optical-derived synthetic supervision but uses SAR-only inference; see D-2026-10-28-049. | Do not repeat this frozen screen or advance to B2/B3. Before proposing another model experiment, resolve local source/parent-scene, acquisition, product-level, and target provenance; establish whether a traceable fall residual-registration audit or a better-specified independent cohort/reference is available. Any later study needs a newly justified protocol and unchanged predeclared gate, with test data inaccessible. | Keep the current route stopped at B1. Reopen only if materially new source, target, or registration evidence supports a new prospective protocol; do not convert this result into a clean-SAR or general optical-utility claim. |
| H2. Pair assignment affects B1F output | The B1F seed0 checkpoint uses optical in a pair-dependent way. | Observed for this checkpoint: matched−permuted PSNR +0.061642 dB; output distances differ across conditions. This is not a scientific utility claim. | Do not repeat merely to accumulate screens; use saved rows. A later independent model can test replication only under a new protocol. | Revise if new model or fixed-pair controls fail to reproduce response. |
| H3. Benefit varies by image and tradeoff | Optical-associated benefit is heterogeneous and aligns with one or more image/ROI properties. | Exploratory: matched vs SAR-only PSNR positive on 156/200; gains decline over post-hoc SAR-only PSNR quartiles, but several structure/target-region measures regress. The descriptive analysis is complete for one failed-screen checkpoint; the target region is target-derived and tie-expanded, not an independent physical-scatterer ROI. | No further post-hoc stratification of these rows. Any follow-up must be prospective, target-free at inference, and evaluated on a new protocol/checkpoint or independent data. | Do not promote stratified claims without prospective replication across seeds/data; drop them if metric trade-offs persist. |
| H4. Geometry transfers more safely than texture | Optical geometry/edge cues provide more reliable help than raw texture for SAR structure recovery. | Unverified assumption; no controlled cue-isolation evidence. HSFMamba's full text confirms that its heavy-haze example is qualitative and says adaptive optical-quality perception is future work; it does not compare geometry to texture, quantify residual shifts, or establish a fallback. VGSRF-Net uses Retinex reflectance/illumination priors to reconstruct intermediate SAR features for fusion, but does not isolate geometry versus texture for SAR-image restoration. The full-text 2015 optical-driven filter uses optical-vector distance in bilateral predictor weights and blends its estimate with SAR-BM3D using a multiscale-ENL mask; 2019 GNLM rejects predictors with a SAR-domain test. Neither isolates geometry versus texture or tests controlled spatial offsets. WMoE-Net is now publisher-verified at abstract level as alignment/quality-gated ship-detection prior art, but full-text protocol and any transfer to pixel restoration remain unverified; see the [updated source screen](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md). Registration-error sensitivity is studied for classification and detection, but local residual error and restoration transfer are unknown. See the [HSFMamba full-text audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md). | Only after H1/target and registration provenance are resolved, compare a prespecified cue against raw optical with matched capacity and budget; separately test geometry perturbations only under an authorized protocol. | Reject if gains are not metric-consistent or cue selection is target-dependent at inference. |
| H5. Utility depends on degradation and task | Optical utility varies with physically defensible degradation and with despeckling-only, SR-only, and joint objectives; it need not be monotonic. | Unverified. Local synthetic representation protocol does not establish real sensor physics. Muzeau et al. (EUSAR 2024) compare noisy/noisy, noisy/denoised, and denoised/denoised SR supervision on one airborne SLC scene; this informs task decomposition but does not compare sequential restoration order. The paper reports 100 SR epochs, while its linked repository history has no versioned 100-epoch config; this does not identify the actual paper-run configuration. See the [history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md). | Define degradations and subtask targets first; screen a few defensible conditions with common fixed inputs. | Do not expand a matrix when degradation/target validity is unresolved or compute is unavailable. |
| H6. Local reliability can prevent negative transfer | An inference-available, target-free local score can reduce harm while preserving matched utility. | Not supported for the local joint-SR task. Phase-1 R/S proxies did not yield a deployed positive-PSNR rule; T oracle was target-assisted and failed structure criterion. In the saved B1F seed0 diagnostic, all-zero optical underperformed its SAR-only base on all eight recorded metric contrasts (paired ΔPSNR `−0.014729 dB`, 95% CI `[−0.021834, −0.007967]`); B1F code has no explicit reject state or zero-optical bypass. A no-training cue diagnostic tested `1 − mean(normalized entropy)` of the nine offset matcher weights on 200 fixed validation images. It failed its fixed all-eight-positive direction rule: Spearman ρ was `+0.156` for PSNR, `+0.108` SSIM, `+0.098` EPI, `+0.107` edge ratio proximity, `+0.016` TBR error reduction, `+0.212` bright-peak ratio proximity, `−0.094` bright-energy ratio proximity, and `−0.094` bright-region MAE reduction. This specific entropy cue is not supported as a global optical-utility/rejection signal on that failed-screen checkpoint; it does not rule out all cues. The full-text 2015 optical-driven filter uses a target-free multiscale-ENL classifier to blend optical-guided bilateral filtering with SAR-BM3D for despeckling; GNLM rejects predictors with a SAR-domain statistical test and reports one qualitative stale-guide case. These are task-specific rejection/fallback mechanisms, not calibrated expected-restoration-gain estimates or controlled geometric-mismatch evaluations. GCOANet's publisher abstract adds diffusion-based reference generation in optical-assisted blind-spot SAR despeckling, but does not clarify the inference-time optical path or establish geometric robustness or a calibrated utility cue. The despeckling methods do not establish quantitative SAR-only fallback for joint SR under geometric mismatch. Onat's 2026 publisher abstract describes metric-driven pixelwise fusion of multiple SAR despeckling outputs with local quality maps and pseudo-reference generation inspired partly by optical-assisted restoration. The PLFSID landing page/README are accessible, but README says `main.m` is demo-only and the algorithm scripts are withheld; the actual source and full article remain inaccessible. This narrows generic local output selection for SAR despeckling but does not establish optical input at inference, geometry robustness, or joint-SR fallback. Correspondence uncertainty (matching), discrepancy-gated fusion (WEMFusion), and Retinex-guided SAR-feature reconstruction (VGSRF-Net) address adjacent tasks. | No new selector training until an observable proxy and prospective metric gate are specified for the joint-SR task. | Reject proxies that require targets at inference, increase normal-condition loss beyond threshold, or fail multimetric validation. |
| H7. Diffusion adds value beyond regression | A conditional residual diffusion model improves uncertainty/fidelity tradeoffs in a failure regime where direct regression is inadequate. | Unverified and deferred. The reviewed RSDM/LSIR paper establishes SAR-only joint SR/despeckling diffusion prior art, while 2025 GCOANet's publisher abstract describes diffusion-based cross-domain reference generation in optical-assisted SAR despeckling. The official IGARSS 2025 program also lists a physics-inspired SAR-despeckling diffusion paper, but only existence-level metadata is accessible; no technical comparison is possible. The GCOANet abstract does not establish whether optical is an inference input or whether that design improves optical-guided joint SR under the local target protocol. Other SAR diffusion papers report both sampling acceleration and task-specific hallucination/scattering-fidelity concerns. This evidence neither supports local utility nor rules diffusion out. | Reopen only after documenting a regression failure and a valid target. Before any experiment, audit the method/code protocol and GPU/sampling budget, then preregister SAR-only vs SAR+optical controls. | Stop if it only increases visual sharpness, hallucinated bright scatterers, runtime, or stochastic variance without verified utility. |
| H8. Joint vs cascade tradeoff | Under one valid matched protocol, a jointly optimized model will offer a better predeclared quality–cost tradeoff than the strongest sequential and end-to-end cascade baselines while meeting structural and bright-scatterer guardrails. | Not supported as a local superiority claim. Phase-A2 Case B reported 17.3751 dB for the best `cnn→swinir` cascade and 17.3258 dB for SwinIR-Joint, about `+0.0493 dB` to the cascade on PSNR. A later overlap audit found the fixed development partitions share target pixels, so this historical comparison is reference evidence only and not independent generalization evidence. It specifically rules out presuming `Joint > Strong Cascade`; it does not prove that cascades are generally superior. Amieva et al. (2026) add same-sensor IW→SM supervised joint SR/despeckling and compare two-stage with end-to-end training for one architecture, but do not compare both task orders or optical guidance; the study's three-image test and pre-split full-image-set normalization also constrain protocol comparability. Lee and Son's MONTI 2026 paper supplies ordered structure restoration→speckle synthesis and reports comparison with single-stage baselines. Muzeau et al.'s author-hosted EUSAR paper further evaluates three ×2 training regimes—Noisy2Noisy, Noisy2Denoised, and Denoised2Denoised—by changing whether LR/HR inputs and targets are MERLIN-denoised within a single SR network. This is denoising-state supervision analysis, not a comparison of `despeckle→SR` versus `SR→despeckle` or a matched end-to-end cascade versus jointly optimized system. Its MERLIN outputs are denoised pseudo-references; the experiments use one ONERA SETHI scene and a patch-level train/validation split, so they do not establish scene-independent generalization. See the [author-hosted paper](https://www.jeanphilippeovarlez.com/publications/ewExternalFiles/Papier_Superresolution.pdf) and [author-linked code repository](https://github.com/muzmax/SAR_super_resolution). | After source/target provenance is resolved, a prospective source-group-disjoint development split is frozen, and suitable compute is available, register five concrete variants: stage-wise `D→SR` and `SR→D`, end-to-end `D→SR` and `SR→D`, and direct joint restoration. This crosses task order with training coupling for cascades; see the [operator-level comparison design](restoration_order_analysis.md). Use identical fixed inputs, targets, masks, and evaluator; predeclare model-specific tuning and compute budgets, seeds, paired metrics, intermediate-output diagnostics, runtime, memory, and parameter counts. Keep test data inaccessible. | If joint restoration fails the predeclared multimetric and cost criteria against the strongest valid cascade, make no joint-superiority claim; report the observed task- and budget-specific tradeoff. |
| H9. SAR observation-model consistency | With a verified SAR representation and forward operator, adding a data-fidelity/unrolled step will reduce held-out observation inconsistency while preserving structure and bright-scatterer metrics relative to an otherwise matched reconstructor. | Proposed and currently untestable as a physical claim. Local inputs are project-labelled 8-bit PNGs in a normalized representation; source sensor, product level, and calibrated amplitude/intensity/power domain are unverified. The current 256→128 chain is a composite representation-domain stress degradation and the target is a noisy pseudo-target, not an independent clean SAR reference. SAR-CDL and the IGARSS physics-inspired diffusion program item establish adjacent prior art only; neither validates a local forward operator. | First recover authoritative source/product and processing metadata, then define and independently validate the forward operator in its correct signal domain. Only then compare a matched data-driven estimator with a data-consistent/unrolled model under a source-group-valid split, using observation residuals together with edge, radiometric, and bright-scatterer measures. | If the operator or signal domain cannot be verified, restrict claims to representation-domain consistency. Reject the route if residual minimization only matches the noisy pseudo-target, smooths or hallucinates bright scatterers, or harms the predeclared structure/fidelity guardrails. |

## H8 literature addendum — task-order source search (2026-10-29)

The bounded review confirms joint SAR despeckling/SR prior art and comparisons among SR supervision targets. It does not verify a matched task-order and training-coupling experiment. A separate journal-indexed cascade-comparison claim remains unverified because publisher full text is inaccessible and its relation to DSRKD is unresolved. H8 stays an open hypothesis with existing source/target and development-split prerequisites; no local superiority claim follows. See D-2026-10-29-028 and the [source review](../../experiments/literature_audits/sar_task_order_source_review_20261029_v1/report.md).

## H4 theoretical addendum — frequency-specific effect of unknown residual shift (2026-10-25)

Under a single-frequency linear model with an independent, per-example unknown global optical translation, the incremental LMMSE gain after SAR observation depends on the squared magnitude of the shift distribution's characteristic function. A zero-mean Gaussian shift gives `q(k)=exp(−kᵀΣδk)`, so the model predicts stronger attenuation of optical linear utility at higher frequencies along uncertain displacement directions. A known, exactly corrected global translation is invertible and does not remove information in the ideal no-cropping model. This is a model-based derivation, not empirical support for H4, a general nonlinear-network bound, or a claim about local SAR/optical data. Any future test requires authorized source/registration metadata and an independent prospective split; see the [derivation and assumptions](optical_information_analysis.md#frequency-resolved-linear-utility-under-random-residual-translation-2026-10-25).

## H1 theoretical addendum — additive SAR-noise sensitivity (2026-10-25)

For a fixed single-frequency linear SAR transfer and fixed optical noise/cross-modal relation and shift coherence, the derived LMMSE reduction from adding optical input is nondecreasing in additive SAR-noise power. This is a conditional analytic reference only; it does not imply monotonic gains across Gamma-speckle looks, SR scales, finite-network training, or the local representation-domain data. H1 remains not promoted and requires a prospective source-group-disjoint protocol for any severity-by-optical experiment. See the [derivation and limits](optical_information_analysis.md#sar-noise-severity-under-fixed-optical-reliability-2026-10-25).

## Interpretation boundary: conditional information and noisy pseudo-targets

Let `X` denote the latent SAR quantity whose restoration is scientifically desired, `Y` the SAR observation supplied to a model, `O` its optical guide, and `T` the available noisy or filtered pseudo-target. For a square-integrable target `U` and inputs `W`, define the population Bayes risk as `R_U^*(W) = inf_g E||U − g(W)||_2^2`, where the infimum is over measurable estimators. Under squared-error loss, the risks for predicting `T` satisfy

\[
R_T^*(Y)-R_T^*(Y,O)
=\mathbb{E}\!\left[\left\|\mathbb{E}[T\mid Y,O]-\mathbb{E}[T\mid Y]\right\|_2^2\right]\ge 0.
\]

This follows from the conditional-variance decomposition. It describes the value of optical input for predicting **`T`**, assuming optimal conditional-mean estimators and the same population distribution. A finite network may fail to realize this population gain because of model, optimization, or generalization error.

If clean `X` were available, the analogous squared-error reduction would be

\[
R_X^*(Y)-R_X^*(Y,O)
=\mathbb{E}\!\left[\left\|\mathbb{E}[X\mid Y,O]-\mathbb{E}[X\mid Y]\right\|_2^2\right].
\]

This depends on whether optical input changes the conditional mean of `X`. Conditional mutual information concerns the full conditional distribution: it can be positive when `O` changes conditional variance or other aspects of that distribution while leaving its mean unchanged, yielding no Bayes-risk reduction under squared error. Thus `I(X;O\mid Y)` and squared-error utility answer related but different questions.

The `T` identity does not by itself establish improvement for latent `X`. Let `R_U(f) = E||f-U||_2^2` and `ε = T − X`. For SAR+optical and SAR-only estimators `f_{SO}` and `f_S`, respectively,

\[
R_T(f_{SO})-R_T(f_S)
=R_X(f_{SO})-R_X(f_S)
-2\mathbb{E}\langle f_{SO}-f_S,\varepsilon\rangle.
\]

Unless the residual cross-term is known to vanish or is otherwise controlled, their ranking against `T` need not equal their ranking against `X`. No independent clean SAR reference or residual-independence condition has been established for the local protocol; therefore B1/B1F PSNR deltas remain proxy-target consistency evidence.

For optimal conditional-mean estimators, the difference has an exact general decomposition. With `T=X+E`, let `δ_X=E[X|Y,O]−E[X|Y]` and `δ_E=E[E|Y,O]−E[E|Y]`. Then `ΔR_T=ΔR_X+E||δ_E||²+2E⟨δ_X,δ_E⟩`. Thus proxy-target and latent-signal Bayes gains agree only under a condition such as `E[E|Y,O]=E[E|Y]`; otherwise their ordering is unrestricted. This holds for square-integrable variables and optimal population estimators, not for finite MMCANet metric deltas as a direct identification. The shared-input local pipeline does not establish the required residual condition. See the [distribution-free derivation](optical_information_analysis.md#exact-bayes-risk-decomposition-for-a-noisy-pseudo-target-2026-10-25).

The objective's `I(X;O\mid Y)` is a useful statement of possible information in optical beyond SAR, not a quantity identified by saved PSNR/SSIM rows. In the special scalar conditional-Gaussian case, it is linked to fractional posterior-variance reduction through the conditional correlation, while absolute squared-error gain also depends on SAR-only conditional variance; see the [optical-information analysis](optical_information_analysis.md#why-degradation-severity-has-no-general-monotonic-rule). Outside that model, positive conditional information need not change the conditional mean and therefore need not reduce squared-error Bayes risk. Any clean-SAR utility claim needs an independently justified reference or explicit assumptions controlling the target-residual cross-term, as well as source-group-valid evaluation. This clarification changes no historical gate and introduces no experiment. See the [metric audit](evaluation_audit.md) and [optical-information analysis](optical_information_analysis.md).

The hash-bound B1 v2 configuration also shows that the LR observation is generated from the same normalized SAR tensor saved as the HR pseudo-target, followed by representation-domain Gamma perturbation. The one-mode Gaussian result and a multivariate linear counterexample show that shared target residual can make optical risk reduction for `T` differ from reduction for latent `X` in either direction, depending on operator/covariance geometry; neither identifies the local residual covariance or its direction. See the [shared-input derivation](optical_information_analysis.md#shared-pseudo-target-and-lr-input-a-one-mode-gaussian-reference-2026-10-25) and [multivariate counterexample](optical_information_analysis.md#a-multivariate-counterexample-to-one-mode-target-risk-ordering-2026-10-25). H1 remains not promoted; no new experiment or gate is authorized.


## Optical channel analysis

H4 and H5 should be read with the task-specific channel distinctions in [optical_information_analysis.md](optical_information_analysis.md#working-theory-optical-channels-and-task-specific-value-2026-10-19): geometry may support localization, while reflectance texture and semantics are not direct SAR-scattering measurements. The Gaussian derivations there are illustrative special cases, not predictions for the local multimodal data. No local cue-isolation evidence exists, and the split-integrity constraint still blocks new model selection.

## H1 evidence addendum — current-source B1 v2 image-level direction (2026-10-23)

The frozen B1 v2 paired validation CSV shows positive three-seed image-mean PSNR deltas for 129/200 identities (106 positive in all three seeds; 45 negative in all three), but SSIM worsened for 192/200 and edge-strength ratio error worsened for all 200 in each seed. The top 40 identities accounted for 61.4% of the positive PSNR-delta sum. This is exploratory distribution evidence on the same 15-source-group, fall-only development cohort and normalized PNG-derived noisy pseudo-target; it does not alter H1's not_promoted status, identify a cause, or demonstrate clean-SAR utility. See the [audit report](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/report.md).

**Source-folder diagnostic (2026-10-25).** Averaging the three fixed-seed group means, ΔPSNR was positive in 10/15 source folders and negative in five; ΔSSIM was negative in all 15. The identity-weighted mean ΔPSNR remained only +0.001617 dB, below the frozen +0.05 dB gate, and mean ΔSSIM remained −0.000892. This supports a limited observation that the small PSNR direction spans multiple folders while the structure-similarity loss is consistent across them. It does not identify a cause or imply clean-SAR benefit. See the [predeclared group diagnostic](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/report.md) and [D-2026-10-25-123](DECISION_LOG.md#d-2026-10-25-123--keep-the-source-group-b1-screen-at-stop-at-b1).

## H1 evidence addendum — C2 objective/code relationship (2026-10-23)

Across the three C2 seeds, the recorded validation total falls from epoch 1 to 10 while SSIM and the strict selection score fall; residual-contribution RMS rises and the SAR backbone remains frozen. The only active losses are pseudo-target reconstruction and a high-frequency residual penalty, with direct SSIM/edge/gate/guard terms disabled. This makes objective/metric tension plausible but does not establish causation, clean-SAR utility, or that any alternative loss or backbone schedule would pass the frozen gate. D-2026-10-23-46 changed two current model/trainer hashes after the run; the sealed hashes remain the provenance for the historical result. H1 remains not promoted. See the [code audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).

## H6 evidence addendum — source-aligned C2 gate rank (2026-10-24)

On the repeatedly used 200-name C2 validation cohort, input-derived mean gate activation had per-seed Spearman rho +0.8675, +0.8590, and +0.8383 with target-derived per-image actual-minus-base strict-score change; the mean was +0.8549 and the 15-folder cluster-bootstrap interval [+0.7259,+0.9107]. This is bounded support for a rank association with the composite score. Mean seed-wise rho was +0.0551 for ΔPSNR and −0.0604 for ΔSSIM, with direction changes. The score also contains edge-ratio and TBR terms; the association does not identify a component, establish calibration, or show that thresholding the gate reduces harm. The checkpoints were selected using this validation cohort, and the folders are not verified parent scenes. H6 remains open for prospective testing, not supported as a deployable reliability mechanism. See the [protocol and result](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/).

## H8 camera-pipeline literature boundary (2026-10-24)

Qian et al. (ICCP 2022) provide direct prior art for task-order ablations and sequential versus end-to-end processing in Bayer RAW denoising, demosaicing, and SR. Xing and Egiazarian (CVPR 2021) compare serial, pairwise-joint, and fully joint camera restoration under a fixed task order. This rules out novelty claims for order ablations in general imaging, but does not test SAR despeckling/SR order, SAR radiometric/structure safeguards, or optical guidance. H8 remains an unverified, prospective SAR-specific quality–cost hypothesis. Its matched comparison remains gated on authoritative source/target provenance and an independently verified development split; B1 remains not_promoted, so B2/B3 do not open. See the [source audit](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/report.md).

## H8 evidence addendum — MD-Glow (2026-10-25)

MD-Glow adds a SAR-specific shared-representation multi-task precedent, but its indexed paper text provides despeckling metrics rather than a quantitative joint-SR or task-order comparison. Its qualitative despeckle-to-SR example cannot determine whether a joint output is better than a cascade. H8 therefore remains open and gated on valid source/target provenance and an independent development cohort; no experiment is authorized by this literature result. See [D-2026-10-25-08](DECISION_LOG.md#d-2026-10-25-08--record-md-glow-as-sar-multi-task-despeckling-sr-prior-art).

## H8 evidence addendum — ISSMSAR task-order framing (2026-10-29)

ISSMSAR explicitly describes both stepwise despeckling→SR and SR-first auxiliary despeckling as existing SAR processing strategies and qualitatively criticizes the stepwise route. Its reported comparisons do not isolate order or training coupling against a matched direct-joint model. H8 remains a prospective SAR-specific quality–cost hypothesis, not a novelty claim that SAR literature has never discussed task order; the local comparison stays gated on valid source/target provenance and an independently verified development cohort. See [D-2026-10-29-025](DECISION_LOG.md#d-2026-10-29-025--bound-issmsars-sar-task-order-discussion) and the [audit addendum](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md#addendum-2026-10-29-sar-task-order-framing).

## H4/H6 evidence addendum — FAF-CD adjacent fusion boundary (2026-10-24)

FAF-CD is a change-detection precedent for feature rectification, deformable spatial fusion, and adaptive Fourier/Haar branch gating. Its BRIGHT stress tests perturb optical appearance and SAR speckle/radiometry/resolution, with validation-only detection metrics; the reviewed protocol does not inject geometric shifts or estimate whether optical guidance improves SAR reconstruction. This narrows broad alignment/fusion novelty while leaving H4's geometry-versus-texture utility and H6's target-free SAR-restoration fallback hypotheses unresolved. See the [audit](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/report.md) and [decision D-2026-10-24-40](DECISION_LOG.md#d-2026-10-24-40--bound-faf-cd-as-adjacent-change-detection-prior-art).
## H1/H6 evidence addendum — Gaya et al. abstract-level prior art (2026-10-25)

Gaya et al.'s full-text GRETSI 2023 Fuse-MERLIN paper demonstrates prior optical fusion architectures for complex-SAR self-supervised despeckling and reports qualitative fine-road gains on one scene; the authors explicitly state that quantitative validation was not performed. Its preliminary two-pixel registration test is not a controlled shift sweep. The distinct-title 2024 IGARSS paper by the same authors remains abstract-level, and the version relationship is unverified. This narrows broad novelty around optical cooperation and self-supervised SAR despeckling, but provides no task-matched evidence for H1's joint-SR optical utility or H6's quantitative mismatch-safe fallback. Both hypotheses remain open and blocked on the valid prospective protocol. See [D-2026-10-25-66](DECISION_LOG.md#d-2026-10-25-66--classify-fuse-merlin-as-single-scene-qualitative-despeckling-prior-art), the [2023 full-text audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md), and [2024 abstract audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).

## H1/H5 literature boundary — SpeckleDiff-2D (2026-10-25)

SpeckleDiff-2D adds SAR-despeckling diffusion prior art trained on optical-dataset-derived synthetic speckle. The abstract does not disclose inference inputs, so it neither supports nor refutes H1's question about same-scene optical utility over a matched SAR-only joint baseline. It also does not justify local diffusion under the current failed B1 gate. H1/H5 remain untested by this literature record; see the [audit](../../experiments/literature_audits/specklediff2d_abstract_screen_20261025_v1/report.md) and [D-2026-10-25-06](DECISION_LOG.md#d-2026-10-25-06--record-specklediff-2d-and-gaya-thesis-route-boundaries).



## H1/H8 literature boundary — Amieva SM subaperture enhancement (2026-10-25)

A second, distinct Amieva et al. paper studies SAR-only, same-grid Stripmap enhancement from Doppler subapertures and reports a noisy-reference speckle/detail trade-off. This strengthens the requirement that H1 compare optical against a task-matched SAR-only joint baseline; it does not test H1 optical utility or H8 task-order alternatives. Its test-distribution preprocessing caveat and two-scene test limit transfer. H1 remains untested locally and blocked on a valid prospective cohort; B1 remains not_promoted, so B2/B3 remain gated. See the [audit](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/report.md) and [D-2026-10-25-17](DECISION_LOG.md#d-2026-10-25-17--classify-amieva-sm-subaperture-enhancement-as-sar-only-prior-art).


## H1/H8 prior-art boundary — Dong complex-valued subaperture SR (2026-10-25)

Dong et al. add abstract-level prior art for complex-valued SAR SR using subaperture learning/fusion, narrowing broad mechanism-level novelty. The available evidence does not establish a joint despeckling objective or a joint-versus-cascade result, and it does not test optical utility. H1 remains untested locally; this literature screen does not alter B1 or open B2/B3. See [D-2026-10-25-18](DECISION_LOG.md#d-2026-10-25-18--classify-dong-complex-sar-sr-as-abstract-level-prior-art) and the [audit](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/report.md).


## H1/H6 implementation boundary — MMCANetV2 optical absence versus mismatch (2026-10-25)

The source provides an explicit absent-optical model path returning the SAR base, but the optional bounded alignment and structural gate do not establish that the model detects harmful misregistration or predicts optical residual utility. H1 matched optical benefit and H6 target-free mismatch-safe fallback remain untested under a valid prospective protocol; B2/B3 remain gated. See the [source audit](../../experiments/diagnostics/mmcanet_v2_architecture_source_audit_20261025_v1/report.md) and [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness).


## H1 theoretical addendum — fixed linear model with multiplicative Gamma intensity speckle (2026-10-25)

For `Y=XN`, where `X` is a nonnegative latent intensity and `N~Gamma(L, rate=L)` is independent unit-mean multiplicative speckle, the scalar LMMSE reduction from observing `O=cX+U` after `Y` is

\[
\Delta R_{\mathrm{LMMSE}}=
\frac{c^2\sigma_X^4 q(L)^2}{\sigma_U^2+c^2\sigma_X^2q(L)},\quad
q(L)=\frac{\mu_X^2+\sigma_X^2}{L\sigma_X^2+\mu_X^2+\sigma_X^2}.
\]

With latent intensity statistics and optical reliability fixed, this model predicts larger linear-MSE benefit as `L` decreases. It does not apply to the local post-log normalized PNG multiplier, changing SR scales, or a finite learned estimator. H1 remains unverified locally; no severity-by-optical experiment is authorized by this derivation. Full assumptions and covariance steps are in [optical information analysis](optical_information_analysis.md#unit-mean-gamma-multiplicative-noise-under-a-fixed-linear-cross-modal-model-2026-10-25).

## H1 implementation addendum — Gamma sampler moments (2026-10-25)

The canonical sampler matches the predeclared unit-mean Gamma moments for L=1,4,8 in the tested intensity and representation cases. Its amplitude mode uses the square root of the intensity-power multiplier; squared moments match the Gamma law while amplitude-factor mean is below one. This is a code-level distribution check, not evidence that local PNG noise is physically parameterized or that optical utility is monotone with physical ENL. See D-2026-10-25-85 and the [audit](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/report.md).


## 2026-10-25 — Representation-domain boundary

Current fixed-development observations concern a log-compressed 8-bit-PNG representation with synthetic representation-domain noise. They do not test calibrated physical speckle, clean-reference recovery, or optical utility under an independent source-group split. Keep those as open questions; no hypothesis or gate is promoted by this source/config audit. See the [data-domain audit](../../experiments/data_audits/fixed_val_sar_representation_contract_20261025_v1/report.md).


## 2026-10-25 — Dataset lineage is not a hypothesis confirmation

The local overlap pattern raises QXS-SAROPT or similar retiling as a candidate explanation, but it does not verify source identity or scene groups. Treat source-specific transfer and generalization claims as open until a source crosswalk and prospective group-disjoint partition are verified. See the [lineage audit](../../experiments/data_audits/local_2k_source_lineage_candidate_20261025_v1/report.md).


## 2026-10-25 — Exact-byte non-match does not select an alternative dataset

The existing train/validation crosswalk rules out no transformed source: it found no byte-identical SEN1-2 files, while re-encoding and retiling remain plausible and QXS identity is unverified. Keep all source-specific explanations open pending an authoritative manifest or source-group crosswalk. See the [lineage audit](../../experiments/data_audits/local_2k_source_lineage_candidate_20261025_v1/report.md).

## 2026-10-25 — Conditional-mean requirement for noisy-target interpretation

Noise2Noise Eq. 6 states an indexed clean-target condition. In population form, unrestricted squared-loss estimators match when E[Z|X]=E[S|X]; target noise with zero conditional mean given S and independent of X given S is sufficient, and this population conditional mean need not equal a sample-specific S. The local fixed-input builder derives LR and HR pseudo-target from one preprocessed SAR observation, and no latent-target/noise model establishes the population relation. This is an interpretation prerequisite, not evidence for or against H1 optical utility. Keep H1 unverified and proxy-target metrics scoped to target consistency. See D-2026-10-25-178 and the population-mean audit.

Noisier2Noise is a separate candidate for a known, sampleable source-noise model; its additive correction and multiplicative Bernoulli special case do not characterize the local normalized PNG noise. This does not change H1 or create a new model hypothesis. See D-2026-10-25-179.

## H4 prior-art boundary update (2026-10-26)

**Observation:** H4 remains unverified. The 2015/2019 optical-guided despeckling methods use optical gating or SAR-domain predictor rejection, but the reviewed evidence does not compare geometric cues against texture under controlled registration error. The 2024 Gaya abstract likewise does not expose such a protocol. This leaves the geometry-versus-texture hypothesis open. See [D-2026-10-26-046](DECISION_LOG.md#d-2026-10-26-046--add-three-optical-guided-sar-despeckling-records-with-unresolved-historical-membership).

## H1/H4 B1 v2 multi-metric observation (2026-10-26)

**Observation:** In a read-only summary of the 15 fixed B1 v2 source groups, EPI correlation increased in every group while edge-strength and bright-peak ratio errors increased in every group. TBR ratio error increased in all 15 group means; bright integrated-energy ratio error and bright-region MAE increased in 12/15 groups.

**Interpretation boundary:** This is a development-set observation within one fall ROI-season, with nominal source matching but unverified fine pixel registration and noisy 8-bit pseudo-targets. It cannot distinguish geometric mismatch from optimization, target, or metric-domain effects. H1 remains unresolved; no optical utility, reliability mechanism, or physical scattering claim is supported. The B1 v2 gate remains `not_promoted`, and B2/B3 remain gated. See D-2026-10-26-123.

## H-SLC-1 — Sentinel-1 IW I/Q self-supervision after product-specific processing (unverified)

**Hypothesis:** After correct TOPSAR deramping/demodulation, spectrum handling, and product-aware preprocessing, Sentinel-1 IW real/imaginary components may satisfy a useful independence condition for MERLIN-style self-supervision. **Status:** untested and not currently supported for this cohort. The primary MERLIN paper says direct IW use is invalid and leaves extra I/Q decorrelation preprocessing for future work. Do not treat this as a method or start data access/training; first freeze processing and a training-only statistical diagnostic under an authorized, source-grouped protocol. See D-2026-10-26-147 and the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).


## H11 — Cross-attention pair-specific marginal effect (2026-10-27)

**Hypothesis:** With direct optical fusion retained, bypassing the trained optical cross-attention residual updates changes predictions more under matched optical than under the frozen cross-folder permutation.

**Status/evidence:** Not supported by the descriptive v2 intervention on the fixed C2 checkpoints: prediction-change relative RMS was 2.461e-05 matched and 2.467e-05 permuted. Internal attention updates were nonzero, but the final prediction changes were very small. No target or quality metrics were used.

**Interpretation limit:** This does not test optical utility because the direct optical path remained active, and bypassing trained modules is an out-of-training-distribution intervention. It does not establish that cross-attention is generally unnecessary, that optical is ignored, or that the model is mismatch-robust. The repeatedly used development cohort is not for model selection or promotion. Keep B1 `not_promoted` and B2/B3 gated. See [D-2026-10-27-002](DECISION_LOG.md#d-2026-10-27-002--record-interpretation-only-cross-attention-bypass-diagnostic).

## H12 — Direct optical fusion-input marginal sensitivity (2026-10-27)

**Hypothesis:** With both trained cross-attention updates active, zeroing the optical-HR feature and optical-context channel groups at the fusion-stem input changes fixed-checkpoint predictions; the change may or may not depend on whether the optical pairing is matched.

**Status/evidence:** Supported only as fixed-artifact sensitivity. The direct-channel bypass changed prediction relative RMS by 0.0002428 (0.02428%) for matched optical and 0.0002428 (0.02428%) for the frozen permutation; the folder-equal means differed by about 1.5e-8 in relative RMS. The SAR base was unchanged. The fusion features changed more than the final output. This intervention's output delta was roughly an order of magnitude larger than the separate attention-bypass delta, but the two interventions are not additive and cannot be interpreted as a unique causal decomposition.

**Interpretation limit:** The bypass is out of training distribution and uses the same repeatedly reused, one-ROI-season development cohort. No targets or quality metrics were used. This does not establish optical utility, image-quality benefit, mismatch robustness, or generalization, and cannot support model selection or promote B1. B2/B3 remain gated. See [D-2026-10-27-004](DECISION_LOG.md#d-2026-10-27-004--record-interpretation-only-direct-fusion-input-bypass).

**CUDA repeat:** The same frozen bindings were rerun on `cuda:0`. Folder-equal prediction-change relative RMS remained 0.00024281485 matched and 0.00024282937 permuted. The paired CPU/GPU per-sample maximum absolute difference was 6.49e-9 for this scalar. This is a same-cohort hardware consistency check, not a new seed or an independent validation. It does not change H12's sensitivity-only interpretation or the promotion gate; see [D-2026-10-27-014](DECISION_LOG.md#d-2026-10-27-014--record-the-cuda-repeat-as-interpretation-only-hardware-consistency-evidence).

## 2026-10-27 reverse-task evidence boundary

The SAG-CR and CGD-Net source review does not support or reject the project’s optical-utility hypotheses because their target modality is optical. No hypothesis status or promotion threshold changes. Their alignment/reliability mechanisms may inform a later candidate only if task-matched SAR-output evidence justifies it and the existing phase gates permit the work.

## H1 split metadata boundary — 2026-10-27 correction

The 1,800-pair assignment is disjoint across 15 official source-scene folder IDs per partition. `ROIs1970_fall` denotes the fall distribution and its random-sampling seed; the paper reports 78 globally distributed fall scenes. These source-folder facts do not provide per-scene coordinates or acquisition metadata and are not evidence for or against matched optical utility. H1 and B1 remain unchanged. Do not infer geographic separation or acquisition independence without a scene-to-coordinate/product crosswalk. See D-2026-10-27-027 and the [scene taxonomy audit](../../experiments/data_audits/sen1_2_scene_folder_taxonomy_20261027_v1/report.md).

## H1/H6 prior-art boundary — Sim2Real SAR restoration (2026-10-28)

De Paepe et al.'s supervised MOCEM method jointly addresses SAR despeckling and sidelobe reduction, but has neither optical input nor spatial SR. It therefore supplies no evidence for H1 optical utility, mismatch robustness, or an inference-time SAR-only fallback; it does narrow broad claims of being the first joint SAR-restoration method. Its effective reflectivity target and qualitative-only real-image sidelobe assessment further bound interpretation. No local hypothesis gate changed. See D-2026-10-28-040 and the [audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).

## 2026-10-30 文献审查影响

CoCoGen、PISD 与 CoCoS 为 PDE 约束生成提供方法先例或测度审查问题，但不构成本项目 SAR/optical 假设的直接证据。本轮不新增、不晋级也不否定 optical utility、错配抑制或局部拒绝假设；原有数据分割和 B1/B2/B3 门槛继续生效。

## 2026-10-30 文献审查影响：联合恢复与少步先例

直接 SAR JDSR、光学引导联合恢复、自然图像参考自适应扩散及通用少步求解均已有相关工作。本轮文献证据收窄“首次联合/通用门控/通用少步”候选，不构成 H1–H6 的实证检验；所有光学效用、生成范式和错配回退假设维持原状态，不晋级也不否定。


## 2026-10-30 假设状态核对

C1 v2 清单元数据核验不构成 H1–H6 的模型或数据观察；未新增、接受或拒绝任何研究假设。train/dev 来源组互斥不足以代替空间/采集独立性证明，详见[审计记录](../../experiments/data_audits/mmcanet_phase_c1_train_dev_metadata_boundary_20261030_v1/report.md)。

## 2026-11-05 条件流逆问题文献不改变本地假设状态

两篇新审查的条件流工作研究一般图像逆问题的观测算子条件化或条件样本覆盖；它们没有测试 SAR 去斑、SAR 联合超分、匹配 optical 的增量效用或错配回退，故不改变 H1–H6 的本地证据状态。C3-A 入口本轮在读取图像前因其他 CUDA 进程阻塞退出，没有新增模型观察或指标。实验仍绑定冻结 C1 v2 train/development，confirmation 继续封存；文献边界见[审计报告](../../experiments/literature_audits/flow_operator_posterior_support_20261105_v1/report.md)，资源状态见[GPU 预检 v6](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。


## 总纲 v1.6 假设映射 — 2026-10-30

Q1–Q6 已由正式总纲采纳为研究问题；本条只固定问题框架，不把任何假设标记为已支持/已拒绝。每条假设仍须关联预先声明的协议、可证伪结果和来源证据。

## 2026-11-06 题名校订与 C3-A 等待不更新假设

arXiv:2609.12953 的当前版本题名校订为 FIRM；这只更新书目事实，不改变 H1–H6。C3-A 等待器 v8 已启动但 GPU 门槛未满足，训练和 development 评价均未开始，故没有新增模型结果：H3 仍未检验，H4 仍缺经核验的观测似然，H1/H2 仍无新增配对光学证据。confirmation/test/optical 载荷未访问。见[条件流文献审计](../../experiments/literature_audits/flow_operator_posterior_support_20261105_v1/report.md)及[v8 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261106_v8/manifest.json)。


## 2026-11-07 文献更新后的假设状态

FlowSGS 与 FMPS 已作为通用 flow 后验方法先例登记。文献先例会收窄方法新颖性措辞，但不支持或否定本项目关于 SAR 域任务条件 flow、光学增益、观测相容性或失配回退的任何假设；这些假设状态仅由对应协议下的证据决定。


## 2026-11-07 文献边界补记：乘性噪声生成模型

ICMLA 2025 的对数正态乘性噪声 SDE 工作没有检验本项目的光学增量价值、SAR 联合超分、错配回退或真实 SAR 观测一致性。Gamma 与对数正态乘子的一次方差匹配合成对照只支持“分布不能仅凭均值/方差视为相同”，不更新任何 Q1–Q6 假设的证据状态。当前假设仍待本项目合格实验检验。
