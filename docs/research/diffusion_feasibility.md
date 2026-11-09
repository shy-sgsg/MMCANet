# 本项目的扩散方法可行性

## 2026-11-09 已知图像分布下的多样本评价量检查

纯合成 `[0,1]` 图像域 Gamma 边缘构造显示，逐像素 CRPS 能区分均值塌缩和乘性噪声过分散，却不能识别每个像素边缘分布相同、空间依赖错误的共享随机因子候选；归一化整图能量分数和邻接变差分数在该已知案例中可识别该差异。二维双峰诊断同时表明，样本数会影响有限集合分数的波动。评分规则一手来源指出，能量分数虽具适当性，对部分相关结构失配的敏感性可能有限；变差分数对相关结构更敏感，但不是严格适当分数，故不能单独验证完整联合分布。结果只验证指标实现/解释的玩具已知案例，不是扩散或流模型效果、SAR 物理证据、样本校准或推荐 draw 数。C3-A v3 和所有 gate 不变。详见[评分规则来源核验](../../experiments/literature_audits/proper_scoring_rules_multisample_eval_20261109_v1/report.md)、[图像域指标诊断](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/report.md)、[二维分布诊断](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/report.md)和[样本数诊断](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/report.md)。

## 非生成式 SAR 联合恢复先例不更新扩散可行性 — 2026-11-07

Amieva 等人的全文确认直接 SAR-only 联合去斑/超分先例，但其 SAN/ESRT 路线为非生成式，不检验 diffusion/flow 的样本质量、数据项一致性或 SAR-only C3-A 效果。因此它只影响先例边界，不改变生成方法设计、C3-A gate 或训练状态。见 D-2026-11-07-055 与[全文审查](../../experiments/literature_audits/amieva_direct_fulltext_followup_20261107_v1/report.md)。

## IOP 直接 SAR 联合扩散先例线索待全文核实 — 2026-11-07

检索发现题为 *Frequency-enhanced diffusion prior network for joint SAR image despeckling and super-resolution* 的 IOP DOI `10.1088/1361-6501/ae8686`；当前 IOP/DOI 正文路径受限，ResearchGate 只显示摘要级二手信息。故暂把它列作高相关开放先例，不将二手摘要中的级联结构、数据集、指标或效果写成事实，也不据此改 C3 方法设计或 gate。见[来源审查报告](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## 多 draw 像素均值收益依赖输出相关性 — 2026-11-07

共同 Gamma 分量纯合成诊断保持每个 draw 的 `Gamma(L,1/L)` 边际不变，仅改变 draw 间相关系数。K=8 时，PSNR 增益从独立 draw 的约 9.03 dB 降至相关系数 0.5 时约 2.49 dB、0.9 时约 0.40 dB，完全相关时为零。该实验只验证预设数学构造，不代表训练模型 draw 相关性，也不证明样本均值的分布真实性。冻结 C3-A v3 evaluator 逐 draw 计算标量指标、报告像素标准差，不保存均值图或显式依赖统计；这是协议范围而非缺陷。未来若列出均值点估计、分布或依赖估计，须另行预登记并明确所需输出。见[CPU 诊断](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/report.md)与[协议/源码审查](C3A_MULTI_SAMPLE_DEPENDENCE_AUDIT_20261107.md)。

## SAR 两阶段潜扩散超分辨率已有全文先例 — 2026-11-07

Lee 与 Son 的 MONTI 2026 非归档作者稿先用超分网络恢复结构，再以盲点去噪和条件潜扩散生成 SAR 散斑。它收窄“SAR 潜扩散超分”及“结构恢复后生成散斑”的宽泛新颖性，但不是本项目的光学引导 JDSR 或 SAR 观测一致性证据。全文还显示，四个输出的像素均值可以比单个扩散样本获得更高 PSNR/SSIM；因此生成样本质量、分布真实性与均方误差点估计需分别报告。当前 C3-A 评价器已经按 draw 计算指标，不平均预测图；不改冻结协议。见[全文审查](../../experiments/literature_audits/monti_two_stage_sar_sr_20261107_v2/report.md)和[Gamma 均值 CPU 诊断](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/report.md)。

## SAR-to-optical latent flow 是反向输出任务先例 — 2026-11-07

SAR 条件 flow matching 已用于从 SAR latent 生成 optical latent，并以 ODE 输出 optical 图像。它说明通用 SAR 条件 flow 和笼统的结构保持说法不能作为独占贡献；但它不验证由 SAR 观测恢复 SAR 图像、联合去斑/超分、optical 对 SAR 输出的增量价值或观测一致性。C3-A 本地 SAR-only Flow 效果仍未知，训练暂停期间不据该文调整模型或 gate。详见[一手审查](../../experiments/literature_audits/sar_conditioned_flow_sar_to_optical_review_20261107_v1/report.md)。

## DiffusionSAR 已覆盖条件扩散 SAR 去斑，不等于本地 JDSR 证据 — 2026-11-07

2024 年 TGRS 的 DiffusionSAR 以带斑 SAR 观测为条件，采用 Gamma speckle 合成数据和 log-Yeo–Johnson 处理，并在多时相 SAR 上微调。它说明宽泛的“扩散 SAR 去斑”已有直接先例；但它不做 SR，推理不使用匹配 optical，也不验证本地 8-bit PNG/noisy pseudo-target 或 SR 观测一致性。论文真实数据的 single-look intensity/SLC 术语不一致，需避免将产品物理域当成已确定。C3-A 仍需按冻结协议回答本地任务条件流是否可行；当前训练暂停，本轮不移植其变换或改协议。见 D-2026-11-07-026 与[全文审查](../../experiments/literature_audits/diffusionsar_conditional_sar_despeckling_fulltext_20261107_v1/report.md)。

## 拆分身份元数据复核不改变扩散方法判断或阶段门槛 — 2026-11-07

C1 v2 的 source group 与文件身份元数据互斥支持既有 train/development 使用边界，但不提供生成模型质量、扩散/Flow 方法适配性或观测似然证据。C3-A 仍无真实训练结果，C4/C6 不因该数据审计提前启动；见 D-2026-11-07-003。

## 新增 SAR 回波域扩散后验与流轨迹优化先例 — 2026-11-07

[UAV SAR Imaging via Diffusion Posterior Sampling](https://arxiv.org/abs/2610.14516) 报告在 MIMO-OFDM SAR 回波算子和复数反射率变量上做扩散后验成像，并联合处理未知轨迹误差；当前可核验来源是 2026-10-10 的 arXiv v1 预印本，具体效果为作者仿真报告。[MS-Flow](https://proceedings.mlr.press/v306/denker26a.html) 是 ICML 2026 的通用流先验逆问题求解方法，以中间轨迹状态和多重射击优化降低对全轨迹反传的依赖。两者分别属于物理回波成像和通用潜变量流逆问题，不等于本地固定退化下的条件 SAR 去斑/超分模型。没有原始复数回波和定标信息时，不将 UAV 论文的似然直接移植至本地 8-bit PNG，也不以这些文献解锁 C4 或 C6。全文边界见[审查报告](../../experiments/literature_audits/sar_echo_and_flow_trajectory_prior_art_20261107_v1/report.md)。

## C3-A 补充采样结构诊断已冻结 v2；资源等待继续 — 2026-11-05 17:24（中国标准时间）

多次采样协议 v2（SHA-256 `8d886f26023f9d929265b3f70ad9f51314f94788b1e4a01f269af5c1f0a589fc`）除每身份四次固定噪声 draw 外，固定并复用来自 LR/noisy pseudo-target 的 edge mask，报告 EPI、边缘强度比及 mask manifest。这些指标仅反映开发集 pseudo-target 表示域中的结构相似性。AISAR C3-A 专项测试 18 项通过，加入共用 evaluator 回归后合计 44 项通过；等待会话 `77673` 最近观察到 4 个 CUDA 进程、1,186/4,096 MiB 空闲，尚无训练或完整开发集结果。原 C3-A v1 训练协议与选轮规则不变，confirmation/test/optical 继续关闭。见[协议 v2](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v2.json)与[等待记录](../../experiments/diagnostics/c3a_gpu_wait_20261105_v3/manifest.json)。

## C3-A 补充采样稳定性已预登记；资源等待继续 — 2026-11-05 16:46（中国标准时间）

针对原筛选每身份只采一个固定样本的限制，新增四次固定噪声 draw 的 development-only 补充评价，并冻结相同身份的 C2 SAR-only 对照、确定性设置和来源组 bootstrap。补充协议 SHA-256 为 `9c00accfc9cd046783323735509db9505a09a6351bcf55d93e5699ebf9970b77`。代码与 16 项合成/单元验证通过；训练和评价尚未开始，GPU 等待会话 `28461` 最近仍观测到 3 个 CUDA 计算进程、1,167/4,096 MiB 空闲。该筛选是受限的 SAR-only 方法探索，不表示正式 C3 晋级，也不改变 C2/B1 gate 或解锁 B2/B3。见[实验登记](EXPERIMENT_REGISTRY.md)与[等待记录](../../experiments/diagnostics/c3a_gpu_wait_20261105_v2/manifest.json)。
## C3-A SAR-only 条件流筛选已预登记；GPU 训练待资源 — 2026-11-05

依据用户已批准的方法探索任务，已冻结一个独立的 SAR-only conditional rectified flow 小筛选，直接沿用 C1 v2 `x2_light` train 1,600/92 组和 development 200/19 组，不重新切分。协议固定 5 轮、batch=4、单 seed、双线性 SAR 条件、`0.1 * N(0,I)` 输入中心化 source、velocity MSE 和 8 步 Heun；对照为同一开发输入上的既有 C2 SAR-only checkpoint。脚本只读取 SAR LR 和 noisy pseudo-target，不读 optical。

这不是正式 C3 晋级：C2 gate 仍未判定，B1 仍停在原门槛；本筛选不能回答 optical utility、SAR 观测一致性或错配安全回退。6 项定向测试、Python 编译和真实 train/development 单样本 CPU 计算通路均已通过，但不含任何完整训练指标。GPU 首次预检空闲 1.06/3.67 GiB，低于 3 GiB 门槛；后两次检测到 3 个其他 CUDA 计算进程。训练尚未启动，未创建 run 目录。待资源可用后运行[冻结入口](../../scripts/run_phase_c3a_sar_flow.py)，证据见[登记](EXPERIMENT_REGISTRY.md)、[显存预检](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v1/manifest.json)、[进程预检一](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v2/manifest.json)和[最终预检](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v3/manifest.json)。

输入中心化 flow source 已在自然图像超分辨率中出现，例如 [RFMSR](https://arxiv.org/abs/2607.12753) 将 source distribution 放在低分辨率 latent 周围并采用两阶段 flow 训练。此先例收窄通用机制的新颖性；它没有验证 SAR 去斑、光学引导、SAR 观测模型或本地 noisy pseudo-target。

## confirmation 复核不检验生成模型，也不解锁 C3 — 2026-11-05

本轮沿用 C1 v2 冻结划分复核既有确定性 MMCANet checkpoint，没有实现或运行 Flow/Diffusion，也没有测量生成质量、观测相容性或采样效率。matched optical 对 SAR-only 的结果未达到预设增益门槛，且执行记录存在载荷访问后的字段修订及技术恢复重跑。因此该结果不构成 C3 证据、不改变 C2 v2 原 gate，C3 仍关闭；B1 仍为 `not_promoted / stop_at_b1`，B2/B3 也继续关闭。后续是否进入生成模型研究须依据总纲阶段条件另行决定。见 D-2026-11-05-001 与[实验登记](EXPERIMENT_REGISTRY.md)。

## C2 v2 探索结果不解锁 C3 — 2026-10-31

冻结 C1 v2 train/development 输入上已完成 C2 八组 ×30 epoch 训练，但正式 screen 因 median TBR 未定义而未判定；非门槛观察尚未显示 matched optical 相对 SAR-only 的明确 PSNR 增益。该轮使用的 MMCANet 对照并非生成式模型，不提供 C3 的生成质量、观测相容性或效率证据。未创建或运行 Flow/Diffusion 模型；C3 仍等待 C2 有效效用证据及独立冻结的后续协议。详见 D-2026-10-31-007 与[实验登记](EXPERIMENT_REGISTRY.md)。

## C2 合成指标验收不启动生成模型路线 — 2026-10-30

本轮没有运行或比较生成模型。C1 固定影像、CUDA 和完整 C2 runner 前置验收仍未满足；C3 继续等待已通过的数据与基线阶段，不从代码合同测试推断生成式方法可行。

更新日期：2026-10-30。状态：暂缓，尚未实现。

## Flow Matching 后验采样已有多种通用方案，本地仍不具备证据 — 2026-10-30

D-Flow SGLD 在预训练流先验上采用似然引导或源空间后验采样；贝叶斯流匹配以先验—观测模拟样本训练摊销条件流。它们收窄了通用生成式后验采样的新颖性，但不验证 SAR 表示域似然或光学引导恢复。贝叶斯流匹配报告了集中后验及先验预测尾部观测下的近似误差；源空间方法需要匹配的似然／前向算子。当前项目的可追溯数据域和 C2 执行验收仍未满足，故不启动 Flow／Diffusion 实验。候选 v1.4 仍待批准。见 D-2026-10-30-007 和[审查报告](../../experiments/literature_audits/flow_posterior_sampling_frontier_20261030_v1/report.md)。

## OGSFlow README 先例不构成本地生成方法证据 — 2026-10-30

作者 README 描述了光学引导的两阶段结构估计与条件流生成，并称 RGF 用于两个阶段。这使通用的该类架构不能单独构成新颖性，但未证明其代码行为、协议或效果，也未验证其适用于本地 SAR 联合恢复。候选 v1.3 的审批仍待定；本地生成式路线继续暂缓，阶段门槛不变。见 D-2026-10-30-006 和[审查报告](../../experiments/literature_audits/ogsflow_readme_details_20261030_v1/report.md)。

## DiSpeckle 书目线索未达到方法审查证据门槛 — 2026-10-30

ORCID 中 Crossref 来源的记录仅支持题名与 DOI 元数据；出版方全文及作者稿未取得。当前不能核验其 SDE、似然、数据域或效果，也不能据此改变本地扩散路线判断。候选 v1.2 保持待批；项目模型实现仍暂缓。见 D-2026-10-30-005 和[审计报告](../../experiments/literature_audits/dispeckle_metadata_route_20261030_v1/report.md)。

## 自适应参考控制已有先例，但不是本地扩散结果 — 2026-10-30

Ada-RefSR 在单步自然图像参考超分辨率中使用可学习摘要 token 相关性门控，并报告无关参考、匹配比例、模糊及仿射实验。STMP-DiT 按超分辨率阶段组织文本与 DINO 先验，并报告语义图像描述失效案例。这些来源缩小了通用自适应条件引导的可主张新颖性，但未证明扩散方法适用于 SAR 输出 JDSR，也未建立 SAR 观测似然、雷达保真度或 光学/SAR 回退。仅凭这些文献不能使本地方法获得实验资格；按现有研究章程和数据门槛继续暂缓生成式实验。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。

## VFM 是条件图像逆问题先例；BFM 仅涉及效率 — 2026-10-30

ICML 2026 的 VFM 联合训练流映射和观测条件噪声适配器。完整论文在已知算子和加性高斯观测噪声条件下评估 ImageNet 逆问题，包括双三次 ×4 超分辨率。这是 C3/C4 文献中的条件逆问题方法参考，但其高斯似然与本地 C2 不匹配。[参考目标审计](REFERENCE_TARGET_AUDIT.md)将 B1 v2 目标确认为已含噪的归一化表示；冻结的 C2 使用同类目标，并规定合成中心 FFT/降采样/负值截断/Gamma 退化链。D-2026-10-30-003 已推导该伪目标的候选连续域 Gamma 似然，但完整 VFM 后验目标、零值支撑数值行为及 float32 观测语义仍未验证。该模型描述归一化伪目标分布，不是干净 SAR、定标辐射量或物理散斑。NeurIPS 2025 的 BFM 是通用 ImageNet 生成效率先例，只能用于后续成本比较。两项结果均不使本地实验获得资格；在批准的研究方向、数据/目标依据、光学效用资格和方法协议门槛满足前，继续暂缓生成式研究。见 D-2026-10-30-001/002/003 和[来源审查](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md)。

## GLCdiffcr adds optical-output conditional-consistency diffusion prior art — 2026-10-29

The publisher abstract describes SAR-guided diffusion for optical cloud removal with an optical pixel/SSIM consistency loss. It is adjacent conditional-diffusion evidence but does not constrain a SAR restoration observation model. Full text was inaccessible and data are on request. It does not change the deferred status or authorize a local method run. See D-2026-10-29-018 and the [audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).

## DMDiff is reverse-task multimodal conditional-diffusion prior art — 2026-10-29

DMDiff conditions diffusion on SAR and cloudy optical imagery to reconstruct cloud-free optical output. It provides direct precedent for SAR–optical conditional diffusion and learned gating of cloud-corrupted pixels, but does not test SAR-output restoration, JDSR, or geometric mismatch/fallback. The publisher-indexed text reports 250 sampling steps and three datasets, while leaving the real Luojia train/test ROI assignment unclear in the retrieved passage. The paper-linked public repository has only a README and no executable model code at the reviewed snapshot. It does not make a local generative route eligible or change its deferred status. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).

## CDD is adjacent conditional optical–SAR diffusion prior art — 2026-10-29

The official abstract describes conditional dual diffusion that generates optical images from SAR for multimodal clustering. This narrows generic cross-modal diffusion novelty, but offers no SAR-output restoration or JDSR evidence and no controlled mismatch/fallback result. The full paper could not be retrieved; code attribution is unverified. This does not make diffusion eligible locally or change its deferred status. See D-2026-10-29-016 and the [audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).

## SDIF is optical–SAR fusion prior art, not SAR-output restoration — 2026-10-29

Zhang et al.'s [SDIF](https://www.nature.com/articles/s41598-024-78778-5) uses SAR-BM3D preprocessing, variational diffusion fusion, and FLCNet semantic guidance to produce fused images for fusion/classification evaluation. It is relevant prior art for optical–SAR diffusion fusion, but does not evaluate a SAR-output restoration, spatial SR, a matched SAR-only JDSR control, or controlled optical mismatch. Its reverse-sampling semantic loss uses labels whose inference-time availability is not clarified. This review narrows broad fusion novelty only; it does not make diffusion eligible locally or change the deferred status. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).

## SAR split-Gibbs diffusion is measurement-domain posterior prior art — 2026-10-29

Gao et al.'s full [arXiv v2 manuscript](https://arxiv.org/html/2512.02768) alternates a complex-domain likelihood Langevin step under a row-subsampled Fourier operator/additive Gaussian-noise model with a DDIM-style learned SAR prior over real/imaginary channels. The simulated study uses phase-randomized 32×32 MNIST intensities; the real-scene branch uses two Sentinel-1A SLC scenes in 256×256 patches. It reports 20 repeated evaluations, but gives conflicting real-scene sampling counts (192 in the text and 24 in the tables) and does not report patch counts, source products, train/test allocation, or overlap handling. Reported gains are author claims. A same-author ICASSP 2026 record has a different title and abstract results; the accessible sources do not establish version identity. This is direct prior art for combining SAR measurement likelihoods with diffusion priors, not optical-guided restoration, image-space despeckling/SR, or validation of a local normalized-PNG likelihood. No local reproduction or gate change follows. See D-2026-10-29-014 and the [full-text audit](../../experiments/literature_audits/sar_split_gibbs_posterior_imaging_fulltext_20261029_v1/report.md).

## γ-Bridge adds SAR-only Gamma-diffusion prior art — 2026-10-29

Hu et al.'s arXiv preprint models multiplicative-Gamma intensity corruption with an exact look-indexed Gamma–Lévy bridge. It trains on natural images with synthetic single-look corruption, then reports zero-shot SAR despeckling on 33 PNG/JPG tiles using no-reference metrics. This narrows broad claims about Gamma-structured generative SAR despeckling, but it does not test optical guidance, spatial SR, or the local normalized-PNG likelihood. The reported 25-step reverse also under-shrinks relative to its five-step result, which the authors attribute to train–inference mismatch. Include it in any future C3/C4 SAR-only method comparison only if the preceding data and optical-utility gates are authorized. Its linked code URL returned 404 during this review; no code was run. See the [arXiv preprint](https://arxiv.org/abs/2607.22719) and [D-2026-10-29-008](DECISION_LOG.md#d-2026-10-29-008--add-gamma-bridge-as-sar-only-gamma-diffusion-prior-art).

## Flow and posterior guidance are candidate tools, not a local model decision — 2026-10-29

The primary-method review separates three choices: Flow Matching defines a training objective over probability paths ([Lipman et al.](https://arxiv.org/abs/2210.02747)); Rectified Flow studies straighter transport paths ([Liu et al.](https://arxiv.org/abs/2209.03003)); DPS and FlowDPS apply approximate posterior guidance to a learned diffusion/flow prior ([DPS](https://openreview.net/forum?id=OnD9zGAGT0k), [FlowDPS](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_FlowDPS__Flow-Driven_Posterior_Sampling_for_Inverse_Problems_ICCV_2025_paper.html)). Adaptive diffusion guidance has general posterior work at [AISTATS 2026](https://proceedings.mlr.press/v300/azangulov26a.html). The [MeanFlow](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html) average-velocity formulation and [Align Your Flow](https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/) flow-map distillation are few-step efficiency candidates, not reasons to begin training. None of these works validates the project's normalized-PNG pseudo-target likelihood, physical speckle model, SAR-output fidelity, or optical reliability/fallback. Flow matching remains deferred until the current optical-utility and data/target gates justify a SAR-output candidate. Faster sampling is an efficiency question after quality and consistency pass. See the [supporting audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md).

## MTDN is non-diffusion adjacent despeckling prior art — 2026-10-28

MTDN jointly uses SAR temporal-change detection and dual-polarization despeckling, without a diffusion model, optical inference input, or spatial SR. It adds no evidence about diffusion feasibility, sampling cost, or optical-conditioned generation. Diffusion remains deferred; see D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).

## DPS1-SAR dataset screen does not change diffusion feasibility — 2026-10-28

DPS1-SAR is a subscription-gated SAR-only despeckling dataset with temporal pseudo-references; its source grouping and reference-stack membership are unresolved. It adds no evidence about conditional diffusion, optical utility, or local sampling cost, so diffusion remains deferred. See D-2026-10-28-042 and the [source audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).

## Non-diffusion SAR-SR run-lineage note (2026-10-28)

The Muzeau et al. EUSAR study is non-diffusion SAR super-resolution with denoising-state supervision. Its author-repository history contains no versioned 100-epoch configuration corresponding to the paper's description, so paper-to-run lineage remains unresolved. This does not supply evidence about local diffusion feasibility or change the deferred status; see the [history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).

## MD-Glow is non-diffusion SAR multi-task prior art (2026-10-25)

MD-Glow uses an invertible Glow encoder/decoder and L1-trained task-specific feature translation; it does not use diffusion or provide evidence about diffusion feasibility. Its despeckling-centered metrics and qualitative SR example leave the local questions of optical-conditioned generation, data consistency, scatterer preservation, and sampling cost unanswered. Local diffusion remains deferred and B1/B2/B3 status is unchanged. See the [audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md).

## Adjacent InSAR regularization precedent (not diffusion; 2026-10-25)

The 2026 JSTARS [MuChaPro-based InSAR article](https://doi.org/10.1109/JSTARS.2026.3722046) uses spatial regularization for interferometric phase/coherence restoration, according to its abstract. It is relevant to structured inverse restoration but is not a diffusion study, optical-guided intensity-SAR SR, or validation of a forward operator for the local PNG representation. Full text and exact protocol remain inaccessible in this review; see the [audit](../../experiments/literature_audits/muchapro_insar_regularized_inversion_abstract_20261025_v1/report.md).

## MONTI 2026 indexed-text follow-up (2026-10-23)

The official workshop page confirms Lee and Son's paper is accepted at MONTI 2026 and that workshop papers are non-archival. Indexed text from the official OpenReview PDF reports IW-trained structure restoration followed by MM-BSN and conditional latent-diffusion speckle synthesis, with real EW inputs evaluated by separate structure/speckle FWD distributions and a synthetic paired PSNR/SSIM table. The stage-2 average improves PSNR/SSIM over stage 1, while individual samples score lower; the paper notes averaging can smooth away high-frequency speckle. This adds a concrete two-stage SAR-SR precedent but does not establish a reversed order or joint SAR despeckling/SR advantage. Direct PDF review, exact train/validation allocation, and code remain unavailable. See the [indexed-text audit](../../experiments/literature_audits/monti_two_stage_sar_sr_20261023_v1/report.md) and [official workshop page](https://sites.google.com/view/monti2026/home).

## Relevant prior work

### SAR-DDPM: full-text and code review (2026-10-25)

Perera et al.'s author manuscript reports conditional DDPM despeckling trained on synthetic single-look L=1 Gamma-speckled DSIFN imagery. Its cycle-spinning results are author-reported on 10 synthetic images; the real-image check is two Sentinel-1 scenes with four manual ENL regions and no clean SAR reference. Optical imagery is only synthetic training supervision, not an inference input. The task is despeckling rather than SR or joint restoration. The linked repository's mutable main branch does not expose a revision-to-table binding; static review found unresolved class-conditioning and SAR-conditioning/input-channel inconsistencies. Do not treat it as a verified reproduction or use it as evidence for optical-guided JDSR. See the [full-text/code audit](../../experiments/literature_audits/sar_ddpm_fulltext_code_audit_20261025_v1/report.md).

### ECDM: conditional diffusion for SAR despeckling (publisher-indexed text)

Guo et al.'s 2025 ECDM is SAR-only despeckling prior art. Its conditional DDPM predicts noise and a variance-modulation parameter, uses wavelet downsampling, and reduces inference from a 1,000-step training schedule to 50 steps. The reported MS-SAR pairs use optical scenes as synthetic clean targets and Gamma-speckled inputs at L=1/2/4; real fine-tuning pairs one Sentinel-1 GRD date with a ten-date SAR average, while evaluation uses no-reference ROI metrics on Sentinel-1 SLC imagery. Thus optical is used to make synthetic supervision, not as an inference guide. Neither synthetic optical reflectance nor a multi-date SAR average establishes clean SAR truth. The paper does not study SR or optical-guided joint restoration. These details come from publisher-indexed sections; direct HTML returned HTTP 429 and PDF/XML were not accessible. The bounded source review and unresolved split, date-membership, product-transfer, and code questions are recorded in the [audit](../../experiments/literature_audits/ecdm_sar_despeckling_fulltext_audit_20261024_v1/report.md).

- **SAR despeckling diffusion.** [SAR-DDPM](https://arxiv.org/abs/2206.04514) (Perera et al., IEEE GRSL 2023) uses conditional DDPM and cycle-spinning; its reported tests include synthetic and real SAR, with real-scene evidence based on no-reference region metrics rather than clean SAR truth. [R-DDPM](https://arxiv.org/html/2401.03122) is an arXiv v1 regional conditional-DDPM despeckler that averages overlapping regional noise estimates and uses DDIM. Its manuscript describes Sentinel-1 VV+VH data near Shanghai and synthetic observations formed by adding different levels of Gaussian noise; it gives no sample counts or explicit train/validation/test allocation. Real-image evaluation has no clean reference and uses ENL/EPI. The listed region parameters (`m=64`, `n=16`) conflict with the same section’s statement that `n` is a multiple of `m`. This is SAR-only despeckling, not SR or optical-guided joint restoration, and the synthetic noise statement does not establish a Gamma-speckle observation model. See the [full-text audit](../../experiments/literature_audits/rddpm_fulltext_audit_20261026_v1/report.md).;  [DiffusionSAR](https://doi.org/10.1109/TGRS.2024.3419083) explicitly diffuses toward multiplicative Gamma noise and uses log-Yeo–Johnson preprocessing; its authors report synthetic and real SAR comparisons and multitemporal fine-tuning to bridge a synthetic-to-real gap. [Paul and Savakis (2025)](https://www.mdpi.com/1424-8220/25/7/2149) study shortened/non-uniform sampling and sample aggregation. They report that unmodified SAR-DDPM underperforms most evaluated methods on their synthetic benchmark and discuss hallucinated structures; their modifications improve the diffusion results. [ECDM](https://doi.org/10.3390/rs17172970) (Guo et al., 2025) combines variance prediction, wavelet downsampling, and accelerated sampling, and reports synthetic pretraining followed by real multitemporal fine-tuning. Yang et al.'s 2025 [GCOANet](https://www.sciencedirect.com/science/article/pii/S1569843225003243) publisher abstract reports optical-assisted SAR despeckling in which conditional diffusion generates cross-domain references using optical pixel correlation, followed by a multiscale blind-spot network. This is direct diffusion-based reference-generation prior art in optical-assisted despeckling; the official indexed abstract names non-local filtering, interval downsampling, and all-directional gradient loss, reports real and simulated experiments, and says data are available on request. A 2026-10-24 secondary patent-index lead (CN120450996A) describes SAR-plus-optical reference generation and use of the trained generator at prediction, but the official patent text and its identity with the journal article remain unverified; it does not resolve the article's inference inputs. The journal record remains abstract-level and leaves detailed data/degradation, geometry tests, metrics, code, and runtime unresolved. The 2026 [SMSDNet](https://ieeexplore.ieee.org/abstract/document/11495191) publisher abstract describes FFT/DCT multiscale features with a lightweight stochastic feature-space diffusion refinement for residual speckle; full protocol and code were not reviewed here. These are despeckling studies, not optical-guided joint restoration.
- **SAR super-resolution diffusion.** Chen, Zhang, and Xiong’s [2024 IEEE ICSP article](https://doi.org/10.1109/ICSP62129.2024.10846499) describes residual prediction and MSTAR/SARBake evaluation in its publisher abstract; the proceedings TOC verifies the author list and page 340. The full protocol remains unreviewed; see the [metadata audit](../../experiments/literature_audits/icsp2024_sar_diffusion_sr_metadata_20261028_v1/report.md). [SharpSAR](https://doi.org/10.7527/S1000-6893.2025.32804) (Wang et al., *Acta Aeronautica et Astronautica Sinica*, 2026) uses latent diffusion with wavelet and direction-aware structure modules and a speckle-plus-blur synthetic degradation. Its official publisher tables report PSNR below SwinIR on both listed datasets (`SD`: 21.68 vs 22.23 dB; `KOR`: 22.50 vs 22.62 dB), alongside best listed SSIM/LPIPS/FID. The architecture ablation shows a mixed metric response between W and W+DAE, while the table labels a `Real-SharpSAR` variant without enough accessible detail to identify that change. The abstract's 8.12% aggregate-improvement basis is unspecified on the accessible page. The result tables are visible, but the article body, exact dataset identities, split and degradation protocols, and code were not verified. This is a metric tradeoff in single-image SAR SR, not evidence for optical-conditioned joint restoration. A 2026 [Lee and Son manuscript](https://openreview.net/pdf?id=5iKl7IWJj1) proposes two-stage latent-diffusion SAR SR, separating structural reconstruction and speckle synthesis, with a Fréchet Wavelet Distance evaluation; its publication status and code availability were not confirmed in this review.
- **SAR-only 联合超分与去斑扩散。** Kuang 等人的 2026 IEEE JSTARS 论文 [Unsupervised SAR Image Super-Resolution With Despeckling via Region-Specific Diffusion Models](https://doi.org/10.1109/JSTARS.2026.3655710) 是直接相关的 SAR-only 联合任务先例。项目既有记录审阅过[ResearchGate 作者稿](https://www.researchgate.net/publication/399908263_Unsupervised_SAR_Image_Super-Resolution_with_Despeckling_via_Region-Specific_Diffusion_Models)，本轮未重新打开或核实其与 IEEE 正式版本的对应关系。按该作者稿，RSDM 用两个 Stable Diffusion v2.1 U-Net 分支，经 LoRA 分别适配均匀和非均匀 SAR 区域，并冻结其余权重；LSIR 插值 LoRA 参数与文本嵌入，并对潜噪声使用 Bézier 插值。作者报告在 SAR-Airport-1.0、SARBuD-1.0 和 FUSAR-Map 上做 ×4 重建，比较 30 幅 SAR 图像；指标为 NIQE/BRISQUE/NRQM 以及 ENL/CV/EPI/TS，并称没有 HR 真值。作者报告使用两幅代表图像适配，LoRA 200 步、rank 16、学习率 `2e-4`；将输入上采样到目标尺寸后以 30 步 DDIM 推理；单图推理时间 26.93 秒，模型加载 3.44 秒，设备为 RTX 4090。以上为作者稿报告，不是独立复现。无参考指标不能证明逐像素 HR 保真或排除无依据结构；受审稿件未建立受控退化核、场景组划分、物理定标或公开代码。该方法收窄 SAR-only 联合扩散新颖性，但未测试 optical 引导或与本项目固定输入 ×2 协议匹配的性能。IEEE 正式来源与作者稿版本对应关系仍开放，详见[来源复核](../../experiments/literature_audits/region_specific_sar_jdsr_source_recheck_20261107_v1/report.md)。

  These are author-reported results, not independently reproduced here. The no-reference perceptual and region statistics do not establish pixelwise high-resolution fidelity or rule out unsupported structure; that is a metric limitation, not a demonstrated failure. The text reviewed does not specify a controlled degradation kernel or scene-group split, and it does not establish full-resolution SAR physical units. No official code repository was verified. This paper narrows novelty for SAR-only joint diffusion but does not evaluate optical conditioning or provide a matching baseline for this project's fixed-input ×2 pseudo-target protocol.
- **Bibliographic-only SAR joint-diffusion lead; technical claims and reuse terms unverified.** Crossref confirms the 2026 *Measurement Science and Technology* record for [Frequency-enhanced diffusion prior network for joint SAR image despeckling and super-resolution](https://doi.org/10.1088/1361-6501/ae8686) and exposes IOP HTML/PDF resource links. Both publisher URLs currently redirect metadata-only HEAD requests to a bot-manager challenge; no article body is available through this review route. OpenAlex lists only the IOP PDF and no repository copy, but its CC BY-NC-ND tag is not verified against the article-specific IOP terms; Crossref's license URI points to IOP's general copyright/permissions page, so reuse terms are unresolved. An exact-title secondary ResearchGate listing exposes abstract-like method/data/results text, but those details remain unverified and are excluded from technical conclusions. The paper is not classified as established SAR-only prior art in the matrix. See D-2026-10-28-014/015/016 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).
- **A relevant non-diffusion counterpoint.** [ProSR](https://arxiv.org/abs/2609.02377), whose [author repository](https://github.com/KAIST-VICLab/ProSR) reports ECCV 2026 acceptance, encodes HR detail residuals as discrete tokens and uses semantic-prototype-guided MaskGIT for SAR ×4 SR. The authors motivate this against smooth predictions merging nearby scattering peaks. Their Umbra SLC protocol uses spectral-cropped LR input and a source-disjoint train/validation split; it reports no separate held-out Umbra SR test, and the repository says model code is unavailable. ProSR reports lower PSNR than UPSR but better perceptual/distribution scores, illustrating metric trade-offs rather than universal diffusion failure. Its task and representation do not match this project's optical-guided joint restoration protocol. See the [full audit](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/report.md).
- **Adjacent multimodal diffusion.** Hu et al.'s 2025 [DB-CR](https://doi.org/10.1109/TGRS.2025.3604654) uses SAR and cloudy optical input to reconstruct cloud-free optical imagery. The author manuscript assumes aligned inputs and reports no controlled residual-shift test or SAR-only fallback, so it does not establish optical-conditioned SAR restoration. Its reported 122,218 SEN12MS-CR triplets conflict with split counts summing to 129,131; preserve this as an unresolved paper-internal discrepancy. See the [MERL manuscript](https://www.merl.com/publications/docs/TR2025-138.pdf) and [record](https://www.merl.com/publications/TR2025-138).
- HSFMamba and OGSRN already cover optical-guided SAR joint SR/denoising or training-time optical feedback. GCOANet adds conditional-diffusion reference generation in an optical-assisted despeckling method, based on the publisher abstract; the inference-time optical requirement is unclear. These methods do not establish that diffusion improves optical-guided joint SAR SR. No direct optical-guided joint SAR despeckling/SR diffusion method was identified in this bounded primary-source pass; DB-CR is adjacent multimodal diffusion with optical output.

Together, this literature establishes active diffusion work for SAR despeckling (including abstract-level diffusion reference generation in an optical-assisted method), single-modality SAR SR, and SAR-only joint SR/despeckling. It reports both efficiency improvements and fidelity/scattering concerns. No direct optical-guided joint SAR despeckling/SR diffusion method was identified in this bounded primary-source pass. This does not show that diffusion will solve the local task, nor does it justify ruling diffusion out.

### Posterior sampling and measurement-consistent SAR imaging

[Diffusion Posterior Sampling for SAR Despeckling](https://doi.org/10.1109/TGRS.2025.3541013) (Wang, Han, and Zhang, IEEE TGRS 2025) treats despeckling as an inverse problem. Its accessible abstract describes estimating a likelihood score from the reverse-transition distribution under a multiplicative-noise model, using a pretrained diffusion prior without paired clean/noisy training or an SAR-domain transform. The authors report synthetic and real SAR evaluations and robustness checks across speckle intensity, initialization, and algorithm parameters. The publisher full text and code were not available in this review; details beyond the abstract remain unverified.

[Synthetic Aperture Radar Deep Statistical Imaging Through Diffusion Generative Model Conditional Inference](https://doi.org/10.1109/TGRS.2024.3498442) (Wang et al., IEEE TGRS 2024) conditions diffusion generation on a maximum-a-posteriori formulation using the received SAR signal, with reported simulated and RADARSAT imaging experiments. [Diffusion-Driven Proximal Posterior Sampling for Synthetic Aperture Radar Imaging](https://doi.org/10.1109/ICASSP55912.2026.11460851) (Gao et al., ICASSP 2026) proposes proximal posterior sampling with a more explicit SAR measurement likelihood; its IEEE abstract reports over 5 dB PSNR improvement on simulated data at 2 dB SNR and a 56.25% sampling ratio. These are SAR image-formation/reconstruction problems from received or undersampled measurements, not restoration of an already formed 128×128 noisy raster to the project's 256×256 pseudo-target. The 2024 full protocol and code were not reviewed. For the 2026 IEEE conference paper, only the abstract was accessible; its full protocol and code were not reviewed. A separately titled arXiv manuscript by the same five authors was reviewed, but the available records do not establish version identity with the IEEE paper.

The distinction matters for “data consistency”: likelihood consistency with raw/undersampled SAR measurements is not the same operator as consistency with the project's synthetic image-space degradation. None of these studies establishes optical-conditioned SAR-output reconstruction, a matched SAR-only joint baseline, or safety against unsupported bright scatterers under this project's target protocol.

The official [IGARSS 2025 program](https://2025.ieeeigarss.org/view_paper.php?PaperNum=5431&SessionID=1136) lists Danwei Lu, Junjun Yin, and Jian Yang's *Physics-Inspired Diffusion Model for SAR Image Despeckling* as an oral presentation (TH1.M3.5). D-2026-10-19-29 explicitly records its inclusion in the original matrix at program-level evidence; the reconstructed row carries only the title, authors, session, and presentation time verified by the event page. The manuscript body, abstract, DOI, code, and technical details remain unavailable. Separately, Crossref confirms the citation metadata for DA-PhysDiff, but its official IEEE article text remains unavailable; that distinct 2026 item stays outside the technical evidence matrix. See the [metadata audit](../../experiments/literature_audits/da_physdiff_crossref_metadata_20261025_v1/report.md) and [source-boundary note](literature_review.md#posterior-sampling-measurement-consistency-and-consistency-models).

### Consistency models

[SAR-Conditioned Consistency Model for Effective Cloud Removal in Remote Sensing Images](https://doi.org/10.3390/rs17223721) (Han, Huang, and Li, *Remote Sensing* 2025) distills a SAR-conditioned score-based teacher into a consistency-model student for cloud-free optical prediction, then applies progressive multistep resampling. The paper uses SEN12MS-CR (169 ROIs) and 256×256 patches at stride 128; it reports up to 40× inference acceleration alongside roughly 80× the parameters of a standard DDPM. The article's accessible protocol did not clearly establish ROI-disjoint train/test assignment. It is a useful remote-sensing consistency-distillation example, but its output is optical imagery rather than restored SAR; the speed and quality claims are author-reported and were not independently reproduced.

Garber and Tirer's [CM4IR](https://openaccess.thecvf.com/content/CVPR2025/html/Garber_Zero-Shot_Image_Restoration_Using_Few-Step_Guidance_of_Consistency_Models_and_CVPR_2025_paper.html) (CVPR 2025) combines consistency-model priors with back-projection guidance and noise injection for zero-shot natural-image SR, deblurring, and inpainting, reporting operation with four neural-function evaluations. Its [author repository](https://github.com/tirer-lab/CM4IR) supports bicubic SR, Gaussian deblurring, and inpainting and documents a Python 3.8/PyTorch 1.9 environment. This is a general few-step restoration/measurement-guidance reference, not evidence of SAR-domain priors, coherent-scatterer preservation, or a SAR data-consistency guarantee.

Together, these additions broaden the diffusion taxonomy but do not supply a direct optical-guided SAR joint-restoration method. Posterior sampling, raw-measurement consistency, and consistency distillation remain candidate families only; the invalid target/source-group evidence and failed B1 gate still preclude a local model screen.

### Flow-matching work in the reverse SAR-to-EO direction

The 2026 arXiv preprints [ReFlowSET](https://arxiv.org/abs/2609.00968) and [ContraFM-S2O](https://arxiv.org/abs/2609.31378) extend conditional flow matching to SAR-to-EO generation. ReFlowSET uses a selected frozen latent codec, a from-scratch conditional DiT, and training-only alignment of noisy-EO features to clean-EO teacher features; its QXS-SAROPT experiment uses a separate 16,001/3,999 train/test split and 50 Euler function evaluations. ContraFM-S2O predicts average velocity with contrastive regularization for one-step EO generation and reports a QXS/SAR2Opt comparison. Both predict optical output from SAR input. They therefore do not establish optical guidance for SAR output, SAR-scatterer fidelity, despeckling, or SAR fallback. ReFlowSET mentions possible local SAR–EO misalignment as motivation, but the reviewed paper does not report a controlled shift sweep. These are adjacent generative priors, not local-task baselines; no local experiment is justified by them while the source-group and target-validity gates remain open.


### C-DiffSET: reverse SAR-to-EO latent diffusion

C-DiffSET (IEEE TCSVT 2026) is adjacent SAR-to-EO translation, not optical-conditioned SAR restoration. Its pixelwise variance channel weights the diffusion noise objective during training; the public inference path does not use that channel to gate SAR output or provide SAR-only fallback. The public QXS split samples individual files without source grouping, and the training entrypoint selects checkpoints by PSNR on its loader named test; linkage from these public scripts to published tables is unresolved. Do not transfer its confidence head as evidence of restoration utility or its split as a group map. The full review and source links are in [literature_review.md](literature_review.md#c-diffset-reverse-sar-to-eo-diffusion-and-confidence-caveat).

### CRDRS: residual diffusion in reverse SAR-to-optical translation

Zhang et al.'s 2026 CRDRS paper and [official code](https://github.com/Mizar29/Residual-Decomposition-Generation-Strategy-for-S2O) use a conditional GAN to produce coarse optical output, then diffuse an optical correction residual conditioned on SAR and the coarse pseudo-optical image. The public implementation constructs the final optical estimate as pseudo-optical minus the sampled residual. This is residual diffusion for SAR-to-optical generation, not a residual model for restored SAR. The paper's reported variance reduction and degraded-guidance robustness do not establish a benefit for the project's noisy SAR pseudo-target or an optical-guided SAR fallback.

The public trainer evaluates up to 64 examples from `TestDataset` and logs FID/PSNR/SSIM/LPIPS every 200 steps by default. The repository README says a split file contains train/validation/test mappings, but that file was not inspected; loader semantics and code-to-table lineage remain unknown. Do not allege leakage from this evidence. See the [full review](literature_review.md#crdrs-residual-diffusion-for-sar-to-optical-translation). No local data, splits, or experiments were accessed.

### DualCDM: reverse SAR-to-optical conditional diffusion

Ma et al.'s 2026 [DualCDM](https://doi.org/10.3390/s26134183) is a dual-domain diffusion model conditioned on SAR and trained to generate optical RGB/NIR targets. Its SFHRB uses timestep-modulated complex Fourier-domain convolutions alongside spatial convolutions; its SAR encoder uses adaptive frequency recalibration; frequency losses constrain predicted noise and a one-step optical estimate. It reports 25-step deterministic accelerated sampling, 135.14M parameters, and about 20.046 s per 500-sample-test image on an A40. Its experiment tables and paired statistical tests concern optical-output metrics, not SAR fidelity. The S1S2 dataset description reports Refined Lee filtering, 15 city image pairs, random patch-level train/test selection, and no verified source-group split; the separate large-image test is a disjoint region within one image pair. The linked code repository is visible, but its source revision/table lineage and both code/data reuse terms are unresolved. This narrows generic SAR-conditioned optical-generation novelty and does not establish optical-conditioned SAR restoration or local diffusion feasibility. See the [audit](../../experiments/literature_audits/dualcdm_sar_to_optical_audit_20261024_v1/report.md).

## Potential formulation

A narrow future test could define the SAR-only baseline output `x0 = G_SAR(z)` and model a residual distribution conditioned on SAR, then compare:

1. direct regression residual `r(z)`;
2. SAR-conditioned residual diffusion `p(r | z, x0)`;
3. SAR+optical-conditioned residual diffusion `p(r | z, o, x0)`.

This is a proposal only. The target residual must be meaningful in the stored representation and aligned protocol; no calibrated reflectivity interpretation follows automatically. The third arm is necessary to isolate whether optical input adds information beyond generative capacity.

## Main feasibility risks

- **Target validity:** Current HR target is a noisy pseudo-target. A diffusion model could reproduce its noise or learn the synthetic degradation pipeline rather than a clean SAR posterior. Noise2Noise's conditional-mean requirement is not established for the local shared-observation LR/target pair; a generative objective does not remove that target-identification problem. See [D-2026-10-25-177](DECISION_LOG.md#d-2026-10-25-177--bound-noisy-pseudo-target-interpretation-by-self-supervision-assumptions).
- **Hallucinated scatterers:** Plausible high-frequency output can be unsupported. Bright-scatterer preservation, SAR observation consistency, local residual analysis, and multiple-sample variability are mandatory.
- **Metric mismatch:** PSNR/SSIM may reward a posterior mean, while one stochastic sample can lose fidelity. Report deterministic expected output policy, uncertainty, and full metric tradeoffs.
- **Pairing leakage:** Optical conditioning can improve target similarity through spatial/semantic correlation without proving physical SAR scattering fidelity. Matched, permuted, missing, and SAR-only controls are required.
- **Compute:** Iterative sampling is costly, and the current shell has no usable CUDA device. The 2k development subset is the intended scale for exploratory screens, though it is limited support for claims about a rich residual distribution. No training should begin now because the required hypothesis, valid target, controls, and compute are not in place.
- **Novelty overlap:** Conditional diffusion exists for SAR despeckling and single-modality SAR SR; any claim must be precisely about the combined conditioning/task/robustness tested.

## Minimum decision gate before implementation

Do not implement until (a) a regression failure that diffusion could plausibly address is documented, (b) a valid training/reference target is defined, (c) direct-regression and SAR-only controls are frozen, (d) a small compute budget and sampler are specified, and (e) primary related work and any available reference code are reviewed. Then run a small validation-only feasibility screen with a fixed sampling rule, multiple seeds where feasible, and a stop criterion. Otherwise keep the route deferred.

## OGSFlow: optical-conditioned flow for SAR super-resolution (repository-level lead; 2026-10-24)

The public [OGSFlow README](https://github.com/AG-WDS/OGSFlow) describes two-stage UAV-SAR super-resolution with an optical-guided S1 estimate and a conditional velocity network in S2. It says the S2 conditioning includes upsampled SAR and registered optical and gives 10 Euler steps for its ×4 example. The README maps the stages and RGF to model filenames and advertises a CPU synthetic-data smoke command; the code remains inaccessible and the command was not run. This is a forward optical-to-SAR flow-matching lead, unlike the reviewed SAR-to-EO flow preprints, but the repository does not yet provide a paper citation. The README does not document joint despeckling/SR, SAR measurement consistency, mismatch robustness, or fallback. Linked protocol/code files were inaccessible to this review, so do not infer the paper's degradation or physical noise model. Keep this as prior-art context only; it does not justify a local diffusion screen while B1 and provenance gates remain closed. See the [audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md).

## OGSFlow source-version follow-up (2026-10-25)

A root-page snapshot crawled yesterday lists an `assets/paper` folder and four commits, but the commit SHA and file listing were unavailable; a separate README snapshot crawled five days ago says the public paper link and final citation are forthcoming. No commit binds the snapshots. No manuscript/code was retrieved or reviewed. This preserves OGSFlow as README-level forward-flow prior art and adds no diffusion-feasibility evidence. See the [versioned-source audit](../../experiments/literature_audits/ogsflow_author_repository_versioned_review_20261025_v1/report.md).

## SAR-conditioned flow matching for SAR-to-optical generation

Ryu, Oh, and Seo's 2026 IEEE GRSL early-access record (DOI [10.1109/LGRS.2026.3742531](https://doi.org/10.1109/LGRS.2026.3742531)) and same-title ICLR workshop manuscript describe SAR-conditioned flow matching with an optical output. The indexed workshop Table 1 favors the proposed method on SSIM, LPIPS, and FID but reports lower PSNR than its latent-diffusion row, contrary to its blanket all-metric prose. The IEEE full text and workshop/journal version relationship remain unverified. This is reverse-direction prior art, not evidence for SAR-output diffusion, optical utility, or SAR-only fallback. See the [source audit](../../experiments/literature_audits/sar_conditioned_flow_matching_s2o_20261024_v1/report.md).

## SpeckleDiff-2D: complex-valued SAR despeckling diffusion (abstract-level; 2026-10-25)

The IEEE abstract for [SpeckleDiff-2D](https://ieeexplore.ieee.org/document/10641516/) describes a complex-valued diffusion network whose forward noise accumulation is intended to mimic speckle generation. It says training uses synthetic speckled images generated from an optical dataset and that evaluation includes simulated and real SAR. The accessible sources do not identify the exact representation/noise law, optical source, targets, splits, metrics, code, or inference-time inputs. The abstract therefore adds SAR-despeckling diffusion and optical-derived synthetic-supervision prior art, but it does not show optical conditioning at inference or joint SAR SR/mismatch/fallback behavior. Keep local diffusion deferred; see the [audit](../../experiments/literature_audits/specklediff2d_abstract_screen_20261025_v1/report.md).



## SAR-only subaperture enhancement is not diffusion evidence (2026-10-25)

Amieva et al.'s CVPRW 2026 subaperture method is discriminative SAR-only enhancement, not diffusion. Its reported noisy-reference trade-off helps narrow the restoration landscape but supplies no reason to implement or train a local diffusion model. The existing data/protocol/compute gate and B1 status remain unchanged. See the [audit](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/report.md).


## Dong et al. subaperture SAR SR is not diffusion evidence (2026-10-25)

The abstract-level Dong et al. screen concerns complex-valued SAR SR with subaperture learning/fusion and provides no evidence about diffusion, posterior sampling, or generated SAR texture. It does not change the deferred status of local diffusion or the B1/B2/B3 gate. See the [audit](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/report.md).


## PolSAR/SinSAR diffusion super-resolution source lead remains unverified (2026-10-25)

A search surfaced a possible item under the title PolSAR Super-Resolution by Fusion of LR-PolSAR and HR-SinSAR Based on Diffusion Model, but its identity and technical content remain unverified. A single bounded direct GET to IEEE record 11414190 returned HTTP 202 without structured citation or abstract fields; the earlier web-reader attempt exposed only a challenge shell. Keep the lead outside the technical synthesis and feasibility assessment until an authoritative record or manuscript is readable. See the [source-route audit](../../experiments/literature_audits/polsar_hr_sinsar_diffusion_source_route_20261025_v1/report.md) and the [direct-record check](../../experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/report.md).


### SMSDNet: feature-space diffusion for SAR despeckling (abstract-level)

The 2026 IEEE GRSL [SMSDNet record](https://ieeexplore.ieee.org/abstract/document/11495191) describes stochastic iterative residual refinement in feature space after multiscale FFT/DCT processing. This adds adjacent SAR-despeckling prior art, not evidence for optical conditioning or joint SR. Since the article and PDF routes were inaccessible and no author manuscript was verified, its claimed gains, denoising fidelity, runtime, data protocol, and implementation remain unreviewed. Do not use this abstract to justify a local diffusion run; see the [full-text route audit](../../experiments/literature_audits/smsdnet_publisher_fulltext_route_20261026_v1/report.md).

### Sentinel-1 IW self-supervision route boundary (2026-10-26)

The asset-free review of a separate S1 IW SLC route does not change the diffusion decision. MERLIN-style complex self-supervision is not ready for this cohort because S1 TOPSAR/IW deramping and I/Q decorrelation remain unresolved; no local training route is authorized. Diffusion remains deferred. See D-2026-10-26-147 and the [SLC protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).

## 2026-10-27 adjacent cloud-removal literature boundary

The reviewed SAG-CR and CGD-Net sources are deterministic optical-output reconstruction methods; they provide no evidence for or against diffusion on SAR-output restoration. Diffusion feasibility and the existing B1/B2/B3 gate are unchanged.

## IGARSS 2025 physics-inspired SAR diffusion source boundary (2026-10-27)

The official program verifies an existence-level SAR despeckling diffusion paper, but its `View Manuscript` label exposes no manuscript link/body in the available reader. A candidate DOI surfaced only through secondary records and was not confirmed by IEEE/Crossref; the third-author rendering also differs across those secondary records. Do not use this lead to assert a method, result, DOI, optical conditioning, or joint SAR-SR contribution. Keep the existing program-level row and reopen only when a primary publisher record, manuscript, or versioned author code is accessible. See the [route recheck](../../experiments/literature_audits/igarss25_physinspired_sar_diffusion_route_recheck_20261027_v1/report.md).

**2026-10-28 direct joint-SAR lead recheck:** Crossref confirms the DOI and bibliographic record, and exposes IOP HTML/PDF links. Metadata-only requests to both links return a bot-manager challenge rather than article text; a bounded primary/author-route search found no manuscript or versioned code. Secondary method/results remain excluded. Do not add a technical matrix row until primary article content can be reviewed; see D-2026-10-28-011/014/015.

## Adjacent non-diffusion SAR restoration boundary — 2026-10-28

The Sim2Real paper uses supervised DRUNet/SEDRUNet models trained on MOCEM simulations; it is not diffusion prior art and does not change the project's diffusion-feasibility assessment. Its physical-reference and real-image sidelobe-evaluation limits are recorded in the [full-text audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md) and D-2026-10-28-040.

## 2026-10-29 H8 任务顺序文献更新

限定范围的 SAR 来源审查确认已有联合去斑/超分工作及替代监督目标，但没有证据据此选择扩散或其他模型家族。扩散路线仍待任务匹配的目标、来源证明和合格开发协议具备后再评估。见 D-2026-10-29-028。
## 2026-10-30 文献补充：物理约束与后验测度

CoCoGen 与 PISD 表明 PDE 反问题中可把离散方程、部分观测或物理更新融入 score/diffusion 采样；它们只支持 PDE 场景中的方法可行性。CoCoS 预印本进一步指出，精确硬约束下“满足方程”与“采到目标后验”可能不是同一件事。当前没有证据说明这些方法可直接作用于项目 8-bit SAR PNG、乘性散斑或 optical 引导。若未来进入 C4，须先核验表示域、推导观测似然/约束测度并验证算子；当前 B1 gate 未通过，不开展 B2/B3。

## 2026-10-30 补充：SAR 联合扩散与少步求解先例

已确认的光学引导 SAR JDSR 先例主要包含判别式/结构融合方法；SAR 生成式 JDSR 的一条 2026 DOI 线索尚待出版方正文核验。Ada-RefSR 是自然图像参考扩散 SR，Blockwise Flow Matching/MeanFlow 是通用生成，LLE 是一般扩散逆问题求解；它们都不能替代 SAR-only 基线或本项目的光学效用对照。少步方法只有在质量、结构、亮回波与观测一致性过关后才可评估。当前 B1 未晋级、C2 运行器验收不完整，不启动 C3 训练。


## 2026-10-30 C1 数据门槛状态

本轮未训练或比较生成模型。C1 v2 的精细重叠与来源组地理/采集独立性未验证，因此本轮不推进 C3，也不改变生成模型可行性判断。证据边界见[审计报告](../../experiments/data_audits/mmcanet_phase_c1_train_dev_metadata_boundary_20261030_v1/report.md)。


## 总纲 v1.6 采纳与 C3 阶段边界 — 2026-10-30

总纲已获批准，但 C3 仍在 C1/C2 之后；本轮没有生成或比较任何生成模型。当前 C2 固定输入绑定、完整 runner、可用影像和 CUDA 条件未满足，不能据文献或合成测试宣称生成路线通过。

## 2026-11-05 补充：逆问题条件流先例与本地适用边界

2026 年两项条件流工作分别研究已知线性算子和加性 Gaussian 噪声下的条件恢复，以及增广传输对条件样本欠离散的影响。它们支持把“测量一致性”和“条件样本覆盖”纳入生成模型审查，但不提供 SAR/JDSR 直接基线。当前本地输入是归一化 PNG 和 noisy pseudo-target，尚无可支持 Gaussian 似然或校准后验的观测模型；因此 C3-A 只做 SAR-only 方法可行性筛选，C4 仍需等待表示与算子核验。见[一手文献审查](../../experiments/literature_audits/flow_operator_posterior_support_20261105_v1/report.md)。

## 2026-11-06 状态更新：SAR-only Flow 等待 GPU

当前 C3-A 是 v1.6 路线内单独预登记的 SAR-only 可行性筛选；C2 gate 仍未判定，不据此进入 optical-conditioned C3-B。等待器 v8 已恢复，但截至 10:19:26 UTC 空闲显存为 1,453/4,096 MiB、其他 CUDA 进程 2 个，训练尚未启动。已修正的 FIRM v2 题名不改变本地模型比较方案；posterior spread 也不作为后验校准证据。资源变化、启动前门槛和访问边界见[v8 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261106_v8/manifest.json)。


## 2026-11-07：区分任务条件 flow 与无条件先验后验采样

FlowSGS 和 FMPS 扩展了通用 flow prior 的后验采样方法谱系；二者以预训练无条件先验和观测/条件步骤为中心，与本地 C3-A 的配对任务条件监督 flow 不是同一比较对象。它们没有提供 SAR 域可行性或光学收益证据，不构成扩大 Flow/Diffusion 预算的依据。C3-A 仍按冻结协议等待用户恢复训练；C4 仍依赖先行阶段和有效观测模型。

## 2026-11-07：log-Yeo–Johnson Gamma 高斯化纯合成诊断

按独立拟合/评估协议，对 `Gamma(L, scale=1/L)`、`L∈{1,2,4,8}` 比较 log-only 与仅由拟合样本确定 `lambda` 的 log-Yeo–Johnson。20 个留出评估结果的等权平均 KS 距离差为 `−0.040896`，四个 L 均在五组重复中下降。结果表明该变换在本次抽象 Gamma 模拟中降低了对标准正态的经验 KS 距离；它不验证真实 SAR 噪声或本地 8-bit PNG/noisy pseudo-target，也不构成训练、迁移或阶段晋级依据。详见[冻结协议与实验报告](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/report.md)。

## 2026-11-07：共享 log-Yeo–Johnson 参数对乘性观测的信号强度敏感

第二项纯合成实验固定 `L=8`，对 `Y=sN` 的五档 `s` 用等权训练混合拟合单一 `lambda`，并按信号强度独立校准和留出评估。五档的 KS 距离均下降，但幅度不同。这说明“Gamma 乘子经逐档拟合变换后更接近正态”不能直接转写为对混合信号观测的统一结论；也不验证真实 SAR 或本地 C3-A。见[共享参数实验报告](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/report.md)。


## 2026-11-07 补充：对数正态乘性噪声 SDE 先例

ICMLA 2025 的 *Perception-based multiplicative noise removal with Diffusion models* 为在乘性噪声下构造 SDE/score 去噪提供直接先例，也显示感知指标与逐像素指标可能取舍。它的对数正态乘子、CelebA/UC Merced 合成评测，与本项目 C2 的 Gamma 表示域退化和 SAR–optical 联合任务不同。纯合成 CPU 对照在均值、方差匹配后仍观察到不同高阶矩；此处仅用于避免错误迁移，不构成生成模型效果证据，不改变 C2/C3 阶段门槛。见[审查报告](../../experiments/literature_audits/multiplicative_noise_sde_fulltext_20261107_v1/report.md)和[分布对照](../../experiments/diagnostics/multiplicative_noise_distribution_cpu_20261107_v1/report.md)。


## ICSP 2024 SAR 扩散超分摘要级先例（2026-11-07）

IEEE Xplore 官方摘要称，`Super-Resolving SAR Images with Diffusion Models: A Dual Evaluation of Metrics and Applications` 使用残差预测的条件扩散进行 SAR 单图超分，在 MSTAR/SARBake 上报告传统指标与下游语义分割结果。官方 PDF 直连未能读取；划分、退化、训练预算和分割评估独立性仍未知。该文只支持单模态 SAR 超分先例，不支持联合去斑或 optical 效用。详见[来源审计](../../experiments/literature_audits/icsp_sar_diffusion_sr_abstract_pdf_route_20261107_v1/report.md)。

## 2026-11-09 补充：Flow 推理控制先例与金标准后验评价

ICML 2026 MPC-Flow 在通用图像逆问题中用模型预测控制引导 Flow 先验，包含超分任务；本轮仅能核验 PMLR 官方摘要，故只记录方法类别与任务范围，不引用未读正文的细节。它不证明 SAR 斑点似然、光学引导或 SAR 输出质量，不使本地 C3/C4 路线获得资格。仅当任务匹配的生成先验和观测模型分别通过现行门槛后，才可把训练后控制求解器列入后续比较。

ICLR 2026 的统计基准使用可精确采样的离散 Lévy 后验评测 DPS，并以覆盖率和最优性差距直接检查分布级质量。为核对本地候选评分量，本项目运行了六维稀疏高斯混合的 CPU 已知后验诊断；三类错误分布的 CRPS 均高于精确后验，精确后验的 50%/90% 覆盖分别为 0.4990/0.9014。该结果只支持该解析合成案例，不外推至 SAR、C3-A 模型或物理后验。详见[来源与范围审查](../../experiments/literature_audits/posterior_sampling_statistical_benchmark_20261109_v1/报告.md)及[CPU 报告](../../experiments/synthetic_cpu/posterior_score_calibration_cpu_20261109_v1r1/报告.md)。
