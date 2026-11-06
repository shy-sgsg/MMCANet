# 光学信息效用分析

## C3-A SAR-only flow 不新增 optical utility 证据 — 2026-11-05

当前预登记的 `x2_light` Flow Matching 筛选只读取 SAR LR 与 noisy pseudo-target，不将 optical 作为输入，也不估计 optical 特征贡献。因此它不改变 Q1/Q2：confirmation 描述性复核仍未观察到达到门槛的 matched-optical 增益，development 也已用于选模；C3-A 尚无训练或完整 development 评价结果。补充协议 v2 将量化 SAR-only Flow 的条件采样波动及 pseudo-target 固定边缘区域上的表示域指标，不使用 optical。后续 optical 方法仍须建立在任务匹配 SAR-only baseline 和新的开发集协议上。见 D-2026-11-05-012 与[实验登记](EXPERIMENT_REGISTRY.md)。

## C1 v2 confirmation 描述性复核：未观察到达到门槛的增益 — 2026-11-05

直接沿用冻结的 C1 v2 来源场景文件夹划分。对 200 个 confirmation 样本、20 个来源组按组等权汇总，matched optical 相对 SAR-only 的 PSNR 增益为：x2_light −0.0114 dB（95% bootstrap 区间 [−0.0192, −0.0052]）、x2_severe −0.0108（[−0.0290, +0.0015]）、x4_light +0.0048（[−0.0050, +0.0158]）、x4_severe −0.0098（[−0.0328, +0.0044]）。matched 相对跨来源组置换 optical 的增益依次为 +0.0035（[−0.0027, +0.0092]）、+0.0038（[−0.0074, +0.0139]）、+0.0047（[+0.0003, +0.0085]）、+0.0047（[−0.0014, +0.0098]）dB。四种条件均未达到 matched−SAR-only 的 +0.05 dB 且区间下界大于零门槛；预登记结果为 `inconclusive`。载荷访问后发生两次执行器字段映射失败，最终按同一冻结设置技术恢复重跑，因此本结果只作描述性复核，不改变 C2 v2 原 gate，也不证明 optical 无效或地理泛化。confirmation 已封存，不用于后续方法选择。见 D-2026-11-05-001 与[完整报告](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/report.md)。

## C1 v2 开发集观察：配对敏感性高于对 SAR-only 的增益 — 2026-10-31

沿用冻结 C1 v2 划分训练并在 200 个 development 样本上完成非门槛诊断。matched optical 相对 SAR-only 的来源组 PSNR 增益依序为 x2_light −0.00552 dB、x2_severe +0.00421 dB、x4_light +0.00610 dB、x4_severe −0.00240 dB，四个 95% bootstrap 区间均跨零且点估计低于 +0.05 dB 继续阈值。matched 相对跨组 permuted optical 的增益为 +0.00994 至 +0.01516 dB，四个区间下界均大于零，说明当前模型输出对正确配对有小幅响应；但这不能替代相对 SAR-only 的效用证据。正式 C2 v2 screen 因两个样本的 median TBR 未定义而失败关闭，gate 未判定。以上均为单 seed、同一选模 development 集上的描述性结果，Q1/Q2 仍开放。详见 D-2026-10-31-007 和[诊断报告](../../experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。

## C2 工程验收未产生光学效用观察 — 2026-10-30

本轮没有读取项目影像或运行 optical-guided/SAR-only 模型。固定输入来源绑定和指标代码测试不回答 Q1 的光学增量价值，也不改变历史 B1 gate 或光学负迁移假设。

更新日期：2026-10-30

## 自适应参考抑制已有先例，但不能证明 光学 对 SAR 有效 — 2026-10-30

Ada-RefSR 表明，自然图像扩散式参考超分辨率可以训练自适应相关性门控来调节参考特征，作者报告了无关参考、模糊和仿射变化实验。STMP-DiT 展示了另一类风险：不准确的 LLaVA 语义可能引导自然图像超分辨率生成虚构纹理。这些发现提示后续需同时检验信息效用和信息诱发的损害，但不能回答 H1 所问的 光学 引导 SAR 输出恢复效用，也不能验证 SAR 可靠性信号。本地假设和门控均未改变。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。

## Reliability-aware classification results do not establish SAR restoration utility — 2026-10-29

Vulli et al.'s 2026 SAR–optical gate adapts classification logits under synthetic optical degradation. This is adjacent evidence that modality reliability is an active SAR–optical fusion problem, but it provides no evidence for or against H1: it does not reconstruct SAR or compare optical-guided restoration against matched SAR-only JDSR. Its reported heavy-noise gate instability reinforces the need to measure failure and fallback, not infer reliability from gate weights. The article's source distribution is almost entirely one region-season, and its reported results remain author claims. See D-2026-10-29-021 and the [publisher-indexed source audit](../../experiments/literature_audits/reliability_aware_sar_optical_classification_20261029_v1/report.md).

## GLCdiffcr does not measure optical utility for SAR restoration — 2026-10-29

GLCdiffcr uses SAR to reconstruct cloud-free optical output. Its results provide no evidence for or against H1, which concerns whether optical input helps reconstruct SAR over a matched SAR-only baseline. It does not test geometric mismatch or SAR fallback. See D-2026-10-29-018 and the [abstract-level audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).

## DMDiff does not test optical utility for SAR output — 2026-10-29

DMDiff uses SAR to help reconstruct cloud-free optical imagery. Its cloud-gating and diffusion results provide no evidence for or against H1, which asks whether optical input improves SAR-output restoration over a matched SAR-only baseline. Cloud masking also does not test geometric misregistration or fallback. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).

## CDD does not measure optical utility for SAR-output restoration — 2026-10-29

CDD conditions optical generation on SAR and evaluates learned features for multimodal clustering. It does not measure whether optical improves SAR-output restoration over a matched SAR-only joint model, so it provides no evidence for or against H1 and does not test SAR fallback under mismatch. See D-2026-10-29-016 and the [abstract-level audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).

## SDIF fusion results do not measure optical utility for SAR output — 2026-10-29

Zhang et al.'s SDIF takes optical and SAR images into a diffusion-based fusion objective and reports fusion/classification results. Its output and metrics do not test optical's marginal value for SAR-output joint restoration against a matched SAR-only baseline, so they provide no evidence for or against H1. The article reports a semantic-guidance tradeoff between classification cross-entropy and PSNR for its fusion task; this cannot be transferred to local restoration utility. Keep H1 and all gates unchanged. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).

## New SAR-to-EO flow-matching papers do not test optical utility for SAR output — 2026-10-29

ContraFM-S2O and ReFlowSET condition optical/EO generation on SAR. They add no evidence for or against H1, which concerns optical input to SAR-output joint restoration against a matched SAR-only baseline. ReFlowSET's discussion of local misalignment is not a measured mismatch-tolerance result. B1 remains unchanged; B2/B3 remain gated. See D-2026-10-29-010 and the [audit](../../experiments/literature_audits/sar_to_eo_flow_matching_preprints_20261029_v1/report.md).

## γ-Bridge does not test optical marginal utility — 2026-10-29

The look-parametric Gamma–Lévy bridge is SAR-only despeckling, without optical conditioning or spatial SR. It narrows generative SAR-denoising novelty but adds no evidence for or against H1, which still requires a matched SAR-only joint-restoration comparison on eligible data. See D-2026-10-29-008 and the [supporting audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md).

## Proposed generative direction does not alter the optical-utility gate — 2026-10-29

Flow matching, diffusion, and posterior guidance are candidate formulations; they do not answer whether correctly paired optical improves SAR restoration. Prior art already includes optical-guided joint SAR SR/denoising and optical-assisted diffusion despeckling. Keep the primary comparison against a matched SAR-only JDSR model, with matched, wrong-pair, and missing-optical controls. Reliability/fallback tests follow only if valid-pair utility is established. The authenticated current cohort is an 8-bit sigma-nought-dB representation without a retained byte-to-dB mapping or raw calibrated values, so only representation-domain consistency is supportable here; see D-2026-10-29-024. The charter proposal does not change B1 or authorize B2/B3. See the [supporting audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md).

## CloudLULC reliability modulation is semantic, not restoration evidence — 2026-10-28

CloudLULC's author-described ORM reduces cloud/shadow/haze-affected optical feature responses while SAR supports target-date LULC classification. This is adjacent evidence that degraded-optical reliability is an active SAR–optical fusion problem. Its semantic output and cloud-contamination protocol do not establish restoration-time optical utility, alignment-aware reliability, protection of SAR scattering structure, or fallback to a SAR-only reconstruction. It adds no evidence for or against H1 and does not change B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-065 and the [audit](../../experiments/literature_audits/cloudlulc_cloud_degraded_semantic_fusion_20261028_v1/report.md).

## MTDN's SAR-derived change cue is not optical reliability evidence — 2026-10-28

MTDN jointly uses two SAR dates and a change-detection mask to limit the effect of changed areas on multi-temporal SAR despeckling. This offers a task-specific example of estimating whether auxiliary observations are suitable for aggregation, but the cue is derived from SAR temporal change, not SAR–optical consistency or optical quality. MTDN uses no optical inference input and does not perform spatial SR; it provides no evidence for or against H1 and no safe-fallback guarantee. Keep it as a conceptual boundary, not an optical-fusion baseline. See D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).

## Bibliography crosswalk adds registration prior art, not H1 evidence — 2026-10-28

Hughes and Schmitt study SAR–optical image matching; Shibata et al. study misalignment-aware cross-modal filtering; Bungert and Ehrhardt jointly estimate reconstruction and structural-side-information registration. These works narrow generic novelty claims but do not compare optical-guided and matched SAR-only joint SAR restoration. They add no evidence for or against H1 and do not change B1 `not_promoted`; B2/B3 remain gated. See D-2026-10-28-043 and the [audit](../../experiments/literature_audits/bibliography_matrix_crosswalk_20261028_v1/report.md).

## DPS1-SAR does not test optical marginal utility — 2026-10-28

DPS1-SAR provides SAR-only temporal references, not optical input or joint SR. Its inclusion in the related-work matrix adds no evidence for or against H1 and does not alter the B1 gate. See D-2026-10-28-042 and the [audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).

## Adjacent alignment/SR data do not establish optical utility for restoration — 2026-10-28

The matrix crosswalk records Capella SAR-only SR, 3MOS and SOMA-1M alignment datasets, and SAR400 as a conditional denoising-data candidate. None compares an optical-guided SAR restoration model with a matched SAR-only joint baseline or tests safe rejection of mismatched guidance. This adds no H1 evidence and does not change the B1 gate. See the [crosswalk audit](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).

## Direct fusion-input sensitivity under matched and permuted optical (2026-10-27)

A predeclared zero-training intervention on the three C2 EMA checkpoints zeroed the optical-HR feature and optical-context groups entering `fusion_stem`, while retaining both LR/HR cross-attention updates, SAR-HR/LR-up features, the reliability and residual heads, and the high-pass output path. It covered 200 fixed development identities, three seeds, and 15 recorded source folders. Folder-equal prediction delta relative RMS was 0.02428% matched and 0.02428% permuted; SAR-base delta was exactly zero. Fusion-feature delta relative RMS was about 36.5%, yet the final output change was small, consistent with downstream attenuation in these checkpoints. The direct-input bypass delta was roughly ten times the separately measured attention-bypass delta, but both are out-of-training-distribution interventions and their effects are not additive.

This result describes sensitivity of these fixed checkpoints only. No target or mask values or quality metrics were used; the repeatedly reused cohort cannot support selection or promotion. It neither establishes optical utility nor proves that the model ignores optical. The bound whole-file `verification.json` digest discrepancy remains unresolved despite the expected payload-hash checks. See the [predeclared protocol](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/protocol.json), [report](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/report.md), and D-2026-10-27-004.

## Lan et al. registration prior art does not establish restoration robustness (2026-10-25)

Lan et al. use Log-TV and hybrid edge filtering to build cross-modal keypoints and estimate optical–SAR transforms. Their manual-control-point RMSE has no stated independence from coordinate refinement, and the separate affine stress is synthetic. These measurements do not show when optical information helps a SAR restoration model, how much local misalignment exists, or whether a restoration fallback works. Keep the local optical-utility question and B1 gate unchanged; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/lan_multilevel_registration_fulltext_20261025_v1/report.md).

## MD-Glow is multi-task SAR restoration, not optical guidance (2026-10-25)

MD-Glow uses SEN1-2 optical images to synthesize speckled training inputs and targets; its model input at inference is SAR, with despeckling and SR task branches. Its ablation is relevant to how auxiliary SR/multiscale training may change SAR despeckling, but it does not compare optical against SAR-only input or test optical mismatch/fallback. It does not alter H1 or the B1 gate. See the [indexed-full-text audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md).

## InSAR spatial regularization does not test optical utility (2026-10-25)

The MuChaPro-based JSTARS article concerns phase/coherence estimation from interferometric SAR projections and uses spatial regularization; its abstract does not evaluate optical inputs or optical-vs-SAR-only utility. It therefore does not support or refute H1/H9 for the local optical-guided intensity-SAR task. See the [bounded audit](../../experiments/literature_audits/muchapro_insar_regularized_inversion_abstract_20261025_v1/report.md).

## Question and scope

Does the failed B1F seed0 feature-fusion checkpoint respond to the correct optical pairing, and does that response improve validation metrics over its SAR-only condition? This is a post-screen, no-training diagnostic over the full `fixed_val_dev2k` list. It is not a promotion test and not an estimate across training seeds.

## Protocol

- Run: `experiments/dev2k_formal/phase_b1f_pairing_diagnostic_20261016_v1/`.
- Model: B1F feature-fusion seed0 checkpoint selected from a failed inner-dev screen.
- Conditions: matched optical; deterministic fixed bijective derangement; missing optical; SAR-only.
- Data: all 200 development-validation identities; 800 per-image metric rows. No test imagery/metrics.
- No training, model selection, or threshold changes. Bootstrap: 2,000 paired resamples, seed 20261016.
- Target: normalized/noisy pseudo-target in the fixed protocol, not clean calibrated SAR.

## Aggregate paired results

| Condition | Mean PSNR | Mean SSIM | Mean EPI correlation | Mean edge strength ratio | Mean TBR mean absolute error | Mean bright peak ratio | Mean bright energy ratio | Mean bright-region MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched optical | 17.358802 | 0.174035 | 0.135504 | 0.132562 | 1.182855 | 0.935780 | 0.779215 | 0.220714 |
| Permuted optical | 17.297160 | 0.170359 | 0.127602 | 0.127263 | 1.208045 | 0.932457 | 0.773062 | 0.226868 |
| Missing optical | 17.311080 | 0.165907 | 0.128915 | 0.118167 | 1.205227 | 0.925681 | 0.765877 | 0.234051 |
| SAR-only | 17.325809 | 0.174666 | 0.134395 | 0.136054 | 1.164371 | 0.942874 | 0.782503 | 0.217426 |

| Paired comparison | Mean Δ | Bootstrap 95% CI | Direction / caution |
|---|---:|---:|---|
| Matched − SAR-only PSNR | +0.032993 dB | `[+0.026632, +0.039101]` | Better proxy-target PSNR; below B1's +0.05 dB mean gate, and checkpoint/gate are not comparable as a promotion run. |
| Matched − permuted PSNR | +0.061642 dB | `[+0.056248, +0.067337]` | Correct assignment matters to this checkpoint's PSNR. Does not establish that the information is physically SAR-valid. |
| Matched − SAR-only SSIM | −0.000630 | `[−0.001207, −0.000040]` | Lower. |
| Matched − SAR-only EPI correlation | +0.001109 | `[−0.000081, +0.002327]` | Small interval includes zero. |
| Matched − SAR-only edge-strength ratio | −0.003493 | `[−0.004455, −0.002546]` | Moves farther below the ideal ratio of 1. |
| Matched − SAR-only TBR mean absolute error | +0.018484 | `[+0.015698, +0.021338]` | Higher error is worse. |
| Matched − SAR-only bright peak ratio | −0.007094 | `[−0.008823, −0.005428]` | Mean ratio is below 1 in both conditions; lower is further from target. |
| Matched − SAR-only bright integrated-energy ratio | −0.003288 | `[−0.004132, −0.002490]` | Both ratios are below 1; decrease indicates further energy deficit. |
| Matched − SAR-only bright-region MAE | +0.003288 | Not summarized here | Higher error is worse. |

### Zero-optical condition versus the SAR-only base

The diagnostic's `missing` condition passes an all-zero optical image through the normal B1F model. The paired `sar_only` condition is the same model call's frozen decoder base output, which the runner verifies is identical across matched, permuted, and missing-optical conditions. This is therefore a direct check of whether the trained zero-optical path returns to its SAR-only base on these validation samples.

| Paired comparison: missing − SAR-only | Mean Δ | Bootstrap 95% CI | Direction |
|---|---:|---:|---|
| PSNR | −0.014729 dB | `[−0.021834, −0.007967]` | Worse. |
| SSIM | −0.008758 | `[−0.009464, −0.008082]` | Worse. |
| EPI correlation | −0.005481 | `[−0.006537, −0.004435]` | Worse. |
| Edge-strength ratio | −0.017887 | `[−0.018982, −0.016779]` | Farther below the ideal ratio of 1. |
| TBR mean absolute error | +0.040856 | `[+0.037413, +0.044329]` | Worse. |
| Bright peak ratio | −0.017192 | `[−0.019065, −0.015239]` | Farther below 1. |
| Bright integrated-energy ratio | −0.016626 | `[−0.017701, −0.015625]` | Farther below 1. |
| Bright-region MAE | +0.016624 | `[+0.015529, +0.017723]` | Worse. |

On this repeated development-validation set and this one selected checkpoint, the zero-optical condition is not a safe fallback to the SAR-only output: all eight saved metric comparisons move adversely, and the paired PSNR interval is below zero. This is conditional diagnostic evidence, not a multi-seed or independent confirmation result.

Metric meanings were checked against `Early-FusionCascadeJDSR/eval_metrics.py`. `edge_strength_ratio` and bright ratios are not “higher is always better”; closeness to 1 matters. TBR and bright-region absolute error should decrease. EPI correlation should increase, but its bootstrap interval includes zero.

## Per-image heterogeneity

- PSNR improved for 156/200 samples and fell for 44/200. Mean gain was `+0.032993 dB`; median `+0.028823`; standard deviation `0.045560`; 5th and 95th percentiles were `−0.037105` and `+0.109429`.
- SSIM improved for 80/200, EPI for 105/200, edge-strength ratio for 59/200, TBR mean absolute error worsened for 165/200, bright peak ratio decreased for 128/200, bright energy ratio decreased for 145/200, and bright-region MAE worsened for 145/200.
- Descriptive quartiles of SAR-only baseline PSNR had mean matched PSNR gains of `+0.0485`, `+0.0448`, `+0.0214`, and `+0.0173 dB` from lowest to highest quartile. This is a post-hoc sample description, not a validated rule for sample selection or a new promotion threshold.
- Mean output MAE vs matched output was 0.01085 for permuted, 0.00978 for missing, and 0.01226 for SAR-only. This confirms changed outputs; it does not identify which specific optical feature caused the change.

## Interpretation

The diagnostic supports two limited statements: (1) this checkpoint uses optical input in a pair-dependent way; (2) on this repeated development-validation set, matched optical raises mean PSNR relative to SAR-only and fixed-permutation conditions. It does **not** support stable multi-metric improvement: SSIM, edge strength, TBR error, and bright-scatterer ratios/MAE move adversely or are mixed. It does not change B1/B1R/B1F status, clear their gates, validate a reliability mechanism, or show robustness to geometric shift or stale optical.

Because both the checkpoint selection and development set were already used in research iteration, bootstrap intervals are conditional and may not be interpreted as independent confirmation. The target is a noisy pseudo-target, not clean physical truth. The response could reflect learned pair correlations, proxy-target structure, or other training effects; the run does not distinguish these explanations.

## B1F architecture and fallback audit (2026-10-18; source review only)

- `MMCANetV2/phase_b1r.py` normalizes SAR/optical feature vectors, scores a 3×3 neighborhood, then applies a softmax over its nine offsets. In B1F, the optical input is average-pooled from 256×256 to the 128×128 LR feature grid before matching, so the neighborhood covers only ±1 feature cell per location (±2 native optical pixels at ×2 scale); it is local feature matching, not a global shift/warp estimator. The weights always sum to one; the matcher has no explicit all-optical-rejected state or reliability scalar. This does not prevent the downstream adapter from learning a small correction, but the matcher alone is not a reject/fallback gate.
- `MMCANetV2/phase_b1f.py` concatenates SAR and locally matched optical features, predicts an unconstrained 60-channel `delta`, adds it to the decoder's early `F60`, then runs the frozen SwinIR decoder. The correction is at feature level; the implementation does not explicitly high-pass the correction or constrain it to output-space high frequencies.
- `scripts/train_phase_b1f.py` trains with final-output SmoothL1 only. The B1F config specifies paired fit augmentation, but no optical dropout or shift/mismatch augmentation. The source returns a SAR-only `base` alongside the fused prediction, but prediction always passes through the fusion path; the inference `forward` has no missing-optical bypass that selects `base`.
- The zero-initialized adapter establishes identity at initialization only. It does not guarantee that a trained zero-optical input returns to the SAR-only output; the saved diagnostic directly observes a deviation and worse metrics under that condition.

These source facts narrow what the B1F result can support. They do not identify the cause of its failed gate, demonstrate that an explicit high-frequency residual would work, or justify a new model screen while the existing data/target provenance and experiment gates remain unresolved. No new inference or training was performed for this audit; the only numerical fallback evidence above comes from the already-saved diagnostic. Its manifest records `test_set_used=false` and `test_set_accessed=false`.

## Follow-up status

The planned descriptive per-image and baseline-PSNR quartile analysis is complete; its distributions, sign counts, and limitations are recorded above and in `EXPERIMENT_REGISTRY.md`. Repeating that analysis on the same checkpoint and validation rows would add little information. The 2026-10-18 mask audit further limits “bright-target” interpretation: these rows measure a target-derived high-value region with ceiling-tie expansion, not a sparse physical-scatterer ROI. They cannot be turned into an inference-time selector because the regions are defined from the target.

The 2026-10-23 source-aligned B1 v2 C2 failure analysis is now consolidated in the [root-cause synthesis](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/report.md). It confirms the frozen PSNR/SSIM gate failure but does not isolate a cause; the same development cohort is not a valid place to retune or select another cue.

The matcher-entropy cue study below is now complete as a narrow Level-0 diagnostic. Any further inference-time selector or cue-isolation study needs a separate prospective question, a target-free inference rule, complete metric criteria, and an unchanged independent confirmation boundary. The failed aligned-optical gate and unresolved source/registration metadata remain active constraints; no new model or robustness claim follows from these descriptive results.

## B1F matcher-entropy utility diagnostic (2026-10-18)

The fixed target-free cue was `1 − mean(normalized entropy)` of the B1F nine-offset local matcher weights for each correctly paired SAR/optical image. Higher values mean the offset weights are more concentrated; they are not calibrated probabilities of restoration benefit. The run reused saved per-image outcomes from the B1F pairing diagnostic for 200 fixed `fixed_val_dev2k` validation identities. It performed inference to capture the cue, but no training, checkpoint selection, threshold fitting, metric recomputation, or test-set access. The exact protocol and row-level results are in [the run artifact](../../experiments/dev2k_formal/phase_b1f_matcher_entropy_utility_20261018_v1/).

| Oriented matched-optical utility | Spearman ρ | Descriptive image-bootstrap 95% interval |
|---|---:|---:|
| PSNR gain | +0.156 | [+0.018, +0.290] |
| SSIM gain | +0.108 | [−0.041, +0.250] |
| EPI correlation gain | +0.098 | [−0.044, +0.233] |
| Edge ratio proximity gain | +0.107 | [−0.035, +0.254] |
| TBR mean absolute error reduction | +0.016 | [−0.131, +0.165] |
| Bright peak ratio proximity gain | +0.212 | [+0.080, +0.336] |
| Bright energy ratio proximity gain | −0.094 | [−0.237, +0.056] |
| Bright-region MAE reduction | −0.094 | [−0.237, +0.052] |

The fixed interpretation rule required all eight associations to be positive; it failed. In particular, TBR association is near zero and the bright-energy and bright-region MAE associations are negative. This cue is not supported as a global optical-utility or rejection signal for this checkpoint and validation data. The result does not rule out every possible inference-time cue. Intervals resample images without known source-group identifiers and are descriptive, not population intervals. The checkpoint failed its original inner-dev gate, this development split has been repeatedly used, the target is a noisy PNG-derived pseudo-target, and several outcomes use target-derived masks. No threshold, model, B1 gate, or B2 boundary was changed.

## Data-integrity addendum — 2026-10-19

The later exact-overlap audit found shared target pixels between fixed train and validation in 580 unique geometries, covering all 200 validation identities. This pairing diagnostic is therefore conditional on a spatially contaminated development split and cannot establish independent generalization or support future selection/promotion. Its saved observations remain unchanged; the audit does not estimate metric impact. The separate B1F fit/inner-dev audit found 440 exact target-overlap geometries, affecting 159/160 inner-dev identities. See the [fixed-split overlap audit](sen1_2_train_val_spatial_overlap_audit.md).


## Working theory: optical channels and task-specific value (2026-10-19)

This section separates mechanisms that are often collapsed into “optical information.” It is a set of physical/statistical hypotheses, not evidence that any cue helps this dataset. The local inputs are normalized PNGs, the targets are noisy pseudo-targets, and source-group and residual-registration provenance remain unresolved.

### A shared scene does not make the modalities interchangeable

Let Z summarize shared scene properties such as object layout, boundary geometry, material, and land cover; X be the current SAR image quantity of interest; O be optical reflectance acquired at some time; and Y be the degraded SAR observation. Shared Z can make X and O statistically dependent, while sensor-specific geometry, incidence/look direction, surface roughness, dielectric properties, speckle, acquisition time, and scene change affect each observation differently. Therefore, dependence between O and X does not imply a pixelwise optical-to-SAR intensity map. An optical edge may locate a boundary without determining whether that boundary is bright, dark, layovered, or shadowed in SAR.

This distinction is consistent with QXS-SAROPT's construction: its paper describes manually selected cross-modal tie points and registers optical subregions to SAR before cropping pairs. That documents a deliberate registration procedure for that dataset; it does not establish the residual alignment or physical mapping of the local PNG subset. See the [primary QXS-SAROPT paper](https://arxiv.org/abs/2103.08259).

| Candidate optical channel | Plausible conditional contribution | Main failure mode | What a positive result would support |
|---|---|---|---|
| Geometry: boundaries, junctions, elongated layout, coarse shape | Spatial support for edges or object extent that are weak in a degraded SAR observation | Residual shift, different viewing geometry, SAR layover/shadow, or a real surface change | A structure-localization benefit under the tested alignment and scene conditions, not SAR amplitude prediction |
| Reflectance/texture: color, local contrast, fine optical texture | Correlation with material or land-cover classes may inform a prior where SAR evidence is ambiguous | Optical reflectance is not radar backscatter; texture can be sensor-specific, seasonal, or unsupported in SAR | A statistical proxy benefit for the tested target and domain, not a physical texture transfer law |
| Semantics/context: object or scene category | Context may regularize broad shapes or distinguish plausible structures | A category prior may overwrite current SAR evidence, especially for stale or changed scenes | Contextual prediction only under tested temporal and scene conditions; not local semantic safety |

These channels are conceptual. A learned feature tensor may mix them, so naming a feature “geometry” or “reliability” does not isolate or validate that mechanism.

### The task changes which information is useful

For SR, the SAR degradation removes or attenuates spatial frequencies. Optical geometry could help localize a boundary whose position is ambiguous in Y, but it cannot supply the missing SAR scattering amplitude without an empirically supported cross-modal relation. For despeckling, optical may help predict a latent reflectivity field only where that field is conditionally associated with optical structure or material. Under a multiplicative-speckle model with speckle independent of O given latent reflectivity, O does not reveal the particular speckle realization; it can at most inform the latent conditional estimate. Joint restoration must satisfy both goals, so a sharper edge is not enough if brightness, target/background contrast, or bright-scatterer energy moves away from SAR evidence.

A noisy pseudo-target T changes the estimand. Under the illustrative additive model T = X + ε and squared-error training, the conditional-mean predictor decomposes as E[T | Y,O] = E[X | Y,O] + E[ε | Y,O]. If the target error is not conditionally mean-zero given the model inputs, a lower risk for T can partly reflect prediction of that error rather than recovery of X. The local target-generation process is not established well enough to assume this additive model, and no clean X is available here. Thus a gain against T is proxy-target consistency; it does not by itself identify which part of O informed clean SAR X. The squared-error conditional-risk distinction and its assumptions are formalized in [research_hypotheses.md](research_hypotheses.md#interpretation-boundary-conditional-information-and-noisy-pseudo-targets).


### Shared pseudo-target and LR input: a one-mode Gaussian reference (2026-10-25)

**Code/configuration fact.** The [`run manifest`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/run_manifest.json) and its [`input binding`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/input_binding.json) bind train and validation inputs to archived manifests; the exact [`train`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/input_metadata/fixed_train/manifest.json) and [`validation`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/input_metadata/fixed_val/manifest.json) manifest hashes match the run record. Both configs use the same chain: `sar_norm_mode=log`, `log_scale=20`, FFT truncation with a 129-bin window (`±64`), 2× bilinear downsampling (`align_corners=False`), negative-value clamping, then a Gamma multiplier with `looks=2` in the representation domain; post-multiplier upper clamping is disabled. In [`prepare_fixed_degradation.py`](../../scripts/prepare_fixed_degradation.py), the HR target `T` is the output of `preprocess_sar_tensor(sar_raw, ...)`, and the LR builder receives that same tensor. [`degradation.py`](../../MMCANet/src/degradation.py) implements the configured filter/downsample/clamp/multiplier order. The LR and target therefore share the preprocessed SAR tensor by construction; the multiplier is a generated representation-domain perturbation. The PNG domain does not identify the native residual or establish physical speckle statistics. This review used code and small manifests only; no image, metric, or test payload was opened.

The following scalar model isolates one possible consequence; it is not a local-data model. Let mutually independent zero-mean Gaussian variables satisfy

\[
X\sim\mathcal N(0,S),\quad E\sim\mathcal N(0,R),\quad N\sim\mathcal N(0,Q),\quad U\sim\mathcal N(0,V),
\]

and define a noisy target, a degraded SAR observation, and an optical guide by

\[
T=X+E,\qquad Y=aT+N,\qquad O=cX+U.
\]

Here `a` is a scalar transfer for one spatial mode, `N` is an independent additive approximation to post-degradation perturbation, and `O` is a direct linear proxy for the latent `X`. Set

\[
D=a^2(S+R)+Q,\qquad W=c^2S+V-\frac{a^2c^2S^2}{D},
\]

with `D>0` and `W>0`. Gaussian conditioning gives

\[
\operatorname{Cov}(T,O\mid Y)=\frac{cSQ}{D},\qquad
\operatorname{Cov}(X,O\mid Y)=\frac{cS(a^2R+Q)}{D},\qquad
\operatorname{Var}(O\mid Y)=W.
\]

Thus the squared-error Bayes-risk reductions from adding `O` are

\[
\Delta R_T=R_T^*(Y)-R_T^*(Y,O)=\frac{(cSQ/D)^2}{W},\qquad
\Delta R_X=R_X^*(Y)-R_X^*(Y,O)=\frac{(cS(a^2R+Q)/D)^2}{W}.
\]

For non-degenerate `S>0`, `c≠0`, `R>0`, `a≠0`, and `Q>0`, this model yields `ΔR_T/ΔR_X = Q²/(a²R+Q)² < 1`: an optical gain in estimating clean `X` can be attenuated when the metric target contains residual `E` that also entered `Y`. If `R=0`, the two gains agree. If `a=0` and `Q>0`, the absolute gains agree for the unobserved mode, while target residual `E` remains irreducible. These limits are counterexamples to any blanket claim that noisy-target metrics always understate or always overstate optical utility.

The local operator is nonlinear in places, uses a multiplicative representation-domain Gamma perturbation, and includes 2-D filtering/downsampling and possible aliasing; the decomposition `T=X+E` and its covariance parameters are not identified. The equations therefore do not quantify local bias or predict a metric trend. They make the existing boundary more specific: proxy-target PSNR/SSIM cannot alone identify clean-SAR optical utility when the LR input is derived from the same noisy pseudo-target. No gate, experiment, or training decision changes.

### A multivariate counterexample to one-mode target-risk ordering (2026-10-25)

The one-mode result does not give a direction that survives spatial mixing. Let independent centered Gaussian vectors satisfy `X~N(0,S)`, `E~N(0,R)`, `N~N(0,Q)`, and `U~N(0,V)`, and define `T=X+E`, `Y=A T+N`, and `O=C X+U`. Assume the displayed inverses exist. Given `X` and `Y`, the conditional mean of the target is

\[
\mathbb E[T\mid X,Y]=B X+L Y,\qquad
L=R A^\top(ARA^\top+Q)^{-1},\qquad B=I-LA.
\]

Since `O` is conditionally independent of `T` given `(X,Y)`, adding `O` changes the conditional mean of `T` by `B` times its change to the conditional mean of `X`. Define

\[
M=\operatorname{Cov}\!\left(\mathbb E[X\mid Y,O]-\mathbb E[X\mid Y]\mid Y\right)\succeq0.
\]

For squared Euclidean error in this jointly Gaussian model, the optical risk reductions are therefore

\[
\Delta R_X=\operatorname{tr}(M),\qquad
\Delta R_T=\operatorname{tr}(BMB^\top).
\]

In a single scalar mode, `B=Q/(a²R+Q)`, recovering the attenuation result above. In multiple dimensions, `B` need not be a Euclidean contraction when the degradation operator and residual covariance do not share directions, so the two traces have no universal ordering.

An explicit two-dimensional example has `S=I₂`, `R=diag(100,1)`, `A=[1 1]`, and `Q=1`. Then

\[
B=\frac1{102}\begin{bmatrix}2&-100\\-1&101\end{bmatrix},\qquad
P=\operatorname{Cov}(X\mid Y)=I_2-\frac1{104}\begin{bmatrix}1&1\\1&1\end{bmatrix}.
\]

Take a scalar optical measurement `O=c^\top X+U` with `c=P^{-1}e_2` and independent `U~N(0,v)` for any `v>0`. Its incremental posterior-mean covariance is proportional to `e₂e₂ᵀ`, so

\[
\frac{\Delta R_T}{\Delta R_X}=\|Be_2\|_2^2=\frac{20201}{10404}\approx1.942>1.
\]

Thus even with an independent additive Gaussian target residual, a shared-input linear model can make optical gain against the pseudo-target larger than gain against the latent signal; the scalar case gives the opposite ordering. This is a mathematical counterexample, not evidence that the local B1 metrics overstate clean-SAR utility. The local pipeline is nonlinear, uses a representation-domain Gamma multiplier and clamping, and has unidentified residual covariance and signal domain. No local direction or correction factor follows; no experiment, gate, or training decision changes.

### Exact Bayes-risk decomposition for a noisy pseudo-target (2026-10-25)

The ordering question has a distribution-free identity that complements the Gaussian examples. Let `T=X+E` be square-integrable vectors in the same metric space, and compare optimal conditional-mean estimators from nested inputs `Y` and `(Y,O)`. Define

\[
\delta_X=\mathbb E[X\mid Y,O]-\mathbb E[X\mid Y],\qquad
\delta_E=\mathbb E[E\mid Y,O]-\mathbb E[E\mid Y].
\]

For any square-integrable `U`, the squared-error Bayes-risk reduction from adding `O` is `E||E[U|Y,O]-E[U|Y]||²`. Since the conditional-mean increment for `T` is `δ_X+δ_E`,

\[
\Delta R_T=\Delta R_X+\mathbb E\|\delta_E\|_2^2+2\mathbb E\langle\delta_X,\delta_E\rangle.
\]

The direction of the target-risk change relative to latent-signal risk is therefore controlled by how optical input changes the conditional mean of the target residual and by its alignment with the latent-signal increment. If `E[E|Y,O]=E[E|Y]` almost surely, the gains agree; independence of `E` from `(Y,O)` is one sufficient condition. Otherwise the cross-term can be positive or negative, so neither risk reduction universally bounds the other. This identity does not assume Gaussian variables, but it concerns optimal population estimators; it does not convert finite-network PSNR changes into clean-SAR utility. In the local pipeline the LR input is derived from the same preprocessed noisy pseudo-target, and no residual-independence condition is established. The identity supplies no local correction factor or empirical direction and changes no experiment, gate, or training decision.

This boundary matches the primary self-supervision literature. Noise2Noise Eq. 6 states an indexed clean-target condition; in population notation, noisy-target and clean-target squared-loss estimators align when E[Z|X]=E[S|X]. Target noise with zero conditional mean given S and independent of X given S is a sufficient construction, but the population conditional mean need not equal a sample-specific S. SAR2SAR explicitly assumes independent noisy pairs before adapting the method to multiplicative log-speckle, temporal change, and residual temporal correlation. Multi-temporal MERLIN constructs a target component excluded from its inputs and applies SLC-specific decorrelation steps. The local fixed-input builder instead reuses one preprocessed SAR tensor as the basis for both input and target, so these papers do not establish the local latent-target relationship. See D-2026-10-25-178.

### Why degradation severity has no general monotonic rule

A deliberately idealized scalar Gaussian example makes the hidden assumptions explicit. Suppose the SAR-only posterior after observing Y is X | Y ~ Normal(m_y, v_y), and suppose O = X + η where η is independent Gaussian noise with variance v_o. The posterior variance after adding O is v_y v_o / (v_y + v_o), so its squared-error reduction is v_y² / (v_y + v_o); the conditional mutual information is one half log(1 + v_y / v_o). In this special direct-observation model, both increase as SAR uncertainty v_y increases.

A more general scalar, conditionally jointly Gaussian model shows what drives that result. For a fixed observed `Y=y`, let `v_y = Var(X | Y=y)`, `w_y = Var(O | Y=y)`, and `ρ_y = Corr(X,O | Y=y)`, with nonzero conditional variances and `|ρ_y| < 1`. The conditional posterior variance and the squared-error Bayes-risk reduction from observing `O=o` are

\[
\operatorname{Var}(X\mid Y=y,O=o)=v_y(1-\rho_y^2),\qquad
\Delta R(y)=v_y\rho_y^2.
\]

For the same scalar Gaussian distribution,

\[
I(X;O\mid Y=y)=-\tfrac12\log(1-\rho_y^2).
\]

Thus conditional mutual information describes the fractional posterior-variance reduction through `ρ_y`, while absolute squared-error gain also scales with the SAR-only uncertainty `v_y`. Across images or degradation conditions, both `v_y` and the residual conditional association `ρ_y` can change. A stronger SAR degradation may raise `v_y` but lower `|ρ_y|` if optical alignment or scene correspondence worsens, so the direct-observation monotonic result is only one special case. These identities assume an ideal conditional-mean estimator and a scalar conditionally Gaussian model; they are not measurements or predictions for the local PNG/pseudo-target data.

That monotone result does not transfer automatically to SAR/optical imagery. O is not a noisy direct measurement of X; its relationship to X can change with geometry, material, time, misregistration, and degradation. A more severe SAR degradation can increase the room for a valid optical cue while also increasing ambiguity, mismatch sensitivity, and model error. The project should therefore estimate a degradation-by-optical interaction rather than assume one direction. Conditional mutual information, Bayes risk against a pseudo-target, finite-network metrics, and SAR physical validity are distinct quantities.

### Frequency-resolved linear utility under random residual translation (2026-10-25)

The scalar analysis above does not show how uncertain geometry could affect different spatial frequencies. The following two-dimensional, single-frequency model makes that dependence explicit. It is a linear-estimation derivation, not a physical SAR observation model or a claim about the local PNG cohort.

For one spatial frequency `k`, let the latent SAR coefficient be `X_k`, the SAR observation be `Y_k = H_k X_k + N_k`, and a linearly related optical coefficient be `O_k = c_k X_k + U_k`. Assume zero-mean, mutually independent signal/noise terms, with powers `S_X`, `S_N`, and `S_U`. Let the optical image have an unobserved per-example global shift `δ`, independent of those terms, so the observed coefficient is `O'_k = exp(−i kᵀδ) O_k`. Write `φ_k = E[exp(−i kᵀδ)]`, `q_k = |φ_k|²`, and `S_Y = |H_k|²S_X + S_N`. The SAR-only linear residual variance is

\[
V_Y=S_X-\frac{|H_k|^2S_X^2}{S_Y}.
\]

Using the covariance formula for a linear least-squares estimator, the reduction in mean-square error from adding `O'_k` after `Y_k` is

\[
\Delta R_{\mathrm{LMMSE}}(k)=
\frac{q_k |c_k|^2 V_Y^2}
{|c_k|^2S_X+S_U-q_k|H_k|^2|c_k|^2S_X^2/S_Y}.
\]

For a zero-mean Gaussian shift with covariance `Σ_δ`, `q_k=exp(−kᵀΣ_δk)`, so the linear gain decreases with frequency in directions where displacement uncertainty is larger. More generally, set `A=|c_k|²V_Y²`, `D=|c_k|²S_X+S_U`, and `B=|H_k|²|c_k|²S_X²/S_Y`. Then `ΔR_LMMSE=qA/(D−qB)` and its derivative is `AD/(D−qB)^2>0` when the powers are nondegenerate. Within this model, less phase coherence means less LMMSE gain. This is consistent with geometry being most fragile at fine spatial scales, but the optical–SAR coefficient `c_k` and optical residual power `S_U` still determine whether any useful cross-modal relation exists.

A known global translation can be inverted exactly in the ideal no-cropping Fourier model; it does not destroy information. The attenuation above concerns a per-example shift that is unknown and marginalized by a frequency-wise linear estimator. An estimator that jointly infers the shift from the whole image may recover some of that utility. Local deformation mixes frequencies and is outside this derivation. The result therefore motivates a falsifiable interaction between shift uncertainty and spatial frequency; it does not predict the behavior of a learned fusion network, prove that high-frequency optical texture maps to SAR scattering, or explain any historical B1 result.

### SAR-noise severity under fixed optical reliability (2026-10-25)

The preceding model can also isolate SAR observation-noise severity while holding registration and optical quality fixed. For the same single-frequency model, define the SAR signal fraction

\[
r=\frac{|H_k|^2S_X}{|H_k|^2S_X+S_N}\in[0,1].
\]

The incremental LMMSE reduction becomes

\[
\Delta R_{\mathrm{LMMSE}}(k)=
\frac{q_k|c_k|^2S_X^2(1-r)^2}
{S_U+|c_k|^2S_X(1-q_kr)}.
\]

For a positive denominator, its derivative with respect to \(r\) is

\[
\frac{\partial\Delta R_{\mathrm{LMMSE}}}{\partial r}
=-\frac{q_k|c_k|^2S_X^2(1-r)
\left[2S_U+|c_k|^2S_X\left(2-q_k(1+r)\right)\right]}
{\left[S_U+|c_k|^2S_X(1-q_kr)\right]^2}\le 0,
\]

because \(0\le q_k,r\le1\). With \(H_k\ne0\), \(r\) decreases as additive SAR-noise power \(S_N\) increases, so this model predicts a nondecreasing absolute MSE benefit from optical input as SAR noise rises, saturating at

\[
\lim_{S_N\to\infty}\Delta R_{\mathrm{LMMSE}}(k)
=\frac{q_k|c_k|^2S_X^2}{S_U+|c_k|^2S_X}.
\]

If \(H_k=0\), the SAR observation contains no information about this coefficient and the predicted optical gain is already independent of \(S_N\). At the opposite endpoint, when \(H_k\ne0\), \(q_k=1\), \(S_U=0\), and \(S_N\to0\), the displayed ratio is degenerate but its continuous limit is zero, as expected when the SAR observation already determines \(X_k\).

This monotonic result requires fixed \(H_k,c_k,S_U,q_k\), a common linear latent coefficient, and additive independent noise. It is not a prediction that optical gains increase with a Gamma-look parameter or with SR scale: multiplicative speckle is signal-dependent, and changing scale changes sampling, aliasing, and the frequency transfer. In real optical/SAR data, correspondence, residual geometry, and cross-modal relation may also vary with degradation. Thus a measured nonmonotonic interaction would not contradict this restricted result. It remains an analytic reference for a future severity-by-optical experiment, not local evidence or a learned-network bound.

### Falsifiable implications for a future valid protocol

1. A geometry-only cue should primarily change boundary localization/edge measures; a change in SAR intensity, TBR, or bright-target energy needs separate evidence and cannot be inferred from edge alignment.
2. Raw texture or semantic conditioning should be rejected if it improves proxy PSNR while worsening predeclared SAR-domain structure or bright-scatterer guardrails.
3. A severity sweep must use common fixed LR observations and a matching SAR-only model at each severity. Report the observed interaction and uncertainty; do not fit a monotonic trend after seeing results.
4. Any usefulness or reliability score must be computed without the target at inference and tested against paired per-image optical-minus-SAR-only utility on a source-group-disjoint development set. A score correlated with alignment alone is not a utility estimate.
5. If authorized source and registration metadata become available, test whether measured per-example residual-shift uncertainty predicts frequency-stratified optical utility beyond a fixed SAR-only baseline. Separate known/corrected shifts from unknown residual shifts, predeclare frequency bands and all SAR structure/bright-region guardrails, and do not use the current contaminated development partitions.

These implications do not authorize another screen on the current contaminated partitions. First resolve source groups and target/registration provenance, then decide whether a prospective experiment can test one of them. No model, threshold, B1 gate, or B2 status changes follow from this analysis.

## WMoE-Net repository-level mismatch and quality-gating lead (2026-10-24)

The official publisher record now verifies a WMoE-Net article and abstract describing wavelet-directed alignment and condition/quality-gated expert fusion for oriented ship detection. The abstract reports MSOSD-1.0 and QXS-SAROPT-SHIP experiments and author-claimed gains under low illumination/clouds/fog. Full text, code-level implementation, numeric result tables, and a controlled offset protocol remain unavailable. Detection results do not establish target-free optical usefulness for SAR pixel restoration, structural fidelity, or a SAR-only fallback. The separate QXS-SAROPT-SHIP archive still has unresolved source lineage and rights. See the [updated source screen](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md).


## B1 v2 C2 gated-residual diagnostic (2026-10-24)

On the same 200 repeatedly used development-validation identities, the three frozen C2 residuals yield mean actual ΔPSNR `+0.001621 dB`, ΔSSIM `−0.000893`, and ΔEPI `+0.001311` relative to their matching SAR-only bases. Increasing the shared residual scale to the largest tested value, `α=1.2`, gives mean ΔPSNR `+0.001805 dB`, but the frozen strict-score optimum remains `α=0` for every seed. A target-assisted pixelwise oracle gives mean ΔPSNR `+0.027348 dB` and ΔSSIM `+0.004462`; this only shows that some residual pixels better match the noisy pseudo-target when target values are available. It neither defines an inference-time reliability signal nor establishes clean-SAR recovery. Overall classification is mixed/inconclusive, with per-seed labels D/D/A. Do not infer optical absence of utility or safe fallback from this screen; preserve the failed gate and provenance limits. See the [full diagnostic](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/report.md).

## C2 gate-activation rank diagnostic (2026-10-24)

A follow-up used the input-derived mean reliability-map activation as a target-free cue and per-image actual-minus-base strict-score difference as the target-derived outcome. Seed-wise Spearman rho values were +0.8675, +0.8590, and +0.8383; their mean was +0.8549, with a 15-source-folder cluster-bootstrap 95% interval [+0.7259,+0.9107]. This satisfies the predeclared positive-rank rule for the composite score on this cohort. The per-image strict score is PSNR + 5·SSIM − 2·|ln(edge_strength_ratio)| − |ln(tbr_mean_ratio)|; mean seed-wise rho was only +0.0551 for ΔPSNR and −0.0604 for ΔSSIM, with seed-specific direction changes. This supports a bounded association with this composite pseudo-target score, not all-metric utility or calibrated reliability. The aggregate C2 residual result and B1 gate do not change. Checkpoints used this repeatedly reused validation set for selection; the 15 folders are not verified parent-scene/acquisition units. Do not threshold this cue or retune the cohort. See the [protocol](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/protocol.md) and [report](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/report.md).

## Optical-derived synthetic training in SpeckleDiff-2D (literature boundary; 2026-10-25)

The IEEE abstract says SpeckleDiff-2D trains on synthetic speckled images generated from an optical dataset, but the reviewed source does not state whether optical imagery is present at inference. Treat this as synthetic-supervision prior art, not as evidence for same-scene optical utility, registration robustness, or optical-based fallback in SAR restoration. No local optical diagnostic or project hypothesis changes; see the [audit](../../experiments/literature_audits/specklediff2d_abstract_screen_20261025_v1/report.md).



## SAR-only Doppler-subaperture prior art (2026-10-25)

Amieva et al.'s distinct CVPRW study shows a SAR-only route to same-grid effective-azimuth enhancement and reports an ENL-versus-PSNR/SSIM trade-off against noisy full-aperture references. It reinforces the need for a task-matched SAR-only baseline and multimetric safeguards, but provides no evidence for or against optical benefit. The local H1 question remains open; the failed B1 screen and provenance gate are unchanged. See the [audit](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/report.md).


## Dong et al. subaperture SAR SR does not test optical utility (2026-10-25)

The accessible abstract describes complex-valued SAR SR from subaperture/subband representations but does not establish an optical input or optical-versus-SAR-only comparison. It therefore neither supports nor refutes H1's matched-baseline optical-utility question. Publisher full text and the exact protocol remain unavailable; no local model-selection or gate decision follows. See the [audit](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/report.md).


## MMCANetV2 fallback semantics at the model boundary (2026-10-25)

The current source has a deterministic absent-optical path: None becomes zero optical with false presence, the optical residual is masked out, and the output equals the sigmoid-bounded SAR base. This is distinct from the old B1F zero-optical diagnostic and does not show that misregistered/wrong optical is detected or safely rejected. The optional gate target is structural edge-direction agreement with the target, not direct residual utility; the optional alignment field has no direct offset-supervision term in the inspected trainer/loss path. No model execution or data was used. Keep H1 optical utility and H6 mismatch-safe fallback unproven; see [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness) and the [source audit](../../experiments/diagnostics/mmcanet_v2_architecture_source_audit_20261025_v1/report.md).


### Unit-mean Gamma multiplicative noise under a fixed linear cross-modal model (2026-10-25)

The additive-noise result above does not model intensity speckle. A separate scalar calculation makes the stronger assumptions needed for one multiplicative case explicit. Let nonnegative latent intensity `X` have mean `μ>0` and variance `σ²>0`. Let `N ~ Gamma(L, rate=L)`, independently of `X`, so `E[N]=1` and `Var(N)=1/L`; set the observed SAR intensity to `Y=XN`. Let a scalar optical observation be `O=cX+U`, where `U` is zero-mean noise independent of `X` and `N`, with variance `σ_U²`. Hold the distribution of `X`, `c`, and `σ_U²` fixed as `L` varies.

The second moments are

\[
\operatorname{Cov}(X,Y)=\sigma^2,\qquad
\operatorname{Var}(Y)=\sigma^2+\frac{\mu^2+\sigma^2}{L},\qquad
\operatorname{Cov}(Y,O)=\operatorname{Cov}(X,O)=c\sigma^2.
\]

For the best linear estimator, the reduction in mean-square error from adding `O` after `Y` is the squared partial covariance divided by the conditional optical variance:

\[
\Delta R_{\mathrm{LMMSE}}=
\frac{\operatorname{Cov}(X,O\mid Y)^2}{\operatorname{Var}(O\mid Y)}
=\frac{c^2\sigma^4 q(L)^2}{\sigma_U^2+c^2\sigma^2q(L)},\qquad
q(L)=1-\frac{\sigma^2}{\operatorname{Var}(Y)}
=\frac{\mu^2+\sigma^2}{L\sigma^2+\mu^2+\sigma^2}.
\]

For a nondegenerate optical observation with positive denominator, `dq/dL < 0`, and

\[
\frac{d\Delta R_{\mathrm{LMMSE}}}{dq}
=\frac{c^2\sigma^4 q\left(2\sigma_U^2+c^2\sigma^2q\right)}
{\left(\sigma_U^2+c^2\sigma^2q\right)^2}\ge 0.
\]

Thus, in this restricted linear model, reducing `L` increases the absolute linear-MSE benefit of optical input, with a finite limiting gain. This is a property of unit-mean multiplicative noise on a latent intensity under fixed cross-modal reliability; it is not a general theorem for SAR restoration.

The local project instead applies a Gamma multiplier to a post-log, resized, normalized 8-bit representation tensor, with a noisy pseudo-target. Its `degrade_speckle_looks` parameter is therefore not an identified physical ENL. Super-resolution also changes the sampling operator, while real registration and optical reliability may vary with severity. This derivation neither authorizes a local look sweep nor predicts finite-network PSNR/SSIM; it only supplies a falsifiable reference model if a future protocol can establish the physical intensity domain and independent source groups.


### Source-group direction in the frozen B1 v2 screen (2026-10-25)

The source-folder diagnostic averages the three fixed-seed group means without ranking folders. Mean ΔPSNR was positive in 10/15 folders; mean ΔSSIM was negative in all 15. The identity-weighted mean ΔPSNR was +0.001617 dB, below the frozen +0.05 dB requirement, and mean ΔSSIM was −0.000892. This is a representation-domain pseudo-target observation on one fall ROI-season. It does not show that optical information improves clean SAR or establish why SSIM declined. The frozen screen remains `not_promoted`; fine pixel co-registration is still unverified. See the [diagnostic report](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/report.md).

### Product-level geolocation metrics do not measure cross-sensor residuals (2026-10-25)

Official Sentinel-2 reports specify absolute geolocation against GCP/reference imagery and separate inter-band or multi-temporal S2 registration measures. The older report's initial GCP result was better than 12.36 m at 95% on 107 products; the later baseline-specific table reports under 11 m at 95.5% for baseline 02.04 and under 0.3 pixel for inter-band registration. These metrics are not S1-to-S2 tie-point residuals. Sentinel-2 GRI refinement was deployed operationally after the 2017 SEN1-2 source period, but the SEN1-2 record does not expose exact product IDs or processing baselines. The Sentinel-1 GRD specification describes terrain geocoding but does not establish a compatible pairwise error distribution. Therefore absolute product accuracy, S2-only registration, and nominal 10 m grid spacing must not be converted into a local optical shift estimate. Fine co-registration remains unverified; see the [official specification audit](../../experiments/literature_audits/sentinel12_official_geolocation_specs_20261025_v1/report.md).

### Independent control/check-point validation precedent (2026-10-25)

Liao et al. (2025) describe separating georeferenced control points used in block adjustment from manually identified, spatially distributed checkpoints used to report pointwise residuals. That is a stronger validation pattern than reporting matching scores or inlier counts alone. The article's four regions use GF-03 with GF-07/ZY-3, not SEN1-2; its reference-map accuracy and checkpoint annotation uncertainty remain unverified in this partial full-text review. This does not quantify the local optical shift or support a B2 setting. See the [primary-source audit](../../experiments/literature_audits/sar_optical_independent_tiepoint_validation_20261025_v1/report.md).
### Registration-error evidence from land-cover classification (2026-10-25)

Wu et al. compare pixel-, feature-, and decision-level SAR–optical fusion under synthetic cardinal translations for land-cover classification. The results depend on fusion level, reference modality, scene heterogeneity, and spatial resolution; they also show that multimodal fusion need not outperform optical-only classification. Their sub-pixel alignment figures are fitted GCP residuals, not held-out checkpoint accuracy. These observations motivate careful task-specific measurement, but they neither show how SAR restoration metrics respond nor justify a local shift threshold. See the [full-text audit](../../experiments/literature_audits/sar_optical_registration_error_sensitivity_fulltext_20261025_v1/report.md).
## Training-time optical supervision versus inference-time optical input (2026-10-25)

The publisher abstracts for OGRN and OGSRN describe optical-derived training guidance/feedback followed by SAR-only inference. These works establish that optical data have been used as training supervision for SAR SR, but do not establish that a paired real optical image improves the local inference-time joint restoration task. OGRN's data/protocol details remain unverified; OGSRN's accessible abstract does not identify dataset/version, target construction, degradation, split, metrics, or implementation. SciOpen labels the paper Open Access (CC BY-NC-ND 4.0), but the extracted page still exposes no PDF body/link; direct publisher routes were unavailable through the current reader. Keep the local H1 question and task-matched SAR-only joint baseline separate from these training-only routes. See [D-2026-10-25-181](DECISION_LOG.md#d-2026-10-25-181--classify-ogrn-as-training-time-optical-guidance), [D-2026-10-25-183](DECISION_LOG.md#d-2026-10-25-183--keep-ogsrn-at-the-publisher-abstract-boundary), the [OGRN audit](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/report.md), and the [OGSRN audit](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md).

## OGSRN author-group patent route (2026-10-26)

The CJA OGSRN author roster matches the inventors listed on Shenzhen University patent CN113421190B. The patent describes same-scene HR optical assistance during training, then feeds bicubic-upsampled/channel-expanded LR SAR into the optical branch during testing. This is a related patent embodiment, not verified proof of the article's exact implementation. It reinforces the distinction between training-time optical supervision and inference-time use of real paired optical imagery; it does not address joint despeckling/SR or misregistration fallback.

## Optical-guided despeckling safeguards are prior art (2026-10-26)

**Fact:** The 2015 Optical-Driven Nonlocal filter limits optical use with a homogeneous/heterogeneous classifier; the 2019 Guided patchwise nonlocal filter applies an SAR-domain test to reject risky predictors; Gaya et al. (IGARSS 2024) adds abstract-level self-supervised optical-assisted SAR despeckling. **Inference:** These records establish optical-use safeguards as prior art. **Open:** The reviewed sources do not report controlled global shifts/local warps or a joint SR output and do not validate geometric mismatch detection or fallback for the current joint-restoration task. See [D-2026-10-26-046](DECISION_LOG.md#d-2026-10-26-046--add-three-optical-guided-sar-despeckling-records-with-unresolved-historical-membership).

### S1 IW / Sentinel-2 geometry boundary (2026-10-26)

The separate SLC metadata screen's ≤24-hour pairing and Sentinel-2 cloud-cover cutoff are candidate filters, not evidence of optical correspondence. S1 IW SLC remains in slant-range/azimuth geometry with burst structure; product-specific geocoding and measured residual registration are prerequisites. No optical utility or mismatch result follows from the metadata screen. See D-2026-10-26-147 and the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).


## B1 v2 cross-attention bypass sensitivity (2026-10-27)

The predeclared diagnostic bypassed both trained optical cross-attention residual updates while preserving the direct optical feature/context path. On the fixed development-validation cohort, the equal-weight prediction-change relative RMS was 2.461e-05 for matched optical and 2.467e-05 for cross-folder-permuted optical. The internal LR/HR attention updates were nonzero, but the final prediction change was very small and nearly the same across pairing conditions.

This is an inference-time sensitivity observation for the fixed C2 checkpoints. It does not measure image quality, establish that optical is ignored or useful, or demonstrate safe handling of mismatches; bypassing trained components is an out-of-training-distribution intervention. The source-folder labels are not verified scene/acquisition groups, and the cohort cannot support model selection or promotion. See [D-2026-10-27-002](DECISION_LOG.md#d-2026-10-27-002--record-interpretation-only-cross-attention-bypass-diagnostic) and the [diagnostic report](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261027_v1/runs/attempt_001/report.md).

## Reverse-task fusion evidence boundary — 2026-10-27

SAG-CR and CGD-Net use SAR to aid optical cloud removal. Their alignment supervision and reliability gates are useful adjacent mechanism references, but they do not establish that optical geometry or texture improves SAR restoration. Any transfer hypothesis remains untested and cannot bypass the matched SAR-only baseline or B1 gate. See the [source audit](../../experiments/literature_audits/reverse_task_cloud_removal_alignment_review_20261027_v1/report.md).

## H12 CUDA hardware repeat — 2026-10-27

The frozen direct-fusion bypass was repeated on `cuda:0` with the same input and checkpoint bindings as the earlier CPU run. Folder-equal prediction-change relative RMS was 0.00024281485 for matched optical and 0.00024282937 for the frozen permutation; the corresponding fusion-feature relative RMS values were 0.3650986 and 0.3658403. SAR-base change remained zero. Paired CPU/GPU maximum absolute difference for prediction relative RMS was 6.49e-9, with a maximum source-folder-summary difference of 6.03e-10.

This supports only numerical consistency of this fixed-checkpoint sensitivity diagnostic across the two execution devices. The intervention remains out of training distribution and uses the repeatedly reused development cohort. It does not measure image quality, optical utility, mismatch rejection, or generalization and does not change the B1 stop or B2/B3 gates. The whole-file `verification.json` digest discrepancy remains unresolved; payload hashes passed. See [D-2026-10-27-014](DECISION_LOG.md#d-2026-10-27-014--record-the-cuda-repeat-as-interpretation-only-hardware-consistency-evidence) and the [CUDA report](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/report.md).

## Muzeau source and run-lineage boundary — 2026-10-28

Muzeau et al. provide SAR-only SR supervision evidence, not optical-input evidence. Their paper reports 100 SR epochs, but the author-linked repository's 14 reachable commits contain no versioned 100-epoch configuration; this leaves paper-to-run lineage unresolved and does not affect the local optical-information hypotheses. See [D-2026-10-28-038](DECISION_LOG.md#d-2026-10-28-038--keep-the-muzeau-paper-to-run-mapping-unresolved-after-repository-history-review) and the [history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).

## Sim2Real does not test optical utility — 2026-10-28

De Paepe et al.'s MOCEM-trained joint despeckling/sidelobe method uses SAR and acquisition metadata, with no optical input and no SR task. Its results therefore do not support or refute optical's marginal value for this project and do not resolve mismatch-safe fallback. See D-2026-10-28-040 and the [full-text audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).


## MONet: optical-derived synthetic supervision versus optical inference input — 2026-10-28

MONet uses grayscale UC Merced optical imagery as a clean reference to synthesize single-look noisy training pairs; the trained despeckler takes SAR alone at inference. This is evidence about one supervised data-construction strategy, not about the marginal value of a correctly paired optical input during SAR reconstruction. Its stated inability to simulate extremely heterogeneous dominant-scatterer/geometry cases also limits transfer to SAR structure preservation. The follow-up study compares synthetic, multitemporal, and hybrid training references, but only its institutional abstract and author README were accessible. This does not alter local H1 or B1. See D-2026-10-28-049 and the [audit](../../experiments/literature_audits/monet_optical_training_reference_20261028_v1/report.md).

## SAR task-order literature does not update optical-utility evidence — 2026-10-29

The bounded H8 search reviewed SAR-only joint-restoration and supervision-target papers. It adds no evidence that optical inputs improve SAR restoration, and does not change H1 or the B1/B2/B3 gates. See D-2026-10-29-028 and the [source review](../../experiments/literature_audits/sar_task_order_source_review_20261029_v1/report.md).

## 2026-10-30 文献边界记录

本轮核查的 CoCoGen、PISD 与 CoCoS 研究对象为 PDE 科学计算或其硬约束后验测度，没有检验匹配 optical 是否提升 SAR 联合去斑与超分辨率，也没有检验 optical 错配下的安全回退。因此它们不更新 optical utility 的正负证据，也不改变当前 B1 未晋级结论。

## 2026-10-30 文献边界更新

HSFMamba 提供了光学引导 SAR 联合超分/去噪的直接发表先例；OGSRN 与 OGSFlow 分别提供光学空间反馈训练和仓库层面的光学引导条件流超分先例。Ada-RefSR 的可靠性门控来自自然图像参考超分。它们都没有回答本项目匹配 optical 相对 SAR-only JDSR 是否有增益，也没有验证本项目错配回退，因此不更新本地 optical utility 的支持/反对证据。


## 2026-10-30 C1 数据状态与光学效用主张

本轮没有运行光学/SAR 模型或读取图像，因而对正确配对光学是否改善 SAR 恢复没有新增观察。C1 精细重叠与来源独立性仍未验证；不得据本次元数据核验支持或否定光学效用假设。见[审计报告](../../experiments/data_audits/mmcanet_phase_c1_train_dev_metadata_boundary_20261030_v1/report.md)。


## 总纲 v1.6 采纳状态 — 2026-10-30

总纲正式采纳不会改变 H1（匹配 optical 是否产生增量价值）的证据状态。历史 B1 未晋级，C1/C2 前置证据待完成；本次没有新增光学效用观察。

## 2026-11-05 文献边界：条件流逆问题未检验光学效用

新增的算子条件流与条件后验覆盖文献都没有比较 SAR 输出任务中的匹配 optical 与 SAR-only 输入，也没有研究错配 optical 下的安全回退。因此它们不增加或减少 H1 的本地证据；H1 仍由合格的本地配对实验回答。C3-A 当前只筛查 SAR-only flow，GPU 预检受其他 CUDA 任务阻塞，尚无新增图像质量结果。见[文献审查报告](../../experiments/literature_audits/flow_operator_posterior_support_20261105_v1/report.md)和[预检记录](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。

## 2026-11-06 光学效用假设状态

C3-A 仍只读取 SAR LR 与 noisy pseudo-target。v8 等待尚未开始训练，因此本轮没有新增 optical 效用或 SAR-only Flow 结果；H1/H2 不变，C2 原 gate 仍未判定，optical-conditioned C3-B 不启动。FIRM/Twist Flow 的文献更新均未检验 SAR–optical 增量效用或错配回退。数据和资源边界见[v8 等待 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261106_v8/manifest.json)。
