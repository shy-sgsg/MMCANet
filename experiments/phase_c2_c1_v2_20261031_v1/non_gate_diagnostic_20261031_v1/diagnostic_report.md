# C2 v2 开发集补充诊断（非冻结门槛）

## 目的与限制

原 C2 v2 配对 screen 因必需的 median TBR 在少数开发样本上未定义，按冻结规则 fail-closed。此补充运行保留全部开发样本，仅把 `tbr_median_abs_error` 和 `abs_log_median_tbr_ratio_error` 标记为不可用；不加 epsilon、不删样本、不作晋级或资源决策。

协议：`mmcanet_phase_c2_screen_20261029_v2`。结果是单 seed、19 个开发来源组上的描述性观察；同一 development 已用于训练期 checkpoint 选择，不能作为独立泛化证据。

## 未定义中位数覆盖

| 条件 | median TBR 未定义的开发样本数 |
|---|---:|
| x2_light | 2 / 200 |
| x2_severe | 2 / 200 |
| x4_light | 1 / 200 |
| x4_severe | 1 / 200 |

## 开发集 PSNR 配对差值

正值表示候选方法优于对照；bootstrap 按来源组等权。

| 条件 | 对比 | 来源组平均增益（dB） | 95% 来源组 bootstrap 区间 | 正增益来源组 |
|---|---|---:|---:|---:|
| x2_light | optical_guided_matched_vs_sar_only | -0.0055 | [-0.0166, +0.0028] | 8 / 19 |
| x2_light | optical_guided_matched_vs_optical_guided_permuted | +0.0152 | [+0.0093, +0.0211] | 16 / 19 |
| x2_severe | optical_guided_matched_vs_sar_only | +0.0042 | [-0.0043, +0.0134] | 11 / 19 |
| x2_severe | optical_guided_matched_vs_optical_guided_permuted | +0.0143 | [+0.0078, +0.0210] | 16 / 19 |
| x4_light | optical_guided_matched_vs_sar_only | +0.0061 | [-0.0046, +0.0177] | 10 / 19 |
| x4_light | optical_guided_matched_vs_optical_guided_permuted | +0.0106 | [+0.0062, +0.0150] | 17 / 19 |
| x4_severe | optical_guided_matched_vs_sar_only | -0.0024 | [-0.0105, +0.0057] | 8 / 19 |
| x4_severe | optical_guided_matched_vs_optical_guided_permuted | +0.0099 | [+0.0044, +0.0152] | 16 / 19 |

## 解释边界

本次生成 3200 条逐图控制指标与 3200 条逐图配对对比。所有已定义指标均覆盖完整的 200 个 development 身份；median TBR 两项明确缺失，因而冻结 C2 v2 决策函数没有运行。
该诊断不改写原失败的 screen、不改变冻结协议、不构成 continue/stop/inconclusive 决策、不晋级 B1，也不支持独立泛化或辐射物理结论。
未定义 median TBR 之外的结构/回波护栏包括：ssim, epi_corr, abs_log_edge_strength_ratio_error, tbr_mean_abs_error, abs_log_mean_tbr_ratio_error, abs_log_bright_peak_ratio_error, abs_log_bright_integrated_energy_ratio_error, bright_region_mae。完整结果见 CSV。

## 访问边界

只读取 train/development 固定输入和八组 best checkpoint；confirmation/test 影像、测量和指标均未读取。
