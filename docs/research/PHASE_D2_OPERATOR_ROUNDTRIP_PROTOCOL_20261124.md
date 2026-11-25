# Phase D2 固定退化算子往返核验方案

**状态：执行前固定**  
**日期：2026-11-24**  
**目的：** 在选择 observation-aware 生成路线前，检验固定 C2 `x2_light` 的 SAR `target`、记录随机种子和 SAR `lr` 是否与清单描述的表示域退化链相容。

## 范围和输入

- 只逐样本读取 train 1,600/92 组与 development 200/19 组中的 SAR `target`、SAR `lr`，以及各自角色 `files.csv` 中的 `sample_id`、`source_group_id`、`patch_id`、`sample_seed` 与 SAR 文件哈希。
- 不读取 optical 数组；不读取 confirmation/test 角色、影像、清单成员或指标；不生成新的固定输入，不修改现有文件。
- 按 train/development manifest 中冻结的 C2 `x2_light` 设置执行：`target` 为 `[1,256,256]` 的归一化 noisy SAR pseudo-target；对它依次执行 129×129 中心频域截断、非负 clamp、双线性 ×2 下采样（`align_corners=false`）、`Gamma(shape=8, rate=8)` 表示域乘法；不做上界 clamp。每个样本使用自身清单里的 `sample_seed`。
- `MMCANet/src/degradation.py` 的 SHA-256 与角色 manifest 记录一致（`b7c0e920…e0cb7bd1`）。本地生成 wrapper 当前哈希不同于数据生成时记录的 `33af6723…7582cc3b`，因此本核验只主张对 manifest 设置和已哈希算子依赖的往返检查，不声称旧 wrapper 源码完整重放。

## 预声明判据

对每个样本计算 `max_abs_error = max(abs(lr_regenerated - lr_stored))`、MAE、RMSE 与 bitwise-equal 标记。由于本机 Torch/CUDA/GPU 与数据生成环境记录相同，预期全部样本 `max_abs_error ≤ 1e-7`。任何样本超过该阈值都记录身份与误差并判定为 `NOT_REPRODUCED`；不调阈值、不修改历史数据。

该结果只验证固定表示域处理链的可重复性。即使通过，也不证明真实传感器成像、物理辐射量或 clean-SAR 观测模型正确；它只支持在这个合成归一化表示域中研究明确的条件似然候选。

## 运行边界与资源

- 仅做逐样本 GPU 前向算子核验；不反向传播、不训练、不推理、不调用模型。
- 按用户明确指示，不以未知 CUDA 进程或 3 GiB 空闲显存作为启动拦截条件；记录启动前后显存和本进程峰值，若进程实际发生 CUDA OOM 则停止并保留错误。
- 输出写入新目录 `experiments/diagnostics/phase_d2_operator_roundtrip_20261124_v1/`；不覆盖历史结果。

