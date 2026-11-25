# SAR 观测约束扩散 Pilot 实施计划

> **执行要求：** 实施者逐任务执行本计划；各步骤使用 `- [ ]` 勾选。当前由主代理在隔离 worktree 内连续实施。

**目标：** 实现并执行一个 SAR 表示域观测约束扩散 Pilot，检验固定退化似然能否约束生成样本并改善相对 SwinIR 的探索性恢复结果。

**架构：** 复用仓库已有小型时序 U-Net 与固定 C1 v2 数据加载器，训练 HR noisy pseudo-target 的无条件 epsilon 先验；采样时用经哈希核验的频域截断/双线性下采样算子、正值 Gamma 似然和 clamp 零值单侧约束进行后验引导。所有 Pilot 结果限定为重复使用的 development 探索证据。

**技术栈：** Python、PyTorch、NumPy、AISAR 环境；不新增依赖。

**依据文档：** [`PHASE_D2_METHOD_SELECTION.md`](../../research/PHASE_D2_METHOD_SELECTION.md)

## 全局约束

- 仅可读取 C1 v2 `x2_light` train 1,600/92 组与 development 200/19 组的 SAR `lr`、`target`；不得读取 optical、confirmation/test 图像、清单成员或指标。
- `target` 为 `[0,1]` noisy pseudo-target；LR 是未上界裁剪的表示域值，可大于 1。
- `degradation.py` SHA-256 必须等于 `b7c0e9208a132792824bdde4296d7d3650587a144bf7ab4ec34bb161e0cb7bd1`。
- Pilot 训练固定 seed `20261124`、最多 5 epochs、batch 4、LR patch 64→HR patch 128、AdamW `lr=1e-4`/weight decay 0、EMA `0.995`。
- 先验为 1000-step epsilon DDPM，beta `1e-4` 到 `0.02`；采样为 20-step deterministic DDIM。
- 正值 Gamma 观测稳定下限固定 `1e-8`；零观测约束固定 `relu(z)^2`；正值项与零约束项分别对各自 mask 求均值，组合权重固定 1。
- 未知 CUDA 进程和 3 GiB 空闲显存不是启动门槛；记录资源，实际 CUDA OOM 时保留日志并停止该 run。
- 不改论文、不修改主工作区、不覆盖已有 checkpoint/结果、不删除历史产物；结果使用唯一新目录。

## 审查关注与对应测试

1. 零观测是否按 clamp 支持点处理：测试 `y=0,z≤0` 无零值违反，`y=0,z>0` 违反率为 1 且梯度指向更低 `z`。
2. 正 Gamma 项是否能把预测均值推向观测：已知 `a=y>0` 梯度为零；`a≠y` 梯度方向正确。
3. LR 大于 1 是否被错误裁剪：测试 `y=1.5` 仍保留原值且似然有限。
4. 退化是否保持批次、通道、空间和梯度契约：测试单图及批次输出尺寸、有限值与反向梯度。
5. 保护数据是否被静默纳入：数据加载测试只允许 train/development，并对文件集合、SHA 与 group 数不匹配 fail-closed。

---

### Task 1（任务 1）：观测似然与可微算子

**文件：**
- 新建：`experiments/phase_d2_observation_aware_20261124_v1/observation_model.py`
- 新建测试：`tests/test_phase_d2_observation_model.py`

**接口：**
- `apply_representation_operator(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`，返回未 clamp 的下采样值 `z` 与 clamp 后的 `a`；批次维内部逐样本调用固定算子。
- `observation_energy(prediction: torch.Tensor, observation: torch.Tensor) -> dict[str, torch.Tensor]`，返回 `positive_nll`、`zero_constraint`、`zero_violation_rate` 与 `total`。
- 内部 `_energy_from_intermediates(z, a, observation)` 只实现正值 Gamma 项与零支持 hinge，供小张量数学测试使用；正式入口必须由固定算子生成 `z/a`。

- [x] 写已知张量测试：正观测处 `a=y` 时 Gamma 项有限且对 a 的一阶梯度为 0；`a` 高/低于 y 时梯度符号相反。
- [x] 写零支持测试：零观测对应 `z≤0` 时约束为零；`z>0` 时违规率为 1、梯度非负。
- [x] 写 LR 大于 1、输入尺寸、有限性和梯度传播测试；运行测试确认按预期失败。
- [x] 用冻结 `degradation.py` 的 `return_steps=True` 生成 `z/a`；按固定 `eps=1e-8`、零 hinge 和权重 1 实现最小目标。
- [x] 运行定向测试和 `py_compile`，确认测试/编译通过；另记录到 `z<0,y>0` 时正值项对 `z` 梯度为零的源码缺陷。

### Task 2（任务 2）：无条件 HR 先验训练

**文件：**
- 新建：`experiments/phase_d2_observation_aware_20261124_v1/train_prior.py`
- 新建测试：`tests/test_phase_d2_prior_training.py`

**接口：**
- `prepare_training_batch(records, order, rng, lr_patch=64) -> tuple[torch.Tensor, torch.Tensor]`，返回归一化至 `[-1,1]` 的 HR crop 与零条件。
- `train(args) -> Path`，在不存在的唯一 run 目录内写配置、命令、环境、输入哈希、日志和 `last/best` checkpoint。

- [x] 测试 paired crop 正确对应 LR→HR 的 2×坐标及增强同步；检查 seed 固定与 batch shape。
- [x] 实现数据读取合同，只加载 SAR train/development；训练仅用 train；按 epsilon MSE 保存 best 与 last。
- [x] 写 CPU 小批次前向/损失测试，再运行相关 tests 与 `py_compile`。
- [x] 冻结 Pilot 配置及新 run 目录后，在 CUDA 上执行最多 5 epochs；不中断用户其他 GPU 任务，不以未知进程或 3 GiB 门槛阻止启动。
- [x] 保存实际 CUDA 状态、峰值 allocated/reserved、磁盘余量、训练日志与任何异常；实际未发生 OOM。

### Task 3（任务 3）：后验引导采样与统一评价

**文件：**
- 新建：`experiments/phase_d2_observation_aware_20261124_v1/sample_posterior.py`
- 新建测试：`tests/test_phase_d2_posterior_sampler.py`

**接口：**
- `ddim_posterior_sample(model, observation, alpha_bar, timesteps, seed, guidance_weight=1.0) -> torch.Tensor`。
- `evaluate_development(predictions, records, baseline_predictions, observation_metrics) -> tuple[dict, dict]`，输入按 sample ID 对齐的四 draw、SwinIR prediction 和观测项遥测，输出单次 draw、pixel-mean、来源组配对统计。

- [x] 用小尺寸确定性 toy model 测试采样输出 shape/range/finite、同 seed 可复现及似然梯度方向。
- [x] 在每个冻结 DDIM 步计算 denoised estimate、表示域 Gamma/零支持代价梯度与确定性 DDIM 更新；模型参数冻结，只对状态启用梯度。
- [x] 先运行 sampler 单元测试、D0/D2 定向 tests、`py_compile`。
- [x] 训练 Pilot 完成后在全部 200 张 development 上生成 4 个固定 seed draw，记录逐图、逐来源组 PSNR/SSIM/RMSE/EPI/边缘/亮值/背景、观测 NLL/零支持违规率、运行时间与峰值显存。
- [x] 与同一固定输入的 SwinIR 输出做配对比较；不以像素均值代替单次质量，不宣称独立泛化或 C3 晋级。

### Task 4（任务 4）：研究记录、决策与交付

**文件：**
- 新建：`docs/research/PHASE_D2_OBSERVATION_AWARE_PILOT_REPORT_20261124.md`
- 更新：`docs/research/CURRENT_STATE.md`、`MASTER_RESEARCH_PLAN.md`、`EXPERIMENT_REGISTRY.md`、`DECISION_LOG.md`、`NEXT_ACTIONS.md`、`RESEARCH_RISKS.md` 及相关文献/假设记录。

- [x] 对照固定 Pilot 协议、run manifest、checkpoint、逐图 CSV 与资源日志写 Research Decision Review。
- [x] 明确 `CONTINUE/PIVOT/STOP/BLOCKED`；阻止把 development 结果写成独立泛化或 clean-SAR 恢复证据。
- [x] 最后运行定向与完整项目测试，审查仅本目标文件、提交并推送隔离分支。

## 执行结果说明（2026-11-24）

本计划的 D2 observation-aware diffusion Pilot 在五轮先验训练和四 draw × 全 200 张 development 的后验评价后，按冻结协议停止当前设置。后来对实现的边界复核发现 `z<0,y>0` 时正值观测项经双重 clamp 对 `z` 梯度为零；该缺陷限制该次 guidance，但不是唯一失败原因，也不追溯修改旧运行或门槛。新代码、测试、checkpoint、逐图指标、资源与 Research Decision Review 已记录于 Phase D 研究材料。当前定向及项目测试以最新 `tests/` 目录结果为准。

## 自审

- 观测模型、先验训练、后验采样和决策记录分别有任务与验收步骤；没有把 EDSR baseline 实现混入本方案，EDSR 作为独立 D1 子项目记录。
- 审查关注中的零点、正值、LR>1、张量/梯度、数据边界分别由任务 1–3 的测试或数据加载合同覆盖。
- 任何模型收益、novelty 或阶段晋级都不是计划假设；Pilot 的反证结果同样保留。
