# C-Transition 第一阶段理论、数据与 C3-A 结果审查（2026-11-23）

## 结论

第一阶段的 LMMSE 理论边界核验已完成，C3-A SAR-only Flow 的冻结训练、development 推理、多样本评价与 Research Decision Review 也已结束。Flow 未达到冻结的追加第二 seed 条件，当前像素域配置停止；结果仅是被重复使用的 development 集上的探索性观察。新的独立 development 集仍受数据来源阻塞，因此不进入 Optical 条件 C3-B。训练按用户明确授权覆盖 3 GiB 空闲显存预检运行；C3-A 协议文件未修改，资源例外在 run manifest 中披露，全程没有 OOM。

## 1. LMMSE 理论工作包

### 边界核验

新增 [边界测试](../../tests/test_optical_shift_frequency_lmmse_boundaries.py)，覆盖：

- 光学耦合 `c=0` 时风险增益为零；
- 极大平移使相干功率数值下溢至零时，光学增益消失；
- 零平移及复数 SAR 传递系数 `H`、复数光学耦合 `c`，与独立构造的观测协方差矩阵解相符；
- 接近奇异且相关的平移协方差保持有限、非负且稳定；
- 极低/极高 SAR 噪声及光学噪声下风险项保持有限且有序；
- 非半正定平移协方差被拒绝。

原 LMMSE 测试与边界测试共 **18 项通过**。这关闭的是固定复高斯单频假设内的公式/数值边界核验，状态为 `THEORY_VERIFIED_WITHIN_ASSUMPTIONS`；不证明真实 optical utility，也不支持或反证 H1–H6。

### v1r2 复现烟测

以 `30931f15dc42d02ff36ef414faf48c18ea998191` 的归档文件构造临时最小副本，协议、模型和运行脚本哈希均与 v1r2 manifest 相同。运行 30 个既定频率条件，每条件 50,000 个样本，共 1,500,000 个 CPU/NumPy 复高斯样本；所有 5SE 检查通过，最大标准化误差分别为 SAR-only MSE `2.431`、联合 MSE `2.448`、风险增益 `1.725`。临时运行生成了 `cases.csv`、`summary.json`、`报告.md`、`manifest.json`、命令和环境文件，字段契约完整；原有 1,500 万样本结果未运行、未覆盖。

被归档的历史提交缺少 `scripts/cli_zh.py`，而运行脚本会导入该中文参数解析器。烟测将当前工作区的解析器加入临时副本后完成。本次把该小型解析器作为已核验依赖纳入提交；历史 v1r2 manifest 未记录该依赖的哈希，因此本次烟测证明的是补齐依赖后的软件链路可运行，不改写历史 manifest 或声称其原始环境完全干净复现。临时输出已自动清理，哈希和运行结论保存在[核验记录](../../experiments/synthetic_cpu/optical_shift_frequency_lmmse_boundary_cpu_20261122_v1/报告.md)。

### 近期合成诊断的收尾边界

| 工作 | 已完成的核验 | 不能据此声称 |
| --- | --- | --- |
| [随机平移频率分辨 LMMSE](../../experiments/synthetic_cpu/optical_shift_frequency_lmmse_cpu_20261114_v1r2/results/报告.md) | 30 条件/1,500 万样本及协方差矩阵交叉核对通过；本轮边界测试与小规模软件烟测通过 | 真实图像频率收益、配准规律或当前模型的 optical 增益 |
| 相关 draw 的 CRPS（2026-11-10 合成记录） | 合成实验显示 draw 相关性会影响有限集合评分与均值误差 | C3-A 模型的真实 draw 独立性或后验质量 |
| [已知后验评分/校准](../../experiments/synthetic_cpu/posterior_score_calibration_cpu_20261109_v1r1/报告.md) | 已知分布构造下的公式和有限样本行为已有合成记录 | 真实模型后验校准、SAR 目标分布真实性 |
| [空间变差/多尺度敏感性](../../experiments/synthetic_cpu/c3a_spatial_variogram_anisotropy_cpu_20261109_v1/report.md) | 合成相关结构下，评分排序会依赖 lag、方向和空间构造 | 真实 SAR 空间相关或正式 gate 指标；多尺度方案仍是候选 |
| [亮点阈值敏感性](../../experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/报告.md) | 阈值、饱和值并列会改变表示域目标区大小与指标解释 | 独立物理散射体 ROI 或辐射定标亮点保真 |

这些结果用于结束纯合成公式/评分诊断，不再追加大型 Monte Carlo 扫描。它们均未访问本地模型图像或 checkpoint，也不改变 C2/C3 gate。

## 2. C1 数据协议与新 development 集

冻结 [C1 v2 来源组协议](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/protocol.json)及[汇总](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/summary.json)的清单哈希为 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`。元数据分配为 train 1,600 对/92 组、development 200 对/19 组、confirmation 200 对/20 组；131 个官方来源组已全部分配，三个角色的来源组不相交。来源与身份依据见 [C1 官方元数据审计](C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)、[train/development 交叉核验](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/report.md)和[数字 PNG 映射审计](../../experiments/data_audits/c1_v2_numeric_png_source_crosswalk_20261107_v1/report.md)。这些是既有 metadata/hash 审计；本次没有重新读取影像或复算交叉表。

回审发现 train/development 交叉核验脚本会覆盖当前目录的 `results.json`。现已改为检测到已有结果便在读取输入前失败，并添加单元测试；原历史审计结果保留。复跑依赖本地受控元数据，提交不包含样本级来源清单，不能从干净检出单独重跑。

边界必须保持清楚：

- 来源场景文件夹互斥不证明地理位置、采集日期、轨道或 patch footprint 独立；精细配准仍未验证。
- development 已被重复用于训练/选择，只支持范围受限的探索性结论。
- C1 v2 的 confirmation 曾在 2026-11-05 被访问作描述性复核，现已永久封存，不能称为 untouched，也不得用于选模或晋级；C1 v2 没有独立 test 角色。
- 项目 `data/sar`、`data/opt` 的数字编号 2,000 对 PNG 尚未映射到冻结 C1 身份；既有 train/development 精确哈希交叉为 0 命中，不能据编号或图像外观推断来源。
- 当前官方清单 131 组已全分配，既有候选 30 组也已用完；可审计的 SEN12MS 地理映射路线尚未建立对应关系。因此目前无法从已有来源构造新的未使用、可核验、来源组隔离的 development 集。

**新独立 development 的最小解锁条件：** 取得尚未用于 train/development/confirmation 选择的新官方来源组；获得逐文件来源与配对 manifest；核实需要的地理/采集/patch-footprint 映射及配准边界；在查看新组指标前冻结角色分配和评价规则。不得重新打开 confirmation/test 载荷或继续追溯无身份映射的数字 PNG。

该数据阻塞影响正式泛化、C3-B optical 晋级和真实频率效用主张；它不自动取消 C3-A v1 已冻结的探索筛选协议。

## 3. C3-A SAR-only Flow 冻结训练与运行复核

继续复用 [C3-A SAR-only Flow v1 协议](../../protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json)，不创建重复模型，也不改历史配置：

| 项 | 冻结内容 |
| --- | --- |
| 数据 | C1 v2 train/development；固定 `x2_light`；仅 SAR LR `[1,128,128]` 与 noisy pseudo-target `[1,256,256]`，不读 optical |
| 数据域 | 8-bit PNG 的归一化表示域；target 是 noisy pseudo-target，不是 clean SAR 或定标功率 |
| 模型 | 紧凑时间条件 U-Net，base channels 16、time dimension 64；约 323,617 参数 |
| Flow | `x0 = bilinear(SAR_LR) + 0.1ε`，固定线性路径和 velocity MSE；8 步 Heun、每 draw 16 次速度场评估 |
| 训练 | seed `20261105`；5 epochs；batch 4；Adam，学习率 `1e-4`、weight decay 0；无增强、无 AMP、0 workers；每 epoch 400 步、总计 2,000 步 |
| 选轮 | 按来源组等权 development PSNR 最大化，平局选较早 epoch |
| 继续条件 | 只有来源组等权配对 ΔPSNR ≥ `+0.05 dB`、组级 bootstrap 95% CI 下界 > 0、平均 ΔSSIM ≥ 0，才值得另做第二 seed；此条件不是晋级 gate |
| 资源门槛 | CUDA；至少 3 GiB 空闲显存；不得有并行 GPU 训练 |
| 运行目录 | `experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1_attempt2_20261123`；独立唯一输出，没有覆盖既有结果 |

运行目录现已包含完整训练日志、`last`/`best` checkpoint、环境、恢复历史和结果 manifest。训练在前次 checkpoint 加载遇到 PyTorch `UnpicklingError` 后，从已记录的 `last.pt` 正常续跑；该失败不是 CUDA OOM，失败日志与 SHA-256 均保留。C3-A v1 的确定性 SAR-only baseline `experiments/phase_c2_c1_v2_20261031_v1/matrix/runs/x2_light__sar_only/checkpoints/best.pt` SHA-256 为 `8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165`。

### 训练与评价当前状态

用户授权在已运行的 CUDA 评价任务旁启动 C3-A，并以 OOM 为停止边界。run manifest 记录的空闲显存依次为 2,946,433,024、2,624,978,944 和 2,630,483,968 字节（约 2.74、2.44、2.45 GiB），低于协议 3 GiB 资源预检值；协议 JSON 保持原样，运行命令使用显式授权覆盖参数。外部 PID `589855` 为用户确认的评价任务。C3-A 训练于 2026-11-23 完成 5 轮，记录运行时间 1,663.71 秒，峰值已分配 GPU 显存 189,086,208 字节；未发生 OOM。

按冻结验证规则选择 epoch 2。Flow 相对确定性 SAR-only baseline 的来源组配对 ΔPSNR 为 −3.0724 dB（95% CI [−3.4720, −2.6922]），ΔSSIM 为 −0.17818（[−0.20823, −0.15062]）；多样本 v3 的 ΔEPI 为 −0.16871（[−0.18814, −0.14697]）。冻结的第二 seed 条件未满足，停止当前配置且不追加 seed。完整解释见[Research Decision Review](C3-A%20SAR-only%20Flow%20决策复核_20261123.md)。

## 4. C3-A 评价方案与缺口

按 [多样本稳定性 v3](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v3.json) 和 [样本集合补充 v3](../../protocols/mmcanet_phase_c3a/sample_set_evaluation_c1_v2_20261114_v3.json)，仅评价了 200 个 development 身份/19 组；每身份 4 个 draw，seed `20261105`，8 步 Heun；身份内配对、来源组级 10,000 次 bootstrap（seed `20261106`）。已报告单次 draw 的 PSNR/SSIM/EPI/边缘强度比/裁剪比例、采样稳定性、像素均值点估计及 Energy/variogram plugin 与 IID fair 评分，没有用这些结果选 checkpoint 或修改 gate。结果见 `experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3_attempt2_20261123` 与 `experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3`。四 draw Flow 推理耗时 458.28 秒，基线 4.60 秒；详细性能、亮值代理及样本归档结果见 Research Decision Review。

现有协议未覆盖 TBR、亮区 MAE 和前向观测残差。不得把它们写成已完成指标，也不追溯改写冻结 v3。若未来研究问题确需这些指标，应在有资格的数据和训练/评价授权下另立版本化协议；对当前 8-bit 表示域，残差只能称表示域一致性诊断，不能称辐射物理一致性。

## 5. Optical-conditioned Flow 对照方案

满足 C2 对 optical 效用的阶段条件、具备合格开发依据并恢复训练授权后，再冻结同一 SAR 主干、训练预算、采样器、评价样本与随机种子，比较：

1. SAR-only Flow；
2. 正确配对 optical；
3. 在同一 optical-conditioned checkpoint 上按预先冻结的跨来源置换提供错误 optical；
4. 在该 checkpoint 上提供缺失/屏蔽 optical。

匹配、错配、缺失条件使用同一个 optical checkpoint，并对同一 SAR 身份配对生成底噪；必要时另设参数量匹配 SAR-only 对照。按来源组汇总 ΔPSNR、ΔSSIM、ΔEPI、边缘误差、TBR、亮区 MAE、表示域观测诊断与成本，并用来源组级配对区间报告。不因当前复用 development 集上的旧筛选观察训练 optical 模型，也不把注意力权重或 gate 激活当作 optical 有效性的证据。

## 6. LMMSE 频率理论向真实数据的转化

不直接将 `q_k=exp(-kᵀΣδk)` 当作真实配准规律。待取得未使用、来源可追溯且具有必要地理/采集/配准元数据的新 development 后，先冻结频段、平移/局部形变压力条件、匹配/置换对照、同身份生成种子和来源组统计；再检验实测配准不确定性是否与频带级 optical 增益相关，以及错误 optical 是否产生负迁移。若无合格新组或可验证配准，T4 真实数据实验保持阻塞，不在当前重复使用的 development 上追加 screen。

## 7. SAR 物理一致性边界

当前固定输入是归一化 8-bit 表示域 PNG，训练 target 是 noisy pseudo-target；未找到可用于本阶段的原始定标幅度/强度/功率与已验证成像观测模型。因此不能宣称真实 Gamma speckle、辐射一致性或真实传感器物理一致性。C3-A checkpoint 已完成表示域生成与评价，但没有计算或验证前向观测残差；后续残差须先明确作用域、退化链和测量域。

## 8. 阶段决策与解锁顺序

1. 保留 `THEORY_VERIFIED_WITHIN_ASSUMPTIONS`；停止扩大纯合成 LMMSE 扫描。
2. C3-A v1 训练和 v3 评价已完成；当前像素域 SAR-only Flow 未达到第二 seed 条件，按冻结停止规则结束。
3. C1 v2 的 131 个来源组均已分配，现有 development 已重复用于选择，confirmation 已封存；先取得有可核验身份、来源组和配准依据的新 development，才重新讨论 Optical 效用和 C3-B。
4. C2 v2 gate 仍为 `unjudged`；本轮 C3-A 结果不回答 Optical 增量，也不晋级 C2 或 C3-B。

本次 C3-A 仅读取 train 与 development 的 SAR LR 和 noisy pseudo-target；未读取 Optical、confirmation 或 test 影像载荷/测量。训练与评价加载了运行 checkpoint 和确定性 baseline checkpoint。训练期间未发生 OOM。
