# C-Transition 第一阶段准备度审查（2026-11-22）

## 结论

第一阶段的理论收尾、C1/C3-A 状态审计和下一轮实验准备已完成。下一项模型实验仍是按冻结协议运行 C3-A SAR-only Flow；它只支持复用 development 集上的探索性比较。训练暂停仍有效，且本次 GPU 快照低于协议要求的空闲显存，因此没有启动训练。建立新的独立 development 集也受数据来源阻塞。本文记录的是准备审查，不代表模型训练或真实模型评价已经完成。

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

## 3. C3-A SAR-only Flow 冻结训练准备

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
| 运行目录 | `experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1`；只能新建唯一输出，不覆盖已有结果 |

检查点状态：该 C3-A 运行目录和有效 `best.pt` 当前不存在；既有确定性 SAR-only baseline `experiments/phase_c2_c1_v2_20261031_v1/matrix/runs/x2_light__sar_only/checkpoints/best.pt` 存在，SHA-256 为 `8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165`。本轮只读取该固定 baseline 文件的大小和 SHA-256 以核验身份，没有加载权重参数。训练入口的协议/hash/preflight/resume 契约已按当前工作区代码和定向测试审查；现有工作区中的 C3-A 相关 CPU 定向测试 **82 项通过**，未读取项目图像。这些 C3-A 实现和测试改动在本任务开始前已存在，本次提交不包含它们；该回归结果针对当时的工作区快照，不是本次提交单独可复现的测试集。

### 训练与评价当前状态

用户此前要求暂缓训练，授权并未恢复。本轮提权只读设备查询为：NVIDIA GeForce RTX 3050 Laptop GPU、驱动 `580.178.04`、PyTorch `2.8.0+cu128`、CUDA `12.8`、`torch.cuda.is_available=True`、1 张卡；采样时总显存 4,096 MiB、空闲 1,971 MiB、利用率 98%，有三个进程占用。它证明 CUDA 可见可用，但不满足 C3-A 的 3 GiB 空闲及无并发计算要求。没有启动训练、等待器、模型推理或项目 GPU 计算。

正式恢复训练仍需要用户明确解除暂停，并在启动前重新满足冻结资源门槛。资源满足也不替代用户训练授权。

## 4. C3-A 评价方案与缺口

保留 [多样本稳定性 v3](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v3.json) 和 [样本集合补充 v3](../../protocols/mmcanet_phase_c3a/sample_set_evaluation_c1_v2_20261114_v3.json)：在有效 C3-A best checkpoint 生成后，仅评价 200 个 development 身份/19 组；每身份 4 个 draw，seed `20261105`，8 步 Heun；使用身份内配对、来源组级 10,000 次 bootstrap（seed `20261106`）。报告单次 draw 的 PSNR/SSIM/EPI/边缘强度比/裁剪比例、采样稳定性、像素均值点估计及协议 v3 定义的 Energy/variogram plugin 与 IID fair 评分；不选 checkpoint、不改 gate。对应输出分别为 `experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v3` 和 `experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3`。

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

当前固定输入是归一化 8-bit 表示域 PNG，训练 target 是 noisy pseudo-target；未找到可用于本阶段的原始定标幅度/强度/功率与已验证成像观测模型。因此不能宣称真实 Gamma speckle、辐射一致性或真实传感器物理一致性。任何后续前向残差须先明确作用域、退化链和测量域；当前可说的是受控表示域一致性，且尚未在真实 C3-A checkpoint 上评价。

## 8. 阶段决策与解锁顺序

1. 保留 `THEORY_VERIFIED_WITHIN_ASSUMPTIONS`；停止扩大纯合成 LMMSE 扫描。
2. 用户恢复训练后，优先按原冻结协议完成一个 C3-A SAR-only Flow 筛选，再执行原定 C3-A v3 描述性评价；训练结果只能解释为复用 development 上的探索性观察。
3. 只有满足协议继续条件，才按单独冻结协议做第二 seed；不得因结果不理想而修改原 gate。
4. 新独立 development 是正式 optical 增益、频率迁移和泛化主张的必要数据解锁；当前没有可用的未分配来源组。
5. C2/C3-B 的已有门槛保持不变；本报告和理论核验不晋级任何阶段。

本轮在项目盘仅剩约 11 GiB 时未生成大文件；临时 smoke 输出已自动清理。未读取 confirmation/test 影像载荷或测试测量，未加载任何 checkpoint，也未读取项目图像。
