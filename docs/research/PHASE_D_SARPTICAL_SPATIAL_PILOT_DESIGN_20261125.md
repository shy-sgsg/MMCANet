# SARptical 单城区 SAR 空间分组 pilot 设计

**状态：用户已批准有限 SAR-only pilot（2026-11-25）；实施日期：2026-11-26**

**日期：2026-11-25**
**研究范围：** Phase D SAR-only 空间分组方法 pilot；不进入 C3-B，不读取 confirmation payload。

## 1. 研究问题与可证伪假设

本 pilot 检验：在 SARptical 柏林城区数据上，按空间块隔离 train/development/confirmation 后，SAR-only ×2 表示域重建能否稳定训练；以低分辨率 SAR 的双三次上采样为起点的条件 flow bridge，是否能比任务适配 EDSR 和 SwinIR 更好地重建同一 noisy SAR pseudo-target，同时保留结构和高回波。

**假设 H-D-BRIDGE：** 双三次上采样提供低频 SAR 结构锚点，条件 flow 只需学习从带噪锚点到 SAR 高分辨率 pseudo-target 的修正，因此比从独立标准高斯噪声生成整幅图像更少发生整体亮度漂移，并能在不损害 PSNR、SSIM、EPI 和亮回波代理的条件下改善至少一项重建指标。若候选在预先冻结的 development 评价上不优于最强确定性对照，或结构/亮回波明显恶化，则停止当前 bridge 配置。

该假设针对新数据、新表示域、空间划分和锚定路径，**不是**对已停止 C3-A 像素域 Flow 配置的续训或同配置重复。即使通过，也只说明本次单城区实验值得后续研究，不代表独立城市泛化、真实去斑能力、SAR 辐射物理一致性或 Optical 增量价值。

## 2. 数据来源和已核实事实

- 数据来源为官方 [SARptical 仓库](https://github.com/zhu-xlab/SARptical)所链接的 `SARptical_data.zip`，论文为 [The SARptical Dataset](https://arxiv.org/abs/1801.07532)。
- 首次本地归档：`data_external/SARptical_20261125/SARptical_data.zip`；大小 `1,115,515,719` 字节；SHA-256 `eb4c9191d95c1334ab385f3989a8718ada5cfbac464fab29d8544f9a7da97772`。首次运行在读入 train 样本 `point_454242` 时发现 ZIP 本地文件头损坏，尚未开始模型训练；失败证据保存在外部盘目录 `phase_d_sarptical_spatial_pilot_20261125_v1/`，原归档与该失败目录均保留。
- 按同一官方链接重新下载的归档位于 `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/SARptical_data_reacquire_direct_20261126.zip`；大小同为 `1,115,515,719` 字节；SHA-256 `044cfa600e301f3b1f3d72d38f2182c4e74453d9403a0aab66076d6308c567f9`。新旧归档的坐标 MAT SHA-256 均为 `f5a3ddef4df2b8c9251fc9c53ed8b7245593815925eade7a1c58378b5bd674ac`，空间分组及 1,222 项 train/development SAR 清单完全一致；新归档的 1,222 个获准 SAR MAT 均通过 ZIP CRC、`ampCrop` 字段、`112×112` 尺寸及有限值检查。该扫描未读取 confirmation、test 或 Optical 影像载荷。此后 pilot 固定使用新归档 SHA-256。
- ZIP 中有 8,840 个 SAR `*_ampPatch.mat` 与 10,108 个 Optical `.png`。每个 SAR 点可能对应多个 Optical 视图，因此 Optical 文件数不是独立 SAR 样本数。
- 作者 README 将 SAR 文件描述为 dB 单位的幅度 patch；数据集论文/README 描述为柏林城区 TerraSAR-X 与 UltraCAM aerial 数据，SAR–Optical 中心对应约 1–2 m，但影像仍保留不同成像几何、只经缩放/旋转作近似对齐。该配准说明不支持 Optical 像素级监督。
- `selectedPntXYZ.mat` 提供 8,840 个 SAR 点的 UTM 坐标。坐标范围约为东西 1.55 km、南北 1.26 km。metadata-only 网格审计未读取图像数据。
- 仓库未发现明确的独立数据许可证文件；本次只在本机研究使用，不将数据归档或影像载荷提交到 Git，也不随代码分发。

## 3. 空间划分候选

1. 以 UTM 坐标系原点对齐的 `250 m × 250 m` 全局网格分组；对每个点要求其中心距网格边界至少 `75 m`。该内缩距离大于约 `100 m` patch 的半宽，用于避免跨集合 patch 足迹重叠。
2. 当前坐标审计得到 28 个有样本的网格、1,528 个符合内缩条件的 SAR 点。网格样本数依次为：385、155、134、126、98、89、84、78、67、66、34、31、30、29、19、19、15、15、12、10、7、6、4、4、4、3、2、2。
3. 角色数量冻结为 train/development/confirmation = 12/8/8 个完整网格组；样本量目标为 60%/20%/20%，development 和 confirmation 各至少 200 个样本。
4. 为使分配可复现，先按 `(UTM 网格东向索引, 北向索引)` 排序 28 组，再用 `numpy.random.Generator(numpy.random.PCG64(seed)).permutation(28)` 产生候选顺序。按 seed `0..199999` 搜索，只使用组样本数，要求角色组数为 12/8/8 且 development/confirmation 各至少 200 个样本；最小化
   \[
   J=\sum_{r\in\{train,dev,conf\}}\left(\frac{n_r/N-p_r}{p_r}\right)^2,
   \quad (p_{train},p_{dev},p_{conf})=(0.6,0.2,0.2),
   \]
   并列取最小 seed。metadata-only 搜索得到 seed `187762`，对应 916/306/306 个样本、12/8/8 个组。固定 NumPy 版本和 PCG64 实现后，首 12/次 8/末 8 个网格分别分配给 train/development/confirmation。
5. 该算法只使用坐标、网格编号和 SAR 点名录；不读取影像内容，不按模型结果挑划分。获批后生成版本化 split manifest 并保存哈希；训练程序只加载 train/development 的 SAR MAT 文件。confirmation 仅保留在 manifest 中，训练、选模、指标脚本均拒绝其角色。

划分保护可排除所定义网格间的局部 patch 重叠，但不构成独立采集场景或独立城区划分；该数据覆盖同一柏林城区，空间组数也有限。结果只能作为有限的空间块探索证据。

## 4. 任务定义与输入变换

- 仅使用 SAR 幅度 dB patch。Optical 文件不进入数据加载器或模型。
- 目标为 SARptical 原始 `112 × 112` dB patch。构造 ×2 输入时，使用 PyTorch `F.interpolate(mode="bicubic", antialias=True, align_corners=False)` 将目标统一降到 `56 × 56`；这是固定的表示域受控退化，不声称模拟真实 SAR 传感器采样。
- 使用 train 目标像素的全局均值与标准差对 LR/target 做同一线性标准化；统计量在划分冻结后只由 train 计算，development 使用同一组值。模型输出为无界线性值，不使用 sigmoid、逐图归一化或裁剪。
- 不额外注入 Gamma speckle，也不把目标称为 clean SAR。该设置评价的是对 noisy SAR pseudo-target 的 ×2 重建，不能单独证明去斑。
- 评价的 `data_range` 固定为 train target 的 `P99.5−P0.5`，PSNR 与 SSIM 全方法共用；预测不裁剪。以 train LR 双三次上采样 `b` 的像素分布预先固定 `P99` 高回波、`P10` 背景阈值，以 `|Sobel(b)|` 的 `P90` 阈值固定边缘 mask；development 每张图都从输入构建同样的 mask，任一 mask 少于 16 像素时该项报 `NA` 并计数，不静默剔除样本。保存 PSNR、SSIM、RMSE、EPI、高回波绝对误差/P95、背景偏差、边缘幅值和使用相同降采样算子的 LR 重投影 RMSE。

## 5. 方法与对照

### 确定性对照

- Bicubic ×2：固定退化的逆向插值基线，无训练。
- EDSR SAR 单通道 ×2：沿用 vendored EDSR 主体，`num_feat=64`、`num_block=16`，以线性输出适配 dB 标准化域；身份标记为 `TASK_ADAPTED_PUBLISHED`。
- SwinIR SAR 单通道 ×2：沿用仓库中已审查的 SwinIR 架构主体并适配单通道、×2 和线性标准化域；身份标记为 `TASK_ADAPTED_PUBLISHED`。均不称为作者权重或原论文完整复现。

### 单一生成候选：双三次锚定条件 Flow Bridge

复用现有 `SarConditionalFlow` 时间残差 U-Net 主体（`base_channels=16`、`time_dimension=64`），输入为状态、单通道双三次锚点和时间。令 `b` 为标准化 LR 的双三次上采样，`x` 为 HR noisy pseudo-target，`ε ~ N(0,I)`，固定 `σ=0.5`：

\[
x_0=b+\sigma\epsilon,\qquad
x_t=(1-t)x_0+t x,\qquad
v^*(x_t,t)=x-x_0,\quad t\sim U(0,1).
\]

网络以 MSE 学习条件速度场。推理从 `b+0.5ε` 开始，使用固定 8 步 Heun 积分得到单次 draw；最终同时报告单 draw、4 draw 像素均值和 draw 间差异，不把像素均值当作生成样本。此桥接参数化的待检验机制是保留 SAR 低频锚点、生成高频修正；它不是已验证的创新，也不保证比确定性重建更好。

## 6. 训练、选模与评价预算

- 所有训练方法使用相同 train 样本、相同固定 LR/target、相同增强（同步水平翻转、垂直翻转、转置）、batch size 4、LR patch `48 × 48` / HR patch `96 × 96`、随机种子 `20261125`、最多 30 个 epoch；patch 坐标及样本顺序由固定 generator 决定。
- EDSR/SwinIR 使用 Adam、`1e-4`、weight decay `0`、L1 目标；Flow Bridge 使用 Adam、`1e-4`、weight decay `0`、速度 MSE。每种训练最多约 6,870 次更新；记录每轮 loss、gradient norm、时间和 CUDA 峰值显存。
- 每轮在全体 306 个 development 样本上按来源网格等权计算固定单 draw 的 PSNR，作为所有方法共同的 best checkpoint 选择规则；Flow 对每个 `sample_id` 使用由固定 seed 导出的同一 development 初始噪声，跨 epoch 不改变。保存 `best`、`last`、优化器和随机数状态。
- 最终对 306 个 development 样本评价确定性输出及 Flow 的 4 个预设 draw。按 8 个空间组等权报告均值、逐样本配对差和 10,000 次来源组 bootstrap 95% 区间；8 组统计只作探索性描述。
- 报告参数量、训练更新数/时长、推理单张中位数与总时长、峰值 CUDA allocated/reserved、checkpoint/预测大小和训练后磁盘余量。

## 7. 预先决策规则

- **CONTINUE（仅进入另行设计的复核）**：Flow Bridge 相对最强确定性对照的来源组等权单 draw `ΔPSNR ≥ +0.05 dB`、8 组 bootstrap 95% 区间下界大于 0，且 `ΔSSIM ≥ 0`、`ΔEPI ≥ 0`；高值区域均值/P95 绝对误差、边缘幅值绝对误差和背景绝对偏差不得出现一致恶化。一致恶化预先操作化为：8 个空间组中有 **5 个或更多**的配对误差增大；任一护栏有效组不足 8 组则该护栏不通过。四 draw 均值和单 draw 分开判读。
- **STOP 当前 bridge 配置**：不满足上述继续门槛，出现系统性幅度偏移/虚假结构，或代价相对精度收益不合理。不得以“训练还不够”自动续训、加 seed 或改阈值；如果存在源码可定位的实现错误，先记录并另行冻结修复方案。
- **实际 CUDA OOM**：保留失败 checkpoint/log/manifest 并停止本次 GPU run。按用户明确要求，不因未知 CUDA 进程或 3 GiB 空闲显存门槛拦截，不终止其他任务。
- 任意 NaN/Inf、尺寸/配对集合错误、checkpoint 写入失败或磁盘写满均停止并保留诊断。不得读取 confirmation payload；不运行 Optical 条件模型、不改变 C2/C3 阶段门槛。

## 8. 资源与产物

- 训练结果目录固定为 `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/phase_d_sarptical_spatial_pilot_20261126_retry01`，目录存在即失败，不覆盖先前失败目录或历史结果。
- 2026-11-26 启动前实测项目根分区约余 `6.1 GiB`，外部盘约余 `89 GiB`。训练缓存、checkpoint、预测和结果统一写入上述外部盘唯一目录，不写入 Git；脚本要求输出盘至少满足预计峰值新增 `3 GiB + 5 GiB` 余量，训练期间持续记录磁盘余量。GPU 型号及显存状态写入 run 环境记录；按用户指示，不以并行 CUDA 进程或固定空闲显存门槛阻止启动，不终止其他任务；只在实际 CUDA OOM 时停止本次 run。`/tmp` 仅作可重建临时空间。
- 保存中文 `protocol.md`、配置、命令、环境、git 状态、split/input manifest、数据哈希、checkpoint、训练日志、逐图/来源组 CSV、四 draw 预测、资源记录及 Research Decision Review。
- 只清理本次可明确重建的临时下载分块和临时解压副本；保留原始归档、固定训练/开发缓存、所有 checkpoint、失败证据和评价结果。论文目录保持不变，不覆盖任何既有论文。

## 9. 本次授权边界

用户已批准 SARptical 上的有限 SAR-only 空间分组 pilot。按本方案执行 train/development 全流程，confirmation 保持封存；完成后形成 Research Decision Review，并继续寻找更广来源数据。该 pilot 不授权 C3-B、Optical 载荷使用或任何论文改写；既有论文文件不得覆盖，本轮不修改论文、论文表格或投稿产物。

## 10. 执行前数据归档复核记录（2026-11-26）

官方链接重新获取的归档通过了冻结 split 清单及所有获准 SAR 载荷检查，除归档副本和 SHA-256 之外未改动样本角色、分组、训练配置、评价规则、停止门槛或模型。首次损坏归档的失败尝试保留为独立记录，不作为模型失败。

首次启动读取并缓存固定 train/development 输入后，在第一个训练器调用前因 `run_pilot.py` 的脚本入口块早于 `train_method` 定义而触发 `NameError`；该目录完整保留，未执行优化器更新。回归测试先确认入口行号次序错误，再将入口块移动到文件末尾。重试使用全新目录 `phase_d_sarptical_spatial_pilot_20261126_retry01`，不覆盖任何失败证据。
