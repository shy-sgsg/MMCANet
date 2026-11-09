# 项目事实与证据分类

## 2026-11-09 研究总纲 v1.8 获批

- 用户于 2026-11-09 明确批准采纳完整 `v1.8-proposed` 候选，并授权同步正式总纲、`AGENTS.md` 和 `README.md`；本次同步已完成。
- v1.8 增补 MONTI 2026 SAR 结构超分后潜扩散散斑生成先例，并规定单次生成 draw、像素均值点估计和样本分布为不同评价对象。具体差异与审批依据见[采纳记录](RESEARCH_CHARTER_CHANGELOG_20261107.md)。
- 采纳不改变历史结果、C1–C6 阶段资格、数据权限、C2/C3 gate 或训练暂停状态；不授权 confirmation/test 访问或新训练。

## 2026-11-07 Amieva 论文全文核实事实

- 机构仓储中的已发表版 PDF 正文曾成功读取，之后同一 URL 复访超时。正文描述 10 个 Sentinel-1 IW（10 m）/SM（5 m）配对区域、升轨 VV/VH、1–12 天间隔；SM 参考经 Lee 滤波、dB 转换和 GéFolki 配准。
- 按整景划分 5 train、2 validation、3 test；train 有 435874 个不重叠 48×48/96×96 patch，validation 有 244786 个，test 为整景。
- 正文称每 band 第 0.1/99.9 百分位基于完整影像集合计算后再划分；输入/参考统计是否分开或合并未写清楚。此事实只支持“预处理使用保留场景像素值分布”，不证明 test 标签训练或依据 test 指标选模。
- 这是非生成式 SAR-only 联合恢复文献，作者报告值未独立重算。跨模式 Lee-filtered 参考并未被证明为无噪声真值。证据和限制见[审查报告](../../experiments/literature_audits/amieva_direct_fulltext_followup_20261107_v1/report.md)。

## 2026-11-07 SAR 联合扩散论文 DOI 二手线索（尚未核实）

- ResearchGate 搜索结果列出题名 *Frequency-enhanced diffusion prior network for joint SAR image despeckling and super-resolution*、DOI `10.1088/1361-6501/ae8686` 及摘要；该页没有全文。IOP 出版方与 DOI/Crossref 本轮入口不可读。
- 因证据仅为二手页面，这里只记录“存在高相关查证线索”；不据此确认正式元数据、方法、数据集、指标或作者结论。详见[来源审查](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## 2026-11-07 生成样本均值统计与 SAR 潜扩散超分全文

- OpenReview 作者稿与 MONTI 官方页面确认一项 SAR-only 两阶段超分先例：结构恢复后使用条件潜扩散增强散斑；官方页面标注该 workshop 论文为非归档。它不研究推理时 optical 引导或本项目 8-bit noisy pseudo-target。全文审查见[报告](../../experiments/literature_audits/monti_two_stage_sar_sr_20261107_v2/report.md)。
- 纯合成实验在 `L={1,2,8}`、`K={1,4,8}` 上验证 Gamma 独立乘性噪声的均值、方差、ENL 与 MSE/PSNR 关系，共 9 个条件通过预声明核验。没有项目载荷或模型参与。结果见[报告](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/report.md)。
- 后续共同 Gamma 分量诊断固定单 draw 边际、用 `ρ` 控制 draw 相关性，45 个 `L×ρ×K` 条件全部通过；K=8 增益从 `ρ=0` 时约 9.03 dB 降至 `ρ=0.5` 时约 2.49 dB、`ρ=0.9` 时约 0.40 dB、`ρ=1` 时 0 dB。该观察只限合成构造，不测量模型样本相关性。见[相关 draw 报告](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/report.md)。
- C3-A v3 evaluator 按 draw 计算 PSNR/SSIM/EPI/边缘强度比等，再汇总逐身份指标和逐像素样本标准差；没有逐像素平均预测图、分布距离、显式 draw 依赖统计或预测图保存。与冻结协议相符，不是缺陷；需扩展时应另立协议。此为源码/协议静态观察，不是生成模型结果。见[审查](C3A_MULTI_SAMPLE_DEPENDENCE_AUDIT_20261107.md)。

## 2026-11-07 C2 全 development CPU 对比差诊断

- v5 冻结协议覆盖 4 种退化条件 × 每条件 200 个 development 样本，共 800 条条件—样本、19 个来源组；使用 8 个既有 best checkpoint 完成 3,200 次 CPU 前向。
- C2 控制为 matched、permuted、missing、SAR-only。零目标背景中位数每条件分别为 2、2、1、1 条；matched−SAR-only 来源组等权均值对比误差差在四条件为 −0.000191、−0.000939、−0.000862、+0.000677。
- 结果属于重复使用的 development 与归一化 PNG noisy pseudo-target 表示域；只作描述，未运行 bootstrap/显著性检验，不参与 C2 gate、选模或调参。访问标记为 confirmation/test false，CUDA 未初始化。见[实验报告](../../experiments/diagnostics/c2_all_development_contrast_cpu_20261107_v5/report.md)。

## 2026-11-07 SAR 条件 Flow 翻译先例

- OpenReview ICLR 2026 ML4RS Workshop 作者稿描述 SAR 条件潜空间 Flow Matching，将噪声 latent 生成 optical latent；数据为 SAR2Opt，正文给出 1,450/627 个 train/test patch，600×600，TerraSAR-X 与 Google Earth。
- IEEE Xplore 有同题名 GRSL Early Access 索引记录，日期 2026-10-12、DOI `10.1109/LGRS.2026.3742531`。IEEE 全文和版本对应未核实；该方法输出 optical，不是本项目 SAR 输出 JDSR。见[一手来源审查](../../experiments/literature_audits/sar_conditioned_flow_sar_to_optical_review_20261107_v1/report.md)。

## 2026-11-07 08:29（中国标准时间）DiffusionSAR 全文核验事实

- 2024 年 TGRS DiffusionSAR 研究单幅 SAR 条件扩散去斑；观测 SAR `y` 是推理条件，光学参考只参与合成训练，论文没有联合 SR 或 inference-time optical 实验。
- 合成数据含 2,400 个 256×256、10 m、来自 16 个地点的 Sentinel-2 patch，使用 Gamma L=1/2/4；另对真实 SAR 多时相序列进行微调。论文采用 log-Yeo–Johnson 噪声变换，但这不等于验证其适用于 C1 8-bit PNG。
- 论文不同章节分别称真实数据为 single-look intensity 与 SLC，产品表示仍无法核实。全文事实及主张边界见[审计报告](../../experiments/literature_audits/diffusionsar_conditional_sar_despeckling_fulltext_20261107_v1/report.md)与 D-2026-11-07-026。

## 2026-11-07 08:09（中国标准时间）SEN12MS 地理元数据候选路线

- SEN1-2 原始论文按 1158/spring、1868/summer、1970/fall、2017/winter 四组随机 ROI 组织场景；同一作者团队的 SEN12MS 原始论文明确称使用同一批随机 ROI 和季节定义，并将场景导出为有地理参考的 GeoTIFF。
- SEN12MS 是经独立筛选和处理的后续数据版本。论文未明确保证各版本 ROI 子目录数字编号逐一相同；不得把相同目录数字或 patch 数字直接当作已核验的地理/patch 对应。
- 本轮仅确认候选元数据路径，未提取坐标或读取 GeoTIFF。TUM MediaTUM 1474000 返回 HTTP 429。C1 地理、采集与精细 footprint 独立性仍未知；不改变当前 split 或阶段 gate。
- 证据与未核实项见[交叉映射路线审查](../../experiments/data_audits/c1_geospatial_crosswalk_route_20261107_v1/report.md)及 D-2026-11-07-024。

## 2026-11-07 04:06（中国标准时间）C2 固定退化生成器协议绑定事实

- 冻结 C2 v2 退化基种子为 20261032，条件映射为 ×2/×4、Gamma `L=8/2`。原生成器只记录调用方 seed 和硬编码条件，没有强制两者匹配冻结 JSON。
- 现有生成器在 CUDA 检查、目录创建和影像读取前拒绝 seed 或 condition drift；冻结 JSON、退化公式和 gate 不变。
- 两个错配回归先在旧实现上失败，修复后连同来源清单错配保护共 3 项 CPU 定向测试通过，两个 Python 文件 `py_compile` 通过。运行只用临时文本清单，无数据载荷、checkpoint、GPU 或模型访问。见 D-2026-11-07-010 和[C2 核查报告](C2_FIXED_GENERATION_PROTOCOL_BINDING_20261107.md)。

## 2026-11-07 03:46（中国标准时间）C1 v2 train/development 官方来源链更正事实

- 冻结 C1 v2 来源协议使用 TUM MediaTUM 官方条目 1436631、`m1436631` 模块和官方 `checksums.sha512` 文件索引；协议按春季/冬季根目录与 source-scene 文件夹构造互斥组。
- 2026-10-31 train/development 暂存 manifest 记录 1,600/200 个身份、92/19 个来源组、3,600 个文件完成暂存；暂存代码按冻结清单逐文件核对字节数与 SHA-512。786 个复用局部文件在合并前也经 SHA-512 核验。confirmation/test 访问标记为 false。
- 2026-10-19 上游哈希审计范围仅为旧 `fixed_train_dev2k` / `fixed_val_dev2k` 的 3,600 个文件，字节级精确匹配为 0；不能将该结果外推到 C1 v2 train/development。`./data/` 数字命名 2,000 对与 C1 v2 的逐文件身份映射仍未知。
- 地理坐标、采集日期、轨道、产品 ID 与 patch footprint 交叉表仍缺；当前仅支持来源场景文件夹及 train/development 文件身份互斥，不声称地理或采集独立。更正依据见 D-2026-11-07-009 和[C1 来源链核查](C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)。

## 2026-11-07 02:13 C3-A 表示域范围审计与输出越界诊断事实

- 按冻结 C3-A `x2_light` 输入，CPU 核验 train 1,600 个样本/92 个来源组和 development 200 个样本/19 个来源组；只读取 `lr` 与 `target`，逐文件 SHA-256、shape、dtype、有限性和值域均通过。
- 来源组等权平均的 LR 像素大于 1 比例为 21.57%/22.16%；按模型实际双线性上采样后的 condition 大于 1 比例为 14.94%/15.29%。target 恰为 1 的比例为 1.23%/1.28%，大于等于 0.99 的比例为 1.73%/1.72%；target-condition MAE 均值约 0.1451/0.1454。
- 这些数字描述 8-bit PNG 派生表示域，不说明定标物理量、真实亮点饱和或干净 SAR 恢复。审计未设置阈值、不改变训练/裁剪/选模规则。
- 多次采样评价协议 v3 将总越界比例分解为 raw 输出低于 0 与高于 1 的比例，保留 v2 原件；原质量指标、数据划分及阶段门槛不变。合成定向回归 30 项通过；无真实训练/模型结果。
- 设备仅 CPU；仅访问 train/development SAR 数组。未读取 optical、confirmation、test；未启动训练、正式评价或 GPU 等待器。证据见[审计运行清单](../../experiments/diagnostics/c3a_representation_range_audit_20261107_v1/manifest.json)。

## 2026-11-07 00:26（中国标准时间） C3-A epoch 续训合同 CPU 验证事实

- C3-A runner 支持显式按完整 epoch 续训，checkpoint v2 绑定冻结协议、输入清单、代码与关键环境，并记录模型/优化器、Python/NumPy/PyTorch RNG、专用 DataLoader/Flow generator、最佳轮次和 train/development 日志。
- 当 `best.pt` 比 `last.pt` 更新时，恢复选择器把已通过严格校验的最新状态原子写入 `last.pt` 后返回，修复两文件写入间中断的路径不一致。
- CPU 合成训练连续执行与恢复执行在第二步参数、Adam 状态及后续 RNG 序列上等价。C3-A 续训、Flow、评价器和等待器定向回归 67 项通过；相关 Python 文件 `py_compile` 通过。
- 未运行真实训练、GPU 等待或正式评价；未访问 optical、confirmation、test 载荷。该结果只证明续训软件合同，不证明收敛、重建效果或 gate 晋级。

## 2026-11-06 23:37 C3-A train/development CPU 数据预检事实

- 按冻结 C3-A `x2_light` 输入，在 CPU 上逐文件核验 train 1,600 个样本/92 个来源组和 development 200 个样本/19 个来源组的 SAR LR 与 target 数组，共 3,600 个 `.npy` 文件。
- 两角色数据均通过冻结文件哈希、预期 shape、float32 dtype、有限数检查及协议值域检查；各角色的固定输入 manifest 与 `files.csv` 哈希见[运行清单](../../experiments/diagnostics/c3a_cpu_data_preflight_20261106_v1/manifest.json)。
- 预检前后 CUDA 均未初始化；confirmation、test 和 optical 载荷访问标记均为否。项目分区结束后可用空间为 21,117,218,816 字节（约 19.67 GiB）。
- 此事实只证明该冻结输入副本符合 C3-A 数据读取合同，不证明模型效果、数据物理定标或来源组的地理独立性；训练仍按用户要求暂停。

## 2026-11-06 23:25 C3-A CPU 张量/manifest 合同与数值核验事实

- 对 C3-A 条件流匹配辅助函数的 CPU 合成回归发现并修复了混合浮点 dtype、无效训练时间以及 Heun 速度场可广播 shape/dtype 输入未 fail-fast 的问题。
- 多次采样评价器此前未核对训练 `input_manifest.json`；现已在任何 development 载荷读取前验证来源路径/哈希、角色集合、固定输入计数、仅 `lr`/`target` 数组及 confirmation/test/optical 标记。
- 实施前 7 个张量合同断言、9 个 input manifest 断言及 2 个路径/光学数组断言先失败；修复后 C3-A Flow、multi-sample evaluator 与等待器相关定向测试共 51 项通过，相关 Python 文件编译通过。
- 合成张量 CPU Adam 单步的 loss/gradient 有限，优化器确实改变模型参数；这只验证训练算子链路，不构成项目数据上的模型训练。
- 清理了本轮验证生成的 26 个可再生 `.pyc` 文件，共 560,672 字节；pytest 缓存、数据、历史实验、checkpoint 和来源不明文件均保留。
- CPU 解析线性 ODE 检查得到 Heun 步数翻倍误差比 3.907/3.970，模型调用数为 2×步数；样本源噪声映射对 seed/sample_id 可复现且与 batch 顺序无关。证据：[诊断目录](../../experiments/diagnostics/c3a_flow_cpu_contract_20261106_v1/)。
- 本诊断不读取项目影像、不初始化 CUDA，不代表 SAR 模型性能、采样稳定性或物理有效性；C3-A GPU 训练按用户指示暂停，所有阶段 gate 保持原状。

## 2026-11-06 v1.7 批准与同步事实

- 用户明确批准 `v1.7-proposed`，批准日期为 2026-11-06；现行规范文本位于 `docs/research/RESEARCH_CHARTER.md`。
- v1.7 沿用 v1.6 研究主题及 Q1–Q6/C1–C6 框架，补充算子感知条件流/后验样本覆盖先例、冻结 C1 v2 train/development 使用范围及 confirmation 已封存边界。
- 本次批准不改变 C2 未判定状态、B1 `not_promoted / stop_at_b1`、B2/B3 关闭状态，也不授权 confirmation/test/optical 载荷访问。
- 截至 20:54（用户暂停 GPU 任务前），C3-A v8 仍等待 GPU；快照为空闲 1146/4096 MiB、3 个其他进程，训练未启动。磁盘当时约 15.06 GiB；本轮没有新增的大型项目结果可清理。

## 2026-11-06 18:35 等待资源与空间回升事实

- v8 于 10:34:26 UTC 的 GPU 快照为空闲 1,453/4,096 MiB、2 个其他 CUDA 计算进程；未启动训练。
- 同一等待记录的磁盘可用空间为 13.611 GiB，独立读数为 13.47 GiB。约四分钟前独立读数曾为 4.88 GiB；中间仓库未写入大文件，变化来源未知。
- 本轮新增缓存两次清理：pytest/Python 测试缓存 26,850 字节，最终 `py_compile` 缓存 22,604 字节；均为当前任务生成且可重建。其他数据/结果均保留。

## 2026-11-06 18:31 资源/清理事实

- v8 等待器 10:30:25 UTC 记录磁盘可用 4.909 GiB，10:30:26 UTC 独立读取为 4.88 GiB；约 11 分钟前等待器记录为 12.135 GiB。空间下降约 7.23 GiB，具体来源未知。
- GPU 同期空闲 1,229/4,096 MiB，3 个其他 CUDA 计算进程，C3-A 训练未启动。
- 2026-11-06 项目目录中没有新增大于 10 MiB 文件。仅删除本轮测试和编译生成的 10 个缓存文件，合计 26,850 字节；未删除任何数据、结果、checkpoint 或其他项目文件。

## 2026-11-06 C3-A 重启与文献版本事实

- v7 等待记录最后一次更新于 2026-11-06 09:32:03 UTC；当时没有训练启动时间。当前未发现对应等待/训练/评价进程，C3-A 训练和评价输出目录均不存在；退出原因未知。
- `scripts/wait_for_c3a_gpu.py` 原先固定使用已存在的 v7 记录目录，旧入口会在覆盖保护处退出。现已要求传入唯一 `--run-id`，v8 记录位于 `experiments/diagnostics/c3a_gpu_wait_20261106_v8/`。新增定向测试 5 项通过，入口和测试文件编译通过。
- v8 最近 GPU 快照（2026-11-06 10:19:26 UTC）：空闲显存 1,453/4,096 MiB，2 个其他 CUDA 计算进程；训练未启动。confirmation、test、optical 和 train/development 载荷均未访问。
- 磁盘等待器与独立读取的短时读数分别为 12.135 GiB 与约 9.46 GiB；变化来源未知。当天未发现项目目录新增大于 10 MiB 文件；本任务没有可确认的大型缓存/结果可清理。
- arXiv:2609.12953 当前 v2 题名为 FIRM: Flow-based Imaging via Regularized Minimization；v1 旧题名为 Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems。该书目版本校订不改变理论假设或本地方法决策。
- 内部稿 `paper/MMCANet/mmcanet_jdsr_draft.tex` 标注为 2026 年 7 月 pilot，并自述三轮、单 seed、无 validation、数据物理域未核实；不是当前主线的投稿证据。仅读了结果章节前的正文，其中含历史 test 文字；未读任何 test 载荷、test 数值或 checkpoint，故本次文字访问记为 `test_set_accessed=true`、`test_set_used=false`。


## C3-A 结构评价协议与资源现状 — 2026-11-05 18:30（中国标准时间）

- 当前多次采样协议为 v2，SHA-256 `8d886f26023f9d929265b3f70ad9f51314f94788b1e4a01f269af5c1f0a589fc`；明确登记固定 edge mask、EPI、边缘强度比和 `edge_mask_manifest.csv`。v1 在训练与载荷访问前被替代。
- AISAR 下多次采样评价、等待器与 Flow 定向测试共 18 项通过；加入共用 evaluator 回归后合计 44 项通过，相关 Python 文件 `py_compile` 通过。真实训练与 development 评价尚未运行，因此没有方法效果、完整运行时间或峰值显存结果。
- 新等待会话 `77673` 的最近查询（2026-11-05 10:29:35 UTC）为 RTX 3050 Laptop GPU 空闲 1,079/4,096 MiB、利用率 97%、4 个其他 CUDA 计算进程；该会话继续按 60 秒间隔等待。train/development 清单、C1 来源清单和 baseline 文件存在性已确认；未读取影像载荷。

## C3-A 多次采样诊断实现事实 — 2026-11-05 16:18（中国标准时间）

- 新冻结补充协议 SHA-256：`9c00accfc9cd046783323735509db9505a09a6351bcf55d93e5699ebf9970b77`；协议将每身份固定四次采样、development 200 个身份和 19 个来源组、组级 bootstrap 与不参与选轮的解释边界锁定。
- 诊断入口核验训练协议、训练 manifest、best checkpoint 哈希与轮次、固定 development manifest 和来源清单；只读取 SAR LR/target，不读取 optical、confirmation 或 test。
- 合成验证为 C3-A Flow、采样汇总及 GPU 等待器资源规则共 16 项通过；Python 编译检查通过。尚未运行训练或真实多次采样评价，因此没有方法效果、运行时或实测显存事实。
- 2026-11-05 16:03 的 GPU 资源事实为 RTX 3050 Laptop GPU、1,303/4,096 MiB 空闲、利用率 96%、3 个计算进程；16:26 UTC 快照为 1,197/4,096 MiB 空闲、利用率 69%、3 个计算进程；08:43:44 UTC 快照为 1,167/4,096 MiB 空闲、3 个计算进程。等待会话 `28461` 正在按 60 秒间隔轮询；这些快照均不是当前实时状态。

## 活动研究进展 — 2026-11-05 15:07（中国标准时间）

- **事实：** 提权训练器入口最新检查发现 3 个其他 CUDA 计算进程并在读取影像前退出；GPU 快照显示 1,192/4,096 MiB 空闲、利用率 96%。等待器当前活动会话为 `25767`，最新轮询为 3 个进程、1,224 MiB 空闲。
- **事实：** C3-A 预登记网络有 323,617 个可训练参数；FP32 参数存储理论占用 1.23 MiB，不含训练状态或激活。
- **事实：** README 已补充 v1.6 对应的 C1–C6 路线、方法候选和状态词汇；没有修改正式总纲或阶段 gate。
- **未知：** C3-A 训练、development 全集指标与生成质量均未验证。

## 最新资源续记 — 2026-11-05 12:01（中国标准时间）

- **事实：** 第八次 C3-A 启动检测到 5 个其他 CUDA 计算进程，入口以退出码 2 在读图前停止；GPU 可见，未读取影像或创建运行目录。详见[预检 v8](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v8/manifest.json)。
- **未知：** C3-A 训练和完整 development 评价仍未完成；没有可报告的 flow 效果结果。
- **规范：** 用户要求目标完成并通过适用验证后，再审查范围、提交并推送；已写入仓库根目录 `AGENTS.md`。

## 本轮科研进展 — 2026-11-05

- **事实：** C3-A 预登记入口的最新一次启动检测到 4 个其他 CUDA 计算进程，并于读图前以退出码 2 停止；GPU 对执行器可见，没有创建训练目录或读取任何图像载荷。访问标记见[预检 v7](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v7/manifest.json)。
- **事实：** C2 代码新增独立的零背景对比差诊断函数；4 个合成边界测试和 39 个既有 C2/evaluator 回归共 43 项通过，相关 Python 文件编译通过。
- **未知：** 新指标尚未用于真实 SAR 开发集预测，故没有新增方法效果观察。C2 v2 gate 仍未判定；B1/B2/B3 状态不变。
- **数据边界：** 本轮指标验证仅使用测试代码中的合成数组；未读取 train/development/confirmation/test 影像，confirmation 仍封存。

## 研究总纲候选审阅 — 2026-11-05

- **事实：** 用户已批准的正式研究总纲仍为 v1.6。完整 v1.7-proposed 候选、逐节差异报告、数据协议审查和一手文献支持审查已准备，尚未批准；受保护的正式总纲、AGENTS.md 与 README.md 未据候选修改。
- **事实：** 本次审阅读取了含历史 test 关联说明的文字记录，任务级标记为 `test_set_accessed=true`（仅文字）、`test_set_used=false`。未读取 test 图像、测量数组、test 指标文件或 checkpoint，也未读取 confirmation 载荷。
- **解释边界：** 文字记录访问不构成 test 指标或样本的模型选择使用；项目累计 test 标记仍按历史状态保留。

## C3-A 启动预检 — 2026-11-05

- **事实：** 用户已明确要求直接使用项目冻结的 C1 v2 train/development/confirmation 划分。本次方法筛选绑定原 C1 v2 `x2_light` 固定输入：train 1,600/92 组用于训练，development 200/19 组用于逐轮评价；没有重新切分，也没有使用 `./data/` 数字文件名子集。
- **事实：** 最新一次在仓库复现环境 `AISAR` 中运行预登记 C3-A 入口时，训练器检测到 3 个其他 CUDA 计算进程，按冻结资源规则以退出码 2 停止；没有读取影像载荷、创建训练运行目录或开始训练。见[最新预检记录](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。
- **边界：** 本次 C3-A `confirmation_payload_accessed=false`、`test_set_accessed=false`、`test_set_used=false`、`optical_payload_read=false`。确认集仍封存，C2 原 gate 与 B1/B2/B3 状态不变。

## C1 v2 confirmation 一次性 C2 复核 — 2026-11-05（访问后修订）

- **事实：** 直接使用冻结 C1 v2 来源场景文件夹划分，没有重新切分：train 1,600/92 组、development 200/19 组、confirmation 200/20 组。confirmation 暂存清单 SHA-256 为 `300b04f307a0f9c12f327442d4e27633104d0909311c81e7b803e4424a4353e5`；四条件固定输入生成执行清单 SHA-256 为 `9e54dfa99a218498cde35fdda95888bcf63ac9c50137d7a9c8da344af5b64003`。
- **观察：** 复用 C2 v2 八个 best checkpoint，对 200 个 confirmation 样本运行四条件 × 四种控制。最终产生 3,200 条逐图控制指标、1,600 条配对对比和 72 条组汇总；每条件 20 个来源组等权 bootstrap。matched−SAR-only PSNR 为 −0.0114、−0.0108、+0.0048、−0.0098 dB；matched−permuted 为 +0.0035、+0.0038、+0.0047、+0.0047 dB。全部 matched−SAR-only 点估计低于 +0.05 dB 门槛，预登记结果为 `inconclusive`。
- **执行偏差：** 首次运行在确认数组核验后、推理前因置换路径字段映射失败；后一次运行完成四条件推理并在内存计算指标，但在写出结果前因阈值字段映射失败。没有查看这些内存指标或据此选模。修正后以相同数据、checkpoint、指标和阈值做技术恢复运行。原协议和两份访问后修订分别见[协议目录](../../protocols/mmcanet_phase_c2/c2_confirmation_20261105_v1/)；失败执行和最终恢复记录分别见[`首轮记录`](../../experiments/phase_c2_confirmation_c1_v2_20261105_v1/)、[`amended_v1`](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v1/)和[`amended_v2`](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/)。
- **解释边界：** 本次 `confirmation_payload_accessed=true`、`test_set_accessed=false`、`test_set_used=false`；confirmation 不得再用于调参或模型选择。结果是来源文件夹划分、单 seed、normalized 8-bit noisy pseudo-target 的表示域描述性观察，不构成原 C2 v2 gate、B1 晋级或地理/辐射物理主张。项目累计 test 标记仍为 `true/true`，由既往旧 `test_100` 污染历史造成。

## C1 v2 划分上的 C2 v2 实际训练与诊断 — 2026-10-31

- 本轮直接沿用冻结 C1 v2 来源场景文件夹划分，没有重新切分：train/development/confirmation 为 1,600/200/200 个样本、92/19/20 个来源组。仅取得并核验 train/development 的 3,600 个 PNG；confirmation 未下载、未访问。暂存与身份校验见[manifest](../../experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json)。
- C2 v2 四种固定退化条件的输入生成完成，共 8 个角色/条件单元、21,600 个数组、约 7.85 GB。冻结八组训练均完成 30/30 epoch，共 48,000 个优化步骤；最佳开发 epoch 为 30、30、28、28、26、30、28、28。见[输入生成 manifest](../../experiments/phase_c2_c1_v2_20261031_v1/fixed_input_generation_manifest.json)和[训练矩阵](../../experiments/phase_c2_c1_v2_20261031_v1/matrix/manifest.json)。
- 冻结 paired screen 因两条 development 身份的 target 背景中位数为零，触发 C2 v2 的 fail-closed 规则；整次正式评价失败关闭，gate 未判定。没有修改协议、删样本或加 epsilon。原始记录见[失败 manifest](../../experiments/phase_c2_c1_v2_20261031_v1/screen/execution_manifest.json)。
- 独立非门槛诊断保留全 200 个 development 身份、19 个来源组和四种控制，仅将两项 median TBR 误差标为不可用。matched optical 相对 SAR-only 的来源组 PSNR 增益为 −0.00552、+0.00421、+0.00610、−0.00240 dB；四个 95% 区间均跨零且低于 +0.05 dB 继续阈值。matched 相对 permuted optical 为 +0.00994 至 +0.01516 dB。此为单 seed 的开发集描述性观察，不能用作 gate、稳定效用或独立泛化结论。详见[非门槛诊断报告](../../experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。
- 本轮 `confirmation_payload_accessed=false`、`test_set_accessed=false`、`test_set_used=false`。这些标记只描述本轮；项目累计 test 污染历史仍按旧记录保留。来源组仅证明文件夹级分离，地理/采集独立性与精细空间重叠仍未知。

## C1 v2 载荷与 C2 矩阵的中间快照 — 2026-10-31（训练结果已由本页首节更新）

以下为载荷取得后、训练和评价尚未全部结束时记录的中间状态；不代表当前运行结果。

- 冻结 C1 v2 train/development 角色各含 1,600/200 个配对样本，来源组为 92/19；从官方来源取得 3,600 个 PNG 后逐文件核验 SHA-512、PNG 模式和尺寸。正式暂存见[并行暂存 manifest](../../experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json)。200 个 confirmation 样本未下载或访问。
- C2 v2 四种固定退化条件的 train/development 输入全部生成，共 8 个单元、21,600 个数组、约 7.85 GB；生成器逐项记录来源身份、输入 SHA-256 与确定性 seed。见[固定输入矩阵 manifest](../../experiments/phase_c2_c1_v2_20261031_v1/fixed_input_generation_manifest.json)。
- 冻结八组 ×30 epoch 训练已启动，训练入口的新鲜 batch=8 合成 CUDA 预检通过；实时进度见[训练矩阵 manifest](../../experiments/phase_c2_c1_v2_20261031_v1/matrix/manifest.json)。paired screen 尚未执行，尚无方法效果结论。C1 来源组—地理/采集交叉表与精细空间重叠仍未解决。
- 本轮 C1 v2 confirmation/test 访问标记为 `false/false`。项目累计 test 标记为 `true/true`，源于旧本地训练误用旧 `test_100` 中 13 个身份；该历史事件与本次 C1 v2 confirmation 无关，旧本地测试子集已失去未触碰测试资格。

## C2 配对屏幕执行链代码事实 — 2026-10-31（载荷取得前的阶段记录）

## C2 配对屏幕执行链代码事实 — 2026-10-31（载荷取得前的历史代码快照）

- 新增角色受限暂存、八组 CUDA 训练矩阵/恢复入口，以及完整开发集 paired screen 入口；screen 检查 200 个 identity 的四控制结果，记录固定掩膜阈值/计数/覆盖率/hash、逐图指标、四项冻结配对对比和来源组 bootstrap。
- 提权 AISAR 下 9 个 C2 定向测试文件共 59 项通过；冻结模型单样本 CUDA 合成推理预检通过，峰值 PyTorch 分配约 20.9 MB，未读取项目数据。
- 此前冻结 batch=8 合成训练预检曾因共享显存 OOM；2026-10-31 重试通过一步合成优化，峰值分配 365.8 MiB、运行后余 52.9 MiB。用户指出的 `data/` 虽有 2,000 对 PNG，但数字文件名与 C1 v2 清单来源路径不一致；train/dev 四类路径匹配均为 0，角色受限暂存在复制前停止。训练矩阵与 screen 尚未在固定 C1 输入上端到端运行。
- 本轮读取研究文档时看到历史 test 关联文字；只记录文本暴露，不读取 test payload/metric/checkpoint，不使用历史数值。当前任务标记 `test_set_accessed=true`、`test_set_used=false`，累计项目标记仍为 `true/false`。
- 详见 [C2 代码/资源审计](../../experiments/diagnostics/c2_execution_acceptance_20261031_v1/report.md)、D-2026-10-31-001/002/003/004。代码通过和合成验证不构成 optical 效用或 B1 晋级证据。

## C2 指标与执行 helper 代码验收事实 — 2026-10-30

- C2 指标 wrapper 显式设置比率图 `min_signal=1e-3`、`eps=1e-8`，并以 fail-closed 规则拒绝任一未定义或非有限必需指标；其 evaluator revision 作为合同字段供运行清单记录。
- 掩膜记录 helper 输出冻结条件 ID、图像形状、每类阈值、像素数与覆盖率；由于完整结果 writer 尚不存在，此字段目前还没有正式 run 落盘证据。
- 四条件审计只读四个 `manifest.json`/`files.csv`，验证 identity、source group、SAR/optical source 与 target/optical array 哈希 parity，不加载 `.npy`。
- 训练 helper 已有共同初始化、固定 seed 的 DataLoader、group-balanced PSNR checkpoint 规则和单 epoch train/development 函数；CPU epoch 验证使用合成数据及测试模型，不代表真实 `C2MatchedJoint` forward/训练。
- 五个 C2 定向测试文件在 AISAR 环境中 29 项通过、3 项 CUDA 项跳过。此事实只描述代码/合成测试，不构成数据、正式训练或模型效果证据。

## 本轮 text-only test exposure — 2026-10-30

- 重读 `CURRENT_STATE.md` 时看到历史 pilot test 关联文字说明；没有打开其底层 test payload、预测、指标文件或 checkpoint。
- 本轮 `test_set_accessed=true`、`test_set_used=false`，未使用该文字支持方法或模型选择；详见 D-2026-10-30-013 和对应[文本暴露审计](../../experiments/data_audits/research_text_read_exposure_20261030_v1/report.md)。

## Authoritative current snapshot (2026-10-29)

Use this snapshot when it conflicts with older dated entries below; those entries preserve their original evidence and remain historical records.

- **Current adjacent optical–SAR prior art:** Liu and Chang's CDD describes SAR-conditioned optical generation for clustering; Zhang et al.'s DMDiff and Cao et al.'s GLCdiffcr use SAR-guided diffusion for optical cloud removal; Vulli et al. describe reliability-gated SAR/optical patch classification under synthetic optical degradation. These tasks do not establish SAR-output restoration/JDSR. GLCdiffcr's publisher abstract describes pixel/SSIM consistency on the optical target, not a SAR observation likelihood; the full article was inaccessible in this review. The matrix contains 138 rows: 78 historical members and 60 additions with unresolved original membership. See D-2026-10-29-016/018/021 and the linked source audits in `literature_review.md` and `RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md`.

- **Research-charter status:** A full C1–C6 candidate (`v1.2-proposed`), sectionwise change report, and bounded primary-source/data-protocol audit are review-only at `RESEARCH_CHARTER_REVISION_PROPOSAL.md`, `RESEARCH_CHARTER_CHANGELOG.md`, and `RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md`. The title and methods are proposed hypotheses; expected academic contributions remain conditional candidates or a bounded null/negative result. D-2026-10-29-019 narrows candidate novelty after OGSFlow's repository README described optical-guided flow SAR-SR and reliability-gated fusion; no paper identity or method result is verified. D-2026-10-29-020 records the explicit expected-contribution wording, D-2026-10-29-023 keeps CN122694669A provisional, and D-2026-10-29-024 limits physical-consistency claims for current PNG data. No protected charter, `AGENTS.md`, or `README.md` was adopted or edited; historical gates are unchanged. See D-2026-10-29-004/019/020/023/024.
- **Physical-consistency title boundary:** D-2026-10-29-024 records that official SEN1-2 PNGs are documented as 8-bit sigma-nought dB; C1 v2's selected files are byte-matched to that source, but the retained documentation has no byte-to-dB mapping or raw calibrated measurements. The separate earlier B1 v2 operator is representation-domain, not a sensor model. “Physics-consistent” remains conditional in the unadopted candidate.

- **Verified C1 v2 scope:** Official spring/winter metadata contains 131 scene-folder groups and 142,506 matched patch indices; the frozen group-first allocation is 92/19/20 groups and 1,600/200/200 pairs. The train/development payload, schema, representation and bounded overlap checks passed. The official SEN1-2 paper supports whole-scene splitting as the intended independence unit, and C1 v2 assigns complete source-scene folders to roles. Direct footprint and acquisition independence for this allocation remain unverified because retained manifests have no ROI coordinates or sensor product/acquisition IDs; a 2026-10-29 check of the official TUM catalog route returned HTTP 429 and surfaced no crosswalk, without proving one is absent. Arbitrary-offset overlap, fine registration and confirmation spatial independence also remain open. Eligibility stays limited to a prospective source-folder-disjoint exploratory development screen, and confirmation payload is untouched. See D-2026-10-28-075, D-2026-10-29-012/022, the [eligibility decision](../../protocols/mmcanet_phase_c1/v2/c1_development_eligibility_20261028.md), the [scene-split review](../../experiments/data_audits/sen1_2_scene_split_metadata_review_20261029_v1/report.md), and the [catalog route audit](../../experiments/data_audits/sen1_2_official_catalog_metadata_route_20261029_v1/report.md).
- **Verified C2 protocol and engineering facts:** A scale-aware parameter-matched candidate, train/development reader, CUDA-only fixed-input generator, and analysis helpers support ×2/×4 and four candidate degradation settings; recorded synthetic contracts passed on the RTX 3050 Laptop GPU. Protocol v2 JSON/Markdown are frozen before data generation. No project-backed C2 arrays were generated, no C2 model was trained, and no performance improvement was measured. The end-to-end training/evaluation runner and remaining protocol acceptance evidence are incomplete. See D-2026-10-29-001/002/003/004 and the [generation contract](../../protocols/mmcanet_phase_c2/fixed_generation_contract_20261029.md).
- **Unchanged historical gate:** Prospective B1 v2 remains `not_promoted / stop_at_b1`; Phase B2/B3 remain gated. Exploratory C2 engineering does not rewrite this decision.

- **Historical C1 v1 inventory fact (superseded for the later official cohort):** The 1,800-pair authenticated train/development cohort spans 30 official source-scene folders, all used by B1 v2. That inspected cache had no unused confirmation groups; a later official spring/winter inventory supplied a separate candidate pool for C1 v2. The local numeric-name 2,000-pair subset remains unmapped. See D-2026-10-28-067/069 and the [C1 v1 inventory](../../protocols/mmcanet_phase_c1/protocol_report.md).
- **Verified representation-profile fact:** All 1,800 frozen train/development composite files and 3,600 decoded panels matched the retained official-source hashes. The combined SAR DN=255 frequency is 2.609%; it is not established sensor saturation. The source is documented as an 8-bit sigma-nought-dB representation, but the current records do not establish byte-to-dB calibration or linear power. The B1 v2 HR target is noisy and normalized, so it is a pseudo-target. See the [profile](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/report.md), [target audit](REFERENCE_TARGET_AUDIT.md), and [degradation audit](DEGRADATION_MODEL_AUDIT.md).
- **Verified validation fact:** Thirty-eight selected synthetic evaluator, normalization, FFT, and Gamma tests passed in 8.24 seconds under the AISAR CPU environment. This did not execute a model or access project/test images and does not establish physical validity or model quality. See the [validation record](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/validation.txt).

- **Verified gate fact:** The prospective source-group B1 v2 screen remains `not_promoted` with next phase `stop_at_b1`. Do not enter B2/B3 or alter any historical gate. The result does not show that optical information is intrinsically useless. See [CURRENT_STATE](CURRENT_STATE.md) and D-2026-10-28-003.
- **Current manuscript review:** The root July paper directory is an internal engineering pilot, not current submission evidence. Its PDF metadata is dated 2026-07-14; the TeX labels a 100-pair mini subset as held-out/test and contains test-associated results, while the local source-group independence is not established. This review saw test-associated prose/numeric values but used none and opened no payload or result file. Current task flags are true/false; see D-2026-10-29-011 and the [audit](../../experiments/literature_audits/manuscript_draft_status_review_20261029_v1/report.md).

- **Verified inventory fact:** The current related-work matrix has 138 data rows and 12 columns: 78 rows have direct pre-corruption membership evidence and 60 are current-evidence additions with unresolved original membership. The exact original inventory count and remaining titles are unknown; the current row count is not completeness evidence. The Vulli et al. (2026) classification article is the latest matrix addition and does not establish SAR-output restoration. See the [supporting charter audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md) and D-2026-10-28-061 for the earlier Ma et al. restoration record.
- **Verified adjacent optical–SAR diffusion-fusion prior art:** Zhang et al.'s SDIF combines SAR-BM3D preprocessing, variational diffusion image fusion, and FLCNet semantic guidance. The article reports fused-image and classification evaluation, not optical-guided SAR-output restoration or joint despeckling/SR. Equation 20 includes semantic labels in reverse sampling, but their inference-time availability is unclear. This narrows generic optical–SAR diffusion-fusion novelty only; it is not local utility or mismatch-safety evidence. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).
- **Verified flow-posterior literature boundary:** FlowDPS is a general flow-prior guidance method evaluated on natural-image inverse problems with linear Gaussian operators. The 2026 posterior-transport preprint gives an exact source-reweighting identity for deterministic diffeomorphic flow priors and a controlled 2D example where its FlowDPS-style guidance misweights modes; its image-scale solver is explicitly heuristic. Neither source validates the local SAR likelihood or optical-conditioned SAR restoration. See D-2026-10-29-013 and the [source audit](../../experiments/literature_audits/flow_posterior_transport_source_review_20261029_v1/report.md).
- **Verified SAR posterior-method boundary:** Gao et al.'s full arXiv manuscript describes split-Gibbs diffusion sampling under a complex Fourier measurement model with additive Gaussian noise. It adds SAR measurement-domain prior art, not optical-guided raster restoration or a local normalized-PNG likelihood. The real-scene sampling count is inconsistent, and the SLC split/source lineage is unreported. The related IEEE ICASSP record is kept separate because a version link was not established. See D-2026-10-29-014 and the [full-text audit](../../experiments/literature_audits/sar_split_gibbs_posterior_imaging_fulltext_20261029_v1/report.md).
- **Verified adjacent reliability-fusion prior art:** Hou et al. (2026) describe RGSNet for SAR–optical semantic segmentation under heterogeneous degradation in an SSRN abstract. Its linked repository says code will be released upon acceptance and currently exposes only a README. This is abstract-level fusion prior art, not SAR-output restoration or misregistration-safe fallback evidence. See D-2026-10-28-045 and the [audit](../../experiments/literature_audits/rgsnet_reliability_guided_sar_optical_segmentation_20261028_v1/report.md).
- **Verified registration prior-art fact:** Lian et al. (2026) propose structure-confidence-weighted phase congruency and coarse-to-fine optical–SAR matching. The evidence is limited to official publisher-indexed text because direct HTML, PDF, and XML routes were unavailable. This is registration prior art only; it does not establish SAR-restoration fidelity, restoration-time optical utility, or mismatch-safe fallback. See D-2026-10-28-030 and the [bounded audit](../../experiments/literature_audits/registration_structure_confidence_phase_congruency_20261028_v1/report.md).
- **Verified OSR-Net citation fact (publisher-indexed):** The MDPI record lists nine authors. Its data statement points to arXiv:2502.01002, which the primary arXiv record identifies as a separate MultiResSAR review by Wenfei Zhang et al., not an OSR-Net manuscript. The OSR-Net low-resolution subset size, split, independent-reference protocol, and code/run lineage remain unresolved; registration metrics do not establish SAR restoration utility. See D-2026-10-28-059 and the [source audit](../../experiments/literature_audits/osr_net_publisher_indexed_text_update_20261028_v1/report.md).
- **Verified adjacent-restoration prior art:** De Paepe et al.'s Sim2Real paper describes MOCEM-trained joint SAR despeckling and sidelobe reduction. Its reflectivity target is explicitly not physically measurable, the background maps are contextual rather than modeled radar backscatter, and real-image sidelobe assessment is qualitative. It does not test SR or optical guidance and is not a task-matched local baseline. See D-2026-10-28-040 and the [full-text audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).
- **Verified adjacent misalignment and despeckling prior art:** Ma et al.'s 2026 arXiv preprint uses a ±2-pixel shift-invariant loss for SAR flood predictions against S2-derived labels and a separate SAR-only CVAE despeckler. Its held-out S2 test performance was used to select patching/stride, and the S1/CVAE data-release terms remain unverified. It does not test optical-guided SAR restoration, joint SR, or a safe fallback. See D-2026-10-28-061 and the [review](literature_review.md#registration-error-sensitivity-fusion-and-confidence-cues).
- **Verified literature-crosswalk fact:** The 2026-10-28 bounded crosswalk added Capella SAR SR, 3MOS, SOMA-1M, and conditional SAR400 coverage as current-evidence additions. Their task/split limitations do not establish optical-guided restoration utility or source-group independence; see the [crosswalk audit](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).
- **Verified dataset-source fact (conditional):** The official IEEE DataPort record for DPS1-SAR lists 12,600 GeoTIFF images across six SAR-input and temporal mean/median-reference packages. File access requires a subscription; the descriptor's described construction does not establish input-excluded references or source-group-disjoint splits. It is adjacent SAR-only despeckling prior art, not optical-guided SR evidence or a local task-matched baseline. See D-2026-10-28-042 and the [source audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).
- **Verified provenance boundary:** The local numeric 2,000-pair PNG subset remains unidentified. Official MediaTUM metadata and archive listings provide no local file-to-source crosswalk; the retained hash audit found zero byte-identical matches for 3,600 fixed train/validation inputs. Transformed copies remain possible. A fresh read of the author-maintained QXS-SAROPT README found a questionnaire-based download link and the paper, but no file-level source map. No new split or model-selection work is justified from numeric names or patch geometry; see D-2026-10-28-025.
- **Verified device fact:** The direct-fusion v2 run completed on CPU because its frozen protocol and runner fix the device to `cpu`, and the runner aborts when `torch.cuda.is_available()` is true. Its successful completion therefore implies that this process did not see CUDA; the run manifest lacks a preflight record to explain why. This was not a silent automatic fallback. The same-input v3 repeat completed on `cuda:0`, and a descriptive scalar cross-device audit is recorded. Future model-forward work must use an approved CUDA profile, a fresh CUDA preflight, and explicit CUDA selection; defer if unavailable. The current ordinary shell cannot import PyTorch or reach the NVIDIA driver. No model process was active in this continuation. See the [cross-device audit](../../experiments/diagnostics/b1_v2_cpu_cuda_output_reproducibility_20261027_v1/report.md).
- **Verified repository-history fact:** The public author-linked Muzeau repository has 14 reachable commits. All seven config-bearing commits set `TRAIN.EPOCH: 10`, three earlier config revisions omit an epoch setting, and no versioned 100-epoch setting was found; the paper's 100-epoch run has no recovered run-bound config/checkpoint mapping. Keep the paper-to-run lineage unresolved. See D-2026-10-28-038 and the [version-history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).
- **Verified connector fact:** Three additional focused metadata-only Google Drive queries for source-group mapping and subset-generation records returned backend `Unknown tool` with no metadata. This is connector failure, not evidence of source-record absence. Keep the local numeric 2k source/group mapping unresolved; see D-2026-10-28-037.
- **Current connector recheck:** A focused metadata-only search and `list_drives` probe both returned backend `Unknown tool`; CUA reported no apps or browsers. No source metadata was returned. Keep the cohort unmapped and model selection paused; see D-2026-10-28-060 and exposure group 512.
- **Bounded cleanup observation:** The 2026-10-28 registered-root cleanup follow-up removed six generated Python cache directories (six bytecode files, 92,505 bytes) and found no additional disposable array, raster, temporary, or smoke-result payload in its audited roots. High-value manifests, configurations, audit records, and cited results remain. This is bounded to those registered roots, not a claim that every workspace file was classified.
- **Verified literature-text fact:** Wiley's OMA-SSR Sections 2.4 and 3.1 say, respectively, that the whole test set supplies training loss and that separate train/test splits were selected. The written protocol therefore uses test targets in training; actual code behavior and possible wording/translation error remain unverified. Treat its test metrics as non-independent unless clarified. See D-2026-10-28-017.
- **Exposure fact:** The OMA source-review task and group 473 manuscript-source review surfaced historical mini-pilot test-metric prose. It was excluded from claims, model selection, and tuning; no test image, payload, or metric file was opened in group 473. Current group 473 flags are `true/false`; cumulative project flags remain `true/false`. Continue to exclude the 100 pilot identities from selection/tuning. See D-2026-10-28-018/031 and exposure groups 462/464/473/474.
- **Current exposure fact:** Group 489 records this continuation’s GPU/runtime and literature-matrix crosswalk review. Historical test-associated text surfaced in continuity records and was excluded; no local test image, metric value/file, prediction, checkpoint payload, or dataset payload was opened or used. Current flags are `test_set_accessed=true`, `test_set_used=false`; cumulative project flags remain `true/false`.
- **Manuscript fact:** `paper/MMCANet_formal/` is an archived 2k preview after the train/validation overlap audit; `paper/MMCANet/` is an internal July engineering pilot. Neither is current submission evidence. B1 remains `not_promoted`; do not refresh publication claims from the historical pilot or current failed gate.
- **Bibliographic-only fact:** Crossref confirms the DOI/title/journal metadata for the 2026 *Measurement Science and Technology* record `10.1088/1361-6501/ae8686`. IOP HTML/PDF requests currently return bot-manager challenge HTML; OpenAlex lists only the IOP PDF and no repository copy. Its CC BY-NC-ND label is not verified against the article-specific publisher terms; Crossref's license URI resolves to general IOP copyright/permissions guidance. Technical content and exact reuse terms remain unverified. No matrix row was added; see D-2026-10-28-014/015/016 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).


## Current compute fact (2026-10-27)

The direct-fusion diagnostic completed on its predeclared CPU protocol. Future model-forward/training work requires a successful GPU preflight and explicit CUDA use; defer compute if the driver is unavailable.

## Matrix-recovery lower-bound correction (2026-10-27)

The 119-row pre-corruption read was taken before D-2026-10-25-46 added six bibliography rows. Therefore 119 is not the final known pre-corruption inventory size: at least 125 rows existed after that update, and later changes may have increased the total further. The 46-row partial snapshot is missing at least 79 rows against this lower bound. The current 100-row matrix has 76 confirmed historical members and 24 entries whose historical membership remains unresolved; at least 25 original entries remain unaccounted for. A 25–49 gap applies only if the final original inventory had exactly 125 rows. See D-2026-10-27-006 and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).

## Matrix-integrity fact (2026-10-26)

A pre-corruption read-only audit confirmed 119 data rows and 12 columns in `related_work_matrix.csv`; the current partial file contains 46 rows, so at least 73 records are absent. An identifier-only scan found 67 unique DOI strings in four research records that are absent exactly from the partial matrix. Those are reconstruction leads only, not proof of original-row membership. Seven known missing rows (six entries added by D-2026-10-25-46 and one earlier SFT entry) are reconstructed in a separate sidecar, but at least 66 records remain unresolved. See the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md) and [candidate index](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/recovery_candidate_identifiers.csv).

Updated: 2026-10-25

## Canonical controlled-degradation semantics (2026-10-25)

- **A. Verified facts:** `MMCANet/src/degradation.py` is the shared LR synthesis implementation. It performs FFT low-pass, optional anisotropic blur, bilinear downsampling, a nonnegative policy, optional Gamma multiplication, and optional output clipping. Its default speckle domain is `representation`; the sampler uses `Gamma(concentration=L, rate=L)`, and the amplitude option uses the square root of that multiplier. Thirteen focused synthetic tests pass for canonical import identity, log normalization, FFT behavior, Gamma moments, and blur behavior.
- **B. Bounded observation:** The optional blur is applied in image x/y coordinates with zero padding. Synthetic tests confirm normalized centered-impulse response, rotated x/y spread, and constant-input edge attenuation. Its physical range/azimuth mapping depends on the source-product orientation, and the observed border behavior is not a validated physical rule.
- **D. Open question:** The normalized PNG-derived SAR representation is not established calibrated amplitude, intensity, or power. Gamma moments do not identify local physical ENL; the LR is derived from an already noisy pseudo-target. Optional blur orientation/boundary assumptions need protocol-specific validation if enabled.
- **E. Unsupported claim:** Current synthetic tests do not establish a physical speckle law, real-SAR denoising fidelity, or sensor-limited SR model.
- **Evidence:** [D-2026-10-25-169](DECISION_LOG.md#d-2026-10-25-169--characterize-optional-blur-with-synthetic-tests), [D-2026-10-25-168](DECISION_LOG.md#d-2026-10-25-168--bound-the-canonical-degradation-to-representation-domain-evidence), the [blur audit](../../experiments/data_audits/anisotropic_blur_characterization_20261025_v1/report.md), the [canonical audit](../../experiments/data_audits/canonical_degradation_semantics_20261025_v1/report.md), and [`tests/test_degradation_and_model.py`](../../tests/test_degradation_and_model.py).

## Manuscript and artifact scope (2026-10-25)

- **A. Verified facts:** `paper/MMCANet_formal/README.md` marks its 2k submission preview archived as of 2026-10-22. The July `paper/MMCANet/` source is classified as an internal historical pilot by D-2026-10-25-166. The root README now labels both directories as archives, not current submission manuscripts.
- **B. Bounded observation:** The registered metadata-only artifact inventory found no additional clearly disposable cache, bytecode, smoke, scratch, failed, partial, or replayable-array files. No additional deletion was made in the current pass.
- **D. Open question:** A current manuscript requires admissible prospective evidence; local source/group provenance and fine registration remain unresolved, and the current B1 gate did not promote optical use.
- **E. Unsupported claim:** The archived fixed-set/held-out results do not establish independent generalization, current model promotion, or a submission-ready method.
- **Evidence:** [D-2026-10-25-167](DECISION_LOG.md#d-2026-10-25-167--classify-both-paper-directories-as-historical-archives), the [scope/access audit](../../experiments/data_audits/paper_directory_scope_20261025_v1/report.md), the [formal archive notice](../../paper/MMCANet_formal/README.md), and the [artifact inventory](../../experiments/artifact_cleanup_current_inventory_20261025_v1.json).

## Sentinel-1 product geolocation literature (2026-10-25)

- **Primary-source fact:** Schubert et al. (2015) report DGPS-surveyed corner-reflector calibration for S1A StripMap SLC/GRDF products. Their 2017 S1A/B follow-up reports IW SLC and GRDH absolute-location-error time series and identifies beam/subswath-dependent azimuth offsets. The 2017 article's Swiss-site table reports IW SLC range/azimuth mean±SD of 0.15±0.19 m / 1.44±1.19 m, with a proposed SLC azimuth correction of 0.00±0.55 m; IW GRDH reports 0.20±0.24 m / 0.31±1.31 m.
- **Evidence boundary:** These are product-series absolute-location statistics on surveyed calibration sites. Neither article reports the local SEN1-2 S1-to-S2 mosaic residual or identifies the exact fall products/baselines used in the local cohort. The 2017 article was reviewed through publisher-indexed primary text because direct XML retrieval returned HTTP 429.
- **Open project fact:** Fine pixel co-registration of the local 2017 SEN1-2 GRD IW/VV and S2 mosaics remains unverified; exact product crosswalk and independent common-reference checkpoints are not available. Do not convert absolute ALE statistics into a local shift or B2 condition. See [D-2026-10-25-129](DECISION_LOG.md#d-2026-10-25-129--keep-sentinel-1a-absolute-geolocation-calibration-separate-from-local-s1s2-residuals), [D-2026-10-25-130](DECISION_LOG.md#d-2026-10-25-130--record-quantified-iw-absolute-geolocation-without-transferring-it-to-local-s1s2-registration), and the [2015](../../experiments/literature_audits/sentinel1_geolocation_accuracy_primary_20261025_v1/report.md)/[2017](../../experiments/literature_audits/sentinel1_iw_geolocation_followup_20261025_v1/report.md) source audits.


## CDSE sidecar authentication path (2026-10-25)

- **Observed:** The selected S1A Item record returned HTTP 200 and four calibration/noise asset links with URI scheme `s3` and bucket `eodata`. No object key or full href was retained. A five-item stratified sample and its 11 target sidecars are frozen; zero XML-sidecar HTTP GETs were issued.
- **Official access requirement:** CDSE S3 access requires a registered account and generated credentials; OData product downloads require an authorized token. The active process has no AWS/CDSE credential variable, and no S3 SDK/CLI is installed.
- **Boundary:** Zero XML bytes or image/pixel data were accessed. This explains an access block; it does not establish missing or invalid calibration metadata. Keep the route paused until a secure authorized credential source is available. See [D-2026-10-25-105](DECISION_LOG.md#d-2026-10-25-105--pause-cdse-sidecar-fetch-until-authenticated-access-is-configured) and the [access-path audit](../../experiments/data_audits/cdse_sidecar_schema_review_20261025_v1/report.md).

## CDSE candidate asset metadata and repeated S1 acquisitions (2026-10-25)

- **Verified catalog fact:** All 80 frozen STAC Item metadata requests succeeded (45 S1 GRD, 35 S2 L2A). S1 items expose VV/VH measurement COG descriptors and calibration/noise XML entries; S2 items expose 10 m B02/B03/B04 reflectance descriptors, SCL, and product metadata. No asset href was retained/followed and no product file or pixel was accessed.
- **Rights fact:** Copernicus Sentinel's legal notice and CDSE terms describe general free/full/open availability with source attribution and modified-data wording; EO access through CDSE requires registration. This does not establish that authenticated product assets were downloaded or inspected.
- **Metadata observation:** The deterministic full-coverage reanalysis of 12 AOIs finds at least one S1 group with 3+ distinct datatakes sharing platform, orbit direction, and relative orbit at each AOI. Longest group spans are 47–59 days. This is a candidate temporal-reference route only; temporal stability, speckle independence, calibrated values, and clean-target quality remain unknown.
- **Open processing risk:** Candidate S1 product types include `IW_GRDH_1S` and `IW_GRDH_1S_C`; resolve catalog/processing harmonization before freezing a mixed-platform cohort. The item-level S2 cloud percentage is not an AOI-local mask; SCL pixels were not read.
- **Boundary:** No split, local data, test pixels/metrics, model, training, inference, or gate changed. See [D-2026-10-25-104](DECISION_LOG.md#d-2026-10-25-104--bound-cdse-asset-metadata-and-repeat-stack-evidence), the [audit](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/report.md), and [repeat summary](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/s1_repeat_summary.json).

## CDSE acquisition/footprint graph (2026-10-25)

- **Metadata fact:** The 19 frozen candidate product pairs deduplicate to 17 Sentinel-1 GRD and 12 Sentinel-2 L2A item IDs; all 29 exact CDSE STAC item requests returned HTTP 200 with valid Polygon geometry.
- **Observed graph:** Pairing and shared-datatake edges form nine components. Adding 141 positive-area product-footprint overlap edges connects all 29 products and all four anchors into one component; the result is unchanged at 1 m², 100 m², and 1,000,000 m² thresholds.
- **Boundary:** This rejects the four-anchor product pool as a group-disjoint cohort under the frozen product-footprint rule. The full product footprints may conservatively merge future small AOIs; this is not a general CDSE viability result and does not establish a split, clear patch, pixel alignment, calibration, independent target, or reuse rights. No assets/pixels or local/test data were accessed; B1/B2/B3 remain unchanged. See [D-2026-10-25-102](DECISION_LOG.md#d-2026-10-25-102--reject-the-four-anchor-cdse-pool-as-a-group-disjoint-cohort) and the [graph audit](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/report.md).

## SEN12TS temporal-reference feasibility screen (2026-10-25)

- **Source/document fact:** Official SEN12TS documentation maps current VV to S1 band 2, prior VV dates to bands 6/10/14, the S1 validity mask to band 19, S2 reflectance to bands 1–12, SCL to band 13, and the cloud/shadow mask to band 14.
- **Observed screen result:** A frozen 43-tile train/development pilot plus a second-rank tile from 38 eligible metadata-linkage components passed decode, value-domain, schema, and `<1%` invalid-mask coverage checks. The supplement transferred 259,190,932 bytes in 76 exact HTTP 206 requests and saved no rasters.
- **Evidence limit:** Median local-mean change was similar across the two samples, component-paired changes varied, and normalized-residual correlations remained positive. The deterministic linkage components are not verified parent-scene IDs; this screen does not establish independent speckle, clean SAR truth, or generalization. Stop further sampling for this temporal-reference question; no training, model selection, or B1/B2/B3 gate changed. See [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit) and the [audit report](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/report.md).

## External prior-art update — MD-Glow (2026-10-25)

- **Indexed-text fact:** MD-Glow trains a shared invertible Glow representation with separate SAR despeckling and SR feature-translation tasks; its synthetic training target derives from SEN1-2 optical imagery with single-look Gamma speckle.
- **Bounded result:** The quantitative comparison is despeckling-only. The paper reports its highest listed PSNR/SSIM on the synthetic LR set and second-best results on the original-resolution set; multi-task ablation gains vary with resolution/metric. Its SR example is qualitative.
- **Boundary:** No optical input at inference, source-group separation, quantitative joint-SR advantage, mismatch/fallback, or local-protocol result is established. CVF direct access returned HTTP 403; the review used indexed paper-body excerpts. See [D-2026-10-25-08](DECISION_LOG.md#d-2026-10-25-08--record-md-glow-as-sar-multi-task-despeckling-sr-prior-art).

## External prior-art update — MuChaPro-based InSAR regularization (2026-10-25)

- **Abstract-level fact:** The 2026 JSTARS paper proposes spatial regularization to improve phase/coherence estimation from despeckled multichannel interferometric projections; it reports simulated interferograms and real TerraSAR-X/PAZ experiments.
- **Boundary:** The regularizer, objective, numerical metrics, and full protocol were not reviewed. This is InSAR phase/coherence prior art, not optical-guided intensity-SAR despeckling/SR or evidence for the local forward model.
- **Access/project impact:** IEEE direct access returned HTTP 418 and the HAL record fetch was unavailable. No local data, test material, model, or experiment changed. H9 remains blocked on source/product/domain provenance; B1 and B2/B3 status are unchanged. See [D-2026-10-25-07](DECISION_LOG.md#d-2026-10-25-07--bound-muchapro-insar-spatial-regularization-prior-art).

This document is a compact index of the repository's current scientific facts and evidence boundaries. The single status authority remains [CURRENT_STATE.md](CURRENT_STATE.md); the detailed claim audit remains [claim_evidence_map.md](claim_evidence_map.md). This inventory does not create a new result, change a gate, or replace the linked run manifests.

The categories below follow the project plan: verified facts, bounded observations, hypotheses, open questions, and unsupported claims. A result is only valid for its recorded protocol and sample population. In particular, current development partitions and noisy pseudo-targets do not establish independent generalization or clean-SAR recovery.

## A. Verified facts

### Research and code

- The research objective is optical-guided joint SAR despeckling and super-resolution. Misregistration robustness, optical utility, and any proposed mechanism are questions to test, not established contributions. See [AGENTS.md](../../AGENTS.md) and the [hypothesis register](research_hypotheses.md).
- The current MMCANetV2 implementation has a 2× SwinIRJoint SAR base and an optical feature path that contributes through a learned spatial gate and high-pass residual. The code also exposes optional local cross-attention and alignment components. Their presence in code does not show that the trained model uses them effectively or safely. See [MMCANetV2/model.py](../../MMCANetV2/model.py) and the variant-specific [B1 v2 protocol](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.md).
- The checkout is not a clean source snapshot: HEAD is fc1d2e5 and the worktree contains extensive modified and untracked source, documentation, and experiment files. A commit ID alone does not reproduce the current tree. The sealed current-source B1 v2 run bound 26 dependencies by hash; all matched at sealing, while later runner maintenance changed two current source files. See the [run source manifest](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/source_hashes.json) and [post-run audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).

### Data and evaluation

- The local numeric 2,000-pair subset is represented by single-channel SAR and RGB optical 8-bit PNGs. Project materials label it SEN1-2, but the upstream archive/version and parent-scene mapping are not established. It is representation-domain data, not verified calibrated amplitude, intensity, or power. See [CURRENT_STATE.md](CURRENT_STATE.md) and the [spatial-overlap audit](sen1_2_train_val_spatial_overlap_audit.md).
- **Primary-source publication fact, not local-file attribution:** The SEN1-2 author manuscript describes GRD IW VV σ⁰ dB, orbit/DEM orthorectification, and no speckle filtering; Sentinel-2 RGB bands 4/3/2 are mosaicked by ROI and contrast-stretched after export. At the time, it says GEE held Sentinel-2 L1C, with TOA rather than atmospherically corrected BOA reflectance. The authors call the release co-registered and say scene/season subsets can support independent splits, but give no independent S1-to-S2 checkpoint residuals. Its 256×256, stride-128 tiling produced 298,790 pairs before manual screening left 282,384. None of these statements identifies the local PNG release or verifies its preprocessing or pair accuracy. See D-2026-10-26-136, D-2026-10-26-137, the [author manuscript](https://arxiv.org/html/1807.01569), and [source crosswalk](../../experiments/data_audits/dataset_lineage_source_paper_crosswalk_20261026_v1/report.md).
- The local fixed train/validation partitions have exact spatial/source/target overlap: 580 train–validation geometries affect all 200 validation identities. B1F fit/inner-dev also has 440 exact target overlaps affecting 159/160 inner-dev identities. These partitions are not independent validation for model selection or promotion. See the [overlap audit](sen1_2_train_val_spatial_overlap_audit.md).
- The separate 1,800-pair prospective cohort was matched to official SEN1-2 source pixels and divided by 15/15 documented source folders. Its validation groups are all fall-season within one ROI-season; acquisition dates and fine pixel co-registration remain unverified. This cohort does not establish broad geographic, seasonal, or acquisition generalization. See the [cohort audit](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md) and [timing/registration audit](../../experiments/literature_audits/sen12_pair_timing_registration_20261023_v1/report.md).
- The registered B1 v2 degradation is a controlled representation-domain 2× stress protocol applied to PNG-derived noisy pseudo-targets. It is not a physically calibrated SAR observation model. Exact steps and metric definitions are in the [protocol](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.md) and [evaluation audit](evaluation_audit.md).

### Current-source B1 v2 result

- The frozen protocol completed six runs: three paired seeds, SAR-only reference and C2 optical candidate, with 200 validation identities nested in 15 source folders. The gate returned not_promoted / stop_at_b1 because mean paired ΔPSNR was +0.001617 dB, below +0.05 dB, and mean ΔSSIM was −0.000892, below zero. All three seed PSNR deltas were positive; the other declared guardrails passed. The unchanged gate does not use the descriptive bootstrap interval to override this decision.
- The aggregate and run manifests record test_set_used=false, test_set_accessed=false, and candidate_test_enumerated=false. These flags describe this six-run B1 v2 matrix only. They do not mean that the entire repository has had zero historical test access; prior access incidents are disclosed in [CURRENT_STATE.md](CURRENT_STATE.md) and [RESEARCH_RISKS.md](RESEARCH_RISKS.md).
- No current optical-guided method has passed its declared promotion gate. B2/B3 remain gated. See the [aggregate report](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/aggregate/report.md), [aggregate manifest](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/aggregate/aggregate_manifest.json), and [run registry](EXPERIMENT_REGISTRY.md).

## B. Bounded observations

- A train-only exact-basename audit found 1,467/1,800 local numeric train/validation names in ReFlowSET's current QXS-SAROPT training list, including 157 local validation names. This is a downstream split-membership clue from an unpinned `main` snapshot; it does not authenticate the local dataset/version, identify parent scenes, or confirm leakage. The 333 local names absent from the train list remain unclassified because no other list was accessed. See the [audit](../../experiments/data_audits/reflowset_numeric2k_train_membership_20261024_v1/report.md).
- The current OGSFlow README says its Stage-2 source-noise scale is calibrated on `val` after S1 selection, frames it as a perception–distortion control, and names HR-trained DeepLabV3/SegFormer downstream evaluation. It also maps stage/fusion names to model files and advertises a CPU synthetic-data smoke command; source contents remain inaccessible and no command was run. Its paper citation and linked protocol docs remain unavailable. Whether calibration is separate from later selection/reporting is unknown; semantic task metrics do not establish SAR-output fidelity. This is repository-level prior-art evidence, not a verified joint-restoration protocol. See the [audit update](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md).
- A maintainer-controlled SEN12MS-CR-TS download helper exposes region- and modality-sharded archives, with test archives named separately; the reviewed route contains no standalone train-only metadata crosswalk. The official article copy licenses the paper under CC BY 4.0 and points to the dataset website, but does not state a distinct dataset license. The public author repository root listing contains no `LICENSE` file. This closes only the examined metadata-only route; the candidate remains conditional. See the [route audit](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/report.md).
- The historical B1 mean ΔPSNR was −0.004219 dB across three seeds. B1R reported Δcosine −0.001855 and Δcorrelation −0.001981. B1F seed 0 reported inner-dev ΔPSNR +0.019419 dB, ΔSSIM −0.001381, and TBR-error change +0.021978; it did not pass its gate. These historical values and gates remain unchanged.
- On the current-source B1 v2 cohort, PSNR was positive on 129/200 identities after averaging the three seeds, while SSIM worsened on 192/200 and edge-strength ratio error worsened on all 200 identities in every seed. This describes behavior against the noisy pseudo-target; it does not identify a cause or establish clean-SAR structural harm. See the [per-image audit](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/report.md).
- The C2 gate failure's direct cause remains unidentified. Objective/selection tension, residual geometry or timing, the pseudo-target/degradation, and frozen-backbone capacity remain competing explanations; the existing validation reports do not isolate them. See the [failure-cause synthesis](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/report.md).
- B1 v2's source-group bootstrap rederivation reproduced its retained summaries. It resamples 15 groups conditional on three fitted seeds from one fall ROI-season and is descriptive, not a substitute for the frozen gate or broader uncertainty. See the [bootstrap audit](../../experiments/diagnostics/b1_v2_source_group_bootstrap_rederivation_20261023_v1/report.md).
- A failed-screen B1F checkpoint's matched-optical pairing diagnostic showed a small PSNR increase over SAR-only while several structural and target-region metrics worsened. The matcher-entropy follow-up failed its predeclared all-eight-positive direction rule. These are one-checkpoint diagnostics on repeatedly used contaminated validation data, not confirmation or a promoted method. See [optical_information_analysis.md](optical_information_analysis.md).
- The loss/objective audit found validation loss decreasing while SSIM and the frozen selection score decreased; this is an observed tension, not a demonstrated cause or evidence for a same-cohort retune. See the [objective/code audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).
- A zero-training, three-seed residual diagnostic on the same 200 B1 v2 validation identities found mean actual-residual ΔPSNR +0.001621 dB and ΔSSIM −0.000893. A target-assisted pixelwise oracle improved those pseudo-target metrics more, but cannot provide an inference-time gate. The diagnostic was mixed/inconclusive and does not promote C2 or change the B1/B2/B3 state. See the [residual diagnostic](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/report.md).

- A predeclared follow-up found positive per-seed rank association between target-free mean C2 gate activation and per-image actual-minus-base strict-score change (rho +0.868, +0.859, +0.838; mean +0.855; source-folder cluster-bootstrap 95% interval [+0.726,+0.911]). Per-seed correlations with ΔPSNR/ΔSSIM were weak or changed direction. This is a bounded composite-score rank observation on the repeatedly used validation cohort, not calibrated reliability or a multi-metric utility result; no threshold or gate changed. See the [protocol and result](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/).
- Publisher-indexed sections of Guo et al.'s ECDM article describe SAR-only conditional-diffusion despeckling, synthetic training pairs with optical targets, real fine-tuning against a multi-date SAR average, and no-reference real-scene metrics. This establishes published SAR despeckling prior art, not optical-at-inference guidance, joint SR, or clean-reference validation. The split grouping, input-date membership in the average, and product-transfer details remain unresolved; the direct article/PDF routes were inaccessible in this review. See the [audit](../../experiments/literature_audits/ecdm_sar_despeckling_fulltext_audit_20261024_v1/report.md).
- Publisher-indexed sections of Ma et al.'s 2026 DualCDM article describe SAR-conditioned diffusion that generates optical images, with a dual-domain architecture and optical-domain evaluation. The linked paper/code repository roots were visible; direct code files and a code-to-table revision binding were not verified. Its reported random patch splits do not establish source-group independence. This is reverse-direction translation prior art, not optical-guided SAR restoration or joint SAR despeckling/SR. See the [audit](../../experiments/literature_audits/dualcdm_sar_to_optical_audit_20261024_v1/report.md).

## C. Research hypotheses, not conclusions

- Correctly paired optical may add useful value over a task-matched SAR-only joint baseline in some conditions; neither a stable benefit nor universal lack of value is established.
- Geometry, edges, low-frequency structure, texture, and semantics may differ in cross-modal reliability. Which cues help and when remains untested locally.
- Utility may vary with degradation severity and task definition; a monotonic increase with severity is not assumed.
- A reliability mechanism may reject geometric or content mismatch, but a gate or attention map is not evidence of calibrated reliability or safe fallback.
- Joint restoration may or may not outperform a fair cascade. Diffusion or physics-constrained approaches may be worth studying only when the research gates, data, and compute support a discriminating experiment.

The full, versioned hypothesis and decision records are in [research_hypotheses.md](research_hypotheses.md), [restoration_order_analysis.md](restoration_order_analysis.md), and [diffusion_feasibility.md](diffusion_feasibility.md).

## D. Open questions and risks

- What upstream archive/version, license, parent scenes, and source-group identities produced the local numeric 2,000-pair subset?
- What are the SAR/optical acquisition dates, residual registration error, product levels, physical units, and exact preprocessing for the local pairs and the authenticated 1,800-pair cohort?
- What independent, group-disjoint development and final evaluation data can support model selection after the current B1 stop?
- Can SEN12TP's package rights and train-archive mapping be resolved? Its mutable author loader exposes georeferencing on training TIFF windows, which could support a train-only spatial-overlap audit; repeated dates must stay grouped by parent `roi_id`, and this is not yet an authorized or validated split.
- What caused the B1 v2 metric tradeoff? Current evidence does not separate target/degradation limits, objective-selection tension, optical transfer, registration, or optimization as a sufficient cause. Retuning on the current contaminated partitions is not a valid way to resolve it.
- The original B1 source snapshot remains unauthenticated against the current checkout. Historical metrics are preserved but cannot be silently rebound to current source.
- Several relevant literature protocols and code lineages remain unresolved, including DSRKD, the exact Corley et al. paper-to-current-main sweep binding, whether the secondary-indexed CN120450996A disclosure is the GCOANet paper's implementation, and the completeness/rights of Onat's linked PLFSID release. The matrix records source-level limits; no abstract-only result or patent-index rendering should be upgraded to a confirmed protocol. See the [Corley audit](../../experiments/literature_audits/corley_pretrained_matcher_fulltext_20261024_v1/report.md), [GCOANet patent-route audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md), and [Onat route audit](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/report.md). A newly visible OGSFlow README also lacks a paper citation and detailed source-level protocol; its RGF/flow description is a repository-level prior-art lead only. See the [OGSFlow audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md).
- The project has documented earlier local test-array access through split tests and historical aggregate test-result text exposure. Keep the project-level statement precise: this B1 v2 run had false test flags, but the full repository does not have a zero-test-access history.

See [NEXT_ACTIONS.md](NEXT_ACTIONS.md), [RESEARCH_RISKS.md](RESEARCH_RISKS.md), and the source-specific audits for active work and restrictions.

## E. Unsupported or contradicted claims

- “First joint SAR despeckling and super-resolution” is contradicted by direct SAR-only and optical-guided joint/restoration prior art, including ISSMSAR. See the [literature review](literature_review.md), [related-work matrix](related_work_matrix.csv), and [ISSMSAR audit](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md).
- Stable optical gain over a matched SAR-only joint baseline is not supported by the recorded gates.
- Misregistration robustness, stale-content rejection, local fallback, and calibrated optical reliability are untested or gated.
- Recovery of clean, calibrated, or physically faithful SAR is unsupported by the local 8-bit PNG-derived noisy pseudo-targets.
- General superiority of joint restoration over a strong cascade is not established.
- A local diffusion method or a diffusion advantage is not established.
- Submission readiness is not established. The paper must track a supported method, complete fair experiments, reproducible artifacts, and an independent review before that status can be considered.

## Evidence entry points

- [Current state](CURRENT_STATE.md)
- [Claim–evidence map](claim_evidence_map.md)
- [Experiment registry](EXPERIMENT_REGISTRY.md)
- [Decision log](DECISION_LOG.md)
- [Research roadmap](MASTER_RESEARCH_PLAN.md)
- [Spatial-overlap audit](sen1_2_train_val_spatial_overlap_audit.md)
- [Evaluation audit](evaluation_audit.md)
- [Literature review](literature_review.md)


## External prior-art update — 2026-10-23

SDS-SAR is author-reported self-supervised despeckling from SAR intensity images without clean references, but the inspected current author-repository test path invokes a 3×3 median-filter placeholder and does not load model weights. Record it as SAR-only denoising prior art, not a reproducible local baseline or joint SAR SR/optical-guidance evidence. Publisher full text, data/weight payloads, and reuse terms were not accessed. See the [source audit](../../experiments/literature_audits/sds_sar_primary_source_20261023_v1/report.md).


## External prior-art update — Lattari et al. (2019)

Lattari et al. report SAR-only denoising with synthetic speckle pretraining and temporal-average Sentinel-1 pseudo-reference fine-tuning; their real-domain evaluation lacks a clean reference. Treat this as denoising prior art, not local clean-SAR or joint SR evidence. Exact stack/patch provenance and paper-specific code/data access were not verified. See the [source audit](../../experiments/literature_audits/lattari_sar_despeckling_2019_primary_20261023_v1/report.md).

## External prior-art update — SAR2SAR (2026-10-23)

- **Fact:** The full accepted manuscript describes three-stage semi-supervised SAR despeckling with synthetic Gamma-speckle pretraining, then two change-compensated multitemporal SAR stages. Real-image evaluation lacks clean references and uses ENL/visual inspection; the paper warns ENL favors over-smoothing.
- **Bounded observation:** A related author-group package contains an inference wrapper, model, and checkpoint listing, but the inspected package does not provide a bound training run or checkpoint-to-paper manifest. It is not a matched baseline for local normalized PNGs or optical-guided joint SR.
- **Open:** Product/acquisition IDs, source-group split, stack-to-patch mapping, dataset terms, and package-weight lineage remain unresolved. See the [audit](../../experiments/literature_audits/sar2sar_primary_fulltext_20261023_v1/report.md). No local data or experiment changed.

## External prior-art update — Attention OffsetNet (2026-10-24)

- **Institution-record fact:** University of Seoul lists the 2026 IEEE JSTARS article by Jin Woo Yu and Hyung Sup Jung, DOI `10.1109/JSTARS.2026.3731425`.
- **Author-reported abstract observation:** Patch-offset RMSE is reported as 8.87 px X / 7.99 px Y; full-scene residual after RANSAC aggregation is reported as about 0.8 px X / 0.2 px Y. These are different stages and metrics.
- **Open:** Full article, scene split, offset-label protocol, code, and paper-to-run lineage were not accessible in this screen.
- **Scope:** Registration prior art only; no SAR restoration or local alignment evidence. See the [audit](../../experiments/literature_audits/attention_offsetnet_author_page_screen_20261024_v1/report.md).

## External prior-art update — ProSR (2026-10-24)

- **Manuscript fact:** The full arXiv manuscript describes discrete Gumbel-VQ detail tokens with semantic-prototype-guided MaskGIT for single-image SAR ×4 SR. It motivates the design with the risk of merging nearby scattering peaks under smooth prediction.
- **Benchmark fact:** The Umbra X-band SLC benchmark uses spectral-cropped LR inputs and a 468/34 geographically disjoint train/validation image split. The paper reports per-scene amplitude normalization, clipping, and uint8 quantization, but no separate held-out Umbra SR test split.
- **Reported-result boundary:** ProSR has lower reported PSNR than UPSR and stronger LPIPS/FID, with separate MSTAR classifier-transfer results. These are author-reported, task-specific outcomes; they do not demonstrate exact physical scatterer recovery or local optical utility.
- **Release and reuse:** The author repository says model code is unavailable. The curated dataset card labels the release CC BY-NC 4.0 while Umbra labels its source imagery CC BY 4.0; resolve both terms before considering reuse. No archive or payload was downloaded. See the [full audit](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/access_manifest.json).
- **Scope:** Direct SAR-only discrete-token SR prior art. It does not establish optical-guided joint despeckling/SR, controlled mismatch robustness, or a SAR-only fallback. No local experiment or gate changed; B1 remains `not_promoted`, and B2/B3 remain gated.

## External prior-art update — camera restoration task order (2026-10-24)

- **Paper facts:** Qian et al. (ICCP 2022) study task ordering for Bayer RAW denoising, demosaicing, and SR and report the camera-specific sequence DN→SR→DM. Their joint TENet uses intermediate DN+SR supervision before demosaicing. Xing and Egiazarian (CVPR 2021) compare sequential, pairwise-joint, and fully joint camera restoration under a fixed DN→DM→SR order.
- **Evidence boundary:** These are RGB camera-pipeline studies with camera noise/degradation and demosaicing; they contain no SAR despeckling, optical guidance, or local SAR measurement. Author-reported results were not independently reproduced. See the [audit](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/report.md).

## TUM SEN1-2 catalogue route check (2026-10-24)

- **Catalogue fact:** TUM's current public open-dataset catalogue lists the 2018 SEN1-2 release with 282,384 SAR–optical pairs and links its publication.
- **Open source-lineage question:** The catalogue entry exposes no sample/scene manifest. The expected mediaTUM archive record returned HTTP 429 through the read-only web route. This does not establish that the local numeric 2k files belong to SEN1-2 or recover source groups. See the [route audit](../../experiments/data_audits/tum_sen1_2_record_metadata_route_20261024_v1/report.md).

## External prior-art update — FAF-CD (2026-10-24)

- **Paper and venue:** Wang et al.'s FAF-CD is an arXiv v1 manuscript accepted to the MONTI 2026 CVPR workshop; the workshop site states accepted papers are peer-reviewed and non-archival.
- **Method and task:** It applies feature rectification, deformable EO–SAR fusion, Fourier/Haar branches, and adaptive gating to building-status change detection using pre-event optical and post-event SAR.
- **Stress-test boundary:** Its 697-sample BRIGHT validation stress subset perturbs optical appearance or SAR speckle/radiometry/resolution at five severity levels. The reviewed manuscript reports no controlled geometric shift/warp sweep or SAR-output restoration/fallback evaluation.
- **Classification:** Adjacent feature-fusion and modality-stress prior art; not evidence of optical utility for SAR despeckling/SR. No local data or code was accessed and no project experiment or gate changed. See the [audit](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/access_manifest.json).

## External prior-art update — SEORation (2026-10-24)

- **Paper fact:** SEORation describes RemoteCLIP semantic deduplication followed by RSLIP scene-compatibility filtering to curate SAR–EO retrieval pairs; QXS-SAROPT is described as external to OpenSEP construction.
- **Reported-result boundary:** The QXS results are bidirectional image-retrieval metrics. They do not measure pixel registration or restored-SAR fidelity.
- **Split/provenance boundary:** The manuscript excerpts say validation pools use held-out source-dataset splits, but expose no audited source-group crosswalk or mapping to the local numeric 2k files. Keep local lineage unresolved.
- **Access:** Search-indexed primary manuscript text and official MONTI page reviewed; no local data, code, or archive accessed. See the [audit](../../experiments/literature_audits/seoration_opensep_pair_curation_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/seoration_opensep_pair_curation_20261024_v1/access_manifest.json).
## Gaya et al. optical-assisted SAR despeckling (2023–2024)

- **2023 full-text fact:** The official GRETSI paper describes Fuse-MERLIN early and intermediate optical fusion for complex-SAR MERLIN despeckling. It reports qualitative fine-road gains and a preliminary two-pixel registration test on one co-registered TerraSAR-X/Pléiades Neo scene, while explicitly stating that quantitative validation was not performed. This is not joint SR or a systematic mismatch/fallback study. See the [full-text audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md).
- **2024 abstract fact:** The IP Paris institutional record identifies the IGARSS paper (pp. 2180–2183; DOI `10.1109/IGARSS53475.2024.10642047`) and says optical imagery is used for self-supervised SAR restoration. Its official conference detail page lists no paper resources; the proceedings link redirects to a closed JavaScript-required gateway. The IEEE and HAL full-text routes remain unavailable; its data, target/noise, inference path, registration, metrics, results, and code remain unverified. The reviewed sources do not establish that the 2024 paper is an extended version of the 2023 work. See the [abstract-level/access audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).
- **Project boundary:** Together these works narrow broad optical-assisted/self-supervised despeckling novelty. They do not establish local optical utility, joint SAR SR, controlled mismatch robustness, or quantitative fallback.

## External prior-art update — SpeckleDiff-2D and Gaya thesis route (2026-10-25)

- **SpeckleDiff-2D:** IEEE abstract describes complex-valued SAR-despeckling diffusion trained on synthetic speckle generated from optical imagery; simulated and real-SAR evaluation are reported at abstract level. The inference path and full protocol remain unknown, so this is not evidence for optical-at-inference utility, joint SAR SR, mismatch robustness, or fallback.
- **Gaya route:** Official Télécom Paris pages list Victor Gaya's 2025 SAR-enhancement/multimodal-fusion thesis but provide no direct thesis file. Exact-title/author searches through theses.fr and HAL returned no candidate record in the available search route; this does not prove the thesis is unavailable. The separate 2023 GRETSI Fuse-MERLIN paper does not establish a version link to the abstract-level 2024 IGARSS paper.
- **Boundary:** Literature only. No local data, test material, code, model, checkpoint, metric, or experiment changed. B1 remains not_promoted; B2/B3 and local diffusion remain deferred. See [D-2026-10-25-06](DECISION_LOG.md#d-2026-10-25-06--record-specklediff-2d-and-gaya-thesis-route-boundaries) and the [audit](../../experiments/literature_audits/specklediff2d_abstract_screen_20261025_v1/report.md).

## Standard SEN12MS train-list metadata screen (2026-10-25)

- **Fact:** The author-provided snapshot contains 162,555 unique training rows and 226 provisional ROI/scene keys across four season branches (50/58/77/41). This is filename metadata, not verified geographic separation. The 2019 paper describes 256×256 patches at stride 128 (50% overlap), dual-pol Sentinel-1 σ⁰ dB, and cloud-adaptive Sentinel-2 mosaics. Pairwise acquisition timing, residual registration, and a clean SAR reference were not established.
- **Interpretation and access:** Standard SEN12MS is distinct from SEN12MS-CR-TS and does not map the local numeric 2k subset. Paper-level CC-BY wording and the author repository's research/education-only, non-commercial data license need release-specific reconciliation. No archive or raster was downloaded, no holdout/test-list contents were read, and no local experiment or gate changed. See the [audit](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/access_manifest.json).

## SEN12MS numeric scene-token reuse (2026-10-25)

- **Bounded observation:** In the same hash-bound author training list, 78 of 119 numeric scene/data tokens appear in multiple seasonal branches and five appear in all four. Branch-qualified keys remain 226; bare numeric tokens are not globally unique.
- **Evidence boundary:** This is filename namespace evidence only. It neither proves cross-season geographic overlap nor proves different keys are spatially disjoint. No raster bounds, holdout/test-list contents, or payloads were accessed. See the [follow-up audit](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/report.md).
- **Metadata-route status:** The reviewed author repository files do not expose a train-only geospatial crosswalk or validation list; the listed overview notebook concerns all ROIs and was not opened. A newly surfaced FDL project README documents a bulk GCS sync, not a train-only metadata endpoint; its page body and a bucket listing were not retrievable through the available route, so underlying permissions remain unknown. The official catalog page shows no usable resource/license field, and the TUM record returned HTTP 429. The candidate remains conditional; see the [mirror-route audit](../../experiments/literature_audits/sen12ms_gcs_mirror_metadata_route_20261025_v1/report.md).



## External prior-art update — Amieva et al. SM subaperture enhancement (2026-10-25)

- **Fact:** The separate CVPRW 2026 paper uses three azimuth Doppler subapertures as input and same-acquisition full-aperture intensity as target on the same grid; the target remains speckled. It reports 10 acquisitions with a 5/3/2 split and two full-scene test acquisitions.
- **Observation:** Author-reported PSNR/SSIM measure consistency to the noisy full-aperture reference; ENL increases while iterative refinement lowers PSNR/SSIM. The paper says percentile clipping statistics are computed over the entire dataset before splitting.
- **Interpretation:** SAR-only, same-grid effective-bandwidth/detail enhancement prior art, distinct from the same authors' IW→SM *Acta Astronautica* article. It does not establish optical utility, conventional ×2 SR, geometric robustness, or fallback.
- **Boundary:** ArXiv v1 full text and official CVF bibliographic record reviewed; linked Hugging Face landing page has no card and its viewer fails. No external payload, local data/test material, model, or experiment was accessed. See the [audit](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/report.md) and [D-2026-10-25-17](DECISION_LOG.md#d-2026-10-25-17--classify-amieva-sm-subaperture-enhancement-as-sar-only-prior-art).


## External prior-art update — Dong et al. complex-valued SAR SR (2026-10-25)

- **Fact:** Amieva et al.'s primary arXiv v1 manuscript cites Dong et al. (2025, TGRS 63, pp. 1–14) as complex-valued SAR SR using multi-scale subband information for azimuth detail. An Xidian University-branded index exposes an abstract describing zero-padding alignment, subaperture/subband decomposition, cross-domain learning, hierarchical fusion, and author-reported amplitude/phase improvements.
- **Observation:** The abstract reports 3.78% amplitude improvement and 4.62% better phase accuracy against a competitor; the comparator, metrics, data, and uncertainty were unavailable.
- **Open:** Publisher full text/DOI metadata, exact data/split/scale/metric, speckle objective, code, and run lineage remain unverified. The abstract does not establish joint despeckling, optical guidance, mismatch robustness, or fallback.
- **Boundary:** Literature only; no local data/test material, project code, model, checkpoint, or experiment was accessed. See the [audit](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/report.md) and [D-2026-10-25-18](DECISION_LOG.md#d-2026-10-25-18--classify-dong-complex-sar-sr-as-abstract-level-prior-art).


## MMCANetV2 model-level absent-optical path (2026-10-25)

- **Fact:** MMCANetV2.forward maps optical None to zero optical and false presence. Its optical residual is multiplied by presence; SwinIRJoint returns sigmoid-bounded SAR base, so the no-optical model output equals that base.
- **Fact:** Optional alignment predicts a zero-initialized dense offset bounded by tanh times max_alignment_pixels; optional cross-attention is local-windowed. The optional gate loss target derives from target/optical edge-direction consistency.
- **Interpretation:** This supports an absent-optical interface path only. It does not establish detection or rejection under geometric/semantic mismatch, calibration, optical utility, or data-loader omission of optical payloads.
- **Boundary:** Source-only review; no model execution, data, checkpoint, metric, or experiment. See the [audit](../../experiments/diagnostics/mmcanet_v2_architecture_source_audit_20261025_v1/report.md) and [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness).

## External prior-art update — HSFMamba code-route boundary (2026-10-25)

- **Fact:** The current index in the paper-linked public GitHub repository lists an OGSSR data download, while the Optics-Guided Super-Resolution Algorithm table has placeholder entries and no HSFMamba code link.
- **Open:** This route does not establish whether code exists elsewhere or by request. The linked dataset share, archive contents, and effective data license were not inspected.
- **Boundary:** Public source-page review only; no external implementation/data payload or local data/test material, result, checkpoint, model, or experiment was accessed. See the [full-text/access audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md) and [D-2026-10-25-69](DECISION_LOG.md#d-2026-10-25-69--bound-hsfmamba-implementation-availability-to-the-linked-author-route).

### CDSE global AOI screen 20261025

事实：24 个只读 STAC 查询均成功；在冻结筛选规则下，12 个 AOI 中有 10 个带候选，形成 10 个无跨 AOI datatake 共享的图分量。此结果仅支持进入协议可行性审查，不证明数据权利、产品质量或模型泛化。审计：[report](../../experiments/data_audits/cdse_global_aoi_group_feasibility_20261025_v1/report.md)。

## 2026-10-25 — General misalignment-aware restoration prior art

**Fact:** Shibata et al. (ICCV 2017) propose translated-guidance weighted composition for cross-modal joint filtering; Bungert and Ehrhardt (IEEE Access 2020) jointly optimize inverse-problem reconstruction and structural-side-information registration, with hyperspectral super-resolution among their applications; Hughes and Schmitt (ISPRS Annals 2019) study semi-supervised SAR–optical matching under limited paired data.

**Boundary:** These are registration/filtering/reconstruction prior art, not evidence for SAR despeckling/SR quality, speckle fidelity, optical rejection, or the local model. Generic misregistration-aware reconstruction is not a novelty claim for this project. The Hughes–Schmitt article was checked at official-abstract level only. See the [misalignment source audit](../../experiments/literature_audits/misalignment_robust_reconstruction_prior_art_20261025_v1/report.md) and [bibliography crosswalk](../../experiments/literature_audits/bibliography_matrix_crosswalk_20261028_v1/report.md).


## External prior-art update — DP-SAR abstract-level record (2026-10-25)

- **Fact:** The IEEE indexed abstract lists DP-SAR in the 2026 ICICC proceedings and describes a complex PolSAR SR network with a frozen DnCNN prior, SAPNet, bicubic residual reconstruction, and point-target/main-lobe/Pauli-domain constraints. DOI: [10.1109/ICICC71012.2026.11637669](https://doi.org/10.1109/ICICC71012.2026.11637669).
- **Observation:** The abstract reports RADARSAT-2 evaluation and range/azimuth super-resolution factors; these author-reported results were not independently verified.
- **Unknown:** Accessible author metadata, full text, code/data, source products, target construction, split, and detailed comparison protocol.
- **Boundary:** Adjacent complex PolSAR SR prior art only. No evidence for optical-guided JDSR, misregistration robustness, or the local 8-bit PNG metrics. See the [audit](../../experiments/literature_audits/dp_sar_pol_sar_sr_abstract_screen_20261025_v1/report.md).


## SEN1-2 native cache and prospective split (2026-10-27)

- Observed the sibling SOTANet cache's native train/val filename partition: all 30 filename-prefix groups occur in both, with no exact basename duplicates.
- The separate `sen12_scene_dev1800_v2` cohort assigns 15 source groups to train and 15 to validation; all 1,800 names map to the frozen group IDs and selected pairs are covered by an existing official-source pixel audit.
- This does not resolve the unrelated numeric-name 2k subset or prove acquisition/geographic independence. B1 remains `not_promoted`; B2/B3 remain gated. Details: [audit](../../experiments/data_audits/sen12_native_partition_group_overlap_20261027_v1/report.md).

## 2026-10-30 文献事实补充：联合恢复与可靠性先例

- DSRKD（IEEE TGRS，2024）作者主页列出 SAR 去斑/超分知识蒸馏工作和 IEEE Xplore 链接；ISSMSAR（Remote Sensing，2024）研究多时相 SAR 集成散斑抑制与超分。
- HSFMamba（IEEE JSTARS，2025）直接研究光学引导 SAR 联合超分与去噪；Acta Astronautica（2026）的一项工作以 Sentinel-1 IW/SM 作为同传感器参考，采用非生成式联合恢复。
- OGSRN（2022）以光学空间反馈辅助训练 SAR 超分，推理只需 SAR；OGSFlow 仓库 README 描述光学引导 UAV-SAR 两阶段条件流 SR，但当前仅核验到仓库说明。
- Ada-RefSR（ICLR，2026）研究自然图像参考超分的单步扩散与自适应参考门控；Blockwise Flow Matching、MeanFlow、LLE 为通用少步生成/扩散逆问题求解先例，不是 SAR 验证。
- 相关工作矩阵中的区域专用扩散 SAR JDSR DOI 条目尚未取得出版方正文，本阶段事实状态为“待一手来源核验”。详细证据见[前沿文献对照报告](../../experiments/literature_audits/optical_sar_generative_frontier_20261030_v1/report.md)。


## C1 v2 冻结清单核验事实 — 2026-10-30

- 协议标识：`mmcanet_phase_c1_v2_sen1_2_source_grouped_20261028`。
- 训练/开发身份数为 1,600/200；对应来源组数为 92/19，按 `source_group_id` 的交集为 0。
- 现存块扫描参数为 128×128 窗口、64 像素步长，跨角色匹配签名数为 0；该扫描不覆盖任意位移/窄条。
- 冻结 manifest 记录 `image_payloads_downloaded=false`，载荷审计清单记录 `temporary_payload_retained=false`。
- 来源组元数据未提供地理足迹、轨道或采集日期交叉字段。
- 旧表示审计的协议标识不同，不能作为本划分的像素级验证。
- 本轮仅访问元数据；未读确认集/测试集影像或指标。来源与 SHA-256 见[本轮审计清单](../../experiments/data_audits/mmcanet_phase_c1_train_dev_metadata_boundary_20261030_v1/access_manifest.json)。


## 研究总纲采纳事实 — 2026-10-30

- 用户明确批准采纳候选 `v1.6-proposed` 及其暂定中文题目；正式版本路径为 `docs/research/RESEARCH_CHARTER.md`。
- v1.6 定义六个科学问题和 C1–C6 前瞻路线；它没有重写历史实验结论或自动改变门槛。
- `AGENTS.md` 继续提供工程/数据安全规则，当前阶段由 `CURRENT_STATE.md` 记录；相关同步记录见 D-2026-10-30-011。


## 通用 Flow Matching 后验文献事实 — 2026-11-07

- FlowSGS 的 2026-10-08 arXiv v3 全文描述 Split Gibbs Sampling、Stochastic Interpolants 先验步骤，并包含线性逆问题及非线性 Fourier phase retrieval 实验。
- FMPS 的 2025-08-09 arXiv v3 全文描述预训练 flow 的速度场校正，并包含自然图像超分等通用条件生成实验。
- 两篇来源均未提供本地 SAR/光学/JDSR 效果证据。来源和适用边界见[一手全文审查](../../experiments/literature_audits/flow_inverse_methods_update_20261107_v1/report.md)。

## 2026-11-07：C3-A 固定尺寸合成 CPU 软件核验

- 事实：固定 `batch_size=4`、256×256 输入下，纯合成 CPU 前向、反向、单次参数更新及 8 步 Heun 采样通过；清单记录 CUDA 不可见且未初始化。
- 证据：[运行报告](../../experiments/diagnostics/c3a_full_resolution_cpu_smoke_20261107_v1/report.md)及相邻 `manifest.json`。
- 边界：该结果不涉及项目影像、checkpoint 或真实指标，不能推导训练收敛、重建质量或 GPU 资源可行性。


## 2026-11-07：乘性噪声 SDE 去噪文献事实

作者提供的 ICMLA 2025 会议版 *Perception-based multiplicative noise removal with Diffusion models* 将乘性因子建模为几何布朗运动导出的对数正态扰动，并在 CelebA 与 UC Merced Land Use 图像上评测合成噪声；论文未报告 SAR 数据实验。正式题名与作者另经 ICMLA 官方目录核验。审查及来源见[文献记录](../../experiments/literature_audits/multiplicative_noise_sde_fulltext_20261107_v1/report.md)。
