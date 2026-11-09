# 实验登记

## 局部亮点护栏阈值敏感性解析诊断 — 2026-11-09（合成解析 CPU 方法实验）

- **协议/代码：** [预登记协议](../../experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/protocol.json)、[运行脚本](../../experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/run.py)；命令为 `python experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/run.py`。
- **设计：** 背景均值 0.04；目标均值 `{0.12,0.20,0.30,0.45}`；整数 Gamma 形状 `{4,16,64}`；9 个阈值 `{0.10,…,0.50}`；64×64 单目标、16 draws。按 Gamma 右尾闭式公式计算命中率、单像素误报率及独立像素假设下的整图至少一次误报概率，不生成图像或随机样本。
- **结果：** 共 108 条记录。固定阈值 0.25 的单 draw 命中率范围为 8.72×10⁻¹²–0.999989；整图至少一次误报概率按 Gamma 形状分别为 0.002674、2.19×10⁻²¹、6.39×10⁻⁹³。解析特例、概率边界和阈值单调性检查通过。详见[中文报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/报告.md)。
- **边界：** 只适用于独立像素/抽样的数学 Gamma 构造；整图误报概率依赖该独立性假设。不能迁移到真实 SAR 或模型输出，也不产生阈值建议。不改评分代码、正式协议、C3-A v3、gate 或训练状态。
- **运行范围：** 仅 Python 标准库、CPU 解析计算；PyTorch 未导入，未访问项目图像、模型/checkpoint、confirmation/test 载荷或测量值；没有训练或推理。

## 稀疏亮点退化下的多样本全局评分敏感性 — 2026-11-09（纯合成 CPU 方法实验）

- **协议/代码：** [预登记协议](../../experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/protocol.json)、[运行脚本](../../experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/run.py)；命令为 `CUDA_VISIBLE_DEVICES='' python experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/run.py`。
- **设计：** 64×64 归一化合成图、4 个真实亮点、16 draws、300 次重复；正确、压暗、漏检、右移 1 像素和增加伪亮点五种条件。比较逐像素 CRPS、归一化整图能量分数、邻接变差分数及阈值 0.25 下的局部亮点统计。
- **结果：** 三项全图评分均区分四种错误条件；压暗/漏检/右移/伪亮点的正配对差比例分别为 0.893/1.000/1.000/1.000。精确检出与 3×3 邻域检出区分漏检和错位；真目标区外伪亮点率识别了添加伪点的条件。详见[中文报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/报告.md)。
- **边界：** 仅适用于预设合成图、Gamma 数学随机变量与阈值；不表示真实 SAR 或模型质量，不将 0.25 设为通用阈值。不改评分实现、C3-A v3、阶段 gate 或训练状态。
- **验证/访问：** `py_compile` 通过；1,500 条逐重复记录、有限评分与清单哈希核对通过。仅 NumPy CPU；未导入 PyTorch，未访问项目图像、模型/checkpoint 或 confirmation/test 载荷/测量值；无训练、推理或 GPU。

## 单亮点漏检对全图分数的图幅尺度敏感性 — 2026-11-09（纯合成 CPU 方法实验）

- **协议/代码：** [预登记协议](../../experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/protocol.json)、[运行脚本](../../experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/run.py)；命令为 `CUDA_VISIBLE_DEVICES='' python experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/run.py`。
- **设计：** 单亮点、正确预测与漏检预测、64×64/128×128/256×256 三种图幅；每个图幅 300 次配对重复、每集合 16 draws，共 1,800 条记录。
- **结果：** 亮点密度下降 16 倍时，CRPS 与邻接变差的绝对配对差约缩小 16 倍，整图能量分数约缩小 15 倍。所有配对区间均高于零，正差比例为 0.970–0.993；正确条件的中心点检出率为 0.976–0.979。详见[中文报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/报告.md)。
- **边界：** 共享噪声合成配对中的排序率不是模型检出概率。结论仅适用于预设一个亮点和 Gamma 数学构造；不外推真实 SAR，也不把跨图幅绝对分差直接比较为方法质量。不改评分公式、C3-A v3、阶段 gate 或训练状态。
- **验证/访问：** `py_compile` 通过；1,800 条逐重复记录、有限评分与清单哈希核对通过。仅 NumPy CPU；未访问项目图像、模型/checkpoint 或 confirmation/test 载荷/测量值；无训练、推理或 GPU。

## C1 SEN12MS train_list 正文读取与文件名交叉尝试 — 2026-11-09（只读元数据审查）

- **问题：** 能否通过作者维护的 SEN12MS train_list，将冻结 C1 v2 train/development 文件映射到可核验的地理/patch 元数据？
- **本地范围：** 仅读取 C1 v2 train/development 暂存清单中的 1,800 个配对记录；不打开图像或其他角色清单。
- **来源访问：** [作者 train_list 页面](https://github.com/schmitt-muc/SEN12MS/blob/master/splits/train_list.txt)标为 4.91 MB 且不展示正文；网页 raw 读取受内容长度限制，本机网络请求被拒绝。三个 train-only 文件名搜索未得到可核验结果。
- **观察边界：** 无法判断是否存在跨版本同名文件；精确搜索无结果不代表无匹配。既有 2026-10-25 训练清单审计只保留摘要，未留存可用于当前交叉的清单正文。
- **结果：** 没有建立文件级身份、坐标、采集日期或 footprint 映射；不更新 C1 独立性主张、划分、阶段 gate 或训练状态。
- **访问标记：** 项目影像、confirmation/test 载荷、作者 test/hold-out 清单、模型/checkpoint、训练、推理及 GPU 均未访问或使用。报告：[获取尝试](../../experiments/data_audits/c1_sen12ms_trainlist_access_attempt_20261109_v1/report.md)；清单：[审查 manifest](../../experiments/data_audits/c1_sen12ms_trainlist_access_attempt_20261109_v1/manifest.json)。

## C1 SEN12MS 地理元数据候选路线复核 — 2026-11-09（只读来源审查）

- **问题：** 是否有新的官方 georeferenced SEN12MS 元数据能把冻结 C1 v2 train/development 来源文件连接到坐标或 patch footprint？
- **来源：** SEN12MS 作者论文描述 GeoTIFF、ROI/scene 与命名结构；作者 README 说明提供分类训练清单、测试清单及 ROI 地图 notebook。官方训练清单网页仅显示文件大小，未显示其 4.91 MB 内容。
- **观察：** C1 train/development 本地来源路径与论文中的 ROI/season/patch 词法相似，但无可核验的逐文件交叉结果。精确检索无结果不等于无匹配；不同发布版的名字相似不能证明身份一致。报告：[来源复核](../../experiments/data_audits/c1_sen12ms_georeference_crosswalk_recheck_20261109_v1/report.md)。
- **边界：** 只查看论文、README、训练清单页面元数据及 C1 train/development staging/path 元数据；不读取 test/hold-out 清单内容、notebook 输出或影像载荷，不访问 confirmation/test，未训练或使用 GPU。

## 横纵相关轴交换的变差分数诊断 — 2026-11-09（纯合成方法实验）

- **问题：** 逐边平方误差后汇总的现行变差分数，能否在横纵相关不同且候选预测交换两个方向时检测各向异性错配？
- **协议/输入：** [预登记协议](../../experiments/synthetic_cpu/c3a_spatial_variogram_anisotropy_cpu_20261109_v1/protocol.md)固定 24×24 logistic-normal 场、横纵 AR(1) 相关 0.75/0.35、双向真值、每方向 300 目标、16 draws 与 lag={1,2,4,8}。
- **结果：** 两个方向 h=1/多尺度正确排序率均为 1.000，且平均差 95% 区间下界大于零；h=8 排序率为 0.593；CRPS 与整图能量分数排序接近 0.5。独立 CSV 和 manifest 复核通过。详见[运行报告](../../experiments/synthetic_cpu/c3a_spatial_variogram_anisotropy_cpu_20261109_v1/report.md)。
- **边界：** 仅测一个平稳可分离 logistic-normal 各向异性构造；不能外推到非平稳结构、亮散射点或 SAR 模型。未改评分代码、C3-A v3、评价协议或 gate。

## 多尺度变差分数相关强度敏感性诊断 — 2026-11-09（纯合成方法实验）

- **问题：** ρ=0.8 单点上的空间区分观察能否延伸至较弱和较强的一阶高斯相关，以及不同真值方向？
- **协议/输入：** [预登记协议](../../experiments/synthetic_cpu/c3a_spatial_variogram_correlation_sweep_cpu_20261109_v1/protocol.md)固定 24×24 logistic-normal 场、ρ={0.4,0.6,0.8,0.9}、AR(1)/RBF、每条件 300 目标、16 draws、lag={1,2,4,8}，按目标进行 2,000 次 bootstrap。
- **结果：** 8 个相关强度/真值条件中 7 个同时满足多尺度平均差区间下界大于零且正确排序比例高于 0.65。ρ=0.4、RBF 真值条件平均差区间为 [0.00001407, 0.00003987]，但正确排序比例 0.600，未达到门槛；其余 7 个排序比例为 0.717–0.990。全部 h=1 排序比例为 0.463–0.550。详见[运行报告](../../experiments/synthetic_cpu/c3a_spatial_variogram_correlation_sweep_cpu_20261109_v1/report.md)。
- **边界：** 只覆盖指定平稳可分离 logistic-normal 协方差族与 K=16；不外推 SAR、非平稳结构、散射点、真实模型 draw 或后验校准。不接入 C3-A v3，不决定真实 lag、权重、draw 数或 gate。未访问项目数据、模型、checkpoint、confirmation/test 或 GPU。

## 单尺度与多尺度变差分数 CPU 诊断 — 2026-11-09（纯合成方法实验）

- **问题：** 当像素边际分布与 1 像素相关相同、远距离相关不同时，单邻接变差分数能否识别协方差失配？等权组合多个固定距离是否改善区分？
- **协议/输入：** [预登记协议](../../experiments/synthetic_cpu/c3a_spatial_variogram_multiscale_cpu_20261109_v1/protocol.md)固定 24×24、可分离 AR(1)/RBF logistic-normal 场、一阶相关 0.8、`h={1,2,4,8}`、每方向 300 个目标、16 draws；共同底噪配对预测家族。
- **结果：** 单尺度 `h=1` 正确排序比例 0.490/0.473；四尺度等权变差分数为 0.990/0.857，且两个方向平均配对差 95% 区间均大于零。逐像素 CRPS 排序接近 0.5；整图能量分数的逐目标排序也较弱。详见[报告](../../experiments/synthetic_cpu/c3a_spatial_variogram_multiscale_cpu_20261109_v1/report.md)。
- **边界：** 仅一个低维平稳可分离协方差对照；不外推非平稳 SAR 结构、散射点、真实 draw 依赖或后验校准。未接入 C3-A v3、不决定真实评分权重、draw 数或 gate；无项目影像、模型、checkpoint、confirmation/test 或 GPU 访问。
- **产物：** `protocol.md`、`run.py`、`per_trial.csv`、`results.json`、`manifest.json`；运行前余量约 19.5 GiB，产物约 0.46 MB。

## IOP SAR JDSR 来源路线补查 — 2026-11-09（文献审查，非实验）

- **检索范围：** 精确题名查询 `edu.cn` 页面、机构仓储与 arXiv；未重复请求此前受限的 IOP、DOI 和 Crossref 正文入口。
- **结果：** 没有发现新的可读取一手全文或可验证作者稿；仍只见 ResearchGate 摘要级二手条目。方法、数据拆分、代码和性能均不升级为事实。
- **边界：** 有界检索未命中不证明其他位置不存在作者稿；未访问项目数据、模型或 checkpoint，没有训练/GPU，也未读取 confirmation/test 载荷或测量值。见[来源审查报告](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## C3-A development CPU 验证权重预检 — 2026-11-09（只读元数据检查，非模型实验）

- **发现：** 未识别到 C3-A 正式训练 run 或匹配的 `checkpoints/best.pt`。现有评价器需要 run manifest、输入 manifest、协议绑定与权重身份校验。
- **限制：** 只检查目录名、文件名与元数据；没有打开 checkpoint、图像、split payload 或测量值。没有训练、推理或使用 GPU。
- **决定：** 当前不运行开发集评价；训练恢复且合规 run 已生成后再调用现有评价器。详见[预检报告](../../experiments/diagnostics/c3a_cpu_evaluation_checkpoint_preflight_20261109_v1/report.md)。

## PCSDD 复数 SAR 扩散去斑全文审查 — 2026-11-09（文献审查，非实验）

- **记录：** [中文审查报告](../../experiments/literature_audits/pcsdd_complex_diffusion_despeckling_fulltext_20261109_v1/report.md)。
- **来源：** IEEE 官方 DOI 元数据及作者公开 CC BY 4.0 全文；IEEE 全文页面自动访问受机器人验证限制。
- **结论边界：** 记录为复数 SLC 上 PDE 引导扩散去斑相邻先例。它不涉及 SAR 超分、光学引导或错配回退，不能直接迁移至本地 8-bit 表示域；正文对盲点掩码 `M` 的符号约定与输入构造式/监督集合定义不一致；作者代码未核实，故不宣称无中心像素泄漏已获独立验证。上传版本对应和场景 split 细节未核实。
- **运行范围：** 未运行代码或数据，未访问项目图像/模型/checkpoint，未使用 GPU，未访问 confirmation/test 载荷或测量值。

## 多样本评分工具单图像 CPU 耗时 — 2026-11-09（合成软件性能诊断，非模型实验）

- **协议与命令：** [预登记协议](../../experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/protocol.json)；`python -m py_compile experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/run.py` 后执行 `python experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/run.py`。
- **输入/计时：** 固定种子合成 `[16,256,256]` 集合与 `[256,256]` 观测；每项预热一次、计时七次。数组生成和文件 I/O 不计入函数耗时。
- **结果：** 逐像素 CRPS 中位数 16.700 ms（IQR 15.884–17.801），整图能量分数 7.183 ms（6.925–7.363），邻接变差分数 14.568 ms（14.482–15.653）；21 条逐次记录、每项 7 条，所有评分有限。
- **边界：** 当前主机单图计算时长；不外推完整开发集或 GPU，不包含模型采样与 I/O，不改变 C3-A v3、指标 gate、draw 数或训练状态。无项目图像/结果、模型、checkpoint、confirmation/test 载荷访问。
- **产物：** [中文报告](../../experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/report.md)、[逐次 CSV](../../experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/逐次计时.csv)、[JSON 与 manifest](../../experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/)。

## 多样本评分量隔离原型验证 — 2026-11-09（纯 CPU 软件核验，非模型实验）

- **实现：** [独立 NumPy 评分工具](../../scripts/distribution_metrics.py)；函数为 `pixelwise_crps`、`normalized_image_energy_score`、`adjacent_variogram_score`。只接收 `[K,H,W]` 预测集合和 `[H,W]` 观测，输入须为有限实数且位于 `[0,1]`；越界时报错，不自动裁剪。未接入 C3-A v3。
- **验证：** `python -m pytest tests/test_distribution_metrics.py -q`，10 项通过；`python -m py_compile scripts/distribution_metrics.py tests/test_distribution_metrics.py` 通过。覆盖精确集合、CRPS 手算值、能量分数像素归一化、邻接变差手算值及输入合同失败情况。另以固定种子的 5×7×9 合成数组和独立向量化公式对照，三项分数最大绝对差为 `2.776e-17`。
- **运行边界：** NumPy CPU 合成小数组，无 PyTorch/GPU、训练或项目图像/模型/检查点访问；confirmation/test 载荷与测量值未访问。
- **解释：** 这是未启用补充设计中的候选评分量的软件核验，不验证真实图像分布、校准、物理一致性、draw 数或 gate。未来使用前仍需冻结数据、汇总、空间边、重抽样和保存策略。

## 多样本概率评分规则一手来源核验 — 2026-11-09（文献审查，非实验）

- **记录：** [审查报告](../../experiments/literature_audits/proper_scoring_rules_multisample_eval_20261109_v1/report.md)。
- **来源：** Gneiting 与 Raftery（2007）JASA 适当评分规则全文；Scheuerer 与 Hamill（2015）MWR 变差评分全文，含 NOAA 作者机构存档。
- **结果/影响：** CRPS 支持逐像素边缘评分；能量分数可评价有限集合表示的多变量分布，但对部分相关失配的敏感性可能有限；变差分数可检查所选配对空间统计，但非严格适当。已把三者的互补关系与局限写入补充设计。两篇来源均未验证 SAR 指标或本地模型。

## 归一化图像域的分布评价量已知案例 — 2026-11-09（纯合成 CPU，非门槛）

- **记录：** [协议](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/protocol.json)、[脚本](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/run.py)、[报告](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/report.md)、[manifest](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/manifest.json)。
- **设计/结果：** 32×32 `[0,1]` 合成图、Gamma(shape=4, scale=1/4) 受控乘性变量、每集合 16 draws、100 次重复。边缘分布相同但错误空间依赖候选的逐像素 CRPS 配对差为 −0.0000128，标准误 0.0001670；归一化整图能量和邻接变差配对差分别为 +0.0051899 和 +0.0044720，100/100 次均高于正确候选。
- **边界：** Gamma 参数仅为数学构造，不解释为本地物理视数；只用于已知合成例子的评价量检查，不代表项目模型、SAR 物理或 C3-A 结果。

## 已知双峰分布的多样本评价量诊断 — 2026-11-09（纯合成 CPU，非门槛）

## 已知双峰分布的多样本评价量诊断 — 2026-11-09（纯合成 CPU，非门槛）

- **记录：** [协议](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/protocol.json)、[脚本](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/run.py)、[报告](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/report.md)、[manifest](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/manifest.json)。
- **设计/结果：** 30 次重复、每候选 256 个样本，比较真实双峰、均值塌缩、矩匹配单峰高斯和 0.8/0.2 偏置双峰。均值塌缩均值误差为零，但能量分数 3.0910、模态支持率与成对距离均为零；真实双峰抽样能量分数为 1.8094。均值和协方差匹配仍未匹配双峰形状。
- **边界：** 仅说明指定二维分布下评价量的诊断行为；不表示本地模型、SAR 物理、校准或 C3-A 结果。C3-A v3 与所有 gate 不变，无训练/GPU/项目影像或确认测试载荷访问。

## 已知双峰分布的评价样本数敏感性 — 2026-11-09（纯合成 CPU，非门槛）

- **记录：** [协议](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/protocol.json)、[脚本](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/run.py)、[报告](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/report.md)、[manifest](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/manifest.json)。
- **设计/结果：** 100 次重复，比较 N=4/8/16/32/64/128/256 的能量分数排序。预登记的 0.90 重复比例描述阈值在矩匹配单峰高斯的 N=32、偏置双峰的 N=16 首次达到；重复间分数差波动随 N 增加总体下降。
- **边界：** 样本数结论只适用于该二维合成例子，不能用作真实 Flow/Diffusion 的 draw 数建议、功效保证或阶段门槛。没有项目影像、模型、checkpoint、confirmation/test 载荷访问，无训练或 GPU。

## ICSP 2024 作者主页全文路线复核 — 2026-11-07（来源审查，非实验）

- **编号：** `LIT-ICSP-SAR-DIFFUSION-ACCESS-20261107-FU1`。
- **结果：** ResearchGate 对应作者论文条目标注无全文，仅提供向作者请求的入口；本次未请求。IEEE 仍只有摘要，NUDT 精确检索未找到作者稿；会议目录只提供书目信息。
- **处理：** 保持摘要级证据，不补写协议或结果细节，也不改变 gate。详见[来源记录](../../experiments/literature_audits/icsp_sar_diffusion_sr_abstract_pdf_route_20261107_v1/report.md)。

## Amieva 等人 SAR 联合去斑/超分论文全文补充审查 — 2026-11-07（文献审查，非实验）

- **编号：** `LIT-AMIEVA-DIRECT-FULLTEXT-20261107-v1`。
- **范围：** 直接读取机构仓储已发表版 PDF 正文；随后复访超时。没有访问项目数据、模型、checkpoint 或 confirmation/test 载荷，没有训练或使用 GPU。
- **结果：** 核实 IW→SM 跨模式 SAR-only 非生成式联合恢复、整景 5/2/3 划分及全影像集合百分位先于拆分；保留参考噪声、百分位统计池化方式和 test 使用边界等限制。
- **处理：** 更新原有矩阵行和综述条目，不增加重复文献；不改变 gate 或正式总纲。详见[审查报告](../../experiments/literature_audits/amieva_direct_fulltext_followup_20261107_v1/report.md)。

## IOP SAR 联合去斑/超分 DOI 有界复核 — 2026-11-07（文献来源审查，非实验）

- **编号：** `LIT-FEP-SAR-JDSR-20261107-v1`。
- **结果：** IOP 出版方、DOI 与 Crossref 入口没有提供可读正文；只检索到 ResearchGate 摘要级二手线索。其题名高度相关，但方法、数据、指标与正式出版元数据仍待一手核实。
- **处理：** 记入开放先例矩阵和候选总纲，不把二手摘要升级成方法/效果事实；不改变 v1.7、C1–C6、任何 gate 或训练状态。详见[来源报告](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## C3-A 多样本协议与实现对应核查 — 2026-11-07（静态审查，非模型实验）

- **范围：** 只读核对冻结 protocol v3 与 `scripts/evaluate_c3a_multisample.py` 的采样/统计/输出路径。
- **结论：** 实现逐 draw 计算指标并报告逐像素样本标准差；没有均值图指标、生成分布距离、draw 相关系数或预测图持久化。与协议目标相符，不是代码缺陷，故不修改 evaluator 或协议。
- **边界：** 不能由此推断真实 draw 独立性/分布校准，也不触碰真实模型输出、项目数据或 confirmation/test 载荷。未来增加评价须另行预登记。详见[静态审查](C3A_MULTI_SAMPLE_DEPENDENCE_AUDIT_20261107.md)。

## 相关 Gamma draw 多样本均值敏感性 — 2026-11-07（纯合成 CPU，非门槛）

- **编号：** `C3A-SYN-CORRELATED-DRAW-MEAN-20261107-v1`；[冻结协议](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/protocol.json)。
- **设计：** `n_k=c+u_k`，固定每个 draw 的 `Gamma(L,1/L)` 边际并控制共同分量形成的相关系数 `ρ∈{0,0.25,0.5,0.9,1}`；`L∈{1,2,8}`、K∈{1,4,8}、每条件 500,000 像素。
- **结果：** 45/45 个预声明核验通过。K=8 的 PSNR 增益随 `ρ` 增加而从约 +9.03 dB 下降至 0 dB；详细范围见[报告](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/report.md)。
- **解释边界：** 仅为共同 Gamma 数学模型，不测量真实模型输出相关性、不代表项目 SAR 数据，不改变 C3-A 评价协议或任何 gate；未训练、未用 GPU、未访问项目载荷或 checkpoint。

## Gamma 乘性噪声多样本均值效应 — 2026-11-07（纯合成 CPU，非门槛）

- **编号：** `C3A-SYN-SAMPLE-MEAN-20261107-v1`；协议：[protocol.json](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/protocol.json)。
- **范围：** 仅 NumPy 合成 `n~Gamma(L,1/L)`；L 为 1/2/8，每条件 500,000 个像素、8 次独立 draw，比较 K=1/4/8 的逐像素均值。
- **观察：** 9 个条件的均值、方差、ENL、MSE 比和 PSNR 容差全部通过。K=4 的 PSNR 增益约 +6.0 dB、K=8 约 +9.0 dB；ENL 随 K 增长。仅支持该解析 Gamma 设定下的统计观察。
- **解释边界：** 均值图可作平方误差点估计，但不再是原单样本分布的生成样本。没有项目数据、模型、checkpoint、confirmation/test 载荷访问；没有训练、GPU、选模或 gate 变更。结果与[报告](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/report.md)和[manifest](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/manifest.json)对应。

## MONTI 两阶段 SAR 潜扩散超分全文审查 — 2026-11-07（文献审查）

- **编号：** `LIT-MONTI-TWO-STAGE-SAR-SR-20261107-v2`。OpenReview 作者稿全文及 MONTI 官方录用/非归档状态已核验。
- **影响：** 加强 SAR-only 生成式超分先例边界；不解锁 optical 条件、观测一致性或可靠性阶段，也不改变任何 gate。见[审查报告](../../experiments/literature_audits/monti_two_stage_sar_sr_20261107_v2/report.md)及[manifest](../../experiments/literature_audits/monti_two_stage_sar_sr_20261107_v2/manifest.json)。

## 正式授权预检 CPU 软件核验 — 2026-11-07（非模型实验）

- **范围：** 冻结输入授权、validation checkpoint metadata 绑定、C2 合成清单及 `infer_v4` 光学缺失/全零来源路径。
- **执行：** 隐藏 CUDA，仅使用 pytest 临时合成清单、checkpoint 与固定 LR 数组。级联/SFT/正式套件在资源检查前完成授权核验；validation evaluator 在资源检查前验证 checkpoint 的 split/manifest hash。实际模型执行仍为 CUDA-only。
- **结果：** 定向测试 73 项中 72 项通过；历史 B1 源码哈希完整性用例因当前树内 10 个文件与历史 run 固定哈希不一致而失败。SEN12TS 元数据测试在已装 `rasterio` 的环境 4 项通过，没有读取像素。
- **边界：** 没有训练、正式评价或 CUDA；confirmation/test 载荷和逐样本结果未访问，gate 不变。未重写历史哈希或产物。详见[核验报告](FORMAL_PREFLIGHT_CPU_AUDIT_20261107.md)。

## C2 全 development 对比差 CPU 描述诊断 — 2026-11-07（非门槛）

- **协议：** [`c2_all_development_contrast_cpu_20261107_v5.json`](../../protocols/mmcanet_phase_c2/c2_all_development_contrast_cpu_20261107_v5.json)，运行 ID `mmcanet_c2_all_development_contrast_cpu_20261107_v5`。
- **范围：** 四条件各 200 个 development 样本、19 个来源组；matched/permuted/missing/SAR-only 四控制；读取 development 固定数组和既有 development 指标以核对输入/掩膜身份。未读取 train 数组、confirmation/test 载荷或指标。
- **执行：** 仅 CPU、2 线程、8 个既有 best checkpoint、3,200 次前向，约 121.35 秒；未训练、未选模、CUDA 未初始化，输出无图像副本。逐样本 3,200 行、配对差 3,200 行、来源组对比 304 行、来源组汇总 16 行。
- **观察：** 目标背景中位数为零的计数按 x2_light/x2_severe/x4_light/x4_severe 为 2/2/1/1（每条件 200）。matched−SAR-only 的来源组等权均值对比误差差为 −0.000191/−0.000939/−0.000862/+0.000677；只描述该重复使用的 development，不解释为 optical 总体效用。
- **状态：** `gate_eligible=false`；C2 v2 原 fail-closed 与 gate 未判定状态不变。详见[报告](../../experiments/diagnostics/c2_all_development_contrast_cpu_20261107_v5/report.md)和[运行清单](../../experiments/diagnostics/c2_all_development_contrast_cpu_20261107_v5/manifest.json)。

## C1 v2 train/development 到本地数字 PNG 的哈希映射核查 — 2026-11-07（只读元数据审计）

将冻结 C1 train/development 的 3,600 个来源模态 SHA-512 与仓库既有的上游精确哈希交叉表连接，命中 0 行、无法恢复数字 PNG 文件名。只解析 C1 train/development 字段；其他角色仅读取 `role` 后排除。既有表仅作为来源索引，不使用历史 validation 指标或模型结果。零命中不证明像素不同，不排除重编码内容，也不确认数据集/传感器身份。未访问影像、数组、checkpoint、confirmation/test 载荷；未训练/GPU。详见[审计报告](../../experiments/data_audits/c1_v2_numeric_png_source_crosswalk_20261107_v1/report.md)、[协议](../../experiments/data_audits/c1_v2_numeric_png_source_crosswalk_20261107_v1/protocol.md)与[结果清单](../../experiments/data_audits/c1_v2_numeric_png_source_crosswalk_20261107_v1/manifest.json)。

## RSDM/LSIR 2026 DOI 一手来源复核 — 2026-11-07（文献审查，非实验）

对 DOI 10.1109/JSTARS.2026.3655710 进行新的出版来源检查。DOAJ 二手索引提供题名、作者、卷页和摘要线索；IEEE Xplore 精确检索未返回正文，期刊卷期入口返回 HTTP 418，DOI/Crossref 页未能读取。结论维持“高优先级开放先例线索”，不将二手材料作为正式方法/结果证据，不声称或否定宽泛首次主张。未访问项目数据、checkpoint、confirmation 内容或 test 数据；未训练或使用 GPU。详见[来源复核报告](../../experiments/literature_audits/region_specific_sar_jdsr_source_recheck_20261107_v1/report.md)及[访问清单](../../experiments/literature_audits/region_specific_sar_jdsr_source_recheck_20261107_v1/access_manifest.json)。

## 等速度场调用预算的解析高斯流求解器比较 — 2026-11-07（纯合成 CPU 数值诊断）

运行 c3a_gaussian_flow_solver_nfe_comparison_20261107_v1：同一 200,000 个源样本、三个目标/源标准差比下，比较 Euler 16 步、Heun 8 步、显式中点法 8 步、经典 RK4 4 步，每个样本的速度场调用数均为 16。12 组中，比值 0.25 时显式中点法的归一化端点 RMSE 最低（0.00175149）；比值 1 和 2 时 RK4 最低（分别为 0.00008399、0.00041202）；Euler 在三种条件下均最大。Heun 结果与同日步数实验精确一致。该结果说明方法排序依赖解析速度场，不代表学习型场、SAR 重建质量或 GPU 延迟。未访问项目载荷、checkpoint、结果、confirmation/test 内容，未导入 PyTorch/CUDA，未训练或使用 GPU。未改变冻结协议或阶段 gate。完整信息见[实验目录](../../experiments/synthetic_cpu/c3a_gaussian_flow_solver_nfe_comparison_20261107_v1/)。

## 解析高斯条件流 Heun 端点误差 — 2026-11-07（纯合成 CPU 数值诊断）

运行 `c3a_gaussian_conditional_flow_solver_20261107_v1`：对独立高斯源/目标的线性插值，使用解析条件均值速度场，固定随机种子与 200,000 个样本，比较三个目标/源标准差比及四种 Heun 步数，共 12 组。8 步归一化端点 RMSE 为 0.021147（比值 0.25）、0.001295（比值 1）和 0.009586（比值 2）；所有条件下增加步数均降低数值积分误差。结果仅是解析一维流的离散误差，不能外推为 C3-A 模型误差、SAR 重建质量或物理一致性。未读取项目 payload、checkpoint、结果或 confirmation/test 载荷；未导入 PyTorch/CUDA、未训练、未用 GPU。未改变冻结协议或阶段 gate。完整协议、命令、环境、结果与 manifest 见[实验目录](../../experiments/synthetic_cpu/c3a_gaussian_conditional_flow_solver_20261107_v1/)。

## C3-A train/development 角色与载荷路径静态审查 — 2026-11-07（源码审查，非实验）

C3-A runner 仅构造 train/development，并以 SAR-only 包装器核验/读取 SAR LR 与目标数组；confirmation 需要显式授权，runner 未启用。续训 manifest 角色与数组集合均 fail-closed 约束。只读源码，未打开任何数据、结果或 checkpoint，未运行测试/GPU/训练/评价。详见[审查记录](C3A_DATA_ACCESS_ROLE_AUDIT_20261107.md)。

## C3-A 续训路径与状态恢复回归 — 2026-11-07（CPU 合成软件验证，非模型实验）

发现 `_resume_code_hashes` 引用不存在的 C2 配对模型路径，修正为实际模块路径，并新增路径解析回归。`tests/test_c3a_resume_code_hashes.py` 与既有 `tests/test_c3a_training_resume.py` 共 16 项通过；Python 编译通过。只用合成 checkpoint 和临时目录，隐藏 GPU；未读取项目 checkpoint、数据或结果，未训练/评价，不改变 gate。详见[C3-A 续训审查](C3A_RESUME_BINDING_AUDIT_20261107.md)。

## C3-A 流目标和 Heun 实现静态审查 — 2026-11-07（协议/源码核对，非实验）

冻结 C3-A v1 的源分布、插值路径、速度目标、MSE 归约、sample_id 固定噪声和 8 步 Heun 均与源码相符；无必要代码改动。未运行代码或测试、未访问数据/模型输出/指标、未使用 GPU，不改变 C3-A 质量或阶段判断。详见[C3-A 静态审查](C3A_FLOW_OBJECTIVE_STATIC_AUDIT_20261107.md)。

## C2 paired screen 写入路径静态审查 — 2026-11-07（源码审查，非实验）

核对 C2 高层 runner 的冻结矩阵/development 输入绑定、输出目录互斥检查，以及 CSV/JSON/Markdown 文件的独占创建。常规路径没有发现静默覆盖或输入身份缺口，不改代码或 C2 gate。只读源码，未运行测试、GPU、模型评价，未访问数据/结果/confirmation/test 载荷。详见[C2 静态审查](C2_SCREEN_WRITER_STATIC_AUDIT_20261107.md)。

## C1 train/development 文件名格式补充 — 2026-11-07（角色限定元数据审查，非实验）

仅核对 train/development 的样本数、来源组数与路径字段；命名结构与 SEN12MS 作者仓库示例相容，但未建立场景或坐标一一对应。未读取作者 split 文件、Notebook 输出、影像、confirmation/test 载荷或结果，不改变任何 gate。详见[C1 路线补充](../../experiments/data_audits/c1_geospatial_crosswalk_route_20261107_v1/report.md)。

## DiffusionSAR 一手全文审查 — 2026-11-07（文献审查，非实验）

审阅 University of Twente 仓储的 2024 TGRS 最终版全文。核实其任务为 SAR 条件扩散去斑，合成退化使用 Gamma speckle，包含 log-Yeo–Johnson 变换与多时相 SAR 微调；无超分或推理时 optical 条件。相关工作矩阵已更新，C3-A 与本地 PNG 的差异及真实场景产品术语不一致均已记录。没有项目数据/结果访问、训练、GPU、代码更改或模型评价，不改变任何 gate。详见[报告与 manifest](../../experiments/literature_audits/diffusionsar_conditional_sar_despeckling_fulltext_20261107_v1/report.md)。

## C1 地理元数据交叉映射路线审查 — 2026-11-07（文献/清单审查，非实验）

一手 SEN12MS 论文称其沿用 SEN1-2 的随机 ROI 与季节定义，并提供有地理参考的 GeoTIFF；这建立了候选来源路线，但未证明两版 ROI 子目录编号一一对应。TUM 1474000 本轮返回 HTTP 429，未下载数据或提取坐标。只读取冻结 C1 v2 train/development 来源组元数据；未读取项目图像、SEN12MS 文件、confirmation/test 载荷、checkpoint 或指标，未运行 GPU、训练或评价。不得将此审查写成坐标交叉表、空间独立证据或阶段晋级。详见[报告与清单](../../experiments/data_audits/c1_geospatial_crosswalk_route_20261107_v1/report.md)。

## C3-A v3 bootstrap 统计字段绑定 — 2026-11-07（CPU 合成验证，非模型实验）

C3-A v3 将 bootstrap 配置置于 `metrics`，但评价器在推理后读取不存在的 `statistics`，且各指标 seed 偏移没有协议依据。现改为在 `_read_protocol` 阶段解析、校验 v3 配置，统一使用协议 seed，并检查 19 组覆盖。19 组合成、多指标与少组拒绝用例通过；C3-A evaluator 测试文件共 31 项通过，编译通过。未运行模型或读取结果。详见[报告与 manifest](../../experiments/diagnostics/c3a_v3_bootstrap_contract_cpu_20261107_v1/report.md)。

## C2 confirmation bootstrap 统计合同核查 — 2026-11-07（CPU 合成验证，非模型实验）

确认协议校验器此前未绑定 bootstrap 的来源组单位、组数、组权重、次数、seed 和区间字段。现已加入按当前 `PROTOCOL_ID` 固定的统计合同校验。六项临时 JSON 漂移回归通过；confirmation/screen 三份定向测试共 46 项通过，六个 Python 文件编译通过。运行只用临时 JSON/source manifest，未读确认载荷或结果、未执行旧协议。详见[报告与 manifest](../../experiments/diagnostics/c2_confirmation_statistics_contract_20261107_v1/report.md)。

## C2 来源组 bootstrap 统计合同核验 — 2026-11-07（CPU 合成验证，非模型实验）

按冻结 C2 v2 的 `source_group_id`、19 组等权、10,000 次、seed `20261030` 与 percentile 区间要求核对实现。发现汇总器只读取次数和 seed，未校验其余统计语义或组数；现加入 fail-closed 字段/组数校验。人工差值区分来源组等权均值 `+1 dB` 与 patch 等权均值 `+2 dB`，bootstrap 同 seed 可复现。C2 paired screen 两份合成测试共 24 项通过，源码编译通过。未运行 GPU、训练或模型评价；无项目数据、confirmation/test 载荷或真实指标访问，不改变 C2 gate。详见[报告与 manifest](../../experiments/diagnostics/c2_source_group_bootstrap_synthetic_20261107_v1/report.md)。

## C2 固定退化生成器与冻结参数绑定核查 — 2026-11-07（CPU 合成验证，非模型实验）

对照 C2 v2 冻结 JSON、`prepare_phase_c2_fixed_inputs.py`、共享退化算子和现有测试。确认实现的归一化、频域截断、下采样、振铃裁剪和 Gamma 表示域乘法与冻结定义一致；发现调用方 seed 与协议的 `degradation_seed_base` 未绑定，四个条件倍率/Gamma `L` 由代码常量重复维护。现已加入前置校验，错 seed 或条件漂移会在 CUDA、输出目录和图像读取之前失败；不改变冻结协议或有效退化。

两个新回归先按预期失败于原来的 CUDA 可用性检查；修复后 C1 source-manifest 错配、seed 错配和条件漂移三个 CPU 定向测试通过，相关生成器/测试文件 `py_compile` 通过。仅使用临时 CSV/JSON，未加载图像、checkpoint 或项目指标，未生成 C2 固定输入、训练、运行模型或初始化 CUDA。C2 gate 未判定；用户要求继续暂停训练。详见 D-2026-11-07-010 和[C2 核查报告](C2_FIXED_GENERATION_PROTOCOL_BINDING_20261107.md)。

## C1 v2 官方来源链与旧文件哈希审计适用范围更正 — 2026-11-07（无 GPU，非实验）

静态核对冻结 C1 v2 来源协议、train/development 暂存脚本及已保留获取 manifest。协议将来源绑定到 TUM MediaTUM 条目 1436631 和官方 `checksums.sha512` 文件索引；暂存清单记录 1,600/200 个 train/development 身份、92/19 个来源组、3,600 个文件完成字节数与 SHA-512 核验。只读现有元数据，没有读取影像载荷。

2026-10-19 旧哈希交叉审计的范围是 `fixed_train_dev2k` / `fixed_val_dev2k` 共 3,600 个文件，精确匹配数为 0；不适用于 C1 v2 train/development。`./data/` 数字命名子集仍未映射。来源链核实不证明坐标、采集日期、轨道或 patch footprint 独立；线上 TUM/OAI-PMH 路线本轮返回 HTTP 429。详见[C1 来源链核查](C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)与 D-2026-11-07-009；不改变 gate。

## C3-A 数值域实现合同静态核对 — 2026-11-07（无 GPU，非实验）

只读对照 C3-A v1 协议、训练/采样/评价源码与已有测试定义。LR 可大于 1，target 在 `[0,1]`；条件不裁剪，流速度头无界，Heun 不裁剪，最终评价才裁剪并记录 raw 越界比例，接口相互一致。没有训练、checkpoint/数据载荷读取或 GPU 初始化；本轮未运行测试。无需修改冻结协议或实现，不构成方法效果证据。详见[C3-A 数值域合同审计](C3A_NUMERIC_DOMAIN_CONTRACT_AUDIT_20261107.md)。

## 论文 pilot 主张与当前阶段静态核对 — 2026-11-07（无 GPU，非实验）

只读检查旧英文稿与现行 v1.7/C1 v2/C2/C3-A 状态。稿件标注为 2026 年 7 月 pilot，所述 1,900/100 文件级试验不属于当前 C1 v2；C2 gate 未判定，C3-A 尚无真实模型结果，因此旧稿不构成当前效果或晋级证据。详见[静态审计](PAPER_CLAIM_EVIDENCE_AUDIT_20261107.md)。

边界披露：稿件及其生成 TeX 表中被动可见旧 100 样本 test 汇总；未检查底层 test 输入/逐图 CSV、未复算、调参或使用其做研究判断，也未访问冻结 C1 v2 confirmation/test 载荷。发现后停止读取汇总表；不得据该旧样本集合继续调参或选模。稿件未跟踪，本轮未修改。H1/Q1 与阶段 gate 不变。

## C2 零背景对比差描述诊断 — 2026-11-07（CPU，非门槛）

**类型：** 既有 C2 v2 development 失败样本上的表示域描述诊断；不是训练、正式 screen 重跑或晋级实验。

**预登记协议：** [v4 JSON](../../protocols/mmcanet_phase_c2/c2_zero_background_contrast_cpu_20261107_v4.json) 与[说明](../../protocols/mmcanet_phase_c2/c2_zero_background_contrast_cpu_20261107_v4.md)。选择规则固定为历史 `zero_median_cases.csv` 的全部 6 条“条件×样本”记录，来自 2 个来源组；不得外推为全 development 效果。C2 v2 正式指标、fail-closed 结论和 gate 均未改写。

**运行与输入：** [`c2_zero_background_contrast_cpu_20261107_v4`](../../experiments/diagnostics/c2_zero_background_contrast_cpu_20261107_v4/)。仅读取四种退化条件中选定案例的 development LR、target、matched optical 与冻结置换 donor optical，读取量共 24 个数组；核对四套 development manifest/files.csv、所有数组 SHA-256、重建掩膜 SHA-256、训练矩阵身份及八个 best checkpoint。没有读取来源 split manifest、train 数组、confirmation/test 载荷或对应指标。

**方法与验证：** CPU、2 线程，6 条案例 × 4 控制（matched/permuted/missing/SAR-only），24 次前向全部成功；24 行逐图结果、24 行冻结配对差、16 行逐条件汇总。目标中位数 TBR 返回 `null` 且明确标为 `undefined_target_zero_background_median`，没有加 epsilon。CPU 合成模型与 runner 边界测试 10 项通过，相关 Python 文件 `py_compile` 通过；manifest 中协议、runner、模型源码及输入/权重身份均可核验。总用时约 1.11 秒，输出目录约 52 KiB。

**描述观察：** matched−SAR-only 的均值/中位数对比绝对误差差值依次为 x2_light −0.000569/−0.001011、x2_severe −0.006447/−0.007495、x4_light −0.003836/+0.003685、x4_severe +0.008148/−0.005016。差值为候选误差减对照误差，负数只表示这些案例中误差较低。每个条件仅 1–2 条记录，样本来自按既有指标失败条件选出的子集；不作总体 optical utility、独立泛化、显著性或物理解释，H1/Q1 与 C2 gate 不更新。

**代码变化：** [C2MatchedJoint](../../MMCANetV2/phase_c2_matched.py) 移除 CUDA-only 前置检查，以允许相同计算图在 CPU 上运行；参数、state dict、GPU 路径及输入 shape/dtype/device 检查保持不变。v1–v3 预检/前向失败及更正后的访问记录保留在各自输出目录。遵循用户暂停指示，本轮未运行 GPU 回归。

## C3-A 表示域范围审计与多次采样裁剪分解协议更新 — 2026-11-07

**类型：** C3-A 冻结 train/development SAR 输入的 CPU 表示域描述审计，加一项仅用合成张量验证的评价器改进；不是模型训练、推理或效果实验。

**数据结果：** 只读 `x2_light` 的 train 1,600/92 来源组与 development 200/19 来源组 SAR `lr`、`target`，逐文件 SHA-256、shape、dtype、有限性和值域均通过。来源组等权均值：LR 像素 >1 为 21.57%/22.16%；实际 bilinear 上采样 condition 像素 >1 为 14.94%/15.29%；target ==1 为 1.23%/1.28%，target ≥0.99 为 1.73%/1.72%；target-condition MAE 为 0.1451/0.1454。逐样本数值、manifest、代码和输入清单哈希见[审计目录](../../experiments/diagnostics/c3a_representation_range_audit_20261107_v1/)；数据是归一化 PNG 表示域 noisy pseudo-target，不是物理定标观测或干净参考。

**协议/代码：** 原多次采样 v2 保留不动；新增未运行 v3，SHA-256 `ee83e2cb4dadea7d4fca6576e479d2922d70df44935f1be51116add26e06016c`。评价器为 Flow 与 SAR-only baseline 增加 raw 输出低于 0/高于 1 比例，按身份汇总并按来源组 bootstrap；两项互斥且其和必须等于既有总裁剪比例。PSNR/SSIM/EPI、采样、训练、选模规则和所有研究 gate 不变。详细依据见[协议更新记录](../../experiments/diagnostics/c3a_representation_range_audit_20261107_v1/protocol_update.md)。

**验证与边界：** 先观测到接口缺失和比例不一致未被拒绝的预期失败；实现后 `tests/test_c3a_multisample_eval.py` 30 项通过，评价器、审计脚本、测试文件 `py_compile` 通过。训练、checkpoint 推理、正式评价、GPU 查询/等待均未运行；optical、confirmation、test 载荷未读取。v3 结果目录不存在。未改变 C2 gate 未判定、B1 `not_promoted / stop_at_b1`、B2/B3 关闭状态。

## 元数据审计 C1-META-20261107-01 — 冻结拆分身份字段与角色互斥 — 2026-11-07

**类型：** 仅拆分成员元数据审计，不是图像分析、模型实验或训练。输入为冻结 C1 v2 `sample_manifest.csv`，SHA-256 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`。train/development/confirmation 分别为 1,600/200/200 样本与 92/19/20 来源组；三角色 `source_group_id` 两两互斥。train/development SAR/optical 路径、选择哈希、官方 SHA-512 无交集。

**解释：** 裸 `scene_id`、`patch_id` 非全局身份键。一个 `scene_id+patch_id` 数值组合重复，但位于不同 season root/source group，实际路径与内容哈希不同。支持来源场景文件夹拆分，不支持地理/采集独立结论。仅查看确认集角色成员元数据，不访问任何图像、测量载荷或模型。详见[审计报告](../../experiments/data_audits/c1_v2_split_identity_metadata_20261107_v1/report.md)。

## 文献审计 LIT-20261107-01 — 流轨迹求逆与 SAR 回波扩散后验 — 2026-11-07

**类型：** 一手文献审查，不是项目模型实验。已核对 ICML 2026 PMLR 论文及 arXiv v2 全文 *Trajectory Stitching for Solving Inverse Problems with Flow-Based Models*，并核对 arXiv v1 全文 *UAV SAR Imaging via Diffusion Posterior Sampling under Non-Ideal Motion in ISAC Systems*。相关工作矩阵新增 2 行，现有 151 行增至 153 行；详细来源、作者报告与适用边界见[审计报告](../../experiments/literature_audits/sar_echo_and_flow_trajectory_prior_art_20261107_v1/report.md)及其访问清单。

**结论边界：** 更新通用流逆问题和生成式 SAR 回波成像的先例范围；没有项目数据/模型访问、代码运行或指标结果，不支持本地 C3-A、optical utility 或 C4 结论，不改变任何 gate。

## C3-A epoch 边界续训与 checkpoint v2 CPU 验证 — 2026-11-07 00:26（中国标准时间）

**类型：** 训练状态恢复的软件合同验证，不是模型训练、模型性能评价或阶段 gate 实验。runner 增加显式 `--resume`，严格绑定冻结协议、固定输入清单、代码状态与关键环境；保存 model/Adam、完整随机状态、专用 DataLoader/Flow generator、最佳轮次与可重建日志。恢复会拒绝不兼容或已完成状态。若中断发生于较新 `best.pt` 原子写入后、`last.pt` 更新前，恢复会把已验证的最新 checkpoint 原子同步到 `last.pt`。评价器兼容 checkpoint v2 并校验绑定。

**验证：** CPU 合成连续两步与保存、重建、恢复后第二步等价；C3-A 续训、Flow、multi-sample evaluator 和等待器定向回归 67 项通过（AISAR，3.09 秒）；4 个相关 Python 文件 `py_compile` 通过。此前默认 Python 无 PyTorch，未进入测试收集；切换项目 AISAR 环境后完成测试，该环境差异不是代码失败。可复现命令：`/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_c3a_training_resume.py tests/test_phase_c3a_flow.py tests/test_c3a_multisample_eval.py tests/test_c3a_gpu_waiter.py tests/test_wait_for_c3a_gpu.py`；`/home/shy/anaconda3/envs/AISAR/bin/python -m py_compile scripts/run_phase_c3a_sar_flow.py scripts/evaluate_c3a_multisample.py tests/test_c3a_training_resume.py tests/test_c3a_multisample_eval.py`。临时日志在记录结果后按精确路径清理；结果与命令保留于本登记。

**边界：** 未调用真实训练入口、GPU 等待器或正式评价；不读取 optical、confirmation 或 test 载荷。冻结协议 SHA-256 仍为 `147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`，模型效果、收敛与阶段 gate 均未验证或改变。项目分区当前可用约 21.84 GiB。

## C3-A train/development SAR 输入 CPU 完整性预检 — 2026-11-06 23:37（中国标准时间）

**类型：** 冻结训练输入的数据完整性核验；不是模型训练、验证评价或性能实验。按 C3-A `x2_light` 协议，只经 `SarTargetOnlyDataset` 读取 train/development 的 `lr` 与 `target`。train 为 1,600 个样本、92 个来源组，核验 3,200 个数组文件；development 为 200 个样本、19 个来源组，核验 400 个数组文件。哈希、shape、float32 dtype、有限值和值域合同通过。

**复现与结果：** 运行脚本、命令、环境、代码哈希及固定输入 manifest/files.csv 哈希记录在[预检目录](../../experiments/diagnostics/c3a_cpu_data_preflight_20261106_v1/)。协议 SHA-256 为 `147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`；来源清单 SHA-256 为 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`。

**边界：** 预检前后 CUDA 均未初始化；confirmation、test、optical 载荷访问均为否。共享项目分区预检结束后可用 21,117,218,816 字节（约 19.67 GiB）。本结果只证明冻结输入副本符合所检查的数据合同，不代表 SAR 定标、模型收敛、重建质量或独立泛化。用户要求暂缓训练，因此不启动训练、等待器或正式评价。

## C3-A CPU 张量合同、评价器清单保护与 Heun 数值诊断 — 2026-11-06 23:25（中国标准时间）

**类型：** 合成张量与元数据软件诊断，不是 SAR 模型训练/性能实验。用户要求暂缓 GPU 训练，故 v8 等待器已手动停止；训练目录、评价目录不存在。新增测试确保 C3-A 评价器在开发数据载荷读取前核对训练/input manifest 的 source path/hash、train/development 角色与计数、只读 `lr`/`target` 数组，并拒绝 confirmation/test/optical 标记或角色。实现前张量合同用例 7 项、评价器 input manifest 用例 9 项及后续路径/光学数组用例 2 项按预期失败。修复后 C3-A Flow、多次采样评价和等待器相关测试共 51 项通过，相关 Python 文件 `py_compile` 通过。

CPU 合成训练单步的 loss/gradient 均有限，Adam 后至少一个模型参数发生变化。解析 ODE `dx/dt=x`，初值 0.1、终值 `0.1e`，Heun 16/32/64 步绝对误差为 `0.0001688682 / 0.00004322162 / 0.00001088610`，步数翻倍误差比为 `3.907031 / 3.970349`，NFE 为 `32 / 64 / 128`。同一 seed/sample_id 的噪声精确一致，batch 调序不改变逐样本噪声，不同样本噪声不同；CUDA 未初始化。复现脚本、命令和结果见 [`CPU 诊断目录`](../../experiments/diagnostics/c3a_flow_cpu_contract_20261106_v1/)。本轮没有读取 train/development、confirmation、test 或 optical 载荷；冻结训练协议哈希未变。为控制磁盘空间，清理本轮生成的 26 个 Python `.pyc` 缓存，共 560,672 字节，未触碰 pytest 既有缓存或研究数据。该诊断不支持模型质量、后验校准或物理一致性主张，不改变任何 gate。

## 总纲 v1.7 获批后的 C3-A 资源快照 — 2026-11-06 20:54（中国标准时间）

用户已批准 v1.7；该决策不改变任何实验协议、C2 gate 或历史阶段资格。C3-A v8 等待器仍在运行：GPU 空闲 1146/4096 MiB、3 个其他 CUDA 进程，训练未启动；磁盘当前约 15.06 GiB，低于 2 GiB 时等待器会阻止启动。confirmation/test/optical 与 train/development 影像载荷在等待期间均未读取；没有本轮模型结果或大型缓存可清理。

## C3-A v8 资源恢复快照 — 2026-11-06 18:35（中国标准时间）

2026-11-06 10:34:26 UTC GPU 空闲 1,453/4,096 MiB、2 个其他 CUDA 计算进程；训练尚未启动。磁盘等待器记录 13.611 GiB，独立读取 13.47 GiB；约四分钟前独立读取为 4.88 GiB，仓库期间没有对应大文件增删，原因未知。测试/编译生成缓存已清理，共 49,454 字节。本轮无模型结果或数据载荷访问。

## C3-A v8 等待资源快照与空间管理 — 2026-11-06 18:31（中国标准时间）

v8 等待器于 10:30:25 UTC 记录项目分区可用 4.909 GiB；独立复核于 10:30:26 UTC 为 4.88 GiB。10:19:25 UTC 同一等待器记录为 12.135 GiB，约 11 分钟下降 7.23 GiB；当天仓库没有新增大于 10 MiB 文件，空间变化原因未知。GPU 快照为空闲 1,229/4,096 MiB、3 个其他 CUDA 计算进程，未达冻结门槛，训练启动时间仍为空。仅清理本轮测试产生的 10 个可重建缓存文件，共 26,850 字节；无大型本任务结果可清理，数据、历史结果、checkpoint 和其他项目文件均保留。等待器在磁盘低于 2 GiB 或 GPU 未达门槛时不会启动实验。

## C3-A GPU 等待器 v8 恢复 — 2026-11-06 18:20（中国标准时间）

**实验状态：** UNDER EVALUATION（等待资源；尚未训练）。v7 最后更新于 2026-11-06 09:32:03 UTC，启动时间未写入；当前未发现其 Python 进程，C3-A 训练与评价目录不存在，退出原因未知。旧等待入口固定指向已存在的 v7 目录，复现后确认会在覆盖保护处退出。

**代码与验证：** 给 `scripts/wait_for_c3a_gpu.py` 增加必填唯一编号参数 `--run-id`，并将精确重启命令写入等待记录。测试先因缺少按运行编号创建目录的行为而失败；实现后 `tests/test_wait_for_c3a_gpu.py` 5 项通过，相关文件 `py_compile` 通过。没有变更训练/评价协议、C2 gate 或任何历史实验。

**v8 资源状态：** 工具会话 `42473`，记录目录 `experiments/diagnostics/c3a_gpu_wait_20261106_v8/`。2026-11-06 10:19:26 UTC GPU 空闲 1,453/4,096 MiB、2 个其他 CUDA 计算进程，未达冻结门槛。等待器在 10:19:25 UTC 读取磁盘可用 12.135 GiB，约一分钟后的独立读取约 9.46 GiB；以等待器预设 2 GiB 下限继续监控，低于下限时只等待。

**数据/结果边界：** 训练与评价目录均不存在；没有读取 train/development/confirmation/test/optical 载荷。当天项目内没有新增大于 10 MiB 的文件，也没有可确认属于本任务的大型可清理结果；原始/派生数据、历史结果和 checkpoint 全部保留。

## C3-A GPU 等待器磁盘余量保护 — 2026-11-06 12:00（中国标准时间）

**v5 停止证据。** 为应对共享分区快速波动，v5 于 03:50:26 UTC 在训练启动前安全中断；训练启动时间为空，确认/测试/光学载荷访问标记均为 false，训练与评价输出目录不存在。保留 v5 manifest。

**保护实现与验证。** 等待器曾设 8 GiB 最低可用空间门槛，但该阈值在 5.646 GiB 余量时阻止了实验，而静态网络仅有 323,617 个参数（FP32 参数约 1.23 MiB），训练和诊断只保存 checkpoint 与 CSV，不复制图像。为保留 2 GiB 余量并避免无依据地长期阻塞，门槛改为 2 GiB，并在训练和多次采样评价各自启动前复查；低于门槛只等待，不删除数据或历史结果。针对配置调整的测试先观察到预期失败，修正后 AISAR 下等待器专项 5 项通过，等待器和测试文件 `py_compile` 通过。原 C3-A、多次采样、Flow 与共享评价器 44 项回归此前通过；本轮 pytest/Python 缓存已清理。

**v6 运行状态。** v6 于 03:51:04 UTC 以唯一目录启动，之后在训练前安全中断；运行期间曾因磁盘低于 8 GiB 而等待。GPU、磁盘变化期间均未启动训练或评价。

**v7 运行状态。** v7 于 03:59:39 UTC 以唯一目录启动，工具会话 `87305`。启动快照 GPU 空闲 1,280/4,096 MiB、3 个其他计算进程；磁盘可用 16.032 GiB、门槛 2 GiB。最新快照（06:28:49 UTC）GPU 空闲 1,070/4,096 MiB、3 个其他计算进程；磁盘可用 16.429 GiB。训练仍未启动。

**数据与存储。** 近 20 分钟未发现仓库内大于 20 MiB 的新文件；训练/评价输出目录不存在，`data`、`experiments`、`outputs`、`checkpoints`、`weights` 大小无明显变化。本任务没有可安全清理的大型产物；仅清理本轮测试生成的可重建缓存，原始/派生数据和历史实验均保留。confirmation/test/optical 与 development 载荷访问标记为 false。

## C3-A 跨日 GPU 等待快照 — 2026-11-06 10:11（中国标准时间）

**执行状态。** 等待记录 v4 和工具会话 `23990` 仍在运行；最近 GPU 快照为 2026-11-06 02:11:19 UTC，RTX 3050 空闲 1,153/4,096 MiB、3 个其他 CUDA 计算进程、利用率 100%。未达到启动条件。

**结果与访问边界。** C3-A 训练目录和多次采样评价目录均不存在；train/development 数据载荷尚未读取。confirmation、test、optical 访问标记均为 false。保留原协议、v3 中断现场和所有历史实验。

**磁盘。** 分区可用空间约 17.74 GiB；本任务只有轻量等待 manifest/命令文件，没有生成大型结果或数据。此前清理仅限仓库 Python/pytest 可再生缓存。

## C3-A 多次采样结构诊断 v2 与 GPU 等待 — 2026-11-05 21:32（中国标准时间）

**协议状态。** 当前补充协议为 [`multi_sample_stability_c1_v2_20261105_v2.json`](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v2.json)，SHA-256 `8d886f26023f9d929265b3f70ad9f51314f94788b1e4a01f269af5c1f0a589fc`。v1 在训练启动和任何影像载荷访问前被 v2 替代，保留作协议历史。v2 对每个 development 身份固定 4 次采样，并补入以固定 LR/noisy pseudo-target 生成、在 baseline 与全部 draws 间复用的 edge mask；报告 EPI、边缘强度比和 `edge_mask_manifest.csv`。结构指标只代表相对 noisy pseudo-target 的表示域一致性。

**实现与验证。** [`evaluate_c3a_multisample.py`](../../scripts/evaluate_c3a_multisample.py) 输出协议列明的逐图、身份汇总、来源组 bootstrap、运行环境/时间/显存与 edge-mask manifest。针对协议版本/输出项和失效边界补入测试；AISAR 下 C3-A 定向测试 18 项通过，合入共用 evaluator 回归后 44 项通过，相关 Python 文件 `py_compile` 通过。当前没有真实训练或 development 指标。

**训练数据访问复核。** 训练入口只构造冻结 C1 v2 train/development 数据集，并由 `SarTargetOnlyDataset` 依据文件清单直接载入 `lr` 与 `target`；不会经通用取样器载入 optical。合成测试在没有 optical 文件夹时验证该行为。此次静态复核没有打开任何项目影像。

**等待与访问边界。** 等待记录 v3 的会话句柄失效，进程已退出；检查时 `manifest.json` 为 0 字节，训练和评价输出目录不存在。保留旧现场。等待记录 v4 已启动，见[`c3a_gpu_wait_20261105_v4`](../../experiments/diagnostics/c3a_gpu_wait_20261105_v4/manifest.json)；13:32:31 UTC 快照空闲 1,637/4,096 MiB、3 个其他 CUDA 计算进程，13:39:32 UTC 空闲 1,520/4,096 MiB、4 个进程，13:58:33 UTC 空闲 1,341/4,096 MiB、3 个进程，14:05:34 UTC 空闲 1,125/4,096 MiB、4 个进程，14:12:34 UTC 最新快照空闲 1,349/4,096 MiB、3 个进程；利用率依次为 97%、97%、96%、99% 和 100%，所以训练仍未启动。固定 train/development 清单分别有 1,600/200 行，C1 来源清单与 baseline 文件存在；未打开图像。confirmation/test/optical 和 development 载荷均未读取。

**等待器修订与磁盘。** 为避免覆盖失效的 v3 记录，等待器唯一输出目录改为 v4；先将测试改为要求 v4 并观察到其因实现仍指向 v3 而失败，再修正代码。C3-A 定向与共享评价器回归共 44 项通过，等待器及测试文件 `py_compile` 通过。只清理仓库内可再生 Python/pytest 缓存约 4.83 MiB；未删除数据、历史实验或其他项目临时文件。仓库分区快照约有 17.3 GiB 可用。

**后续磁盘快照。** 2026-11-05 13:37 UTC 仓库分区剩余约 11.2 GiB，较 13:32 UTC 少约 6.1 GiB；C3-A 训练和评价目录仍不存在，JDSR 仓库没有对应的大型新增文件。保留来源不明的用户缓存及其他项目结果，只继续监视。

2026-11-05 13:50 UTC 复查可用空间约 16.2 GiB；13:58 UTC 为 13.37 GiB，14:05 UTC 为 11.18 GiB，14:12 UTC 为 9.56 GiB。C3-A 输出目录仍未创建。空间波动未关联到本实验，不做外部缓存或历史结果清理。

## C3-A 多次采样补充协议与资源等待执行 — 2026-11-05 16:18（中国标准时间）

**前置情况。** 原安全等待会话 `25767` 已停止，C3-A 尚未启动。停止前未满足 3,072 MiB 显存和零其他 CUDA 进程门槛，未读任何影像。2026-11-05 16:03 提权查询记录 RTX 3050 Laptop GPU 空闲 1,303/4,096 MiB、利用率 96%、3 个其他计算进程。

**补充协议。** [`multi_sample_stability_c1_v2_20261105_v1.json`](../../protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v1.json)，SHA-256 `9c00accfc9cd046783323735509db9505a09a6351bcf55d93e5699ebf9970b77`。只在 C3-A v1 训练结束后运行：固定 C3-A best checkpoint，对 200 个 development 身份各采样 4 次；输入限 SAR LR 与 noisy pseudo-target，不读 optical。与既有 C2 x2_light SAR-only best checkpoint 同身份配对。每个身份先汇总 draw，随后按 19 个来源组等权并做 10,000 次 bootstrap；评价固定全局 seed、deterministic algorithms 与 cuDNN deterministic 设置。PSNR/SSIM 采样标准差和最终裁剪预测的逐像素标准差只作条件采样波动描述；不当作训练随机性或分布校准证据，不用于选轮或改变任何 gate。

**代码与可复现性。** [`evaluate_c3a_multisample.py`](../../scripts/evaluate_c3a_multisample.py) 保存 baseline 逐图 CSV、Flow 每次 draw、身份级汇总、组 bootstrap、环境/命令/代码哈希、两方法运行时间和峰值 GPU 显存；在开发载荷校验之前先将访问标记写为 true，失败时保留失败 manifest。修复了协议 schema 与验证器字段不符，以及训练协议中的说明性 `architecture` 键不能直接传入模型构造器的问题。[`wait_for_c3a_gpu.py`](../../scripts/wait_for_c3a_gpu.py) 只查询 GPU 资源，满足零并发进程和至少 3 GiB 空闲后依次执行 C3-A 训练及补充评价；失败或已有半成品输出时停止并保留现场，不覆盖结果。

**验证。** `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_c3a_multisample_eval.py tests/test_c3a_gpu_waiter.py tests/test_phase_c3a_flow.py`：16 项通过；对应六个 Python 文件通过 `py_compile`。以上测试不包含真实完整训练或开发集评价，当前仍无 C3-A 指标。

**自动继续状态。** 等待会话 `28461` 已运行，记录位于[`c3a_gpu_wait_20261105_v2`](../../experiments/diagnostics/c3a_gpu_wait_20261105_v2/manifest.json)。2026-11-05 16:26 UTC 快照为 1,197/4,096 MiB 空闲、3 个其他 CUDA 计算进程、利用率 69%；未启动训练或读取载荷。等待器只会在满足协议门槛后运行训练和评价。

## C3-A 安全 GPU 等待 — 2026-11-05（旧等待会话 25767，已停止）

第九次原训练入口检查检测到 3 个其他 CUDA 计算进程，退出码 2、未读影像。提权状态查询曾显示 1,192/4,096 MiB 空闲及 96% 利用率。为避免高频重启，启动单个只读状态等待器：每 60 秒检查 CUDA 进程和空闲显存；仅当进程数为零且空闲显存至少 3,072 MiB 时才调用原 C3-A 训练入口。最近轮询为 3 个进程、1,224 MiB 空闲。等待器未读取影像，训练器启动时仍进行自身预检。记录见[预检 v9](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v9/manifest.json)及[等待 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261105_v1/manifest.json)。

网络静态资源信息：323,617 个可训练参数，FP32 权重与偏置理论占用 1.23 MiB；未报告 MACs/FLOPs（当前无可靠工具）。这不代表实测训练/推理显存或效率。

## C3-A 第八次 GPU 预检 — 2026-11-05（训练未启动）

按原预登记入口再次尝试。GPU 对训练器可见，但检测到 5 个其他 CUDA 计算进程，训练器以退出码 2 在读图前停止；没有创建训练目录或访问影像、光学、confirmation、test 载荷。记录见[预检 v8](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v8/manifest.json)。C3-A 尚未产生训练结果。

## C2 零背景对比差诊断原型 — 2026-11-05（合成验证）

**问题与实现。** C2 v2 的必需 median TBR 在零背景中位数时未定义。新增独立的 `compute_c2_contrast_diagnostic_metrics`，计算目标区—背景区均值和中位数差值及其预测误差；target 与 prediction 的 mean/median TBR 各自在自身背景分母严格为正时返回数值，否则该侧为 `null`。只有两侧分母均为正时，成对状态才标记为 `defined`；状态字段指出未定义端。实现不接入正式 C2 v2 evaluator，不改变原协议、阈值、失败状态或 B1 gate。

**验证。** [`test_c2_contrast_diagnostic.py`](../../tests/test_c2_contrast_diagnostic.py) 使用合成数组覆盖双侧零背景、正背景、单侧零背景和重叠掩膜。命令 `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_c2_contrast_diagnostic.py tests/test_phase_c2_confirmation_metrics.py tests/test_evaluator.py --tb=short` 结果为 43 项通过；`py_compile` 通过。没有读取 train/development/confirmation/test 影像，也没有重算模型预测。解释范围仅是指标代码性质验证。

设计细节见[C2 零背景对比差诊断](C2_ZERO_BACKGROUND_CONTRAST_DIAGNOSTIC.md)。

## C3-A 第七次 GPU 预检 — 2026-11-05（训练未启动）

按原协议运行 `/home/shy/anaconda3/envs/AISAR/bin/python scripts/run_phase_c3a_sar_flow.py`。GPU 对训练器可见；检测到 4 个其他 CUDA 计算进程，退出码 2。执行器未读取影像、未创建训练目录、未访问 optical/confirmation/test。完整记录见[预检 v7](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v7/manifest.json)。该尝试不产生训练或模型效果结果。

## C3-A SAR-only Flow Matching 筛选预登记与 GPU 预检 — 2026-11-05（训练未启动）

**目的与划分。** 做一个 SAR-only 条件流匹配方法可行性筛选；直接使用冻结 C1 v2 按来源场景文件夹划分，不重新切分。`x2_light` train 1,600/92 组只用于训练，development 200/19 组用于逐轮评价和选择；confirmation/test 均不读取。development 已被 C2 用于选择，因此所有本次对比都属于复用开发集上的探索性观察。

**冻结协议。** 详见[预登记协议](../../protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json)，SHA-256 为 `147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`。方法为像素域 SAR-only conditional rectified flow：condition 为固定 LR SAR 双线性上采样，source 为 condition 加 `0.1 * N(0,I)`，在均匀时间点构造线性插值并以未加权 velocity MSE 训练；8 步 Heun 采样，不裁剪中间状态，最终输出仅为指标计算裁剪到 `[0,1]`。预算为 5 轮、batch=4、单 seed；最佳轮次按来源组等权 development PSNR 选择，较早轮次打破平分。停止/继续标准要求配对 ΔPSNR 至少 +0.05 dB、来源组 bootstrap 95% 区间下界大于 0，且配对 ΔSSIM 来源组均值不低于 0；该标准只决定是否追加第二 seed，不是晋级门槛。

**固定输入与对照。** 输入根为 `data/derived/c2_fixed_inputs_c1_v2_20261031_v1/x2_light/{train,development}`，身份由 C1 v2 sample manifest 绑定。对照为既有 `x2_light__sar_only` best checkpoint，SHA-256 为 `8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165`；完整训练执行后还须复算对照 development score，并与既有 run manifest 核对。C3-A loader 仅读 `lr` 与 `target` `.npy`，不打开 optical payload。

**代码与验证。** 新增 [`phase_c3a_flow.py`](../../MMCANetV2/phase_c3a_flow.py) 和[训练入口](../../scripts/run_phase_c3a_sar_flow.py)。AISAR 中 6 项定向测试通过；相关 Python 文件 `py_compile`、协议 JSON 解析和 `git diff --check` 通过。真实冻结 train/development 各取 1 个身份进行 CPU 通路检查：一次 train 反向传播和一次 8 步 Heun development 推理均有限；无指标用于选模。数据审查确认 SAR LR 可大于 1（固定退化 `upper_clamp=false`），代码不对 LR 上限裁剪。

**资源与状态。** 首次 GPU 预检为 RTX 3050 空闲显存 1.06/3.67 GiB，低于预登记 3 GiB 门槛；后两次预检检测到 3 个其他 CUDA 计算进程。三次启动器均在读图前退出，未创建训练目录、未读取数组、未启动训练。保留[显存预检](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v1/manifest.json)、[进程预检一](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v2/manifest.json)和[最终预检](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v3/manifest.json)。因此当前没有生成质量、运行时、完整开发集指标或假设更新；C2 gate 与 B1/B2/B3 状态均不变。C3-A 本轮 `confirmation_payload_accessed=false`、`test_set_accessed=false`、`test_set_used=false`、`optical_payload_read=false`。

本轮经用户允许的 GPU 提权，在 `AISAR` 环境再次执行同一入口；最新预检仍发现 3 个其他 CUDA 计算进程，训练器在读图前退出（退出码 2）。本次未查询空闲显存数值、未创建训练目录、未读取数组；完整命令和输出见[预检 v6](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。此前系统默认 Python 缺少 PyTorch 的导入错误已纠正。

## C1 v2 confirmation 上的 C2 一次性复核 — 2026-11-05（访问后修订；非晋级结果）

**目的与划分。** 沿用冻结 C1 v2 来源场景文件夹划分，不重新随机切分：train 1,600/92 组、development 200/19 组、confirmation 200/20 组。复用已完成的 C2 v2 八组 best checkpoint。confirmation 来源组文件夹互斥；地理/采集独立性与精细空间重叠未核验。

**协议和输入。** C1 manifest SHA-256 为 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`；父 C2 v2 协议 SHA-256 为 `e7b2eb183968dab3512aa2a5ff65244ebf43764dbdd9baa2d25bfffa0864bd1e`。确认协议初版 SHA-256 为 `b46df6eb33f5b4e122868d337b82441db5263c340610bf5a3e220e09f572c523`；访问后修订一/二的 SHA-256 分别为 `31d27e3f2b2f70d57b9fb726c1f48e34f9faa0348e48e95eba188d1b823fe77d` 和 `ef9fe360c0cc19aa5f72eb2ad8af07381c1bb333f5e71267349498ae8437267c`。两次修订只修复已冻结置换路径和判定阈值的代码字段映射，没有改变样本、模型、checkpoint、退化、指标、阈值或判定规则。修订详情见[协议目录](../../protocols/mmcanet_phase_c2/c2_confirmation_20261105_v1/)。

**执行记录。** 暂存 manifest SHA-256 为 `300b04f307a0f9c12f327442d4e27633104d0909311c81e7b803e4424a4353e5`；四条件 fixed-input 执行清单 SHA-256 为 `9e54dfa99a218498cde35fdda95888bcf63ac9c50137d7a9c8da344af5b64003`。首轮在确认数组校验后、推理前因置换路径字段映射失败；第二轮完成四条件推理并在内存生成 3,200 行指标，但在阈值读取处失败，未写出/查看指标。技术恢复以同一冻结设置重新执行并成功落盘；失败记录分别见[`首轮`](../../experiments/phase_c2_confirmation_c1_v2_20261105_v1/)和[`amended_v1`](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v1/)，最终运行位于[`amended_v2`](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/)。恢复 manifest SHA-256 为 `70f443f897cfbb7849be53a6d164475b19f884051e7d58ef5674ff869592faf3`。

**结果。** 运行覆盖 200 个样本、20 个来源组、四种退化和四种推理控制，输出 3,200 条逐图控制指标、1,600 条配对对比、72 条汇总和 1,440 条来源组指标；EPI 未定义数为 0。按来源组等权的 matched−SAR-only PSNR 增益依次为：x2_light −0.0114 dB（95% 区间 [−0.0192, −0.0052]）、x2_severe −0.0108（[−0.0290, +0.0015]）、x4_light +0.0048（[−0.0050, +0.0158]）、x4_severe −0.0098（[−0.0328, +0.0044]）。matched−permuted 分别为 +0.0035（[−0.0027, +0.0092]）、+0.0038（[−0.0074, +0.0139]）、+0.0047（[+0.0003, +0.0085]）、+0.0047（[−0.0014, +0.0098]）dB。预登记状态为 `inconclusive`，没有条件达到 +0.05 dB 且 bootstrap 下界大于零的 matched−SAR-only 门槛。完整逐图指标、配对区间和 hash 见[中文结果报告](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/report.md)与执行 manifest。

**解释边界。** 这是访问后修订并经过一次技术恢复重跑的单 seed 描述性复核，不是原始严格预注册的晋级确认，也不构成独立地理泛化或辐射物理证据。C2 v2 原 screen 仍保持 fail-closed、“未判定”；B1 仍为 `not_promoted / stop_at_b1`，B2/B3 关闭。confirmation 今后不得用于训练、调参或模型选择。本轮 `confirmation_payload_accessed=true`、`test_set_accessed=false`、`test_set_used=false`；项目累计 test flags 仍按旧污染历史保留。

**验证。** 9 个相关 Python 源码/测试文件通过 `py_compile`；确认链定向测试 29 项通过。排除缺少 `rasterio` 的收集文件后，全仓为 394 项通过、20 项失败、4 项跳过；失败测试与依赖问题见[验证记录](../../experiments/phase_c2_confirmation_c1_v2_20261105_validation_report.md)。

## C2 v2 冻结筛查：C1 v2 train/development — 2026-10-31（训练完成；正式 screen fail-closed；补充诊断完成）

**目的。** 在项目既有 C1 v2 来源场景文件夹划分上，按冻结 C2 v2 协议检验 matched optical 相对 SAR-only 联合基线的开发集增量价值；不重新划分数据，不访问 confirmation。

**数据。** train 1,600 身份/92 来源组；development 200 身份/19 来源组；confirmation 200 身份保留。官方 train/development 共 3,600 个 PNG，按冻结 C1 v2 身份、字节数、SHA-512、PNG 模式与尺寸核验。暂存运行见[下载与核验 manifest](../../experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json)，源样本清单 SHA-256 为 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`。

**协议。** `mmcanet_phase_c2_screen_20261029_v2`，冻结 JSON SHA-256 为 `e7b2eb183968dab3512aa2a5ff65244ebf43764dbdd9baa2d25bfffa0864bd1e`。四种固定退化条件，各运行 `sar_only` 和 `optical_guided` 两个 arm；30 epoch、batch=8、固定 seed、无增强/无混合精度；总计 48,000 个 optimizer steps。协议与参数未改。

**输入与运行。** 四条件 × train/development 共 8 个固定输入单元、21,600 个 `.npy` 数组、约 7.85 GB；[输入生成 manifest](../../experiments/phase_c2_c1_v2_20261031_v1/fixed_input_generation_manifest.json) 全部完成。训练矩阵位于 [`matrix/`](../../experiments/phase_c2_c1_v2_20261031_v1/matrix/)，八组均完成 30/30 轮，共 48,000 个优化步骤；开发集最佳轮次依序为 30、30、28、28、26、30、28、28。GPU 为 RTX 3050 Laptop，环境为 AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8。运行矩阵和 checkpoint 哈希见矩阵 `manifest.json`。

**正式 screen。** 冻结 paired screen 在首个必需指标遇到 `Target background median is zero or too small for a defined ratio` 后 fail-closed；原始失败 manifest 保留在 [`screen/execution_manifest.json`](../../experiments/phase_c2_c1_v2_20261031_v1/screen/execution_manifest.json)。诊断定位到 2 个唯一 development 身份、按条件形成 6 条条件×样本记录；target 背景零像素比例为 53.1%–71.2%，而背景均值仍为正。协议规定此类必需指标未定义时整套 screen 失败；未加 epsilon、未删样本、未修改协议。故正式 C2 v2 gate 为“未判定”，不能表述为通过或停止。

**非门槛诊断。** [补充报告](../../experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)保留全部 200 个 development 身份、19 个来源组和四种控制，生成 3,200 条逐图控制与 3,200 条配对对比；仅将 `tbr_median_abs_error` 和 `abs_log_median_tbr_ratio_error` 标为不可用。matched optical 相对 SAR-only 的来源组等权 PSNR 增益为 x2_light −0.00552 dB、x2_severe +0.00421 dB、x4_light +0.00610 dB、x4_severe −0.00240 dB；四个 95% bootstrap 区间均跨零且点估计均低于预设 +0.05 dB 继续阈值。matched 相对 permuted optical 为 +0.00994 至 +0.01516 dB，四个区间下界均大于零。此为单 seed、同一 development 集上的描述性诊断，不产生 gate decision，也不证明稳定效用或独立泛化。

**访问边界与限制。** 本轮 C1 v2 `confirmation_payload_accessed=false`、`test_set_accessed=false`、`test_set_used=false`。单 seed source-folder-disjoint development screen 不作为独立泛化证据或晋级 B1/B2/B3 的依据。C1 来源组—地理/采集交叉表和精细空间重叠仍缺；仓库 `data/` 数字命名子集未用于本轮。历史 test 污染累计标记仍按既有记录保留。

**未解决限制。** C1 来源组—地理/采集交叉表和精细空间重叠仍缺；`data/` 数字命名 2k 子集与 C1 v2 身份没有可审计映射，未混入本次运行。此前旧本地训练误用 13 个 `test_100` 样本的历史事件仍需在累计访问标记中保留；C1 v2 confirmation 未受影响。

## C2 配对屏幕执行链与 CUDA 资源复核 — 2026-10-31（载荷取得前的代码/合成验证历史记录，非方法实验）

以下记录描述当时数据尚未取得时的代码验收状态；真实 C1 v2 训练与评价结果以本页首条登记为准。

完成 C1 v2 train/development 角色受限暂存入口、八组 CUDA 训练矩阵/恢复入口和完整 development 配对评价入口。配对 screen 按冻结循环偏移 100 构造跨来源组 optical donor，逐图复用固定 mask，输出四种控制、四项冻结 contrast、source-group 等权 bootstrap、mask SHA-256/阈值/像素数/coverage、决策 JSON/中文报告及访问 flags。未修改冻结 JSON 或 gate。

提权 AISAR 下 9 个 C2 定向测试文件 **59 项通过**，14 个 C2 源码/测试文件 `py_compile` 通过。RTX 3050 Laptop GPU 上冻结模型合成 batch-1 推理预检通过（峰值 PyTorch 分配约 20.9 MB，未读取项目数据）。此前 batch=8 合成训练预检曾因共享显存不足 OOM；2026-10-31 重试在运行前可用显存 470.9 MiB 时通过一步冻结合成训练，峰值分配 365.8 MiB、运行后余 52.9 MiB。用户指出的 `data/` 有 2,000 对数字命名 PNG，但 train/development 清单路径匹配为 0；受限暂存未复制文件。故未运行真实输入生成、训练或方法比较。C2 仍无方法结果；B1 仍为 `not_promoted / stop_at_b1`，B2/B3 关闭。详见 D-2026-10-31-004。

本轮读取状态/研究文档时出现历史 test 关联文字，故本轮文本暴露标记为 `test_set_accessed=true`、`test_set_used=false`；没有打开底层 test payload/metrics/checkpoint，累计项目标记仍为 `true/false`。详见[C2 代码/资源审计](../../experiments/diagnostics/c2_execution_acceptance_20261031_v1/report.md)、[文本暴露审计](../../experiments/data_audits/research_text_read_exposure_20261031_v1/report.md)及 D-2026-10-31-001/002。

## C2 冻结来源绑定与 fail-closed 指标合成验收 — 2026-10-30（代码验证，非模型实验）

类型：C2 v2 数据合同、指标和执行 helper 的合成测试，不是正式模型实验。固定输入生成器/读取器核验冻结协议 ID/哈希及 C1 来源清单 pin；指标 wrapper 固定 `min_signal=1e-3`、`eps=1e-8` 与 evaluator revision，未定义/非有限必需指标整项失败；掩膜记录含条件、阈值、像素数与覆盖率。四条件审计只读 manifest/files.csv，核验身份、来源组、target/optical/source 哈希关系，不加载 `.npy`。训练 helper 覆盖共同初始化、seeded DataLoader、group-balanced PSNR 选择及使用合成数据/测试模型的 CPU epoch 测试。AISAR 下 C2 五个定向文件为 29 项通过、3 项 CUDA 跳过；协议和来源清单哈希与冻结值一致。未读取项目图像、确认/测试数据、checkpoint 或正式实验指标；本次 `test_set_accessed=false`、`test_set_used=false`，累计仍为 `true/false`。八次训练/恢复、paired screen 与结果 writer 未实现；B1/B2/B3 状态不变。详见[执行验收审计](../../protocols/mmcanet_phase_c2/execution_acceptance_audit_20261030_v2.md)。

## 流匹配后验采样文献审查 — 2026-10-30（仅文献核验，未做实验）

类型：两篇 2026 年一手论文全文审查及相关工作矩阵更新，不是实验运行。新增 D-Flow SGLD 与贝叶斯条件流后验记录；矩阵共 144 行，78 行旧清单归属已确认、66 行未确认。未产生代码、数据、模型或结果；未访问本地数据、测试／确认集或指标，未运行项目代码、实验或测试。本次测试标记为 `false/false`，项目累计为 `true/false`。见 D-2026-10-30-007 和[审查报告](../../experiments/literature_audits/flow_posterior_sampling_frontier_20261030_v1/report.md)。

## OGSFlow 作者仓库 README 审查 — 2026-10-30（仅文献核验，未做实验）

类型：公开仓库 README 来源审查，不是实验运行。记录了作者对两阶段结构估计／条件流生成以及两阶段 RGF 的描述；源码、固定版本、论文、协议和结果未核实。候选更新为 v1.3-proposed，仍待批准。未新增矩阵行，未产生代码、数据、模型或结果，未访问项目本地数据、测试／确认集或指标，也未运行实验或测试。本次测试标记为 `false/false`，项目累计为 `true/false`。见 D-2026-10-30-006 和[审查报告](../../experiments/literature_audits/ogsflow_readme_details_20261030_v1/report.md)。

## DiSpeckle 书目来源审查 — 2026-10-30（仅文献核验，未做实验）

类型：元数据与全文入口核验，不是实验运行。ORCID 中 Crossref 来源的作品记录支持书目元数据；IEEE 全文及作者稿未取得。未将二手摘要作为方法证据，未添加矩阵行，未产生代码、数据、模型或结果。见 D-2026-10-30-005 和[审计报告](../../experiments/literature_audits/dispeckle_metadata_route_20261030_v1/report.md)。

## 自适应参考引导相关工作审查 — 2026-10-30（仅文献审查，未做实验）

审阅 ICLR 2026 Ada-RefSR 论文记录与稿件、作者可变版本的注意力处理器源码，以及出版方索引中的 STMP-DiT 文本。新增两条当前证据文献记录。自然图像参考超分辨率中已有自适应参考门控、无关参考控制和模糊/仿射压力测试；STMP-DiT 另报告了一次含糊图像描述导致的失效。记录了公式/代码的静态差异，但未据此推断方法失效。未运行项目图像、模型、checkpoint、预测或实验。宽范围本地文本搜索可能遍历历史测试相关文本；当前标记为 test_set_accessed=true、test_set_used=false，未使用任何结果。矩阵共 142 行（78 行确认属于旧清单，64 行归属未确认）。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。

## VFM 目标与似然兼容性核对 — 2026-10-30（只读，未做实验）

将 VFM 目标与目标/表示审计、冻结的 C2 v2 JSON 以及相关采样器、写入器和读取器源码进行了对照。本地高分辨率目标是已含噪的归一化表示；C2 定义了该域内的 FFT、降采样、负值截断和 Gamma 退化。已解析推导候选连续域逐像素似然，包括零点原子和大于 1 的支撑范围。完整后验目标与 float32 舍入语义仍未验证。未访问图像、测试/确认集数据、实验指标、模型或 checkpoint；未运行或修改代码，也未启动实验。见 D-2026-10-30-002/003 和[交叉核对](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md#local-target-and-observation-model-compatibility-crosswalk)。

## 条件流逆问题文献审查 — 2026-10-30（未做实验）

审阅了 ICML 2026 官方 VFM 记录与论文全文，以及 NeurIPS 2025 BFM 论文记录。VFM 是在 ImageNet、已知算子和加性高斯噪声假设下的条件图像逆问题及超分辨率先例；BFM 仅提供通用生成效率证据。新增两条文献记录，矩阵共 140 行（78 行确认属于旧清单，62 行归属未确认）。未访问 SAR 图像、测试/确认集数据、实验 checkpoint、预测或指标；未运行项目代码、模型或实验。候选章程、B1/B2/B3 门槛和 C2 状态均未改变。见 D-2026-10-30-001 和[来源审查](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md)。

## Required pilot-paper review text exposure — 2026-10-29 (no experiment)

The source-brief-required read of the internal pilot draft surfaced historical test-associated prose/outcomes, and a bounded formal-paper README search surfaced a fixed-test artifact reference. Text only; no test image, fixed input, prediction, metric file, checkpoint, or test manifest was opened. No result value informed design or selection; exclude the surfaced values and mini identities from future tuning. Current-task flags are `test_set_accessed=true`, `test_set_used=false`; cumulative flags remain `true/false`. No run was started. See D-2026-10-29-030 and [exposure group 518](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md#group-518--source-brief-review-reread-historical-pilot-test-prose).

## Historical gate report review — 2026-10-29 (no experiment)

Read the legacy B1/B1R/B1F reports, source-aligned B1 v2 aggregate, and Phase 1-R/S/T reports requested by the research brief. The recorded failures/stops are preserved by the inactive candidate's §14; no protocol, code, result, or gate changed. The reviewed experiment reports are validation/inner-development or target-assisted diagnostic records; they provide no independent generalization evidence. No test payload or test metric file was opened. See D-2026-10-29-031.

## C2 protocol/source binding and runner acceptance audit — 2026-10-29 (static; no experiment)

Verified the frozen protocol JSON hash and inspected the C2 generator, reader, model, metric helpers, and staging-script inventory. The pinned C1 v2 sample-manifest hash is not enforced by the generator/reader; the available staging script targets an older train/val cohort; and no C2 cross-condition parity checker, trainer/checkpoint selector, complete paired evaluator/result writer, or runner-level CUDA/resource preflight exists. No tests, project payloads, model, metric result, checkpoint, or experiment were accessed. Current-task flags are false/false; cumulative project flags remain `true/false`. C2 remains closed pending implementation and runner acceptance. See D-2026-10-29-029 and the [audit](../../protocols/mmcanet_phase_c2/execution_acceptance_audit_20261029_v1.md).

## Required manuscript review text exposure — 2026-10-29 (no experiment)

The user-requested review of the paper/project entry documents surfaced historical test-associated result prose. This was text-only; no test image, fixed input, prediction, metric file, checkpoint, or test manifest was opened. Values were not used for design or selection. Current-task flags are `test_set_accessed=true`, `test_set_used=false`; cumulative flags remain `true/false`. No run was started and no run manifest exists. See D-2026-10-29-027 and [exposure group 517](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md).

## C2 metric-formula and mask-coverage source review — 2026-10-29 (no experiment)

Static review found that the frozen C2 v2 ratio policy references `ratio_metrics` and `min_signal=0.001` but leaves its effective `eps=1e-8` in the helper default; the helper can return a null edge-correlation value that the C2 wrapper does not reject; and fixed-mask construction returns thresholds without area coverage. No end-to-end runner exists in the inspected source inventory. This records execution-acceptance requirements only; the frozen protocol and code were not changed, and no data, test/confirmation payload, metric result, checkpoint, model, or experiment was accessed. Current-task test flags are false/false; cumulative flags remain `true/false`. See D-2026-10-29-026 and the [acceptance audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md#c2-metric-formula-and-mask-coverage-contract-review-2026-10-29).

## SEN1-2 representation/title evidence review — 2026-10-29 (metadata/protocol only; no experiment)

The retained official SEN1-2 document describes stored SAR patches as 8-bit sigma-nought dB. C1 v2's selected train/development PNGs were byte-matched to official source, but no byte-to-dB mapping or raw calibrated values are retained. The separate earlier B1 v2 protocol uses a noisy normalized pseudo-target and representation-domain Gamma perturbation. These sources do not support calibrated physical-likelihood or physical-speckle claims. No new image, test/confirmation payload, model, metric, or experiment was accessed. See D-2026-10-29-024, the [C1 v2 eligibility audit](../../protocols/mmcanet_phase_c1/v2/c1_development_eligibility_20261028.md), and the [B1 v2 representation profile](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/report.md).

## CN122694669A official publication route check — 2026-10-29 (literature-only; no experiment)

The official CNIPA publication-announcement page opened but returned no item-specific record through the text reader; linked PSS search returned HTTP 412 and the official publication portal timed out. Secondary-index patent text remains unverified and provisional. No scholarly matrix count, model, protocol, hypothesis, or gate changed. No local data/test/confirmation payload, code, metric, or checkpoint was accessed; no experiment ran. See D-2026-10-29-023 and the [route audit](../../experiments/literature_audits/cn122694669a_official_route_20261029_v1/report.md).

## SEN1-2 official catalog metadata route check — 2026-10-29 (metadata-only; no experiment)

The official ISPRS dataset announcement links to the TUM mediaTUM SEN1-2 record. Direct catalog access returned HTTP 429; exact-ID/site-scoped official searches did not return scene coordinates, acquisition/product IDs, or a crosswalk. No archive or image payload was downloaded, and confirmation/test payloads were not opened. C1 remains source-scene-folder-disjoint only; no eligibility, model, gate, or experiment changed. See D-2026-10-29-022 and the [route audit](../../experiments/data_audits/sen1_2_official_catalog_metadata_route_20261029_v1/report.md).

## Reliability-aware SAR–optical classification literature review — 2026-10-29 (publisher-indexed source; no experiment)

Reviewed publisher-indexed content for Vulli et al.'s 2026 sample-dependent Sentinel-1/Sentinel-2 gate for patch-level classification under synthetic optical degradation. This narrows generic reliability-gate novelty; classification does not establish SAR restoration/JDSR or safe reconstruction fallback. Direct publisher page/PDF/XML access returned HTTP 429 or was unavailable, and author results were not reproduced. The related-work matrix now has 138 rows (78 confirmed historical members; 60 unresolved additions). No local data, code, model, test/confirmation payload, or experiment was accessed; current-task flags are false/false and cumulative project flags remain true/false. See D-2026-10-29-021 and the [audit](../../experiments/literature_audits/reliability_aware_sar_optical_classification_20261029_v1/report.md).

## Charter requirement-coverage review — 2026-10-29 (document review; no experiment)

The candidate's §9 already listed conditional contribution candidates, but its expected academic-contribution deliverable was not explicitly labeled. Candidate v1.2 clarifies the expected output without presuming positive results or changing the scientific scope. No experiment, code, data, model, protocol, gate, or test/confirmation payload changed; current-task flags are false/false and cumulative project flags remain true/false. See D-2026-10-29-020.

## OGSFlow candidate-novelty review — 2026-10-29 (literature/document review; no experiment)

The current public repository README describes optical-guided flow SAR super-resolution and reliability-gated guide fusion, which narrows the inactive charter's candidate novelty. The README provides no verified paper citation or inspectable method/protocol source in the reviewed routes. The candidate and supporting audit were updated; no scholarly matrix row, model, run, protocol, gate, or paper claim was added. No local data, test/confirmation payload, metrics, checkpoint, or model were accessed. Current-task test flags are false/false; cumulative project flags remain true/false. See D-2026-10-29-019.

## GLCdiffcr conditional-consistency diffusion literature review — 2026-10-29 (abstract only; no experiment)

The publisher abstract describes SAR-guided diffusion for cloud-free optical reconstruction and a multi-condition consistency loss combining pixel errors with SSIM. This is optical-output consistency, not a SAR likelihood or SAR-output JDSR. It says data are available on request; dataset/split details were not reviewed. Direct publisher opening returned HTTP 403, and no paper-linked code route was verified. No local data/code, test/confirmation payload, metric, checkpoint, model, or experiment was accessed or used. Current-task test flags are false/false; cumulative project flags remain true/false. No hypothesis, gate, charter status, or experiment changed. See D-2026-10-29-018 and the [audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).

## DMDiff conditional optical cloud-removal literature review — 2026-10-29 (source review; no experiment)

Publisher-indexed text describes DMDiff's SAR-conditioned diffusion for cloud-free optical reconstruction, with gated cloudy-optical features and cross-modal attention. It is reverse-task prior art, not SAR-output restoration or JDSR. Airborne, WHU-OPT-SAR, and LuojiaSET-OSFCR datasets are described; the retrieved text does not establish how the two selected Luojia ROIs map to train/test. The paper-linked GitHub page contains only README.md and says code will be accessible later. No local project code/data, test/confirmation payload, model, metric, checkpoint, or result was accessed; no test or run occurred. Current-task test flags are false/false; cumulative project flags remain true/false. No hypothesis, gate, charter status, or experiment changed. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).

## CDD optical–SAR diffusion-clustering literature review — 2026-10-29 (abstract only; no experiment)

The institutional abstract describes a conditional dual-diffusion model that generates optical images conditioned on SAR, with features used for multimodal K-means clustering. This is adjacent optical–SAR diffusion prior art, not SAR-output restoration/JDSR evidence. The official record lists a final paper link, but retrieval returned HTTP 403; a GitHub candidate was not linked to the authors in the reviewed sources. No project code/data, test or confirmation payload, metric, checkpoint, or model was accessed; no test or run occurred. Current-task test flags are false/false; cumulative project flags remain true/false. No hypothesis, gate, charter status, or experiment changed. Matrix: 135 rows, 78 historical-membership confirmed, 57 unresolved. See D-2026-10-29-016 and the [audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).

## SDIF optical–SAR diffusion-fusion literature review — 2026-10-29 (literature only; no experiment)

Reviewed the full publisher article for Zhang et al.'s SDIF. It uses SAR-BM3D followed by variational diffusion fusion and semantic classification guidance; its evaluated output is a fused image, not a SAR restoration or spatial SR result. The authors report two paired datasets, classification and fusion metrics, and a semantic-guidance tradeoff; these reports were not reproduced. Inference-time label availability for the semantic term is unclear. No project code, data, confirmation/test payload, metrics, checkpoints, or model were accessed; no test or run occurred. Current-task test flags are false/false; cumulative project flags remain true/false. No hypothesis, gate, charter status, protocol, or experiment changed. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).

## SAR split-Gibbs diffusion posterior review — 2026-10-29 (literature only; no experiment)

Reviewed the full arXiv v2 manuscript for a complex SAR Fourier-measurement inverse problem with additive Gaussian noise and alternating Langevin likelihood/DDIM diffusion-prior updates. The paper reports synthetic MNIST and two-scene Sentinel-1A SLC experiments; its real-scene sampling count conflicts between text and tables, and its patch/source/split details are incomplete. It is measurement-domain prior art, not optical-guided raster JDSR or local-likelihood evidence. The same-author IEEE ICASSP record is not merged because version identity is unverified. No project code, data, test payload, metrics, checkpoints, or models were accessed; no run or test occurred. Current-task test flags are false/false; cumulative project flags remain true/false. See D-2026-10-29-014 and the [audit](../../experiments/literature_audits/sar_split_gibbs_posterior_imaging_fulltext_20261029_v1/report.md).

## Flow posterior-transport literature review — 2026-10-29 (literature only; no experiment)

Reviewed the FlowDPS full author manuscript and a full 2026 posterior-transport preprint. The sources distinguish posterior guidance from conditional flow training; the preprint reports exact source reweighting for deterministic flow priors, a 2D guidance/mode-weight counterexample, and an image-scale heuristic with no posterior certification. Neither source evaluates SAR multiplicative noise, optical-conditioned SAR restoration, or the local pseudo-target likelihood. No local code, data, checkpoint, model, test payload, or metric was accessed; no test or run occurred. Current-task test flags are false/false; cumulative project flags remain true/false. No hypothesis, gate, charter status, or experiment changed. See D-2026-10-29-013 and the [audit](../../experiments/literature_audits/flow_posterior_transport_source_review_20261029_v1/report.md).

## SEN1-2 C1 scene-split provenance review — 2026-10-29 (metadata only; no experiment)

The official SEN1-2 paper supports ROI/scene-level partitioning as the intended independence unit. The C1 v2 group assignment separates complete source-scene folders across train, development, and confirmation. Local manifests lack coordinates/footprints and product/acquisition identifiers, so direct geographic and acquisition independence remains unverified. No image payload, confirmation payload, metric, checkpoint, prediction, or model was accessed; no run occurred. Current-task test flags are false/false. See D-2026-10-29-012 and the [audit](../../experiments/data_audits/sen1_2_scene_split_metadata_review_20261029_v1/report.md).

## July manuscript status review — 2026-10-29 (document review; no experiment)

Confirmed the root four-page July PDF is an internal historical pilot. Its source describes a 100-pair mini subset as held-out/test and contains test-associated values. This review did not reproduce or use those values, open any test payload or generated result table, or run a model. No paper, gate, or protocol was changed. Current review exposure is recorded as test_set_accessed=true, test_set_used=false; see D-2026-10-29-011 and the [audit](../../experiments/literature_audits/manuscript_draft_status_review_20261029_v1/report.md).

## SAR-to-EO flow-matching preprints — 2026-10-29 (literature only; no experiment)

Reviewed the full arXiv manuscripts for ContraFM-S2O and ReFlowSET and logged both as reverse-task SAR-to-EO generative prior art. The title search also confirmed OGSFlow remains repository-level because no paper link was found. No code, datasets, project test assets, or models were accessed or run. No H1, gate, or charter status changed. See D-2026-10-29-010 and the [source audit](../../experiments/literature_audits/sar_to_eo_flow_matching_preprints_20261029_v1/report.md).

## Charter candidate coverage review — 2026-10-29 (document review; no experiment)

Reread the full source brief, mapped its numbered requirements to the 15-section candidate, and cross-checked C2 v2 settings against the frozen JSON. The C1 candidate now explicitly prioritizes source-verifiable official SEN1-2 data when eligible. The candidate remains inactive pending approval; no gate or run changed. See D-2026-10-29-009 and the [coverage crosswalk](RESEARCH_CHARTER_CHANGELOG.md#6-requirement-coverage-against-the-source-brief--2026-10-29).

## γ-Bridge literature review — 2026-10-29 (literature only; no experiment)

Reviewed the primary arXiv manuscript for a SAR-only look-parametric Gamma–Lévy diffusion bridge. The paper reports natural-image synthetic training and no-reference evaluation on 33 real-SAR tiles; its code link returned 404. This adds prior-art context, not local evidence, and changed no run, result, or gate. See D-2026-10-29-008 and the [supporting audit](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md).

## C2 v2 execution-acceptance inventory — 2026-10-29 (read-only; no experiment)

Static inspection found source for the matched model, fixed train/development reader, CUDA fixed-input generator, permutation, fixed masks, metrics, and group summary. The C2 generator takes a pre-staged root; no C2 staging entrypoint or end-to-end training/evaluation runner/runner test appeared in the inspected inventory. Synthetic component tests were not run in this review. No project arrays, model forward, training, inference, or evaluation occurred. The protocol remains frozen as a design, and execution acceptance remains incomplete. See D-2026-10-29-007 and the [acceptance inventory](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md#c2-execution-acceptance--static-inventory-2026-10-29).

## C2 fixed-input generator synthetic CUDA verification — 2026-10-29 (no model)

The CUDA-only generator built all four candidate scale/severity conditions from one temporary synthetic SAR/optical pair. Its final test passed on the RTX 3050 Laptop GPU (`1 passed in 2.07s`) and checked deterministic replay, shared target/optical arrays, scale shapes, and rejection of tampered provenance, overwrite, confirmation generation and incomplete inputs. No C1 payload, confirmation/test image or metric, model forward, training or quality result was involved. Actual fixed arrays and the full screen protocol remain pending. See D-2026-10-29-003 and the [contract](../../protocols/mmcanet_phase_c2/fixed_generation_contract_20261029.md).

## C2 fixed-input reader synthetic verification — 2026-10-29 (no model)

`C2FixedDataset` binds train/development identities to the frozen C1 manifest and checks sample hashes and scale-specific shapes. Synthetic ×2/×4 data-loader checks passed, including changed file, confirmation role and cross-role group rejection (`2 passed`). Actual C1-backed C2 fixed data have not been generated. See D-2026-10-29-002 and the [reader contract](../../protocols/mmcanet_phase_c2/fixed_input_contract_20261029.md).

## C2 matched model synthetic CUDA contracts — 2026-10-29 (implementation verification)

An exploratory parameter-matched SAR-only/HR-optical model supports ×2 and ×4. AISAR CUDA on RTX 3050 passed two synthetic contract tests; no project data, training or quality comparison was involved. Fixed input generation, reader and the C2 screen protocol remain outstanding. See D-2026-10-29-001 and the [model contract](../../protocols/mmcanet_phase_c2/model_contract_20261029.md).

## C1 v2 prospective development eligibility — 2026-10-28 (decision; no model run)

The source-folder-disjoint C1 v2 train/development split is accepted for a separately frozen exploratory C2 screen, not a geographic or confirmation claim. B1 and B2/B3 gates remain unchanged. Source inspection identified scale-2-only SAR baselines, scale-2 optical backbone and crop logic; ×4 requires a matched versioned implementation. No model, checkpoint, confirmation or test payload/metric was accessed. See D-2026-10-28-075 and [eligibility decision](../../protocols/mmcanet_phase_c1/v2/c1_development_eligibility_20261028.md).

## C1 v2 narrow-overlap audit — 2026-10-28 (complete; bounded)

Session `77692` completed the train/development-only 32×32 anchor screen: 10,125,000 development windows per modality, zero candidate translations, and 115 optical windows excluded as a repeated signature (zero SAR). The full-intersection verification path had no real-data candidates; a synthetic 51-pixel shift and altered-pixel rejection exercised its components. Staged payload hashes matched and temporary images were removed. C1 eligibility remains under review. See D-2026-10-28-074 and the [report](../../protocols/mmcanet_phase_c1/v2/narrow_overlap_20261028_v1/report.md).

## C1 v2 split-integrity assessment — 2026-10-28 (metadata and existing evidence)

Counted role-specific group capacities and selected samples from frozen manifests, and compared the scope of the completed 128×128 grid scan with the historical 51–52-pixel strip-leakage finding. The latter is a failure-mode precedent, not a detected C1 overlap. Selection remains gated pending a narrow/arbitrary-shift scan and residual independence assessment. No images, models or test data were accessed. See [decision report](c1_v2_split_integrity_decision_20261028.md) and D-2026-10-28-072.

## C1 v2 train/development payload audit — 2026-10-28 (complete; bounded)

Session `26461` completed: 1,800 pairs / 3,600 files matched frozen sizes and SHA-512; all PNGs passed 256×256 uint8 grayscale-SAR/RGB-optical schema checks. No cross-role decoded-pixel duplicate or exact shared 128×128 window on the 64-pixel grid was detected. The runner removed temporary payloads; confirmation and test payloads were not accessed. The first transfer timeout and probe are preserved in D-2026-10-28-070; see D-2026-10-28-071 and the [audit report](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/payload_audit_report.md) for the successful bounded audit.

## MMCANet Phase C1 v2 source-group and sample allocation — 2026-10-28 (metadata audit; no model)

Official `ROIs1158_spring` and `ROIs2017_winter` listings and the SHA-512 index were reconciled into retained per-group capacity and file-hash manifests. The inventory contains 131 source-scene folder groups and 142,506 matched SAR/optical patch indices. Exact candidate file duplicates across groups: 0; exact file-hash overlap with historical B1 v2 input samples: 0. Under seed 20261028, whole groups are allocated 92/19/20 and sampled identities 1,600/200/200 for train/development/confirmation. Group roles and sample membership are frozen at metadata level only. Train/development payload byte/schema/representation and decoded-pixel/bounded-window checks remain pending; confirmation payload remains unaccessed. This does not establish geographic/acquisition independence or fine registration and authorizes no model selection. No test image/metric, model, or checkpoint was accessed; task flags are false/false. B1 remains `not_promoted / stop_at_b1`. See D-2026-10-28-069 and the [C1 v2 protocol package](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/report.md).

## Official unused-season source capacity — 2026-10-28 (metadata audit; no experiment)

Recursive official rsync listings for `ROIs1158_spring` and `ROIs2017_winter` yielded 70 and 61 source-scene folders and 142,506 matching SAR/optical patch indices in total, with no folder-level patch-index mismatches. The roots are outside the cited prospective B1 v2 cohort's fall/summer roots and provide candidate groups for C1. This proves folder-level difference from that cohort only; geographic, acquisition/date, and fine-registration independence remain open. The raw listing/per-scene table was not retained, so rerun before assigning groups or relying on per-scene counts. No split, model, or gate changed; test flags are false/false. See D-2026-10-28-068 and the [audit](../../experiments/data_audits/sen1_2_unused_source_capacity_20261028_v1/report.md).

## MMCANet Phase C1 source-group inventory — 2026-10-28 (protocol evidence only; no experiment)

The evidence inventory covers the inherited 1,800-pair, source-pixel-verified train/development cohort and its 30 official scene folders. Every folder was already used in B1 v2; no new confirmation role is available in the inspected cache. A metadata-only rsync listing of an unused official season root failed authentication; no files were transferred. This is not evidence that more official groups do not exist. The C1 protocol is `blocked_no_unused_confirmation_groups`, freezes no new split, and authorizes no model selection or training. Test flags are false/false; B1 remains `not_promoted / stop_at_b1`. See the [protocol report](../../protocols/mmcanet_phase_c1/protocol_report.md), [command](../../protocols/mmcanet_phase_c1/command.txt), and D-2026-10-28-067.

## MMCANet Phase C1 representation profile — 2026-10-28 (descriptive train/development audit; no model)

All 1,800 frozen train/development candidate files and 3,600 decoded SAR/optical panel hashes matched the retained official-source audit. The channel profile records encoded 8-bit DN statistics only; SAR DN=255 occurs at 2.609% and is not established as physical saturation. The source SAR is described as sigma-nought dB, but the byte transfer/calibration is unavailable. B1 v2 uses a noisy normalized pseudo-target and controlled representation-domain degradation. The selected synthetic test subset passed 38/38; no model, checkpoint, test payload, or metric was accessed. See the [profile report](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/report.md), [summary](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/summary.json), [target/degradation/metric audits](DATA_REPRESENTATION_AUDIT.md), and D-2026-10-28-067.

## Google Drive source-lineage route availability — 2026-10-28 (metadata probe; no experiment)

## Google Drive source-lineage route availability — 2026-10-28 (metadata probe; no experiment)

The current tool catalog exposes Google Drive search/list actions, but three metadata-only document searches and one `list_drives` call failed with backend `Unknown tool`; no Drive metadata or file contents were returned. Do not infer that no source mapping exists. The local numeric-name 2k cohort remains unmapped, model selection remains paused, and B1 stays `not_promoted` / `stop_at_b1`; B2/B3 remain gated. No project data, experiment, test set, model, checkpoint, or result was accessed. See D-2026-10-28-066 and exposure group 515.

## CloudLULC cloud-degraded fusion source review — 2026-10-28 (literature only; no experiment)

The arXiv abstract and author-controlled project README describe ORM/HIAA/USMT and label-guided semantic optimization for cloud-contaminated SAR–optical LULC mapping. The output is semantic classification, and the available sources do not establish optical-guided SAR restoration, SR, or geometric-mismatch-safe fallback. Full paper, runnable code, split independence, and journal status remain unverified; arXiv and repository spell one author's name differently. Add one adjacent-prior-art row; the matrix now has 121 rows (78 historical members, 43 unresolved). No local data, test payload, metric, checkpoint, model, or code was accessed; no experiment or gate changed. B1 remains `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-065 and the [audit](../../experiments/literature_audits/cloudlulc_cloud_degraded_semantic_fusion_20261028_v1/report.md).

## Direction-aware SAR–optical matching metadata screen — 2026-10-28 (literature only; no experiment)

The search-indexed Taylor & Francis publisher listing verifies the paper title, five authors, journal volume/issue, and online date; the official ISPRS citation page confirms the DOI. Opening the listing returned HTTP 403, and direct article/DOI pages were inaccessible through the current reader, so method, data, split, reference points, metrics, results, and code remain unverified. Add one metadata-only registration prior-art row. Matrix inventory: 120 rows, 78 historically confirmed and 42 with unresolved original membership; at least five original records remain unaccounted for against the 125-row lower bound. No project data, test payload, metric, checkpoint, model, or code was accessed; no experiment or gate changed. B1 remains `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-064 and the [audit](../../experiments/literature_audits/direction_aware_sar_optical_matching_metadata_20261028_v1/report.md).

## CUDA-only model-run interface repair — 2026-10-28 (code-only; no experiment)

The remaining model-forward train, inference, validation, calibration, checkpoint-selection, benchmark, and matrix entrypoints under `scripts/` now reject CPU/auto and perform a CUDA preflight before reading inputs or creating outputs. The no-device-flag Phase-B1R representation audit also fails closed. Python compilation and static source scans passed. No training, inference, smoke, test payload, metric, checkpoint, or result was read or produced; no gate changed. The previously completed CPU v2 diagnostic remains frozen as CPU, and the v3 CUDA repeat remains complete. Future model work requires the approved GPU-visible profile, a fresh preflight, and explicit `--device cuda`. See D-2026-10-28-062.

## PolSAR/SinSAR diffusion SR source-route recheck — 2026-10-28 (literature only; no experiment)

A newly surfaced citation led to a direct IEEE Xplore page (`11414190`) and DOI route; the IEEE page showed only JavaScript/robot verification and DOI opening was inaccessible. An ORCID search-index result marked a matching work item as Crossref-sourced, but the direct ORCID page was JavaScript-gated. ResearchGate/aggregator method and performance text was excluded. No primary abstract/manuscript/code became readable, so the item remains outside the matrix. No local test payload, metric file, checkpoint, model, or data was used; no experiment or gate changed. See D-2026-10-28-053 and the [audit addendum](../../experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/report.md#addendum--source-route-recheck-2026-10-28).

## ICSP 2024 SAR diffusion SR bibliographic verification — 2026-10-28 (literature only; no experiment)

The IEEE-copyrighted ICSP proceedings table of contents lists Zilong Chen, Siqian Zhang, and Boli Xiong for the paper beginning on p. 340. The IEEE Xplore DOI record provides an abstract describing residual-prediction Markov-chain diffusion for SAR SR, with MSTAR/SARBake and downstream segmentation evaluation. The full paper, protocol, code, and run lineage were not reviewed; publisher claims were not treated as local results. No project data, test payload, metric file, checkpoint, or model was accessed. No experiment or gate changed. See D-2026-10-28-052 and the [audit](../../experiments/literature_audits/icsp2024_sar_diffusion_sr_metadata_20261028_v1/report.md).

## OSR-Net publisher-indexed source correction — 2026-10-28 (literature only; no experiment)

The official MDPI index exposes nine authors and describes OSR-Net as an optical-SAR registration model taking optical, original-SAR, and affine-offset-SAR inputs, with modality-specific ResNet-18 features, multi-scale channel attention, affine prediction, and spatial transformation. The article's cited arXiv DOI resolves to a separate MultiResSAR review, not an OSR-Net manuscript. Direct publisher retrieval remains HTTP 429, so the low-resolution subset size/split, detailed reference-point protocol, and OSR-Net code/run lineage remain open. This is adjacent registration evidence, not SAR restoration utility or a task-matched baseline. No local model, data, result, or test payload was used. See D-2026-10-28-059 and the [source audit](../../experiments/literature_audits/osr_net_publisher_indexed_text_update_20261028_v1/report.md).

## RGSNet reliability-guided SAR–optical segmentation — 2026-10-28 (literature only; no experiment)

The SSRN-indexed abstract describes frequency-evidence reliability-guided fusion for SAR–optical semantic segmentation under SAR speckle and optical cloud degradation. The direct SSRN page returned HTTP 403. Its linked repository identifies itself as the official PyTorch implementation, says code will be released upon acceptance, and currently lists only a README. Record this as abstract-level adjacent fusion prior art; no restoration, SR/despeckling, mismatch-robustness, or SAR-only fallback claim follows. No project data, test payload, metric, checkpoint, or model was accessed; no experiment or gate changed. See D-2026-10-28-045 and the [audit](../../experiments/literature_audits/rgsnet_reliability_guided_sar_optical_segmentation_20261028_v1/report.md).


## MambaR-ESRGAN official implementation-route recheck — 2026-10-28 (literature only; no experiment)

The University of Waterloo IVC project portal, lab publication listing, and accepted author manuscript did not expose an author-controlled, versioned code route for MambaR-ESRGAN. The old `/code` endpoint was inaccessible through the current reader; this does not establish that no implementation exists elsewhere. No repository or data was downloaded or executed, and no local result, test payload, model, or gate changed. Keep the work as SAR-only joint-restoration prior art and code availability unverified. See D-2026-10-28-044 and the [route addendum](../../experiments/literature_audits/mambar_esrgan_fulltext_audit_20261025_v1/report.md#addendum--official-implementation-route-recheck-2026-10-28).

## Bibliography-to-matrix crosswalk — 2026-10-28 (literature only; no experiment)

Restored two general misalignment-aware works previously recorded in D-2026-10-25-107 but absent from the current CSV, and added Hughes & Schmitt's 2019 SAR–optical matching record at official-abstract level. The matrix now has 114 rows × 12 columns (77 historically confirmed; 37 unresolved additions). The Hughes PDF timed out; no protocol details were inferred. This is source-coverage work only: no local data, result, checkpoint, test payload, model, or gate was used. See D-2026-10-28-043 and the [audit](../../experiments/literature_audits/bibliography_matrix_crosswalk_20261028_v1/report.md).

## DPS1-SAR dataset source screen — 2026-10-28 (literature/data metadata only; no local experiment)

The official IEEE DataPort record lists 12,600 GeoTIFF images in six SAR-input and multitemporal-reference packages. The file list is visible, but the payloads require a subscription and were not accessed. The primary data-descriptor's described collection/reducer flow does not establish whether each noisy input is excluded from its mean/median reference. Source grouping, stack membership, representation, and split suitability remain unresolved. This is a conditional SAR-only despeckling candidate, not an optical/JDSR baseline; no project data, result, model, or gate changed. Current task flags are false/false. See D-2026-10-28-042 and the [audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).

## Literature-review/matrix coverage crosswalk — 2026-10-28 (literature only; no experiment)

A bounded crosswalk found four reviewed works absent from the 12-column matrix and added them as current-evidence rows: Capella SAR SR, 3MOS, SOMA-1M, and conditional SAR400. The matrix now has 110 data rows: 77 with direct pre-corruption membership evidence and 33 additions with unresolved original membership. At least 15 original rows remain unaccounted for against the documented lower bound of 125. The entries remain adjacent/conditional prior art; they do not establish source-group independence, optical utility, or a task-matched baseline. No local experiment or project data were used. See D-2026-10-28-041 and the [audit report](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).

## Sim2Real joint SAR despeckling/sidelobe prior-art audit — 2026-10-28 (literature only; no experiment)

The author arXiv v1 full text describes MOCEM-trained joint despeckling and sidelobe reduction with acquisition metadata. It reports 5,000 256×256 tuples, a 4,631/369 train/validation split over acquisition-parameter intervals, and real Umbra/Capella SLC examples evaluated with ENL and qualitative sidelobe inspection; no quantitative real-image sidelobe metric is given. The authors call the reflectivity target an effective ground-truth candidate rather than a physically measurable quantity and state that background maps provide context, not true radar backscatter. No public code or dataset link surfaced in the reviewed manuscript/arXiv record. This is adjacent SAR-only prior art, not an SR or optical-guided baseline; historical matrix membership remains unresolved. No local data, result, experiment, or gate changed. See D-2026-10-28-040 and the [audit report](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).

## Muzeau author-repository configuration-history audit — 2026-10-28 (literature only; no experiment)

The linked public repository has 14 reachable commits; all seven config-bearing revisions use `TRAIN.EPOCH: 10`, three earlier config revisions omit an epoch setting, and no versioned 100-epoch setting was found. The paper's reported 100-epoch SR run has no recovered run-bound config/checkpoint mapping. This literature audit produced no local model result and changed no experiment, split, or gate. See D-2026-10-28-038 and the [audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).

## MMCANetV2 missing-optical output contract — 2026-10-28 (synthetic CUDA check; no experiment)

- **Setup:** Approved AISAR profile; PyTorch 2.8.0+cu128; NVIDIA GeForce RTX 3050 Laptop GPU. Reduced `MMCANetV2` (`embed_dim=16`, one block per stage, `fusion_channels=16`, `use_cross_attention=true`), synthetic LR SAR `[1,1,32,32]` and optical `[1,3,64,64]` tensors.
- **Result:** Missing optical and explicit zero optical with `optical_present=false` each matched `sar_backbone(lr)` bitwise (`max_abs_error=0`). Normal optical produced finite `[1,1,64,64]` output. Forward hooks confirmed LR and HR cross-attention ran for all three calls.
- **Regression coverage:** Retained this contract in `tests/test_mmcanet_v2.py::test_v2_cross_attention_zero_optical_fallback_cuda`; the targeted CUDA pytest passed 1/1 after a fresh preflight, and Python compilation passed. Generated bytecode and pytest cache were removed.
- **Boundary:** This is a source-branch contract check, not a full C2 checkpoint evaluation, optical-utility result, or gate run. No project image, test payload, metric, or checkpoint was opened; no training or inference run was started. B1 remains `not_promoted`; B2/B3 remain gated.

## MHFNet publisher-indexed text recheck — 2026-10-28 (literature only; no experiment)

The official publisher index gives additional method detail: US-Loss downweights detector training examples affected by cross-modal box-label shifts, FAFM performs feature alignment/fusion, and DMP-Fusion combines detection outputs. The evidence remains downstream detection prior art; the publisher names QXS and Suez but does not expose exact splits or a controlled offset sweep in the accessible indexed text. Direct page open returned HTTP 403. No local data, test payload, checkpoint, model, experiment, or gate changed. See D-2026-10-28-021 and the [audit](../../experiments/literature_audits/mhfnet_publisher_indexed_text_recheck_20261028_v1/report.md).

## oma_ssr_test_loss_protocol_recheck_20261028_v1 — primary-text protocol audit (2026-10-28; no experiment)

- **Question:** Does the publisher's OMA-SSR text describe use of the held-out test set in the training objective?
- **Evidence:** Section 2.4 assigns the whole test set to the loss for overall training; Section 3.1 separately defines train/test splits. Section 3.2 states that SAR is BM3D-filtered before bicubic downsampling. No code or author clarification was verified, and no official correction surfaced in the bounded publisher search.
- **Decision:** Treat the written protocol as test-contaminated for evidence eligibility. Do not use the paper's numeric test results as independent evaluation unless a primary correction or implementation clarifies the data flow. Do not assert actual code leakage or intent. While checking the internal pilot draft in this continuation, historical mini-pilot test metrics surfaced in text; they were excluded from use. No local data payload, model, experiment, or B1/B2/B3 gate changed; see D-2026-10-28-018 for exposure flags.
- **Record:** [protocol](../../experiments/literature_audits/oma_ssr_test_loss_protocol_recheck_20261028_v1/protocol.md), [report](../../experiments/literature_audits/oma_ssr_test_loss_protocol_recheck_20261028_v1/report.md), and [access manifest](../../experiments/literature_audits/oma_ssr_test_loss_protocol_recheck_20261028_v1/access_manifest.json).

## Nullable ratio-metric consumer reliability — 2026-10-28 (code-only; no experiment)

**Question:** Do current metric consumers handle the evaluator's explicitly undefined correlation values?

**Method and result:** Source inspection exposed `float(None)` failures and missing coverage accounting in Phase-A2, joint-vs-cascade, and formal aggregation. The consumers now skip undefined values, retain null summaries when no values are defined, and expose coverage counts. Five synthetic regression tests and Python compilation passed.

**Device boundary:** The joint-vs-cascade model audit now accepts only CUDA and defaults to CUDA. No model-forward run was started: the current shell lacks importable PyTorch and an accessible NVIDIA driver. Future runs require an approved GPU profile and fresh CUDA preflight.

**Research boundary:** No project data, checkpoint, or historical metric was read or changed. This is not experiment evidence and does not change B1 `not_promoted` / `stop_at_b1` or the B2/B3 gate. See D-2026-10-28-001.

## SEN1-2 catalog metadata route — 2026-10-28 (metadata-only; no experiment)

**Question:** Does a newly indexed official catalog route provide a versioned source-group crosswalk for the local numeric 2k cohort?

**Evidence:** The GovData record lists TUM University Library as publisher, but its linked resource/RDF metadata remain inaccessible through the connected reader. The official MediaTUM OAI-PMH endpoint now returns record 1436631 as SEN1-2, with creator Michael Schmitt, publisher TUM, metadata date 2017-12-15, and a 282,384-pair collection description. The record exposes only oai_dc and has no scene list or per-sample/source-group crosswalk. Its rights field says restrictedAccess, while the official ISPRS release announcement says the dataset is available under CC-BY; this metadata mismatch needs repository interpretation. The HTML record page returned HTTP 429 in the reader and an Anubis challenge in the shell probe. No local image, archive, project test identity, or measurement was opened or downloaded.

**Decision:** The local cohort remains unidentified. Treat the OAI result as collection-level metadata only; do not build a new split or change B1/B2/B3 from this route. Resume source-group assignment only when an authorized, version-pinned file-to-source manifest is available. See D-2026-10-28-002, D-2026-10-28-005, and the [MediaTUM metadata audit](../../experiments/data_audits/sen1_2_mediatum_oai_metadata_recheck_20261028_v1/report.md).

### Official MediaTUM item export and rsync inventory follow-up (2026-10-28)

The JSON item export reports the CC BY 4.0 license URL, a 43.7-GB/564,770-file data-server package, and an official rsync route. Guest-visible `/children` and `/allchildren` exports return zero records. A nonrecursive rsync `--list-only` root request returns the four season folders, `SupportingDocument.txt`, `.noindex`, and `checksums.sha512` (102,960,516 bytes); no file contents were transferred. The previously retained checksum audit covers 564,768 image entries and found zero exact hash matches for 3,600 fixed train/validation local inputs. A bounded text/code/sidecar search also found no local numeric-filename source map; it does not exclude transformed copies or external subset-generation records. OAI-PMH's `restrictedAccess` tag remains semantically unresolved despite the item-level CC BY 4.0 field. No split, result, or gate changed. See D-2026-10-28-010/013 and the [route audit](../../experiments/data_audits/sen1_2_mediatum_oai_metadata_recheck_20261028_v1/report.md).

## Frequency-enhanced joint-SAR diffusion lead — primary-route recheck (2026-10-28; literature only)

Direct IOP, DOI resolver, Crossref, and exact-title routes were rechecked. IOP remained blocked to the available reader, DOI/Crossref opens were inaccessible, and title search surfaced only secondary ResearchGate result text. The secondary method and performance details were excluded; no matrix row or technical prior-art claim was added. The lead remains unverified pending accessible primary text or a versioned author implementation. No local data, model, or experiment was accessed. See D-2026-10-28-011 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).

## Frequency-enhanced joint-SAR diffusion lead — Crossref metadata recheck (2026-10-28; metadata only)

A direct Crossref REST API request returned HTTP 200 and confirms DOI `10.1088/1361-6501/ae8686`, title, *Measurement Science and Technology* 37(32), article 326112, and online/print dates of 2026-08-12 and 2026-08-14. The deposited author given/family fields appear inverted and need publisher confirmation. The article body and author manuscript remain inaccessible; no method, data, degradation, result, or code claim is accepted. No matrix row, local data, model, experiment, or gate changed. See D-2026-10-28-014 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).

## Frequency-enhanced joint-SAR diffusion lead — official full-text route check (2026-10-28; metadata only)

Crossref's `text-mining` links point to the IOP HTML and PDF article URLs. HEAD requests to both returned challenge HTML from `validate.perfdrive.com`; no article body was retrieved or challenge bypassed. A primary-domain/author-route search across ORCID, Xijing University, arXiv, and GitHub returned no author manuscript or implementation. The article remains a bibliographically confirmed but technically unverified lead; no row, experiment, or gate changed. See D-2026-10-28-015 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).

## SEN1-2 public author/TUM repository route — 2026-10-28 (metadata-only)

Focused GitHub searches scoped to Schmitt/TUM-related namespaces did not surface a SEN1-2 file-to-source-scene mapping. The opened `schmitt-muc/SEN12MS` toolbox is for SEN12MS, not the SEN1-2 release. A third-party SEN1-2 feature-matching derivative describes source filenames for its own 64×64 matching patches but has no mapping to this project's numeric names and was excluded as authoritative provenance. No repository was cloned or dataset archive downloaded; local lineage remains unresolved. See D-2026-10-28-008 and the [audit report](../../experiments/data_audits/sen1_2_mediatum_oai_metadata_recheck_20261028_v1/report.md).

## b1_v2_cpu_cuda_output_reproducibility_20261027_v1 — post-run scalar comparison

**Question:** Did the same frozen direct-fusion diagnostic produce materially different scalar outputs when v2 ran on CPU and v3 ran on CUDA?

**Method:** Pair the retained `per_sample.csv` rows by seed, validation identity, source folder, and optical condition. Require matching fixed-input manifest, input-binding, and checkpoint SHA-256 values; verify false/false test flags and no target/mask/quality-metric use. No model forward pass or image access was performed.

**Observation:** All 1,200 row keys matched. Maximum absolute differences were `4.57e-9` for prediction-delta RMS, `6.49e-9` for relative prediction-delta RMS, `2.64e-6` for fusion-feature delta RMS, and `6.09e-6` for direct optical-context input RMS; SAR-base delta was exactly equal in all 1,200 rows. The outputs are not bitwise identical, and no acceptance tolerance was predeclared or applied.

**Boundary:** This is a descriptive cross-device implementation check, not an independent replicate, quality comparison, optical-utility result, or model-selection evidence. B1 stays `not_promoted`; B2/B3 remain gated. See [report](../../experiments/diagnostics/b1_v2_cpu_cuda_output_reproducibility_20261027_v1/report.md), [manifest](../../experiments/diagnostics/b1_v2_cpu_cuda_output_reproducibility_20261027_v1/manifest.json), and D-2026-10-27-047.

## MD-Glow DOI official-record recheck (2026-10-27; source metadata only)

The secondary DOI candidate `10.1109/WACVW65960.2025.00061` was rechecked through the DOI resolver, an exact IEEE Xplore query, and the Crossref single-work API. The resolver and API were inaccessible in the current reader; the IEEE query returned no record. Keep the DOI unverified, use the official CVF locator, and do not infer a DOI from the secondary listing. No local test data, metric, experiment, or gate was accessed or changed. See D-2026-10-27-045 and the [route audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md).

## Historical pilot test-text exposure — 2026-10-26 (disclosure only; no experiment)

**Event:** A capped source-terminology search in the internal pilot TeX file returned whole matching lines that included historical metric text from its 100-identity test subset.

**Disposition:** Exclude all surfaced and unseen test-result text from current evidence. No value was used for analysis, model selection, tuning, or claims. Record task flags as `test_set_accessed=true`, `test_set_used=false`; no image or metric file was opened. Stop any further selection/tuning on those 100 identities. The recorded B1 v2 gate remains unchanged; no source-overlap claim with the pilot subset is made. See D-2026-10-26-138 and the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).

## SEN1-2 author-manuscript processing follow-up — 2026-10-26 (source-only)

**Question:** What does the original SEN1-2 paper establish about the collection's source products, preprocessing, and patch construction?

**Evidence:** The author-hosted manuscript specifies Sentinel-1 GRD IW VV σ⁰ dB at 5 m azimuth / 20 m range spacing, restituted-orbit and DEM orthorectification, and no speckle filtering. It describes Sentinel-2 bands 4/3/2, ≤1% cloud-selection, per-ROI GEE mosaics, and per-band clipping/scaling/contrast stretch. The paper reports 258 scenes/ROIs, 256×256 patches at stride 128, 298,790 pairs before manual removal of 16,406, and 282,384 final pairs.

**Additional author statements:** The authors call the dataset co-registered, say scene/season splits can support independent data, and note that their Sentinel-2 source was L1C/TOA rather than atmospherically corrected BOA.

**Boundary:** These facts and claims characterize the published collection only. They do not attribute the local numeric 2k PNGs, recover their preprocessing or acquisition times, or verify local pair-level co-registration or split independence. No local data, test payload, test metric, experiment, matrix row, or gate was accessed or changed. See the [crosswalk report](../../experiments/data_audits/dataset_lineage_source_paper_crosswalk_20261026_v1/report.md), [author manuscript](https://arxiv.org/html/1807.01569), and [published article](https://doi.org/10.5194/isprs-annals-IV-1-141-2018).

## SEN12MS author geospatial utility route — 2026-10-26 (source-only)

The pinned author notebook parses SEN12MS GeoTIFF names, reads CRS/affine metadata from label rasters, derives representative region coordinates, and writes a shapefile at runtime. It contains no static crosswalk to SEN1-2 source-scene IDs or the local numeric 2k files. An unresolved community issue reports that some generated region locations disagree with the notebook example and remain misplaced after an x/y swap; do not use this plot as verified geospatial truth. No archive, split list, local image, metric, checkpoint, or test payload was accessed. No experiment, split, or gate changed. See the [source audit](../../experiments/literature_audits/sen12ms_geospatial_utility_route_20261026_v1/report.md), [pinned notebook](https://raw.githubusercontent.com/schmitt-muc/SEN12MS/3a41236a28d08d253ebe2fa1a081e5e32aa7eab4/utils/Sen12MSOverview.ipynb), and [issue #5](https://github.com/schmitt-muc/SEN12MS/issues/5).

## b1_v2_prediction_difference_norms_20261026_v1 — streaming output sensitivity diagnostic (2026-10-26; zero-training)

**Question:** Were near-zero matched-versus-permuted PSNR/SSIM differences caused by effectively identical predictions?

**Protocol:** Compare streamed, full unmasked model outputs for 200 fixed validation identities, 15 source groups, and seeds 0–2 under matched/permuted optical, matched/missing optical, and matched optical versus the SAR-only checkpoint. Save only source-group summaries and aggregate statistics.

**Observation:** Mean relative RMS differences were 0.0105%, 0.1291%, and 0.1290%, respectively. Matched/permuted outputs were not identical, but changed only slightly on this cohort. The SAR-only contrast also changes checkpoint and architecture. Norms are in normalized representation output space; bootstrap intervals resample 15 group means over three fixed seeds.

**Boundary:** One fall ROI-season, development validation only, no test access, training, selection, or gate change. All 1,400 rebuilt validation payload hashes matched run-bound files.csv; the full verification.json digest mismatch remains unexplained. No prediction tensors or per-image norm rows were retained. See the [report](../../experiments/diagnostics/b1_v2_prediction_difference_norms_20261026_v1/report.md), [summary](../../experiments/diagnostics/b1_v2_prediction_difference_norms_20261026_v1/summary.json), [group summary](../../experiments/diagnostics/b1_v2_prediction_difference_norms_20261026_v1/group_summary.csv), and [D-2026-10-26-134](DECISION_LOG.md#d-2026-10-26-134--record-prediction-difference-norms-and-task-artifact-cleanup).


## dataset_lineage_source_paper_crosswalk_20261026_v1 — primary-source comparison (2026-10-26; no experiment)

**Question:** Do the local subset's patch geometry and PNG representation establish SEN1-2 or QXS-SAROPT provenance?

**Evidence:** The SEN1-2 primary paper reports 256×256 patches at stride 128 with 50% overlap. The QXS-SAROPT paper reports 256×256 patches and 20% overlap while phrasing the crop parameter as “stride 52.” Local train/validation overlap records 204–205-pixel shifts and 51–52-pixel shared strips. The SEN12MS follow-on paper states that the SEN1-2 distribution it describes had no geolocation metadata.

**Interpretation:** Local geometry differs from canonical SEN1-2 tiling if preserved and is compatible with a 20%-overlap crop, but it does not identify QXS-SAROPT. Retiling remains a competing explanation. Keep local lineage unresolved; B1 remains `not_promoted`, and B2/B3 remain gated. See the [crosswalk report](../../experiments/data_audits/dataset_lineage_source_paper_crosswalk_20261026_v1/report.md).

## png_header_lineage_20261026_v1 — current-subset PNG metadata audit (2026-10-26; no experiment)

**Scope:** Header/chunk metadata only for `data/sar` and `data/opt`; 4,000 PNGs total. The parser seeks over compressed `IDAT` payloads and does not access test paths.

**Observation:** Both roots contain 2,000 256×256 8-bit PNGs with identical numeric basenames `8000`–`9999`. Only `IHDR`, `IDAT`, and `IEND` chunks were present; there were no text keywords or ancillary metadata chunks. Recursive inventory found no non-PNG files or nested directories under the two roots. This does not identify dataset lineage or prove that external source manifests do not exist.

**Record:** [`summary.json`](../../experiments/data_audits/png_header_lineage_20261026_v1/summary.json), [`report.md`](../../experiments/data_audits/png_header_lineage_20261026_v1/report.md), and [`scan_png_metadata.py`](../../experiments/data_audits/png_header_lineage_20261026_v1/scan_png_metadata.py). `test_set_accessed=false`, `test_set_used=false`, `pixel_payload_read=false`.

## B1 v2 optical-pairing sensitivity diagnostic (2026-10-26; zero-training)

Ran matched, cross-source-group permuted, and missing optical conditions on the existing optical candidate, with the matching SAR-only checkpoint as reference. The fixed development validation contains 200 identities from 15 `ROIs1970_fall` source groups; seeds 0–2 were fixed. Matched versus permuted optical produced near-zero PSNR/SSIM changes (−0.000121 dB and −0.000015, with both source-group 95% intervals crossing zero). Matched versus missing and matched versus SAR-only each produced about +0.00162 dB PSNR but about −0.000892 SSIM and worse edge/TBR/bright-peak ratio errors. No training, selection, gate change, or test access occurred. This is not a geometric misregistration test and does not establish generalization. See D-2026-10-26-130 and the [full report](../../experiments/diagnostics/b1_v2_optical_pairing_sensitivity_20261026_v1/report.md).

## R-DDPM full-text protocol screen (2026-10-26; source-only)

The arXiv v1 manuscript describes regional conditional diffusion, overlapping-window averaging, and DDIM. It identifies Sentinel-1 VV+VH data near Shanghai and Gaussian-noise synthetic observations, but does not state sample counts or split allocation. Real evaluation has no clean reference and uses ENL/EPI. A region-parameter statement is inconsistent, and no implementation URL is cited. The existing matrix row is within the byte-preserved historical prefix; no matrix count or project experiment changed. See D-2026-10-26-112 and the [full-text audit](../../experiments/literature_audits/rddpm_fulltext_audit_20261026_v1/report.md).

## Muzeau et al. SAR restoration prior-art update (2026-10-26; source-only)

The author-hosted EUSAR manuscript supports a SAR-only ×2 SR supervision study using noisy-LR to MERLIN-denoised-HR targets. It reports one ONERA SETHI scene and an 80/20 patch split; no independent scene holdout or comparison of cascade ordering is described. The VDE proceedings route redirected to an inaccessible TIB page, and the paper-to-current-repository configuration lineage remains unresolved. This is prior-art coverage only; no author performance value, local metric, experiment, or gate was used. See D-2026-10-26-110, the [literature review](literature_review.md), and the [claim–evidence map](claim_evidence_map.md).

## SARGANet prior-art claim mapping (2026-10-26; documentation audit)

Added `RW-SR-SAR` to the claim–evidence map, recording SAR-only Sentinel-1 ×2 SR as prior art for the SR subtask. The publisher-full-text audit supports the narrow method classification; the article is not optical-guided or joint despeckling/SR. Its test-subset use for loss-weight selection and unresolved split/target details limit its reported evaluation as external context. No local metric or model result was used. See D-2026-10-26-102 and the [claim–evidence map](claim_evidence_map.md).

## 2022 OGRN DOI locator follow-up (2026-10-26; source-only, no experiment)

The official Shenzhen University profile confirms the 2022 JARS title and article metadata but omits a DOI. A non-primary index surfaced candidate DOI `10.1117/1.JRS.16.036503`; direct DOI-resolver, Crossref API, and SPIE full-text routes were inaccessible. Keep the DOI and SSRN-to-journal version relationship unverified; do not add a formal bibliography citation or separate matrix row. No paper method or metric claim was added. See D-2026-10-26-101 and the [focused audit](../../experiments/literature_audits/ogrn_2022_author_profile_recheck_20261026_v1/report.md).

## Related-work matrix current reconciliation (2026-10-26; documentation audit)

The related-work matrix now contains 97 data rows and 12 columns: 74 rows with direct evidence of pre-corruption membership and 23 current-evidence additions with unresolved historical membership. Against the last confirmed 119-row inventory, 22–45 original rows remain unaccounted for. The byte-preserved 46-row prefix remains unchanged. The SARGANet addition is SAR-only Sentinel-1 ×2 SR prior art, not a task-matched joint-restoration baseline; its publisher audit records the article’s test-subset use for loss-weight selection. No local metric was used, and no experiment or gate changed. See [D-2026-10-26-097](DECISION_LOG.md#d-2026-10-26-097--add-sarganet-as-adjacent-sar-only-super-resolution-prior-art), the [matrix recovery report](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md), and the [publisher audit](../../experiments/literature_audits/sarganet_publisher_fulltext_screen_20261026_v1/report.md).

## Related-work matrix integrity recovery (2026-10-26; documentation audit)

A pre-corruption record confirms 119 data rows; the current partial file has 46. At least 73 records are absent. Seventeen prior rows with membership evidence are reconstructed in a separate sidecar, but at least 56 further rows remain unresolved. Preserve the partial snapshot and do not use it as a complete inventory while restoring/reconstructing from audited sources. See [D-2026-10-26-026](DECISION_LOG.md#d-2026-10-26-026--record-related-work-matrix-row-loss-and-recovery-path) and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).

## OGSRN publisher PDF-route recheck (2026-10-26; source-only, no experiment)

SciOpen marks the article Open Access (CC BY-NC-ND 4.0), but its extracted page exposes no PDF body/link. Direct ScienceDirect and a guessed publisher PDF route were inaccessible through the reader. No full text, local data, metric, checkpoint, or test asset was accessed; protocol details remain unresolved. See [D-2026-10-26-023](DECISION_LOG.md#d-2026-10-26-023--keep-ogsrn-at-publisher-abstract-evidence-after-pdf-route-recheck) and the [audit](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md).

## SEN1-2 patch-matching train-list route boundary (2026-10-26; source-only, no experiment)

The author repository README describes 64×64 SIFT cross-spectral matching annotations and named source folders, but no mapping to local numeric restoration IDs. No list rows, images, archives, test identities, or metrics were accessed. This route is closed for source-group recovery absent a verified local crosswalk; the true source-provenance blocker remains. See [D-2026-10-26-024](DECISION_LOG.md#d-2026-10-26-024--close-the-patch-matching-train-list-route-for-local-source-grouping) and the [audit](../../experiments/literature_audits/sen1_2_patch_train_list_metadata_route_20261025_v1/report.md).

## Source-folder paired bootstrap uncertainty (2026-10-26; descriptive, no model run)

The hash-bound 15-folder/three-seed B1 development aggregate yields group-equal ΔPSNR +0.001581 dB (95% source-folder bootstrap interval [+0.000070, +0.003139]) and ΔSSIM −0.000895 (interval [−0.001112, −0.000679]). This analysis is conditional on the fixed seeds and reused fall-only validation cohort; it does not change B1 (`not_promoted`) or authorize B2/B3. It read no per-image rows or test payload. See [D-2026-10-26-015](DECISION_LOG.md#d-2026-10-26-015--record-conditional-source-folder-bootstrap-uncertainty) and the [diagnostic](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_bootstrap_20261026_v1/report.md).

Official documentation identifies those folders as source scenes inside the `ROIs1970_fall` random ROI-sampling-batch/season root; acquisition and exact geographic independence remain unverified. See [D-2026-10-26-016](DECISION_LOG.md#d-2026-10-26-016--clarify-sen1-2-root-and-scene-folder-units).

## Optional anisotropic-blur characterization (2026-10-25; synthetic unit tests only)

Added two tests for the optional blur: the centered impulse response sums to one, a 90° rotation swaps its x/y spread, and zero-padding attenuates constant-input corners. The combined focused synthetic degradation set passes 13/13. This records current implementation behavior; it does not validate a physical SAR point-spread or border model. No project data/test payload, run, model, metric, or gate changed. See [D-2026-10-25-169](DECISION_LOG.md#d-2026-10-25-169--characterize-optional-blur-with-synthetic-tests) and the [audit](../../experiments/data_audits/anisotropic_blur_characterization_20261025_v1/report.md).

## Canonical degradation code and synthetic-statistics audit (2026-10-25; no model experiment)

The shared degradation helper applies FFT low-pass, optional anisotropic Gaussian blur, bilinear downsampling, a declared nonnegative policy, optional Gamma multiplication, and optional output clamping. Its default speckle domain is representation; the implementation does not establish calibrated local SAR units. `tests/test_degradation_and_model.py` passed 11 focused synthetic checks across canonical import identity, normalization, FFT behavior, and Gamma moments. No local image, test payload, checkpoint, run, prediction, metric file, or model was accessed; no training, selection, result, or gate changed. See [D-2026-10-25-168](DECISION_LOG.md#d-2026-10-25-168--bound-the-canonical-degradation-to-representation-domain-evidence) and the [audit](../../experiments/data_audits/canonical_degradation_semantics_20261025_v1/report.md).

## Paper-directory scope and access audit (2026-10-25; no experiment)

The root README and A–E project fact sheet now clearly identify the formal 2k preview and July pilot as historical archives, not current submission manuscripts. Historical fixed-set/held-out result prose surfaced while reading the README; it was not used for selection or a gate. No result file, image, array, prediction, checkpoint payload, or experiment was opened or changed. The metadata-only artifact inventory found no additional disposable outputs. Cumulative access flags remain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-167](DECISION_LOG.md#d-2026-10-25-167--classify-both-paper-directories-as-historical-archives) and the [scope/access audit](../../experiments/data_audits/paper_directory_scope_20261025_v1/report.md).

## July engineering pilot manuscript scope audit (2026-10-25; source-only, no experiment)

The paper-directory draft is an internal historical 1,900/100 pilot with a held-out mini set, no validation set, single-seed three-epoch training, and test-result prose. It is not admissible as current evidence or a submission-results source. The audit read the main `.tex` only; generated result fragments and the PDF remain unopened. No data/model/run/gate changed; cumulative flags remain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-166](DECISION_LOG.md#d-2026-10-25-166--retain-the-july-pilot-draft-as-historical-only) and the [scope audit](../../experiments/data_audits/paper_legacy_pilot_scope_20261025_v1/report.md).

## SOTANet-named checkout identity recheck (2026-10-25; source-only, no experiment)

The neighboring clone's tracked README is headed CycleGAN/pix2pix, and the bounded GitHub root-page snapshot shows the same heading but cannot be pinned to a current remote SHA. GitHub API and raw README access were unavailable. This route supplies no SAR–optical archive/version, patch-to-source mapping, rights, acquisition/source groups, or split semantics; the local 2k lineage remains unresolved. Only repository and research-document text was read; no data/test payload, metric file, checkpoint, training, inference, or model selection occurred. Truncated broad output is excluded and cumulative flags remain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-165](DECISION_LOG.md#d-2026-10-25-165--close-the-sotanetwork-named-checkout-as-a-data-provenance-route) and the [audit](../../experiments/data_audits/sotanetwork_repo_identity_recheck_20261025_v1/report.md).

## SOTANet helper source-lineage check (2026-10-25; source-only, no experiment)

The tracked SOTANet tree filename search surfaced no QXS/SAR2OPT preparation script. Its tracked history has no `QXS`/`SEN1-2` text match; the single `sar2opt` hit is a visualization example path, not a source record. The local `prepare_sar2opt_data.py` helper is ignored and has no commit history; it hard-codes an absent SEN1-2 source mount/root and writes to a different SEN1-2 output path. These references do not authenticate the QXS-named cache or local numeric 2k files. No helper was run and no data/test payload, metric, checkpoint, model, or gate was accessed or changed. See [D-2026-10-25-163](DECISION_LOG.md#d-2026-10-25-163--do-not-treat-the-ignored-sotanet-helper-as-source-lineage) and the [audit](../../experiments/data_audits/sotanetwork_helper_lineage_check_20261025_v1/report.md).

## Research-state route-search truncation disclosure (2026-10-25; no experiment)

A cross-document planning search returned truncated output that may contain historical result-bearing prose. Conservatively retain `test_set_accessed=true`, `test_set_used=false`; no surfaced value was used and no test payload or raw metric file was opened. Exclude the unseen span. See [D-2026-10-25-162](DECISION_LOG.md#d-2026-10-25-162--disclose-truncated-research-state-route-search) and the [access audit](../../experiments/data_audits/research_state_route_search_truncation_20261025_v1/report.md).

## DSRKD Tsinghua institutional source route (2026-10-25; source-only, no experiment)

Checked the corresponding author's official Tsinghua Department of Electronic Engineering profile, displayed academic achievements, academic-papers subpage, and linked personal homepage. The profile identifies Yuantao Gu as a professor; the displayed achievements end in 2015, the academic-papers page timed out, and the linked personal page was inaccessible. No DSRKD manuscript, protocol, or code was obtained. The dataset/training-source conflict remains unresolved; no local data, result, checkpoint, model, run, or gate changed. See [D-2026-10-25-160](DECISION_LOG.md#d-2026-10-25-160--keep-dsrkd-protocol-unresolved-after-the-tsinghua-institutional-route) and the [route audit](../../experiments/literature_audits/dsrkd_tsinghua_profile_route_20261025_v1/report.md).

## Research-context read truncation disclosure (2026-10-25; no experiment)

Several bounded research-document reads returned truncated output. Conservatively keep `test_set_accessed=true`, `test_set_used=false`; no metric value was used and no test asset, raw metric file, or checkpoint payload was opened. Exclude the unseen spans and keep model/split/gate state unchanged. See [D-2026-10-25-161](DECISION_LOG.md#d-2026-10-25-161--disclose-truncated-research-context-reads) and the [access audit](../../experiments/data_audits/research_context_read_truncation_20261025_v1/report.md).

## Lan et al. optical–SAR registration full-text audit (2026-10-25; source-only)

Reviewed the publisher-formatted full article on the HT dataset pipeline, alignment refinement, control-point RMSE, synthetic affine evaluation, and Data Availability Statement. It is registration prior art only; it supplies no independent residual estimate for the local SEN1-2 subset and no SAR restoration evidence. No local data, result, checkpoint, or model was accessed; no run or gate changed. See [D-2026-10-25-158](DECISION_LOG.md#d-2026-10-25-158--classify-lan-et-al-as-registration-only-prior-art) and the [literature audit](../../experiments/literature_audits/lan_multilevel_registration_fulltext_20261025_v1/report.md).

## Research-document search exposure disclosure (2026-10-25; no experiment)

A broad registration/alignment text search across five research documents returned 593 lines and was truncated. Treat the session as `test_set_accessed=true`, `test_set_used=false`; no payload or raw metric file was opened and no surfaced value was used. The output is excluded from evidence; see [D-2026-10-25-159](DECISION_LOG.md#d-2026-10-25-159--disclose-truncated-research-document-search-output) and the [access audit](../../experiments/data_audits/research_context_search_truncation_20261025_v1/report.md).

## So2Sat LCZ42 cross-EPSG train geometry screen (2026-10-25; no model/data payload)

Audit `so2sat_lcz42_cross_epsg_envelope_screen_20261025_v1` transformed only the pinned training `city`, `epsg`, and `tfw` metadata. It compared 835 different-EPSG city pairs and found one 10 m-buffered city-envelope candidate. A patch-level attempt was rejected: 9,986 transformed boundaries were non-convex, so its provisional 51-pair convex-clipping output was invalid and removed. No exact patch-overlap count is accepted. No image, label, validation/test metadata, or local project data was accessed; this screen changes no split, model, run, metric, or gate. See [D-2026-10-25-154](DECISION_LOG.md#d-2026-10-25-154--keep-so2sat-city-labels-out-of-disjoint-splits), the [report](../../experiments/data_audits/so2sat_lcz42_cross_epsg_envelope_screen_20261025_v1/report.md), and [manifest](../../experiments/data_audits/so2sat_lcz42_cross_epsg_envelope_screen_20261025_v1/access_manifest.json).

## So2Sat LCZ42 train geolocation metadata screen (2026-10-25; no model/data payload)

Audit `so2sat_lcz42_author_train_geo_screen_20261025_v1` processed only the author-associated v4 training `city`, `epsg`, and `tfw` arrays. It recorded 352,366 patches, 42 city labels, 26 EPSG codes, eight groups below 100 samples, and zero detected positive-area overlap pairs across cities sharing an EPSG code. Cross-EPSG intersections were not checked. No image/label payload or local project data was accessed; the test geolocation file was not requested, though its LFS pointer hash/size appeared incidentally on the pinned commit page. This is a source-group feasibility screen only: no split, model, run, metric, or gate changed. See [D-2026-10-25-153](DECISION_LOG.md#d-2026-10-25-153--keep-so2sat-lcz42-as-a-conditional-metadata-route), the [protocol](../../experiments/data_audits/so2sat_lcz42_author_train_geo_screen_20261025_v1/protocol.md), [summary](../../experiments/data_audits/so2sat_lcz42_author_train_geo_screen_20261025_v1/result_summary.json), and [report](../../experiments/data_audits/so2sat_lcz42_author_train_geo_screen_20261025_v1/report.md).

## Noisy pseudo-target Bayes-risk identity (2026-10-25; theory only, no run)

For `T=X+E`, square-integrable variables, squared-error population Bayes estimators, and nested inputs `Y⊂(Y,O)`, the incremental gains satisfy `ΔR_T=ΔR_X+E||δ_E||²+2E⟨δ_X,δ_E⟩`. The residual term has no established sign in the local shared-input protocol, so proxy-target gains do not identify latent-SAR gains. This is a general identity, not local empirical evidence. No data, metrics, model, run, split, or gate changed. See [D-2026-10-25-152](DECISION_LOG.md#d-2026-10-25-152--record-the-exact-noisy-pseudo-target-bayes-risk-decomposition) and the [derivation](optical_information_analysis.md#exact-bayes-risk-decomposition-for-a-noisy-pseudo-target-2026-10-25).

## Sentinel-1/2 measured registration literature audit (2026-10-25; public paper text only)

Ye et al. report six Sentinel-1 IW GRD/Sentinel-2 L1C 10 m pairs with initial shift magnitudes of 26.30–67.40 pixels and third-order-polynomial held-out checkpoint RMSE of 0.38–2.29 pixels across flat-to-mountainous terrain. Their 48 checkpoints are withheld image correspondences, not surveyed external truth. Use this as a local-checkpoint design precedent only; no local residual was estimated, no dataset or experiment was run, and no B1/B2/B3 gate changed. Exact local source products and common-reference points remain unavailable. See [D-2026-10-25-133](DECISION_LOG.md#d-2026-10-25-133--use-sentinel12-registration-checkpoints-as-a-method-template-only), the [report](../../experiments/literature_audits/sentinel12_measured_registration_checkpoints_20261025_v1/report.md), and the [access manifest](../../experiments/literature_audits/sentinel12_measured_registration_checkpoints_20261025_v1/access_manifest.json).

## CDSE sidecar schema access-path review (2026-10-25; no XML retrieved)

The deterministic five-item sample (two S1 product-type/platform strata and three S2 platforms) froze 11 sidecar targets: eight S1 calibration/noise XMLs and three S2 product-metadata XMLs. One selected S1 Item request returned HTTP 200, and its selected calibration/noise assets use `s3://eodata`. The default-sandbox request was denied before HTTP; a reviewed-network attempt and a host-only diagnostic made two successful Item metadata GETs total, but zero sidecar GETs. CDSE's official S3/OData routes require registered credentials; none are configured here. No XML/schema fields, rasters, SCL pixels, or image bytes were inspected. This is an access-path blocker, not evidence that product calibration is absent or invalid. See [D-2026-10-25-105](DECISION_LOG.md#d-2026-10-25-105--pause-cdse-sidecar-fetch-until-authenticated-access-is-configured), the [protocol](../../experiments/data_audits/cdse_sidecar_schema_review_20261025_v1/protocol.md), and the [report](../../experiments/data_audits/cdse_sidecar_schema_review_20261025_v1/report.md).

## CDSE candidate asset metadata and repeat-stack feasibility (2026-10-25; metadata only)

The frozen inventory covers 80 unique Item IDs (45 S1 GRD, 35 S2 L2A); all direct STAC Item requests returned HTTP 200. The items advertise S1 VV/VH measurement COGs plus per-polarization calibration/noise XML, and S2 10 m B02/B03/B04 reflectance, SCL, and product metadata. No href was retained or followed; no product XML, raster, SCL pixel, or imagery was accessed. General Sentinel open-data terms and attribution language were reviewed, but authenticated asset access was not tested. A deterministic reanalysis of raw S1 catalog metadata found a same-platform/orbit-direction/relative-orbit group with 3+ datatakes for all 12 AOIs, spanning 47–59 days. This supports only a multi-temporal pseudo-reference feasibility question; season/change stability and independent truth remain unverified. S1A/S1C product-type strings also differ and need harmonization review. B1 remains `not_promoted`; B2/B3 remain gated. See [D-2026-10-25-104](DECISION_LOG.md#d-2026-10-25-104--bound-cdse-asset-metadata-and-repeat-stack-evidence), the [report](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/report.md), the [manifest](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/access_manifest.json), and the [repeat summary](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/s1_repeat_summary.json).

## CDSE acquisition/footprint graph (2026-10-25; metadata only)

The graph uses the 19 frozen S1/S2 product-pair records from the four-anchor screen, deduplicated to 17 S1 and 12 S2 Item IDs. All 29 exact Item metadata GETs returned HTTP 200. The graph contains 19 candidate-pair edges, two same-datatake edges, and 141 positive-area product-footprint overlaps. All four anchors merge into one component at intersection thresholds 1 m², 100 m², and 1 km²; the pairing/shared-datatake graph alone has nine components. This rejects this candidate pool as a group-disjoint cohort under the frozen product-footprint rule; it does not rule out different, predeclared AOIs. No assets/pixels, local/test data, split, model, metric, checkpoint, selection, or gate were accessed or changed. See [D-2026-10-25-102](DECISION_LOG.md#d-2026-10-25-102--reject-the-four-anchor-cdse-pool-as-a-group-disjoint-cohort), the [report](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/report.md), [graph](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/graph.json), and [access manifest](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/access_manifest.json).

## CDSE four-anchor metadata screen (2026-10-25; metadata only)

The frozen November–December 2025 query covered Hamburg, Berlin, Frankfurt, and Munich; all eight STAC pages were complete. Each anchor returned a candidate S1/S2 pair under IW/VV+VH, S2 cloud ≤20%, and gap ≤24 h. The counts were 38/36, 41/72, 39/19, and 40/35 S1/S2 records respectively. Acquisition IDs cross anchors: S1 `513146` appears at Hamburg/Frankfurt, S1 `41531` at Berlin/Hamburg, and S2 `GS2A_20251226T103501_054903_N05.11` at Hamburg/Frankfurt. This is evidence against assigning by city; it is not a split or a validated cohort. No assets or pixels were accessed. See [D-2026-10-25-101](DECISION_LOG.md#d-2026-10-25-101--reject-city-anchors-as-source-groups), the [report](../../experiments/data_audits/cdse_stac_multi_aoi_metadata_20261025_v1/report.md), and [access manifest](../../experiments/data_audits/cdse_stac_multi_aoi_metadata_20261025_v1/access_manifest.json).

## CDSE live STAC pairing screen (2026-10-25; metadata only)

A fixed-point November–December 2025 STAC v1 query returned complete pages for 41 Sentinel-1 GRD and 72 Sentinel-2 L2A items. Ten S1 records were in the selected ascending relative-orbit-44/IW/VV+VH stratum; nine S2 item records reported cloud cover ≤20% across five datatakes. Three unique S1/S2 datatake pairs fell within 24 h. The selected 2025-11-21 S1C/S2C pair had a 6.602 h acquisition gap and a metadata-only approximately 2.56 km common-footprint probe. This establishes a candidate acquisition route only; no pixels, calibration, independent reference, split, training, or model selection were assessed. The incomplete year-wide first page was excluded. See [D-2026-10-25-100](DECISION_LOG.md#d-2026-10-25-100--advance-cdse-to-live-metadata-candidate-screen), the [report](../../experiments/data_audits/cdse_stac_pairing_live_20261025_v1/report.md), and the [access manifest](../../experiments/data_audits/cdse_stac_pairing_live_20261025_v1/access_manifest.json).

## SEN12TS temporal-reference pilot (2026-10-25; external train/development only)

A metadata-only hash selection froze one tile in each of 43 linkage components (28 train, 15 development). All 86 S1/S2 Range requests completed; masks, value domains, schema, and `<1%` invalid-pixel coverage passed. Median normalized-residual correlations were positive for all temporal pairs, so independence and clean-target validity remained unproven. The predeclared next step was a second tile from eligible components, completed in the supplement below. See [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit), [pilot report](../../experiments/data_audits/sen12ts_temporal_reference_pilot_20261025_v1/report.md), and [`access_manifest.json`](../../experiments/data_audits/sen12ts_temporal_reference_pilot_20261025_v1/results/access_manifest.json).

## SEN12TS second-tile temporal audit supplement (2026-10-25; paired descriptive screen)

The frozen second-rank sample covers 38 of 43 metadata-linkage components (26 train, 12 development); five components had only one candidate. All 76 object requests completed with exact HTTP 206 ranges in one attempt, and the schema/coverage gates passed. Median local-mean changes were similar to the pilot, while component-paired deltas varied and residual correlations remained positive. Coarse t−2/t−3 shift diagnostics each had one search-boundary estimate. Per protocol, stop further sampling; do not interpret the temporal mean as clean or independent SAR, and do not train/select or change B1/B2/B3 gates. Reproduction uses [`command.txt`](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/command.txt), the [report](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/report.md), and the paired component CSV/JSON.

## SEN12TS Sumatra header and footprint audit (2026-10-25; metadata only)

All 4,000 exact 4 KiB Azure Blob ranges parsed for the frozen 2,000 S1/S2 pairs. Every pair is 256×256 with exact serialized CRS/affine equality; all 2,000 tile groups pass the conservative 400 m footprint bound (minimum clearance lower bound 1,225.281 m). S1 files have 19 float32 bands and S2 files 14 int16 bands; the headers provide no band descriptions. No pixels, labels, local test material, model, or metrics were accessed. See [D-2026-10-25-95](DECISION_LOG.md#d-2026-10-25-95--advance-sen12ts-to-a-verified-spatial-candidate-cohort) and the [audit](../../experiments/data_audits/sen12ts_geotiff_header_audit_20261025_v1/report.md).

## SEN12TS linked-component train/development manifest (2026-10-25; metadata only)

The candidate's 88 SAR date/track/pass proxies and 58 S2 dates form 43 conservative linkage components. A frozen hash assignment creates 1,101 train pairs across 28 components and 899 development pairs across 15 components; there are zero shared SAR proxies or S2 dates and no test partition. The split is not ready for model selection until band semantics and temporal pseudo-reference validity are audited. See [D-2026-10-25-96](DECISION_LOG.md#d-2026-10-25-96--create-a-prospective-sen12ts-metadata-split-by-linkage-components) and the [split audit](../../experiments/data_audits/sen12ts_sumatra_acquisition_group_split_20261025_v1/report.md).

## QXS author issue status refresh (2026-10-25; literature/source metadata only)

Official issue #13 remains open with a request for an alternate mirror and no maintainer response. It provides no versioned archive or source-group mapping; local source identity and model selection remain unresolved/paused. No local data, test payload, model, experiment, or gate changed. See [D-2026-10-25-93](DECISION_LOG.md#d-2026-10-25-93--keep-local-source-unresolved-after-qxs-issue-status-refresh) and the [issue refresh](../../experiments/data_audits/qxs_author_issue_status_refresh_20261025_v1/report.md).

## Sibling-repository page truncation disclosure (2026-10-25; access audit only)

A combined read of three SOTANet GitHub pages was truncated and cannot be reconstructed; its content is excluded from claims. No local JDSR data or test payload was opened in that event. Preserve cumulative session flags `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-94](DECISION_LOG.md#d-2026-10-25-94--disclose-truncated-sibling-repository-page-read) and the [access audit](../../experiments/data_audits/sotanetwork_page_truncation_20261025_v1/report.md).

## Continuity excerpt exposure (2026-10-25; documentation only)

The combined bounded-prefix read of `CURRENT_STATE.md` and `NEXT_ACTIONS.md` was truncated and could have surfaced historical result-bearing text. Conservatively record `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no value was used. Historical flags and cohort exclusions remain unchanged. See [D-2026-10-25-76](DECISION_LOG.md#d-2026-10-25-76--disclose-truncated-continuity-excerpt) and the [access audit](../../experiments/data_audits/session_context_excerpt_exposure_20261025_v2/access_manifest.json).

## SEN12TS temporal-reference feasibility (2026-10-25; literature only)

Primary dataset documentation and SAR2SAR/multi-temporal MERLIN papers were reviewed to assess whether repeated SAR observations can form an independent target. Keep SEN12TS conditional: four dates over 36 days motivate only a leave-input-date-out pseudo-reference study after versioned stack mapping and temporal-dependence checks; the coarse coherence layers do not certify pixelwise independence at 10 m. No dataset or local test payload was accessed, no experiment was run, and no gate changed. See [D-2026-10-25-75](DECISION_LOG.md#d-2026-10-25-75--keep-sen12ts-temporal-references-conditional) and the [audit](../../experiments/literature_audits/sen12ts_temporal_reference_feasibility_20261025_v1/report.md).

## SEN12TS source/task-fit screen (2026-10-25; metadata only)

The dataset documentation and current product listing describe georeferenced Sentinel-1 RTC γ⁰ and Sentinel-2 tiles, repeated dates grouped by tile-center/date names, and CC BY-NC 4.0 data terms. The dataset does not supply an independent clean SAR restoration target or a general restoration split. The documentation's abstract/listing report 31,398 tiles, while Table 2 prints 27,698 even though its AOI rows sum to 31,398. Keep SEN12TS as a conditional future source-group candidate, not current B1 evidence. No dataset payload or local test asset was accessed; no experiment or gate changed. See D-2026-10-25-73 and the [audit](../../experiments/literature_audits/sen12ts_source_task_fit_screen_20261025_v1/report.md).

## Session context excerpt exposure (2026-10-25; documentation only)

Continuity reads requested bounded prefixes of `CURRENT_STATE.md` and `NEXT_ACTIONS.md`, but the combined tool response was truncated. Conservatively record `test_set_accessed=true`, `test_set_used=false`. No test payload or metric file was opened and no value was used. Preserve prior cohort exclusions; see D-2026-10-25-74 and the [access audit](../../experiments/data_audits/session_context_excerpt_exposure_20261025_v1/access_manifest.json).

## SEN12TP publisher identifier and rights-scope follow-up (2026-10-25; literature only)

The publisher article links dataset availability to Zenodo `7342059`; the current Zenodo SEN12TP v1 landing page identifies `10.5281/zenodo.7342060`, while the 7342059 page returned HTTP 429. The publisher's CC BY 4.0 article notice does not resolve the separate dataset record's blank license field. No dataset archive or metadata JSON was accessed, no external test identity or payload was accessed, and no local project test data or metrics were read. SEN12TP stays conditional; see [D-2026-10-25-72](DECISION_LOG.md#d-2026-10-25-72--keep-sen12tp-conditional-after-publisher-identifier-recheck) and the [audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).

## HSFMamba author-linked code-route follow-up (2026-10-25; literature only)

The linked GitHub repository index lists the OGSSR dataset share but has no HSFMamba entry in its Algorithm table. The data share was not opened, and no model code, external data, local data/test payload, result, checkpoint, model, or experiment was accessed. This confirms only that this inspected route does not expose implementation code; no gate or local experiment changed. See [D-2026-10-25-69](DECISION_LOG.md#d-2026-10-25-69--bound-hsfmamba-implementation-availability-to-the-linked-author-route) and the [audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md).

## Run-interface decision-log text-search disclosure (2026-10-25; no experiment)

A broad historical decision-log query returned truncated text and could have surfaced result-bearing prose. Conservatively record `test_set_accessed=true`, `test_set_used=false`; no test imagery, arrays, predictions, metric files, or checkpoint payloads were opened and no value was used. Historical run flags and cohort exclusions are unchanged. See [D-2026-10-25-67](DECISION_LOG.md#d-2026-10-25-67--disclose-truncated-run-interface-decision-log-search) and the [access audit](../../experiments/data_audits/run_interface_approval_decision_log_text_exposure_20261025_v1/access_manifest.json).

## Gaya et al. Fuse-MERLIN full-text follow-up (2026-10-25; literature only)

The official GRETSI 2023 full text confirms early/intermediate optical fusion for complex-SAR MERLIN despeckling. It reports one-scene qualitative fine-road improvements and a preliminary two-pixel registration test, but explicitly says quantitative validation was not performed. The separate-title 2024 IGARSS paper shares the same six authors; no reviewed source establishes the version relationship. This was source and manuscript work only: no local data, result, test payload, experiment, model selection, or gate changed. B1 remains `not_promoted`; B2/B3 remain gated. See [D-2026-10-25-66](DECISION_LOG.md#d-2026-10-25-66--classify-fuse-merlin-as-single-scene-qualitative-despeckling-prior-art) and the [full-text audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md).

## Gaya 2025 thesis record route follow-up (2026-10-25; literature only)

Exact-title/author searches through theses.fr and HAL did not surface a candidate record in the available web search. This does not establish that the thesis is unavailable; no record or thesis text was opened. The 2023 Fuse-MERLIN paper remains separate from the abstract-level 2024 IGARSS record unless an authoritative source documents their relationship. See the updated [2024 Gaya route audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).

## Gaya 2024 IGARSS proceedings route follow-up (2026-10-25; literature only)

The official paper detail page lists no resources; its proceedings landing page forwards to a closed JavaScript-required EventsAir gateway. This route did not provide the full text. The 2024 work remains abstract-level, separate from the full-text 2023 GRETSI paper unless an authoritative source establishes version continuity. See [D-2026-10-25-68](DECISION_LOG.md) and the [updated audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).

## Gaya et al. optical-assisted SAR despeckling prior-art integration (2026-10-25; literature only)

The pilot Related Work and C9 trace now cite the 2024 IGARSS abstract-level report of self-supervised multimodal cooperation using optical imagery for SAR restoration. The source screen verifies bibliographic metadata through the institutional record and official program, but the paper's data, target/noise model, inference path, registration, results, and code remain unavailable. No local experiment, test payload, model, metric, selection, or gate changed; B1 remains `not_promoted`, and B2/B3 remain gated. See [D-2026-10-25-65](DECISION_LOG.md#d-2026-10-25-65--integrate-gaya-et-al-as-abstract-level-optical-assisted-sar-despeckling-prior-art) and the [source screen](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).

## Same-observation pseudo-target clarification (2026-10-25; no experiment)

The pilot draft now states that the LR input is generated from the same SAR observation saved as the noisy HR pseudo-target. This documents the code-level target/input dependency and limits full-reference metrics to correlated pseudo-target agreement; it does not quantify residual covariance or establish clean-SAR recovery. No run, output, metric, checkpoint, selection, or gate changed. The bounded manuscript review also surfaced one qualitative held-out-result sentence, recorded as `test_set_accessed=true`, `test_set_used=false` in the [access audit](../../experiments/data_audits/manuscript_method_conclusion_text_exposure_20261025_v1/report.md). See D-2026-10-25-55.

## Gamma sampler notation clarification (2026-10-25; no experiment)

The pilot equation now states `Gamma(shape=L, rate=L)`, matching `torch.distributions.Gamma(concentration=L, rate=L)` in `degradation.py`; the multiplier has unit mean and variance `1/L`. Synthetic moment tests now cover intensity, representation (`L=2`), and amplitude-square multipliers; this is not a local statistical validation or calibrated physical-noise claim. No training run or gate changed; see D-2026-10-25-56. The manuscript access audit records a pooled LR input-range summary of unspecified split scope as potentially test-associated and unused.

## Pilot Related Work integration (2026-10-25; literature only)

The pilot draft now cites DSRKD, ISSMSAR, MD-Glow, and MambaR-ESRGAN as SAR-only joint-restoration prior art, alongside the previously added optical-guided sources. Protocol differences and source uncertainties are recorded in [`claim_evidence_map.md`](claim_evidence_map.md), [`related_work_matrix.csv`](related_work_matrix.csv), and D-2026-10-25-53. This was a citation-only update: no experiment, result, checkpoint, gate, or test payload changed. A repeated view of nearby pilot Introduction text is disclosed in the [access supplement](../../experiments/data_audits/related_work_text_exposure_repeat_20261025_v1/report.md); the historical cohort remains excluded from selection and claims.

## SEN12MS train-list numeric-ID namespace audit (2026-10-25; no experiment)

- **Scope:** Re-read the same hash-bound author training-list snapshot and extracted only the branch and numeric token before the patch suffix. No list rows or identities were printed or retained.
- **Result:** All 162,555 unique rows parsed; SHA-256 remains `b01c34efbbaa0ee5bb720248d8a738fd01a0128124b8fcbd6d7bf3df98503fb8`. There are 119 distinct numeric tokens; 78 appear in multiple season branches and five in all four. Per-branch distinct numeric tokens are 50/58/77/41, matching the 226 branch-qualified provisional keys in the previous audit.
- **Boundary/decision:** Token reuse is a namespace collision, not evidence of geographic overlap. Keep branch/season in keys and do not call the 226 keys spatially independent without a source/geospatial crosswalk. No holdout/test-list contents, raster, archive, local test asset, or measurement was accessed; no model run, selection, or gate changed. See the [audit](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/access_manifest.json), and D-2026-10-25-15.

- **Metadata-route check:** The reviewed author repository tree/README points to a whole-dataset ROI overview notebook but does not provide a train-only spatial crosswalk or validation list. Notebook contents/outputs and whole-dataset coordinates were not accessed. The official catalog page exposed no usable resource/license field; TUM returned HTTP 429. Keep the conditional route open only for train-only metadata access that respects the holdout boundary.

## Standard SEN12MS train-side metadata screen (2026-10-25; no experiment)

- **Scope:** Review the 2019 primary paper, author README/license and repository split listing, GovData catalog metadata, and the author training-list file only. No archive, raster, holdout/test-list content, local test asset, or measurement was accessed.
- **Result:** The 5,144,085-byte training-list snapshot has SHA-256 `b01c34efbbaa0ee5bb720248d8a738fd01a0128124b8fcbd6d7bf3df98503fb8`, 162,555 unique rows, and 226 provisional ROI/scene filename keys: spring 50, summer 58, fall 77, winter 41. These are candidate groups derived from filenames, not verified spatially independent source regions. The paper reports 256×256 patches at stride 128 (50% overlap), dual-pol Sentinel-1 σ⁰ dB, and cloud-adaptive Sentinel-2 mosaics; no clean SAR target, pairwise acquisition timestamps, or measured residual-registration error was established.
- **Rights and interpretation:** The paper's CC-BY wording and the author repository's data-specific research/education-only, non-commercial license differ in scope. Confirm release-specific terms before access or redistribution. Standard SEN12MS is distinct from SEN12MS-CR-TS and does not identify the local numeric 2k subset.
- **Boundary/decision:** Candidate test-list contents and identities were not accessed or enumerated. Project test access/use, dataset download, model run, selection, and gate change are false. Keep standard SEN12MS conditional pending rights, train-side spatial/source validation, and a representation-domain noisy-pseudo-target protocol. B1 remains not_promoted; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/access_manifest.json), and D-2026-10-25-14.

## Authenticated v2 parent-ROI split feasibility (2026-10-25; metadata only)

- **Scope:** Aggregate the frozen 30-group v2 assignment by train/validation and `roi_season`; inspect the cohort manifest and existing official-source audit report. No pair imagery or measurements were accessed.
- **Result:** Train has 1,494 fall pairs across 14 source folders and 106 summer pairs in one folder; validation has 200 fall pairs across 15 folders. The split is source-folder-disjoint but parent-ROI/season overlapping. A parent-ROI split from these selected pairs yields either only 106 summer training pairs or one 106-pair validation cluster.
- **Boundary:** Metadata-only; no candidate test directory/identity was enumerated; `test_set_used=false`, `test_set_accessed=false`. No experiment, model selection, or gate changed. See the [audit](../../experiments/data_audits/sen12_scene_dev1800_parent_roi_feasibility_20261025_v1/report.md) and D-2026-10-25-13.

## Representation-domain Gamma composition derivation (2026-10-25; no experiment)

- **Scope:** Read the canonical normalization/degradation source and the frozen prospective B1 protocol; derive conditional moments for the implemented representation-space Gamma multiplier and the hypothetical unclipped inverse-log transform.
- **Result:** For pre-speckle scalar `q`, `qG` has mean `q` and variance `q²/L`. At `L=2`, the inverse-log expectation is finite only for `q<2/ln(21)` and its second moment only for `q<1/ln(21)`. These are conditional algebraic limits, not measured local-input statistics.
- **Boundary:** No image, array, test payload, measurement, or checkpoint was accessed; no experiment or tests were run. `test_set_used=false`; `test_set_accessed=false`. No B1/B2/B3 gate changed. Retain the condition as a representation-domain stress test; see the [audit](representation_domain_gamma_composition_20261025.md) and D-2026-10-25-12.

## MD-Glow indexed-full-text screen (2026-10-25; literature only)

- **Question:** What does MD-Glow establish about SAR multi-task despeckling/SR, and does it test optical guidance or quantitative joint-SR quality?
- **Evidence:** Official CVF indexed excerpts expose method/loss, training setup, ablations, comparison tables, and conclusion. Direct CVF HTML/PDF access returned HTTP 403; no PDF or code was downloaded.
- **Outcome:** Shared invertible Glow features and task-specific translation provide direct single-modality multi-task prior art. Published quantitative comparisons are despeckling metrics; the reported SR result is qualitative. Multi-task ablation changes depend on resolution and metric.
- **Boundary:** No local data, test payload, metric, checkpoint, model, experiment, selection, or gate changed. B1 remains not_promoted; B2/B3 remain gated. See [D-2026-10-25-08](DECISION_LOG.md#d-2026-10-25-08--record-md-glow-as-sar-multi-task-despeckling-sr-prior-art) and the [audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md).

## MuChaPro-based InSAR spatial-regularization prior-art screen (2026-10-25; literature only)

- **Question:** Does the 2026 JSTARS article establish prior art relevant to structured SAR restoration, and is it task-matched to the current optical-guided joint SAR SR question?
- **Evidence:** IEEE publisher abstract and HAL author-profile metadata; direct IEEE access returned HTTP 418, HAL article fetch returned a cache miss, and no full paper was reviewed.
- **Outcome:** Abstract describes projection-based multichannel despeckling with a spatial-regularization framework for low-coherence InSAR phase/coherence estimation, evaluated on simulated interferograms and real TerraSAR-X/PAZ data. This is adjacent model-based restoration prior art, not an intensity-SAR SR baseline or optical-guidance result.
- **Boundary:** No article payload/code, external data/test payload, local data/test material, model, metric, or experiment was accessed or changed. H9 remains untestable pending local sensor/product/signal-domain provenance; B1 remains not_promoted and B2/B3 remain gated. See [D-2026-10-25-07](DECISION_LOG.md#d-2026-10-25-07--bound-muchapro-insar-spatial-regularization-prior-art) and the [audit](../../experiments/literature_audits/muchapro_insar_regularized_inversion_abstract_20261025_v1/report.md).

Updated: 2026-10-25. Current state is in [CURRENT_STATE.md](CURRENT_STATE.md). Historical metrics and manifests remain evidence; generated pixel outputs and build/cache files may be pruned as recorded in the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).

## ProSR discrete SAR-SR full-text audit (2026-10-24; no local experiment)

ProSR is single-modality ×4 SAR SR using discrete detail tokens and semantic-prototype-guided MaskGIT. Its Umbra SLC benchmark uses spectral-cropped LR input, per-scene normalized/clipped/uint8 amplitude, and source-disjoint train/validation images; the paper describes no separate Umbra SR test set. Author-reported PSNR trails UPSR while LPIPS/FID are stronger; separate MSTAR classifier-transfer results do not establish calibrated scattering-point recovery. Model code is unavailable, and the curated dataset card and Umbra source page state distinct reuse licenses. No archive, code, local data, metric, test artifact, or experiment was accessed or changed. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/access_manifest.json).

## OGSFlow repository-source update (2026-10-24; no local experiment)

The current official README describes validation-split source-noise calibration between S1 selection and S2 training, frames the scale as a perception–distortion control, and names HR-trained semantic downstream evaluations. It also maps S1/S2/RGF to named source files and advertises a CPU synthetic-data smoke command; the source pages remain inaccessible and no release check was run. Whether the calibration `val` is separate from later selection/reporting remains unresolved; this is not a leakage finding. Exact estimator/code, paper citation, and linked training/evaluation documents remain unavailable; no archive, blind split, figure, or numeric external result was accessed. Preserve this as repository-level optical-guided SAR-SR prior art only. See the [updated audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/access_manifest.json).

## B1 trainer validation-path guard (2026-10-24; no local experiment)

Both B1 trainer CLIs now require `--validation-dir` when a B1 `--protocol-id` is supplied; non-B1 calls preserve the legacy default. Four parser regression cases and `py_compile` passed. No model run, checkpoint, metric, or gate changed. The full suite also ran the existing split-integrity test, which read only train/validation/test membership manifests and name lists; no image, array, or metric was read. One full-suite failure remains at the preserved historical source-hash check; see D-2026-10-24-19.


## Prospective B1 aggregation implementation (2026-10-21)

The fail-closed source-group aggregator is implemented at `scripts/aggregate_source_group_b1.py`, with the input/output contract in [`b1_aggregation_contract_v1.md`](../../protocols/sen12_scene_dev1800_v2/b1_aggregation_contract_v1.md). Twelve synthetic-fixture tests and `py_compile` pass. It preserves the frozen B1 gate and requires 15-group bootstrap uncertainty, all three paired seeds, exact canonical validation order/group membership, stable fixed `files.csv` hashes, consistent regenerated full-manifest hashes across all six runs, and false test-access flags. A later authorized SAR-only seed-0 attempt trained on the prospective cohort but aborted during first validation on an evaluator-contract error; no complete run or aggregation exists.

| ID / run | Question and protocol | Outcome | Status and interpretation boundary | Evidence |
|---|---|---|---|---|
| Phase-A2 Case B (historical) | Compare strong SAR cascade and SAR-only joint under the fixed 128→256 development protocol. | Best `cnn→swinir` about 17.3751 dB; SwinIR-Joint about 17.3258 dB. | Reference only. Does not support `Joint > Strong Cascade` and is not an optical comparison. | `docs/MMCANet_PhaseA2_Strong_Cascade_Matrix.md` |
| B1 formal | 3 seeds; calibrated aligned-optical candidate vs matching SAR-only joint; complete 200-image fixed validation. Predeclared mean ΔPSNR ≥ +0.05 dB, all seeds positive, with SSIM/EPI/edge/TBR and historically named bright-scatterer guardrails (the target mask is target-derived and tie-expanded; see audit below). | Seed deltas `−0.005673/−0.006051/−0.000935 dB`; mean `−0.004219 dB`; all negative. | Failed / stop at B1. Gate is unchanged. | `configs/dev2k_formal/phase_b1_aligned_optical_gain.json`; `docs/MMCANet_PhaseB1_Aligned_Optical_Gain.md`; `experiments/dev2k_formal/phase_b1_aligned_optical_gain/` |
| B1R R0 | Test SAR-conditioned structural representation probe; validation-only, 3-seed evidence. | Cosine gain `−0.001855`; correlation gain `−0.001981`; joint improvements `0/3`. | Failed / stop at B1R. R1 was not run. | `docs/MMCANet_PhaseB1R_SAR_Conditioned_Structural_Fusion.md` |
| B1F seed0 inner-dev | Feature-level fusion screen, 1 seed, 1,440 fit / 160 inner-dev. Seed gate +0.020 dB and full metric guardrails. | ΔPSNR `+0.019419 dB`; ΔSSIM `−0.001381`; TBR mean absolute error change `+0.021978`. Zero-init identity error max 0; historical suite 184 tests passed. | Failed / stop at B1F. Do not treat its selected checkpoint as a promoted method. No B1F seed1/2 or formal promotion run. | `docs/MMCANet_PhaseB1F_Feature_Fusion.md`; `experiments/dev2k_formal/phase_b1f_feature_fusion/seed0/` |
| B1F pairing diagnostic `phase_b1f_pairing_diagnostic_20261016_v1` | No training. Reuse failed-screen seed0 checkpoint; fixed 200 validation identities; compare matched, fixed derangement, missing optical, and SAR-only. Bootstrap 2,000 paired resamples. | PSNR means: matched `17.358802`; permuted `17.297160`; missing `17.311080`; SAR-only `17.325809`. Matched−SAR-only `+0.032993 dB` (95% CI `[+0.026632,+0.039101]`); matched−permuted `+0.061642` (95% CI `[+0.056248,+0.067337]`); missing−SAR-only `−0.014729 dB` (95% CI `[−0.021834,−0.007967]`). All eight saved missing−SAR-only metric comparisons move adversely. Static code review finds feature-level `F60` correction, a nine-offset softmax matcher without an explicit reject state, SmoothL1-only training, and no missing-optical bypass; see optical analysis. | Complete diagnostic only. Zero optical is not a demonstrated SAR fallback for this trained checkpoint. Does not modify B1F gate, promote B1, or support confirmation. `test_set_used=false`, `test_set_accessed=false`, `training_performed=false`, `checkpoint_selection_performed=false`. CPU only. | `scripts/diagnose_phase_b1f_optical_pairing.py`; run `experiments/dev2k_formal/phase_b1f_pairing_diagnostic_20261016_v1/{manifest.json,summary.json,per_image_metrics.csv,optical_permutation.csv,command.txt}` |
| B1F matcher-entropy utility diagnostic `phase_b1f_matcher_entropy_utility_20261018_v1` | No training or metric recomputation. On the same 200 fixed validation identities, correlate `1 − mean spatial normalized entropy` of the matched pair's nine-way offset weights with eight saved matched-minus-SAR-only oriented utilities. | Spearman ρ (descriptive image-bootstrap 95% CI): PSNR `+0.156` `[+0.018,+0.290]`; SSIM `+0.108` `[−0.041,+0.250]`; EPI `+0.098` `[−0.044,+0.233]`; edge ratio proximity `+0.107` `[−0.035,+0.254]`; TBR error reduction `+0.016` `[−0.131,+0.165]`; bright-peak ratio proximity `+0.212` `[+0.080,+0.336]`; bright-energy ratio proximity `−0.094` `[−0.237,+0.056]`; bright-region MAE reduction `−0.094` `[−0.237,+0.052]`. The predeclared all-eight-positive rule failed. | Complete Level-0 diagnostic only. This matcher entropy is not supported as a global optical-utility or rejection cue on this checkpoint/data; this does not rule out all possible cues. Repeated development validation, failed-screen checkpoint, unknown source grouping, noisy pseudo-targets and target-derived masks limit interpretation. No B1/B2 gate change; no test access. | `experiments/dev2k_formal/phase_b1f_matcher_entropy_utility_20261018_v1/{protocol.md,summary.json,manifest.json,per_image_cue_utility.csv,command.txt}` |
| Phase-1 R1b | Test learned pixel suppressor with corrected implementation on validation. | Mean ΔPSNR `−0.005871 dB`; SSIM/EPI/Edge/TBR tradeoffs did not produce an accepted full-metric result. | Not promoted. Initial R0 had sigmoid saturation and a background-loss sign bug; R1b corrected those implementation issues but remained negative on PSNR. | `docs/MMCANetV2_Phase1_R_harm_suppression.md` |
| Phase-1 S R1b | Test SAR-derived P0/P1/P2 reliability proxies for optical use. | No deployable proxy yielded positive PSNR gain. | Not promoted. No claim all SAR-derived reliability is impossible. | `docs/MMCANetV2_Phase1_S_sar_background_reliability.md` |
| Phase-1 T oracle | Test whether target-assisted 8/16/32-pixel patch routing indicates potential local optical utility. | Some oracle PSNR ceilings were positive, but edge/structure preservation failed; `oracle_supports_t1=false`. | No T1 training. Target-assisted oracle is not inference-available and cannot establish a deployable method. | `docs/MMCANetV2_Phase1_T_patch_utility_routing.md` |

## Train/validation provenance audit (2026-10-18; no model experiment)

- Scope: only the 1,600 `fixed_train_dev2k` and 200 `fixed_val_dev2k` identities and their source `data/sar`/`data/opt` PNGs. Split lists matched the corresponding CSV rows exactly; all 3,600 current source-file hashes matched the frozen `files.csv` values. Source files were 256×256, SAR mode L, optical mode RGB.
- Method: compute SHA-256 fingerprints for each 128×256 edge strip and 128×128 corner block, then compare opposite regions across train/validation for horizontal, vertical, and diagonal 50%-stride neighbors, separately for SAR, optical, and combined paired bytes.
- Outcome: zero exact SAR block matches, zero exact optical block matches, and zero exact paired matches at those offsets.
- Boundary: this negative exact-byte result applies only to its tested offsets; it does not exclude other geometries or establish group independence. A later exhaustive 51/52-pixel edge scan found exact overlaps at 204/205-pixel translations; see the [2026-10-19 audit](sen1_2_train_val_spatial_overlap_audit.md). Audit scope flags for this earlier scan: `test_set_used=false`, `test_set_accessed=false`.
- Evidence: `scripts/prepare_dev2k_formal_split.py`, `scripts/prepare_fixed_degradation.py`, `fixed_train_dev2k/{manifest.json,files.csv,split.txt}`, and `fixed_val_dev2k/{manifest.json,files.csv,split.txt}`. No test artifacts or test measurements were accessed.

## Fixed-mask and B1 validation evaluator audit (2026-10-18; no model experiment)

- Scope: `fixed_val_dev2k` target arrays/masks and B1 validation metadata, per-image CSVs, and saved predictions for seeds 0–2 in evaluator output versions v1/v2. No test images or measurements were opened; scope flags are `test_set_used=false`, `test_set_accessed=false`.
- Fixed masks: all 800 recomputed ENL/edge/target/background masks matched the saved arrays and recorded hashes; none was empty or full. Target/background and ENL/edge overlap was zero for all 200 samples. The 99.5th-percentile target threshold equals 1.0 for 192 samples; target max is 1.0 for all. Ceiling ties expand the target-mask area to 0.5005%–21.846% (median 3.1555%). Interpret current bright-scatterer/TBR target regions as target-derived high-value/ceiling regions, not sparse independent physical ROIs.
- B1 outputs: six 200-row CSVs and six 200-prediction folders (1,200 files total); zero exact full-image prediction/target matches, non-finite PSNR values, missing targets, or shape mismatches. Manifests bind validation split/manifest and weights but lack evaluator-source and explicit crop/mask hashes; enclosing run metadata says dirty. Current source specifies full 256×256, data range 1.0, and fixed masks, but exact historical evaluator revision is unproven.
- Metric convention: generic ratio-image metric is target/prediction; B1 does not report it. B1's TBR ratio is predicted-TBR/target-TBR (ideal 1). Do not conflate them. No historical values or gates were changed.
- Evidence: `fixed_val_dev2k/{manifest.json,files.csv,masks/}`, B1 validation evaluation manifests/CSVs/predictions, `scripts/evaluate_mmcanet_v2_validation.py`, `Early-FusionCascadeJDSR/eval_metrics.py`, and `OGSRN/metrics.py`. Broader historical validation-only evaluator lineage remains open.

## Canonical fixed-degradation domain and cutoff audit (2026-10-18; no model experiment)

- Scope: canonical normalization/degradation source, fixed train/validation manifests and `files.csv` metadata, and synthetic signals. No sample arrays, test imagery, or test measurements were loaded.
- Protocol: 8-bit grayscale PNG → `[0,1]` → `log1p(20x)/log(21)` → centered FFT truncation → optional blur (disabled) → bilinear 256→128 resize (`align_corners=false`) → negative clamp → looks-2 Gamma multiplier in `representation` domain → no final clamp. The multiplier has theoretical mean 1 and variance 0.5; a seeded 1,000,000-sample CPU draw measured mean 0.99990 and variance 0.49870.
- Bandwidth: requested FFT window 64 is reduced to 63 bins (`k=−31…31`) to preserve conjugate symmetry. A synthetic sinusoid at source-grid bin 31 passed; bins 32, 40, and 64 were removed. A 129-bin symmetric window retained bin 40 and the 2× target-grid Nyquist bin 64. The current cutoff is narrower than conventional 2× Nyquist support; intentional extra blur remains possible but undocumented.
- Range: metadata-only summaries of `files.csv` show `lr_max>1` for 1,600/1,600 train and 200/200 validation samples. Median `lr_above_one_fraction` is 18.13%/18.20%; maximum sample LR values are 9.60/6.84. This follows the unclipped representation-domain multiplier and is not evidence of calibrated intensity-speckle behavior.
- Verification: `conda run -n AISAR python -m pytest -q tests/test_degradation_and_model.py` passed 35 tests in 3.58 s. The suite checks canonical bindings, FFT linearity/realness/energy/negative ringing, Gamma intensity and amplitude statistics, and deterministic degradation. The additional synthetic response diagnostic covers the configured 64-bin cutoff; no code or historical fixed tensor was changed. Test flags for this audit: `test_set_used=false`, `test_set_accessed=false`.
- Interpretation: retain B1 and all historical fixed tensors unchanged. Classify this protocol as controlled composite representation-domain degradation, not physical Gamma speckle. Resolve intended passband and LR clamping/range policy before any prospective degradation matrix.

## Phase-1 R1/S-R1b validation evaluator audit (2026-10-18; no model experiment)

- Scope: three seeds each for `mmcanet_v2_phase1_r1_screen_v1` and `mmcanet_v2_phase1_s_r1b_screen_v1`. The artifact review read manifests, aggregate validation CSVs, runner/evaluator source, and artifact filenames; it did not inspect test imagery or measurements. Separate protocol-test array access is disclosed below.
- Lineage: all six run manifests bind to `fixed_val_dev2k` manifest SHA-256 `17d3361efe4ac54573177ad7f30341c10c128e224a9453c954ed1634aaadfbdc` and split SHA-256 `c175fe500ed312e873bdba459d3ecc80b9935137c1d9d69e3bafbe31bc1cbc0c`; all say `training_complete`, `debug_only=false`, and `test_set_used=false`. They do not record `test_set_accessed`. Configs set `train_crop_size_lr=0` and mark target-derived masks for training/validation diagnostics. The hash-matching runner processes 200 validation items without augmentation/cropping and uses fixed masks on full arrays.
- Evaluator lineage: each manifest records `eval_metrics.py` SHA-256 `7ec47daaace4a774a8bc43cda84cb5cd532fc260eed1899b1c41bae238e5b491`. Current source is `21554642f78e87f2127126bd81d6b3f3ee811a1794ee14948d8801acea16006b`, `HEAD` source is `5207914192925104350ce81cc1dc6b82c5cd0429c4e1fa5bcdb28eb6ada8b617`, and the only additional local archived copy found is `outputs/pilot_archive/pipeline_1900_100_20260714/snapshot/Early-FusionCascadeJDSR/eval_metrics.py` at `367cd576d4d54b4060e823b08d2217c2c8b3855077b13fe75b8f1f96a449940c`; none matches. The recorded evaluator source is not independently recoverable from the searched worktree, Git history, or local archive. Other relevant sources (runner, data/model, `OGSRN/metrics.py`) match their recorded hashes.
- Retained metrics: each run has 10 aggregate epochs and finite base/C2/R1 PSNR cells. All six `validation_per_image/` directories are empty, though the hash-matching runners write per-image CSVs during validation. Exact-match counts therefore remain unverified; finite aggregate PSNR is not a substitute. No saved metric values or promotion gates were changed.

## Test-access incident during protocol verification (2026-10-18)

- Three full repository pytest executions invoked the split test while it constructed `FixedDev2kDataset("fixed_test_dev2k", ...)`; the constructor loaded the first test sample's LR, target, and optical arrays (`8107.npy`). Suite outcomes were 199 passed/1 failed, then 200 passed twice. No inference, metric calculation, checkpoint/threshold selection, or test-result review occurred.
- The split test now validates only split manifests and CSV/text metadata and monkeypatches `np.load` to reject fixed-test paths. The source scan found the other dataset test reads train/validation only; test-path evaluator tests reject before dataset construction. The targeted test passed and the guarded full suite passed (200 passed, 2 warnings) without fixed-test array loads. This accidental pre-fix array access does not change historical run-manifest flags; it is disclosed as local audit/test access.

## Pairing-run integrity summary

- Sample count: 200 unique validation identities, each with 4 conditions; CSV has 800 data rows plus header.
- Per-image metric CSV contains `test_set_used` and `test_set_accessed` for every row, all false. Run manifest repeats false flags.
- Validation file list SHA-256: `f603a8440484bb0b170e0814307da4c20f69416bff577e78ea2c5cb531361b70`; validation manifest SHA-256: `17d3361efe4ac54573177ad7f30341c10c128e224a9453c954ed1634aaadfbdc`; fixed names SHA-256: `c175fe500ed312e873bdba459d3ecc80b9935137c1d9d69e3bafbe31bc1cbc0c`.
- Fixed permutation: lexicographic identity order rotated by `floor(N/2)`, a deterministic bijective derangement.
- This is the same repeatedly reused development validation set. Bootstrap uncertainty describes image-pair variation conditional on this checkpoint and sample set; it does not quantify seed/training variability.

## Exact spatial-overlap audit (2026-10-19; no model experiment)

- Scope: fixed train (1,600), fixed validation (200), and the B1F 1,440/160 fit/inner-dev split. Only train/validation source PNGs, manifests, and fixed target arrays were read. No fixed-test data or measurements were accessed; audit metadata records both test flags false.
- Method: an initial feature search identified near-integer 204/205-pixel translations. An exhaustive, unpruned scan then compared exact 51/52-pixel boundary strips across all 1,800 SAR and RGB optical source images and applied the resulting overlap graph to fixed targets.
- Outcome: SAR and RGB optical have the same 580 exact train–validation pairs; every shared target strip is exactly equal. The 1,800 source images form six overlap-connected components, three of which mix train and validation. These components are not verified original scene/acquisition groups. B1F fit/inner-dev has 440 exact target-overlap geometries, affecting 159/160 inner-dev identities and 392 fit identities.
- Interpretation: fixed validation and B1F inner-dev are not independent of their fit targets. Historical values and gates remain unchanged, but cannot be treated as independent generalization or used for future selection/promotion. The audit does not quantify metric impact; original scene/tile/acquisition lineage is still unresolved. The old 128-pixel check remains valid only at its tested offsets.
- Evidence: [audit report](sen1_2_train_val_spatial_overlap_audit.md), scripts `audit_exact_204_205_overlap_graph.py`, `verify_overlap_graph_target_pixels.py`, and `verify_b1f_inner_dev_target_overlap.py`; machine-readable outputs in `experiments/data_audits/sen1_2_train_val_spatial_overlap_20261019_v1/`.

## Local data-family and corpus-count audit (2026-10-19; no model experiment)

- **Question:** Does the available local directory structure establish that the project-labeled 2k subset or reported ~20k corpus is SEN1-2, and does the ~20k count match another published SAR–optical dataset?
- **Primary-source finding:** The 2018 SEN1-2 paper reports 282,384 corresponding SAR/optical patch pairs. The distinct 2021 QXS-SAROPT dataset paper reports 20,000 256×256 pairs from 1 m single-polarization Gaofen-3 SAR and Google Earth optical across San Diego, Shanghai, and Qingdao. It describes manual tie points, bilinear optical-to-SAR registration, 46,071 candidates, and manual filtering to 20,000; it says “stride 52” produces 20% overlap, an internal wording ambiguity. It explicitly distinguishes QXS-SAROPT from SEN1-2. This makes count/name conflation plausible, not proven. The [QXS-SAROPT author repository](https://github.com/yaoxu008/QXS-SAROPT) links data downloads through a form; no form was submitted and no data were fetched.
- **Patch geometry comparison:** SEN1-2's source paper describes stride 128 and 50% overlap for 256×256 patches. QXS-SAROPT states 20% overlap while calling the 52-pixel quantity a “stride”; under conventional step-size terminology these conflict. The local audit's 51/52-pixel strips at 204/205-pixel translations geometrically match 20% overlap, making QXS or another similarly tiled source plausible without proving lineage.
- **Local metadata:** Filename-only inventory confirms 2,000 same-named files in each of `data/sar` and `data/opt` (8000–9999) and 100 in each mini directory (8000–8099). It did not consult split membership, open PNG payloads, or read measurements. No non-PNG sidecar/manifest appeared through depth 3 outside test-named paths. The [reproducible script, command, and summary](../../experiments/data_audits/sen1_2_workspace_inventory_20261019_v1/) record the inventory.
- **Distribution route:** The QXS-SAROPT [author repository README](https://github.com/yaoxu008/QXS-SAROPT) links to a data-request form. Its public [issue #13](https://github.com/yaoxu008/QXS-SAROPT/issues/13), opened 2026-01-17, requests a Google Drive/Hugging Face mirror because the reporter cannot access the data on Baidu; it remained open with no maintainer response at the time of this check. A separate [ReFlowSET model card](https://huggingface.co/JeonghyeokDo/ReFlowSET) says its two training datasets are not redistributed. A non-author [repository README](https://github.com/Cosmosoc/SAR-Imagery-Colorization) links a file named `QXSLAB_SAROPT.zip` on Google Drive, but the page exposed only a sign-in screen and no archive ownership, contents, version, license, or manifest could be verified. Treat that mirror claim as unverified; no form was submitted and no data were downloaded. None of these routes establishes local QXS lineage.
- **Interpretation:** The inspected `data/` tree holds the project-labeled 2k subset; this inventory does not establish whether the project-reported ~20k corpus exists elsewhere. The local source remains unverified; neither the SEN1-2 nor QXS-SAROPT attribution is established. The 3,600-file SEN1-2 SHA-512 crosswalk found no byte-identical matches and cannot resolve source identity.
- **Decision:** D-2026-10-19-10/11. Call the files “project-labeled SEN1-2” pending provenance, preserve all historical run labels, and identify the local 2k source plus the project-reported ~20k corpus before using upstream grouping metadata. B1 remains failed, B2 gated, and no model experiment or test-image/measurement access occurred.

## Sibling SOTANet QXS-named cache crosswalk (2026-10-19; no model experiment)

- **Question:** Does the neighboring ignored `../SOTANet/datasets/sar2opt-QXS/` cache contain pixel-equivalent files for current fixed train/validation identities?
- **Scope/method:** Crosswalk only `fixed_train_dev2k` and `fixed_val_dev2k` identities. For same-named sibling files in its `train`/`val` folders, decode 512×256 RGB composites and compare the left replicated-grayscale panel and right RGB panel exactly against manifest-bound local SAR/optical PNGs. No resize, interpolation, normalization, or visual matching. The sibling cache's `test/` names are counted only; their payloads are not opened, decoded, or hashed.
- **Outcome:** The sibling cache has 20,000 composite paths in local train/val/test folders (16,000/2,000/2,000). Of 1,800 current fixed train/validation identities, 1,626 same-name train/val composites match both local panels exactly with no shape/content mismatches. The other 174 identities occur in the sibling `test/` folder by filename only (155 project-train, 19 project-validation). Project manifest SHA-256 values: train `9d811f5a768d7f42ff480504c31479423bdb8ac6b2b2d6a8ac886ec941815b1a`; validation `f603a8440484bb0b170e0814307da4c20f69416bff577e78ea2c5cb531361b70`.
- **Interpretation/boundary:** This is exact pixel equivalence to a local QXS-named cache for 1,626 development identities, not proof of official QXS-SAROPT lineage, license, original split semantics, or parent-scene/acquisition groups. The 174 filename-only matches are a possible partition risk, not verified official-test leakage. The primary QXS paper and author repository provide no patch-to-source mapping or canonical full-dataset split. ReFlowSET's README describes separate downstream frozen QXS filename lists, but its `splits/` directory was not retrievable; no list contents, test identities, imagery, or metrics were accessed. No local fixed-test manifest/payload, model, or historical gate was accessed or changed; model selection remains paused pending authorized lineage and a prospective group-disjoint split.
- **Evidence:** [crosswalk report](../../experiments/data_audits/sar2opt_qxs_cache_crosswalk_20261019_v1/report.md), [summary with hashes](../../experiments/data_audits/sar2opt_qxs_cache_crosswalk_20261019_v1/summary.json), [script](../../experiments/data_audits/sar2opt_qxs_cache_crosswalk_20261019_v1/audit_crosswalk.py), [primary QXS paper](https://arxiv.org/html/2103.08259), [QXS author repository](https://github.com/yaoxu008/QXS-SAROPT), and [downstream ReFlowSET repository](https://github.com/KAIST-VICLab/ReFlowSET).

## Sibling QXS-named cache PNG metadata scan (2026-10-19; no model experiment)

- **Question:** Do the sibling cache's train/validation PNG containers embed source, scene, acquisition, georeferencing, or split metadata that could recover source groups?
- **Scope/method:** Read PNG signatures, chunk headers, and non-IDAT metadata from only top-level `train/*.png` and `val/*.png` under `../SOTANet/datasets/sar2opt-QXS/`; seek over IDAT payloads without decompression or pixel decoding. Neither the sibling `test/` directory nor any JDSR fixed-test manifest/payload was accessed.
- **Outcome:** All 16,000 train and 2,000 validation files were 512×256, 8-bit. Across both splits, only `IHDR`, `IDAT`, and `IEND` chunks were present; no text, EXIF, ancillary, geospatial, or nonstandard chunks were found. The scanner skipped 5,025,119,948 IDAT bytes and reported no scan errors.
- **Interpretation/boundary:** Embedded PNG chunk metadata cannot recover source groups for these scanned files. This does not establish upstream archive/version, license, official QXS-SAROPT relationship, local split semantics, acquisition IDs, or parent scenes. Keep lineage unresolved and model selection paused pending authoritative source/group metadata; no historical result or gate changes.
- **Evidence:** [audit report](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/report.md), [summary](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/summary.json), [scanner](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/scan_png_metadata.py), [command](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/command.txt), and the two audit-specific checks in `test_scan_png_metadata.py`.
- **Filename/history extension:** Directory-entry checks found that all 18,000 train/validation files are PNGs with decimal numeric stems and no non-PNG sidecars; train and validation have no identical basenames. A search of all reachable SOTANet history and current tracked path names found no `QXS`/`sar2opt` preparation reference. Numeric IDs and an empty repository search do not recover original scene groups or establish upstream lineage. Commands and counts are in the linked audit report/summary.

## Artifact availability update (2026-10-21; no experiment)

The initial cleanup removed generated `.npy` fixed-input arrays and per-image `.npy`/`.png` predictions from the legacy July 1900/100 `outputs/` paths. The supplementary authorized cleanup removed writable root fixed-input arrays and `formal_test/degradation_sensitivity/fixed_inputs` tensors. A broader pass removed all 4,600 generated `.npy`/`.npz` arrays under `experiments/dev2k_formal/` and 2,800 fixed validation/test `.npy` arrays; their configs, commands, manifests, CSV metrics, logs, figures, and checkpoints remain. On 2026-10-23, a later user-authorized pass pruned the read-only `outputs/pilot_archive/` snapshot of copied weights, legacy test100 metrics/reports, run checkpoints, and compiled/generated paper outputs; it retained the original archive inventory manifest (which describes the pre-prune snapshot), the unique archived evaluator source cited above, and the remaining code/config/run-record snapshot. See the [archive-pruning audit](../../experiments/archive_pruning_20261023.json). Array selection used path/suffix and filesystem metadata only; no array or image contents, test measurements, or metrics were read. The earlier duplicate verification read 100 pairs from the legacy mini test subset and matching source PNGs; those identities remain excluded from future selection; see the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).

## Named SOTANet cache crosswalk (2026-10-21; no model experiment)

Fixed train/validation metadata contained all 100 legacy mini test identities in train and zero in validation. The audit excluded those identities before any source read, leaving 1,500 train and 200 validation pairs. Exact comparison against 12,799 train and 1,601 val composites in the separate `../SOTANet/datasets/sar2opt` cache found zero SAR, optical, or same-composite pair matches. All candidate filename/shape/mode checks passed. No group mapping or new split was produced; no test directory or current fixed-test payload was read. The current audit's test flags are false, and prior cleanup access remains separately disclosed. Evidence: [report and artifacts](../../experiments/data_audits/sen1_2_sotanet_named_cache_crosswalk_20261019_v1/report.md).

## Official SEN1-2 named-cache pair probe (2026-10-21; no model experiment)

The lexicographically first named-cache train composite was fixed before retrieval. Its SAR/optical panels match the two PNGs delivered by the official TUM rsync paths exactly. Both source-file hashes, pixel hashes, URIs, contracts, and matcher/probe hashes are retained; the 134,544 downloaded bytes were temporary and removed. The old checksum index was not available and was not compared. Filename-only inventory has 30 groups in each train/val partition and all 30 shared, so native cache membership cannot serve as scene-independent validation. This verifies one pair only and does not map the numeric local 2k or promote a model. Evidence: [protocol, source, and summary](../../experiments/data_audits/sen1_2_named_cache_official_pair_probe_20261021_v1/report.md). Current probe test flags are false; no numeric-2k source image or test directory was read.


## Official-source prospective cohort authentication (2026-10-21; no model experiment)

A metadata-frozen cohort from the sibling `SOTANet/datasets/sar2opt` cache was split independently of its native train/val labels. The cohort contains 1,600 training identities in 24 source-folder groups and 200 validation identities in six disjoint fall-only source-folder groups. Official TUM SEN1-2 source verification matched all selected SAR/optical pairs at decoded-pixel level: 1,800/1,800, zero failures. Exact full-patch and declared ±128-grid checks found zero cross-split overlaps for SAR or optical. This is a source-folder-disjoint development split, not an untouched final test or proof of geographic/acquisition independence. Test directory enumeration/access, local numeric-2k payload access, training, model selection, and dataset adoption flags are false. The canonical checksum index was not compared. See the [frozen protocol](../../protocols/sen12_scene_dev1800_v1/protocol.md), [audit report and retry evidence](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v2/report.md), and preserved [failed transfer attempt](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v1/summary.json). No historical gates or metrics changed.


## Preferred 15/15 official-source cohort authentication (2026-10-21; no model experiment)

After authenticating the initial 24/6 cohort but before any model metrics, metadata-only capacity analysis showed a 15/15 allocation feasible at the same 1,600/200 sample counts. Protocol v2 selects 15 train source folders and 15 disjoint fall validation source folders (all `ROIs1970_fall`); training keeps the singleton summer group plus 14 fall groups. All 1,800 pairs exactly match official TUM SEN1-2 SAR/optical source pixels. Exact and eight declared ±128-grid checks find zero cross-split matches for either modality. The 24/6 v1 remains a valid retained audit; use v2 for prospective model work because it has more validation source folders. This remains development validation, not a final test or evidence of parent-ROI/acquisition independence. Test flags, candidate-test enumeration, training, and model selection are false. See the [v2 protocol](../../protocols/sen12_scene_dev1800_v2/protocol.md), [v2 audit](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md), and [retained v1 audit](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v2/report.md).


## Prospective B1 revalidation setup (2026-10-21; no model experiment)

A separate protocol is frozen for the source-authenticated v2 cohort: 1,600 train and 200 validation identities across 15 disjoint source-folder groups per split. It uses the predeclared 129-bin, 256→128 representation-domain degradation, noisy SAR pseudo-targets, unchanged historical B1 gates, and source-group bootstrap uncertainty. Synthetic-only resource profiling projected about 2.1 GPU hours of optimizer compute for the declared three-seed screen, excluding validation, I/O, metrics, and checkpoint overhead. The source-group-aware aggregator is implemented. The original fixed-input payloads were removed during cleanup; a later CPU rebuild exactly reproduced and verified train/validation file-list, split, and source-protocol hashes, then removed the generated payloads again. See [the protocol](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v1.md), [input-preparation records](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_revalidation_inputs_20261021_v1/), and [rebuild audit](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_input_rebuild_20261021_v1/). No training, model selection, test enumeration, test payload access, or gate change occurred. The current shell has no visible CUDA device; recheck resources and regenerate payloads before any training run.

## Prospective B1 fixed-input deterministic rebuild (2026-10-21; no model experiment)

The frozen v2 train and validation fixed inputs were rebuilt with the retained commands on CPU. The verifier completed in formal/source-checking mode for all 1,600 train and 200 validation identities, checking each LR, target, optical, and four mask files for hash, dtype, shape, finite values, range, and expected count. Stable `files.csv` hashes (`ec9287a7…b7619` train; `c26d1417…1c6a` validation), split hashes (`f3e26083…a0ed`; `222ff6fc…1cf0b`), and the source-protocol hash (`966ffd3f…6be2a8`) match the retained frozen audit exactly. The source staging manifest hash also matches (`eb3e5f78…5c9b0dd9`). Compact manifests, commands, logs, and verification summary are in the [rebuild audit](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_input_rebuild_20261021_v1/); generated PNG/NPY payloads were removed after verification. The current `AISAR` environment reports PyTorch 2.8.0+cu128, CUDA unavailable, and `nvidia-smi -L` unable to contact the driver. No model run, metric calculation, or selection occurred; test flags are false and candidate test was not enumerated.

## Prospective B1 trainer-to-aggregator interface audit (2026-10-21; no model experiment)

Read-only comparison of both generic trainer outputs with the frozen aggregator contract found they are not directly compatible. The trainers emit root manifests, best-epoch metadata and per-epoch CSVs under `validation_per_image/`; they omit `protocol_id`, role, top-level architecture, `candidate_test_enumerated`, and the `fixed_input_files_csv_sha256` map. The aggregator requires those fields plus a nested `validation/manifest.json` and selected-epoch `validation/per_image.csv`. The protocol's `strict_score` is implemented identically in the shared metric helper, and the training settings/loss terms are expressible through current CLI flags. No trainer, aggregator, or run artifact was modified. Add a tested fail-closed sealing step before any B1 training; no gate or result changes.

## Prospective B1 native-run sealing interface (2026-10-21; synthetic fixtures only)

- **Question:** Can completed native SwinIRJoint and MMCANetV2 trainer outputs be bound to the frozen source-group aggregator contract without changing trainer dynamics or reading image/tensor payloads?
- **Implementation:** `scripts/seal_source_group_b1_run.py` reads the native config and run manifest, `train_log.csv`, the selected epoch CSV, fixed-input manifests/`files.csv`/`split.txt`, and checkpoint bytes for hashing. It verifies protocol architecture/settings, config and checkpoint hashes, CUDA metadata, false test-access flags, fixed-input stable hashes/split/source protocol, exact counts and identities, strict-score best-epoch agreement, and finite validation metrics/order. For candidates, it requires the residual calibration JSON and checks its seed, train-manifest binding, first train identities, same-seed SAR-only checkpoint hash, gradient-norm formula, and equality to the configured residual loss weight. It writes the normalized best-epoch `validation/per_image.csv`, its `validation/manifest.json`, and fields required by the aggregator. Existing canonical validation output causes failure.
- **Verification:** `tests/test_seal_source_group_b1_run.py` exercises both run roles, calibration binding, mismatch rejection, and a six-run synthetic handoff through `aggregate_source_group_b1.py`. The sealer plus the existing aggregator tests pass 21/21; `py_compile` passes. Full suite: 232 passed, 2 failed. One expects `fixed_train_dev2k/lr_sar/8000.npy`, which was removed by the requested cleanup. The other detects stale source hashes for four existing untracked files: `scripts/calibrate_phase_b1_residual_weight.py`, `scripts/train_mmcanet_v2.py`, `MMCANetV2/model.py`, and `JointJDSR/model.py`.
- **Boundary:** No real model run was sealed or aggregated; synthetic success proves interface compatibility only. No training, inference, metric selection, test images/measurements, or gate change occurred. The full suite invoked the existing `test_ogsrn_protocol.py::test_fixed_ogsrn_splits_are_complete_and_disjoint`, which reads only fixed-test manifest/split/file-name membership metadata and patches `np.load` to reject fixed-test array access. CUDA is unavailable in the current shell. Use the project PyTorch environment, rebuild fixed input payloads, then seal each of six runs before the frozen aggregator.
- **Contract:** [B1 aggregation and sealing instructions](../../protocols/sen12_scene_dev1800_v2/b1_aggregation_contract_v1.md).

## Prospective B1 registration-condition binding (2026-10-22; no model experiment)

- **Question:** Can the remaining fine-registration uncertainty be stated in the prospective run record without changing the frozen B1 model configuration or promotion gate?
- **Decision:** Freeze [`registration_condition_addendum_v1.json`](../../protocols/sen12_scene_dev1800_v2/registration_condition_addendum_v1.json). The cohort is source-matched under nominal georegistration; fine pixel co-registration and residual-error distributions remain unverified for its fall-only validation groups.
- **Interface:** `seal_source_group_b1_run.py` writes the addendum ID, relative path, condition, and SHA-256 into the root run manifest and selected-epoch validation manifest. `aggregate_source_group_b1.py` checks both bindings against the frozen addendum and carries the record into aggregate summary, report, and manifest.
- **Verification and boundary:** The two focused suites pass 23 tests, including missing-binding rejection and the six-run synthetic handoff with synthetic metric rows. No real model run, inference, metric evaluation, selection, test-data access, or gate change occurred. The run remains a prospective development screen, not pixel-aligned-optical evidence.


## Interface re-verification and literature update (2026-10-22; no model experiment)

Fresh rerun of `tests/test_seal_source_group_b1_run.py` and `tests/test_aggregate_source_group_b1.py` passed 23/23, including the synthetic six-run handoff. This verifies the packaging interface only; it does not verify a trained prospective run or B1 gate. The same continuation reviewed SpaceNet 9's published full text as registration-only prior art; no local data or experiment results were accessed. See [CURRENT_STATE.md](CURRENT_STATE.md) and [D-2026-10-22-06](DECISION_LOG.md#d-2026-10-22-06--classify-spacenet-9-as-registration-prior-art-only).

## Prospective B1 v1 SAR-only seed-0 evaluator failure (2026-10-22)

- **Protocol and inputs:** Frozen `sen12_scene_dev1800_b1_v1`; 1,600 train / 200 validation. Rebuilt train/validation inputs passed formal/source verification. Stable file-list hashes are `ec9287a07b9c7b708c01b4496b6054fbad6f7d903c1d205cfaef24dad69b7619` (train) and `c26d1417aaee0fd0fd248e5288fd1bf672f7c4231df6e93afb33781ffaa81c6a` (validation); source, split, and manifest hashes are retained in the run record and linked preparation audit.
- **Attempt:** SAR-only seed 0 completed one training epoch (800 batches) on an authorized RTX 3050 Laptop GPU profile, then the first validation pass raised `ValueError` in `tbr_metrics`: target-background median denominator is zero or too small. The target-only diagnostic identifies 2/200 zero medians; all 200 target-background means are nonzero. The frozen strict score and gate use mean TBR, while the shared evaluator also computes median TBR and aborts on its undefined denominator.
- **Outcome:** 18 per-image rows were computed in memory but not persisted. No checkpoint, selection, completed metric table, gate evaluation, or model-quality result exists. The remaining SAR-only seeds and all candidate runs were not started. Test flags are false; candidate test was not enumerated. Historical protocols, thresholds, metrics, and gates remain unchanged.
- **Cleanup:** Rebuilt `.npy`/`.png` payloads (16,200 files; 2,790,467,668 bytes), duplicate metadata, progress logs, and raw traceback were removed. The run manifest, exact fixed-input manifests, native config/command, concise denominator diagnostic, and source-authenticated input audit remain.
- **Next action:** Define and verify a separately versioned evaluator policy for undefined ancillary median TBR before rerun. Keep required mean TBR coverage at 200/200 and the original mean-based gate unchanged unless a future predeclared protocol explicitly supersedes it. See [run record](../../experiments/sen12_scene_dev1800_b1_revalidation_20261021_v1/) and [decision D-2026-10-22-15](DECISION_LOG.md#d-2026-10-22-15--preserve-the-failed-v1-run-and-version-any-evaluator-repair).


## Prospective B1 v2 evaluator-policy interface (2026-10-22; synthetic verification only)

- **Protocol:** `sen12_scene_dev1800_b1_v2`, metric policy `b1_mean_required_median_optional_v1`; it retains the frozen v1 source cohort, models, training settings, strict score, eight required gate metrics, numeric thresholds, and test boundary. A versioned registration addendum remains bound separately.
- **Metric behavior:** Mean TBR is required and finite for every identity. Median TBR is ancillary. When either background median is zero/too small, its three values are serialized unavailable with one explicit status; per-role/per-seed coverage is recorded. Partial median means are omitted. Median availability does not enter selection or the gate.
- **Interface changes:** Both trainers accept `--protocol-id`; B1 evaluation now emits the bright-peak, bright-integrated-energy, and bright-region-MAE fields required by the frozen aggregator. The sealer records policy/coverage in run and validation manifests and preserves optional fields in canonical CSV. The aggregator verifies row statuses and manifest coverage, reports per-role/seed coverage, and retains the unchanged mean-based gate.
- **Verification:** 54 focused tests passed across evaluator, policy, sealer, and aggregator suites; Python compilation and each trainer/sealer/aggregator `--help` entrypoint passed. Synthetic fixtures include six-run v2 sealing and aggregation.
- **Boundary:** No real run, cohort pixels, predictions, metrics, or test data were accessed/generated. This implements the approved interface repair only; protocol v1 remains the recorded failed attempt, no model-quality result or gate decision exists, and B2 remains gated.

## Prospective B1 v2 SAR-only seed-0 baseline (2026-10-22; matrix in progress)

- **Run:** sen12_scene_dev1800_b1_revalidation_20261022_v2, protocol sen12_scene_dev1800_b1_v2, metric policy b1_mean_required_median_optional_v1, registration condition source-matched nominal georegistration; fine pixel co-registration is unverified.
- **Inputs:** 1,600 train / 200 validation identities, 15 source groups per split; source-authenticated staging manifest and per-identity hashes are preserved in the run's input_metadata/. The final train and validation files.csv are byte-identical to frozen references (ec9287a0…b7619 and c26d1417…81c6a). Formal/source verification passed for both splits. The file-list IDs remain the frozen v1 input IDs because v2 changes only evaluator representation of undefined ancillary median TBR; training records bind the v2 protocol/policy.
- **SAR-only seed 0:** 25 epochs completed on CUDA; 25 epoch rows were recorded. The frozen strict score selected epoch 14 (17.3329381133). The native run sealed successfully with a 200-row canonical best-epoch validation CSV, explicit median coverage, best/last checkpoints, and registration-addendum binding. See sar_only_seed0/manifest.json and sar_only_seed0/validation/manifest.json.
- **Pruning:** Removed redundant per-epoch checkpoints (25 files, 552,235,325 bytes) and per-epoch validation CSVs (25 files, 1,626,005 bytes). Retain best.pth, last.pth, train_log.csv, console.log, canonical validation/per_image.csv, exact commands/configs/manifests, environment, and input metadata.
- **Boundary:** This single SAR-only seed is not a paired comparison and has no B1 gate result. The five remaining runs are pending. Run flags test_set_used=false, test_set_accessed=false, candidate_test_enumerated=false; no candidate-test directory or test measurements were accessed. A separate session-level possible text-only historical test exposure is disclosed in session_access_disclosure.json.

## Prospective B1 v2 seed-0 paired runs and loader repair (2026-10-22)

- **Interface repair:** `allow_variable_input` controls only the receiving model's accepted spatial dimensions; it does not affect the parameter shapes or forward computation for a fixed input. The loader now ignores this one config field while retaining run-config/hash, method, seed, fixed-input manifests, all other architecture settings, parameter-count, and strict-weight checks. The focused test was observed failing before the repair; `tests/test_mmcanet_v2.py` passes 21/21 and both modified Python files compile. Revalidated 2026-10-23 in `AISAR` on CPU: 21 passed, one existing meshgrid future-warning; source hashes are unchanged and no data/test set was accessed. Details: [interface repair audit](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/interface_repair_audit.json).
- **Calibration:** Seed-0 residual loss weight was computed from the deterministic first train batch only: `lambda_residual=0.739917531104381`. It binds the seed-0 SAR-only best checkpoint and frozen train manifest; test flags are false.
- **Optical candidate seed 0:** Registered v2_b/C2 training completed 10 epochs. The frozen strict selection score chose epoch 1 (`17.3037874374`); the sealer validated all 200 canonical validation rows, fixed input identities, checkpoint hash, optional-median coverage, calibration binding, and registration addendum. Median-TBR coverage is 198/200. See `optical_candidate_seed0/manifest.json` and `optical_candidate_seed0/validation/manifest.json`.
- **Pruning and boundary:** Ten epoch-level per-image validation CSVs (650,377 bytes), temporary py_compile cache (945,506 bytes), and pytest cache (2,366 bytes) were removed. Best/last checkpoints, train log, all-epoch validation summary, canonical best-epoch per-image CSV, command/config/environment, calibration and manifests remain. Two of six runs are sealed; seed-1/2 and aggregation remain. No gate was evaluated and run test flags are false.



## Prospective B1 v2 seed-1 interruption and fixed-input recovery (2026-10-22)

- **Attempt:** `sar_only_seed1` reached 19/25 completed epochs before its process disappeared. The exit cause is unknown; no traceback or OOM-kill evidence was available. No validation selection, complete run seal, gate evaluation, or test access resulted.
- **Preserved/cleaned:** The attempt is archived at [`interrupted_runs/sar_only_seed1_attempt_20261022/`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/interrupted_runs/sar_only_seed1_attempt_20261022/). Its command/config/provenance, 19-row train log, and concise interruption audit remain; partial checkpoints and 19 epoch validation CSVs were pruned.
- **Recovery:** Recreated train/validation payloads only. Their configs, files.csv, split lists, and manifest bytes match the frozen input references after restoring the exact frozen manifest bytes; `verify_fixed_degradation.py --require-formal --verify-sources` completed for 1,600 train and 200 validation samples. Source staging retained 15 groups per split. The compact [input rebuild audit](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/input_rebuild_after_interruption_20261022.json) records hashes and contract. Redundant full verifier JSON was removed.
- **Boundary and next step:** Recovery performed no training, inference, metric evaluation, model selection, or test access. The from-scratch seed-1 retry completed 25/25 epochs with exit code 0 after fresh GPU preflight and was sealed at best epoch 15. The seed-0 pair plus seed-1 SAR-only are the three sealed runs; the seed-1 optical candidate and seed-2 pair remain pending, and the aggregate gate is unevaluated. Keep B2/B3 gated. See [decision D-2026-10-22-19](DECISION_LOG.md#d-2026-10-22-19--archive-the-unknown-seed-1-interruption-and-retry-from-frozen-inputs).


## Prospective B1 v2 SAR-only seed-1 retry completed and sealed (2026-10-22)

The second `sar_only_seed1` attempt used the same frozen 25-epoch command and exact recovered 1,600/200 train/validation inputs. It completed 25/25 epochs with exit code 0 on the RTX 3050 Laptop GPU. The source-group sealer chose epoch 15 by the frozen strict score (`17.499261243381028`), checked all 200 canonical validation rows, and bound the fixed-input manifests, optional-median policy, and registration addendum. Median-TBR coverage was 198/200. Checkpoint SHA-256: `a2bf8cf3652606fb047d2a0e4ceefa2f9cb60388a5e0091f2c342d8a4940c611`; canonical per-image CSV SHA-256: `f5adc5b907fe42e7f68b57033031ee1cfb998db508c7ad875bedc51c531fd461`. Run test flags are false.

The run took 5,793.67 s (1.609 h) end to end; this is a single-run observation and includes validation/I/O. Removed 25 redundant epoch checkpoints (552,235,325 bytes) and 25 epoch-level validation CSVs (1,625,776 bytes), plus a duplicate temporary console log. Best/last checkpoints, command, config, console/train logs, canonical validation CSV and manifests remain. This is one SAR-only seed and provides no optical comparison or B1 gate result. Next: calibrate seed-1 residual weight from its first train batch only, train/seal the matching optical candidate, then finish seed 2.


## Prospective B1 v2 seed-1 train-only calibration (2026-10-22)

A first invocation omitted `--validation-dir`; its legacy default `fixed_val_dev2k` caused a validation-manifest mismatch. It failed before gradient calibration and wrote no output. The only access to that old path was its manifest metadata hash; no old validation image/array or metric was read. See [`calibration_seed1_invocation_audit.json`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/calibration_seed1_invocation_audit.json).

The corrected invocation supplied the frozen v2 validation directory explicitly for checkpoint-lineage validation and loaded only the deterministic first two samples from fixed train for gradient calibration. It completed with `lambda_residual=0.8083524623361836`, binding seed-1 SAR-only checkpoint `a2bf8cf3652606fb047d2a0e4ceefa2f9cb60388a5e0091f2c342d8a4940c611` and train manifest `36274db5abceca2c8955062ab811812011aee373b27cc51d9404477851763304`. Calibration SHA-256 is `5f8ab67aa443f02f6754acd725587daaf1655d8054702995e97eb8d4d38666a8`; test flags are false. `commands.md` now requires explicit `--validation-dir` for all seed calibrations. No protocol, metric, or gate changed.


## Prospective B1 v2 optical-candidate seed-1 completed and sealed (2026-10-22)

The matched v2_b/C2 seed-1 run completed 10/10 epochs with exit code 0 on the same fixed train/validation cohort and same-seed SAR-only initialization. The source-group sealer validated the candidate checkpoint, seed-1 calibration SHA, baseline binding, metric policy, registration addendum, and all 200 canonical validation rows. Best epoch 1 has strict score `17.43902122669129`; median-TBR coverage is 198/200. Checkpoint SHA-256: `c79932d352f5a7dc7a109b5ddea6e3443fc9415be86f1170228dd92bbd8df217`; canonical per-image CSV SHA-256: `fb20b868442f48dfe9e79375b54adda337f916c7c5a8ab7284f6264192d2d290`. Run test flags are false. End-to-end runtime was 1,619.95 s (0.450 h).

Removed ten redundant per-epoch validation CSVs (650,283 bytes) after sealing; best/last checkpoints, `train_log.csv`, `validation_metrics.csv`, console log, canonical `validation/per_image.csv`, and manifests remain. Four of six runs are sealed; only seed-2 SAR-only/C2 and aggregation remain. No gate was evaluated and B2/B3 remain gated.


## Prospective B1 v2 seed-2 SAR-only interruption (2026-10-22)

- **Attempt:** `sar_only_seed2` reached 19/25 epochs; the best observed strict validation score was 17.445867 at epoch 19. The process was no longer present during inspection, NVIDIA device nodes were absent, and `nvidia-smi` could not communicate with the driver. No traceback or OOM record establishes a cause.
- **Artifact disposition:** Archived at [`interrupted_runs/sar_only_seed2_attempt_20261022/`](../../experiments/sen12_scene_dev1800_b1_revalidation_20261022_v2/interrupted_runs/sar_only_seed2_attempt_20261022/). Retained command, config, manifest, console/train logs, and interruption audit; pruned 21 partial checkpoints and 19 per-epoch validation CSVs (465,091,114 bytes). This was not sealed and its observed score is not a registered result or gate input.
- **Boundary/next step:** Four of six runs remain sealed; seed-2 baseline retry and paired C2 candidate remain pending. Restart the baseline from scratch after GPU restoration because the trainer has no resume interface; then calibrate on the first train batch with explicit frozen `--validation-dir`, train/seal the paired candidate, and aggregate only when all six are sealed. Test flags are false; no test images/measurements were accessed. The root manifest records `paused_gpu_unavailable`. B2/B3 remain gated.


## Prospective B1 v2 payload cleanup during GPU pause (2026-10-23)

Removed the temporary train/validation preparation tree after 13 exact metadata hash comparisons matched retained references: 12,600 `.npy` arrays and 3,600 staged `.png` files, totaling 2,790,467,668 bytes, plus 15 duplicate temp metadata/log files. Canonical manifests, file lists, splits, configs, source manifests, commands, verification summary, and the interruption records remain. The v2 run manifest sets `input_payloads_currently_present=false`; four of six runs remain sealed, seed-2 baseline retry and C2 are pending, the gate is unevaluated, and B2/B3 remain gated. No test payloads or metrics were accessed. See [`artifact_cleanup_20261023.json`](../../experiments/artifact_cleanup_20261023.json) and [decision D-2026-10-23-22](DECISION_LOG.md#d-2026-10-23-22--prune-rebuildable-b1-v2-payloads-during-the-gpu-pause).

## B1 calibration CLI guard (2026-10-23; no model experiment)

After a seed-1 invocation omitted `--validation-dir` and safely failed against the legacy `fixed_val_dev2k` manifest before gradient calibration, the calibrator's CLI and Python API were changed to require an explicit validation directory. The documented v2 command continues to point to the frozen prospective validation directory. The change passed `py_compile`; it creates no run result and does not change any protocol or gate. Calibration uses only the deterministic first train batch and reads validation manifest metadata for checkpoint lineage; test flags remain false. See [D-2026-10-23-23](DECISION_LOG.md#d-2026-10-23-23--require-an-explicit-validation-manifest-for-b1-calibration).

## Prospective B1 epoch-boundary resume interface (2026-10-23; synthetic verification only)

Both B1 trainers now resume an incomplete native run from the same run directory's `checkpoints/last.pth`. The interface verifies run config, seed, all CLI training choices, split identities, fixed-input manifest hashes, and false test-access flags; restores strict model/optimizer/scheduler/scaler/EMA/RNG/DataLoader state; writes checkpoints atomically with epoch summaries and current validation rows; and reconstructs the run logs after an interrupted output write. The run manifest keeps earlier failure details and records each resume command. Checkpoints made before this interface are rejected when their state is incomplete.

The new [resume tests](../../tests/test_training_resume.py) pass 7/7; `tests/test_mmcanet_v2.py` passes 21/21; synthetic sealer/aggregator tests pass 26/26; and all three changed Python files compile. No training, inference, metric evaluation, data/pixel access, or test-set access occurred. Four of six registered B1 v2 runs remain sealed, the gate remains unevaluated, and B2/B3 remain gated. The archived seed-2 interruption cannot use resume because its partial checkpoint and per-epoch validation artifacts were pruned; restart that pair from scratch after a fresh CUDA preflight and fixed-input rebuild.

## MONTI two-stage SAR-SR indexed-text review (2026-10-23; no model experiment)

- **Question:** Does MONTI 2026 alter the prior-art boundary for ordered SAR restoration or the project's joint-versus-cascade hypothesis?
- **Evidence:** The official workshop site confirms acceptance and non-archival status. Indexed text from the official OpenReview manuscript reports IW-trained structure restoration followed by MM-BSN and conditional latent-diffusion speckle synthesis; stage 2 lowers both reported FWD distributions for ESRGAN, Restormer, and DRCT. The separate 1,000-sample synthetic paired table reports stage 1 at 23.4196 dB/0.4389 SSIM, individual diffusion samples near 22.50/0.367, and their four-sample average at 24.0388/0.458. See the [audit report](../../experiments/literature_audits/monti_two_stage_sar_sr_20261023_v1/report.md).
- **Interpretation:** This narrows generic two-stage SAR-SR novelty. It does not compare reversed task order, matched joint restoration, optical guidance, misregistration, or fallback. Real-input FWD values are distributional rather than paired-pixel scores.
- **Boundary:** Direct PDF opening redirected to browser verification. No local data, image, metric, model output, test material, training, inference, selection, or gate action occurred.
- **Decision:** D-2026-10-23-31. Keep H8 open and unchanged; no experiment is authorized by this literature update.

## MONTI public-code search follow-up (2026-10-23; no model experiment)

- **Question:** Does a public official implementation link resolve the MONTI paper's code availability?
- **Evidence:** Exact-title, author-name, OpenReview-ID, GitHub/repository, and SI Analytics-domain searches surfaced the official paper/workshop pages but no author repository. Direct PDF access remains blocked, so the search cannot prove code is unavailable. See the [audit report](../../experiments/literature_audits/monti_two_stage_sar_sr_20261023_v1/report.md).
- **Boundary:** No local data, code checkout, model output, test material, or experiment was accessed.
- **Decision:** D-2026-10-23-32. Leave code availability unverified and do not repeat the same routes without a new access clue.

## SOTANet QXS-named cache local generator/config audit (2026-10-23; no model/data experiment)

- **Question:** Do local source helpers and saved training options bind the QXS-named cache to an upstream release, license, split procedure, or scene-group mapping?
- **Evidence:** At SOTANet HEAD `c677065`, `data/split_dataset.py` is tracked and performs an unseeded per-file shuffle and ratio split without pair/group semantics. `datasets/prepare_sar2opt_data.py` is ignored, points to a different hard-coded SEN1-2 input/output path, and has no recorded invocation link to the QXS-named cache. Two ignored training-option files record `dataroot=./datasets/sar2opt-QXS/` but no dataset hash/version, split-generation command, or source IDs. See the [audit report](../../experiments/data_audits/sar2opt_qxs_local_generator_audit_20261023_v1/report.md).
- **Interpretation:** Local training use of the cache is supported; upstream identity, cache license, local partition semantics, and parent-scene/acquisition groups are not. These files do not alter the earlier exact-pixel crosswalk's scope.
- **Boundary:** Read-only code/config/Git metadata review; no pixel data, metric, model output, test identity, or test payload was accessed. Only top-level split labels were visible. No training, inference, evaluation, selection, or gate action occurred.
- **Decision:** D-2026-10-23-30. Keep lineage unresolved; do not use cache partitions as source groups or the legacy contaminated `fixed_val_dev2k` / B1F inner-dev partitions for selection.


## B1 v2 current-source rerun — `sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1`

- Status: complete; frozen B1 gate returned `not_promoted` and `next_phase=stop_at_b1`.
- Protocol: frozen `sen12_scene_dev1800_b1_v2`; metric policy `b1_mean_required_median_optional_v1`; seeds 0/1/2; unchanged 25-epoch SwinIRJoint SAR-only and 10-epoch MMCANetV2 v2_b/c2 settings.
- Cohort: 1,600 train and 200 validation identities; 15 disjoint source folders per split. Registration is source-matched nominal georegistration; fine pixel alignment is unverified. Targets are normalized PNG-derived noisy pseudo-targets.
- Inputs: formal/source verifier passed for train (1,600) and validation (200). Fixed manifests: train `36274db5abceca2c8955062ab811812011aee373b27cc51d9404477851763304`; validation `88429603e8d69255bbd9627159fec427701c1f22150e7d0060904d0efe2f6279`. Frozen metadata bytes match retained references.
- Degradation cross-check: both exact run-bound configurations specify 129 FFT bins (`±64`), matching protocol v2; `config.json`, `manifest.json`, and `files.csv` hashes match the retained `input_binding.json` reference, and all six child runs use those fixed manifest hashes. The CLI default remains 64 for legacy compatibility, so future v2 preparation must pass 129 explicitly. The separate 2026-10-21 prep archive has matching config/file-list hashes but different manifest bytes and is not the run-bound reference. See the [cross-check audit](../../experiments/data_audits/sen12_v2_degradation_contract_crosscheck_20261024_v1/report.md).
- Source: 26 dependency hashes in the run root. The previous root's four source mismatches are documented in `source_alignment_audit.json`; its four sealed results remain historical and will not be mixed into this aggregate.
- Runs: all six completed and sealed (25 SwinIRJoint epochs and 10 C2 epochs per seed), with 200 canonical validation rows each. SAR-only best epochs are 14/15/19; C2 best epochs are 1/1/1. Checkpoints, per-image results, logs, configs, commands, calibrations, and manifests are retained per run.
- Gate result: mean paired ΔPSNR `+0.001617 dB` (95% source-group bootstrap CI `[+0.000075, +0.003180]`), below the `+0.05 dB` threshold; all three seed ΔPSNR values are positive. Mean ΔSSIM is `−0.000892`, below zero. The other declared guardrails pass. Gate reasons: `mean_delta_psnr_db`, `mean_delta_ssim`; next phase is `stop_at_b1`.
- Read-only diagnostic: independent recomputation from the 600 canonical paired rows matches the retained per-group summary within `1e-12`. All 45 seed×source-group mean ΔSSIM values are negative; positive source-group mean ΔPSNR counts are 7/15, 9/15, and 12/15 by seed. This is descriptive proxy-target evidence, not a causal or clean-SAR analysis, and does not alter promotion or subgroup selection.
- Loss/selection diagnostic: the three C2 configs assign nonzero weight only to Charbonnier reconstruction (`1.0`) and high-frequency residual losses (train-only calibrated weights `0.739918`, `0.808352`, `0.771583` for seeds 0/1/2); each value matches its calibration record. Direct SSIM, edge, and guard losses are zero and the backbone is frozen for all 10 epochs. Recomputing the logged strict selection score from its source formula matches all epochs, with epoch 1 selected for every seed. By epoch 10, validation loss is lower but SSIM and selection score are lower and residual-contribution RMS is higher in all seeds. This suggests a possible objective/metric tension; it is not a causal finding and does not justify a same-cohort loss retune.
- Scope: 1,600 train/200 validation identities, 15 documented source folders per split. Validation is fall-only within one ROI-season; registration remains nominal source matching with fine pixel alignment unverified. Targets are normalized PNG-derived noisy pseudo-targets. This is prospective development evidence, not a clean-reference, radiometric, geographic/acquisition-generalization, or test result.
- Cleanup and source binding: all 26 captured hashes matched the worktree when the six-run matrix was sealed. Subsequent D-2026-10-23-46 maintenance changed exactly two current files (`MMCANetV2/model.py`, `scripts/train_mmcanet_v2.py`); original run hashes and manifests remain unchanged. Six run manifests, six validation manifests, canonical per-image/checkpoint hashes, and five aggregate output hashes were checked at sealing. Removed replayable per-epoch/duplicate run artifacts and 16,218 temporary input-workspace files (2,795,206,153 bytes). The [pruning audit](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/artifact_pruning_audit.json) records exact run artifacts, retained records, and deleted payload totals; the [post-run source/objective audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md) records the later code state.
- Boundary: `test_set_used=false`, `test_set_accessed=false`, `candidate_test_enumerated=false` for all six runs and aggregate. No test imagery or measurements were accessed. B2/B3 were not run. See the run root's `README.md`, `run_manifest.json`, `aggregate/report.md`, `environment.json`, `input_binding.json`, and `commands.md`.

## SEN1-2 pair timing and registration source audit (2026-10-23; no model experiment)

- **Question:** Do official SEN1-2 sources establish acquisition-time matching or fine SAR–optical registration for the 1,800-pair fall cohort?
- **Primary-source evidence:** Schmitt et al.'s full 2018 author manuscript defines the fall window as 1 September–30 November 2017, filters Sentinel-1 IW/VV and Sentinel-2 granules within each season, and mosaics each collection per ROI. The method names `ImageCollection.mosaic()` with last-on-top precedence; it does not give paired product IDs, a date-matching rule, maximum temporal distance, or residual-error measurements. The retained official TUM supporting document defines source-scene folder structure and 8-bit representation but not product timestamps.
- **Local cohort evidence:** All 1,800 selected panels match official TUM source pixels, with 15/15 source-folder separation and passing exact/declared ±128-grid checks. These checks authenticate source pixels and frozen folder grouping; they do not verify simultaneous acquisitions, pixel-level mosaic provenance, or arbitrary residual registration.
- **Interpretation:** The cohort is same-season mosaic pairing under nominal georegistration. Per-pixel mixing across collection members is possible from the documented mosaic implementation, but the actual temporal mixture is unknown. Xu et al.'s spring/winter reprocessed subset does not quantify the fall cohort's residual error.
- **Boundary:** The TUM record returned HTTP 429 on the 2026-10-23 recheck; no further file inventory was obtained. No images, test artifacts, model outputs, or metrics were accessed. No protocol, result, or gate changed. See the [audit report](../../experiments/literature_audits/sen12_pair_timing_registration_20261023_v1/report.md) and D-2026-10-23-38.


## SEN1-2 third-party SIFT-label crosswalk (2026-10-23; no model experiment)

- **Question:** Can public SIFT-derived point annotations provide an independent residual-registration estimate for the exact prospective fall validation images?
- **Source and input:** Streamed only the 600,000-row `finally_train.txt` from the [author repository](https://github.com/YuChuang1205/SEN1-2-patch-dataset), pinned by SHA-256 `d08a21b65cae6fea9c83934dd968238cba48b618004f84f3ff1a4e25110d95a6`. Exact pair names were matched only against the authenticated 200-row validation allowlist (SHA-256 `586b5563cf0b7ae74f70e8ec883a9855c78c54d1e0c6499d5ec23284cc9b5a49`). No raw list was retained.
- **Observation:** 434 list rows matched (219 label 1, 215 label 0) across 124/200 validation pairs and all 15 validation source folders. All 219 distinct positive coordinate pairs have `(dx,dy)=(0,0)`; negative-label median displacement norm is 95.34 pixels. No malformed matched records or duplicate coordinate pairs were found.
- **Interpretation:** The zero-offset positives and large negative offsets are consistent with coordinate-based positive/negative sample construction. The README does not document independent surveyed control points or the annotation-generation rule, so this cannot validate fine registration. It does not address product timestamps or mosaic provenance.
- **Verification and boundary:** One offline regression test passed; Python compilation and the pinned end-to-end stream passed. `test_list_fetched=false`, `test_set_accessed=false`, `test_set_used=false`, `images_accessed=false`. No model, metric, gate, or protocol changed. See the [audit directory](../../experiments/data_audits/sen12_sift_tiepoint_probe_20261023_v1/).
- **Code-route follow-up:** The dataset README's code link resolves to the author's RRL-Net training/testing repository; its README describes prepared dataset archives and does not document point-label generation. The SEN1-2 dataset repository's visible file listing has the README and precomputed train/test lists, not a generator script. This bounded check found no independent-survey provenance; it does not prove that no separate source exists. No test-list contents, images, or measurements were accessed.


## SEN12TP prospective source candidate audit (2026-10-23; no data/model experiment)

- **Question:** Does a separate public SAR–optical source offer traceable acquisition and spatial grouping metadata for a future development protocol?
- **Sources:** Roßberg & Schmitt's version-of-record article, the Zenodo v1 landing page, and the `oceanites/sen12tp` README/dataloader source. See the [audit report](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).
- **Facts:** The paper/record report 2,319 pairs over 1,236 areas, scene timestamps/geometry/ROI IDs, a maximum 12-hour temporal gap (typically about six hours), and spatial splitting. The paper gives 80/10/10 train/validation/test, while the Zenodo record describes ~90/10 train/test; exact archive mapping was not verified. The dataloader uses user-supplied ROI directories and does not enforce grouping of repeated dates by `roi_id`.
- **Assessment:** Useful provenance/timing candidate; not a clean-target SAR restoration benchmark. The 10 m grid has stated ~20 × 22 m S1 effective resolution. The visible Zenodo rights/license field is blank; official component-source terms suggest an attributed reuse route but do not establish rights for the compiled archive or curated metadata. Its prose says four train archives but the file listing shows five, and the uncompressed dataset is ~222 GB.
- **Boundary:** A search result exposed article-level aggregate test-metric text, which was not used. No dataset test identity, image, or raw measurement file, and no local project test data/metrics were accessed. No model, protocol, or gate changed. See the [access manifest](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/access_manifest.json).
- **Decision:** D-2026-10-23-41. Keep conditional; no download or training until rights, archive/split mapping, and a controlled pseudo-target protocol are resolved.

### Rights metadata route follow-up (2026-10-23)

- Official source-component terms were reviewed: Sentinel data are open for reuse with attribution; ESA WorldCover is CC BY 4.0; JAXA AW3D30 is subject to JAXA terms; the `s2cloudless` repository's CC BY-SA 4.0 declaration applies to software. These terms do not establish rights for the complete SEN12TP archive or author-curated metadata.
- The official Zenodo metadata API/export route was inaccessible, and its JSON preview returned “Cannot preview file.” The metadata file and dataset archives were not downloaded. Exact release split membership therefore remains unverified.
- No test identities, imagery, or measurement files were accessed; no local project test data or metrics were accessed. No protocol, model, result, or gate changed. See the [audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/access_manifest.json).

### Train-only geospatial route follow-up (2026-10-25; static source review)

- **Observation:** The mutable author `dataset.py` returns each training raster window's bounding box, CRS, and affine transform. Its default 256-pixel windows use stride 249, so adjacent patches overlap by 7 pixels. Directory enumeration is per ROI-scene folder and does not group multiple acquisition dates by parent `roi_id`.
- **Implication:** If package rights and archive membership are resolved, a spatial-overlap check can use only authorized training TIFF georeferencing and avoid the all-scenes metadata JSON. A prospective split must group all dated scenes by parent `roi_id` before patch extraction; scene-directory splits are insufficient.
- **Boundary:** The source branch is mutable and unpinned; it was read statically, not executed. No archive, metadata JSON, TIFF, or external test identity was accessed, and no local project test asset, model, protocol, or gate changed. See D-2026-10-25-70 and the [updated audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).

### Research-record search output disclosure (2026-10-25; no experiment)

- **Event:** A shell double-quoted `rg` pattern contained backticks, triggering command substitution and broadening the search over seven research documents; the tool output was truncated at 9,181 tokens.
- **Boundary:** Treat possible held-out-related text as exposed: `test_set_accessed=true`, `test_set_used=false`. No image, metric payload, prediction, array, or checkpoint was opened and no value was used. No run, result, protocol, or gate changed. See D-2026-10-25-71 and the [access audit](../../experiments/data_audits/sen12tp_doc_search_exposure_20261025_v1/report.md).


## B1 v2 C2 objective/code audit (2026-10-23; read-only)

- **Question:** What implementation facts can explain the observed validation-loss decrease alongside lower SSIM and frozen selection score?
- **Inputs:** Candidate configs and epoch 1/10 training-log rows for seeds 0–2; run/aggregate manifests and summary; bound source hashes; current model/loss/trainer source; D-2026-10-23-46 repair record.
- **Observation:** Only Charbonnier reconstruction (`1.0`) and train-calibrated high-frequency residual loss (`0.739918`, `0.808352`, `0.771583`) are active. The SAR backbone is frozen for all ten epochs. Validation loss decreases in all seeds, while SSIM and selection score decrease and residual-contribution RMS rises. All seeds choose epoch 1. The unchanged gate remains `not_promoted`.
- **Interpretation:** The objective does not directly optimize SSIM, edge, or gate guardrails, so objective/selection tension is plausible. The sigmoid gate is a modulation value, not calibrated reliability. This is not a causal result and does not justify same-cohort retuning.
- **Source boundary:** All 26 source hashes matched when the run was sealed. Subsequent D-2026-10-23-46 maintenance changed current `MMCANetV2/model.py` and `scripts/train_mmcanet_v2.py`; the run-bound `MMCANetV2/losses.py` hash still matches. Historical manifests remain unchanged. See the [audit report](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).
- **Access boundary:** Validation logs/configuration and source/aggregate records only; no test imagery, prediction, identity, metric, or measurement was read. No training, inference, selection, gate, or protocol change occurred. See the [access manifest](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/access_manifest.json).


## Configured upstream source-root availability probe (2026-10-23; path metadata only)

- **Question:** Is the original data root hard-coded in the neighboring SOTANet preparation helper available in this execution environment?
- **Observation:** The helper's /media/shy/.../SEN1-2/data path is absent. The checked media directory is empty; inspected /mnt locations resolve to the root filesystem, and the bounded home-directory path scan found no alternate SEN1-2 source directory. The known SOTANet QXS-named cache is not an upstream archive or source manifest.
- **Interpretation:** This closes only the current local path route. It does not resolve numeric 2k source lineage or rule out unmounted/external copies.
- **Boundary and reproduction:** Only helper source, path existence, mount targets, and directory names were inspected. No data payload, split, test identity, metric, or model output was accessed. See the [report](../../experiments/data_audits/dev2k_configured_source_root_probe_20261023_v1/report.md), [commands](../../experiments/data_audits/dev2k_configured_source_root_probe_20261023_v1/command.txt), and [access manifest](../../experiments/data_audits/dev2k_configured_source_root_probe_20261023_v1/access_manifest.json).

## Local dev2k PNG embedded-metadata audit (2026-10-23; train/validation metadata only)

- **Question:** Do the local dev2k SAR/optical PNGs preserve embedded source, acquisition, calibration, or preprocessing metadata?
- **Inputs and method:** Read only `splits/dev2k_train.txt` and `splits/dev2k_val.txt`, then their 1,800 named SAR and optical files. The scanner checked signatures and chunk headers and skipped all chunk payloads, including IDAT.
- **Observation:** All 3,600 PNGs were structurally readable and contained only `IHDR`, `IDAT`, and `IEND`. No ancillary/application metadata chunks were present.
- **Interpretation:** This closes the embedded-container metadata route for the local dev2k train/validation files. It does not identify upstream archive, parent scenes, acquisition timestamps, calibration, or target-generation lineage.
- **Boundary and reproduction:** Test split, identities, files, and measurements were not accessed or enumerated; no pixels were decoded. Reproduce with [`audit_png_metadata.py`](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/audit_png_metadata.py); see its [report](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/report.md), [summary](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/summary.json), [command](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/command.txt), and [access manifest](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/access_manifest.json).


## B1 v2 per-image direction and heterogeneity audit (2026-10-23; read-only)

- **Question:** Are the small positive mean PSNR deltas spread across validation identities, and do the registered structure/target-region guardrails move in the same direction?
- **Input and validation:** The sealed 600-row paired validation CSV matched its aggregate-manifest SHA-256; the run and aggregate manifests record false candidate-test enumeration/access/use flags. The audit checked 200 identical identities across three seeds and 15 source groups. Only validation metric CSV and manifests were read.
- **Observation:** Three-seed image-mean PSNR was positive for 129/200 identities; 106 were positive in every seed and 45 negative in every seed. The top 40 identities contributed 61.4% of the positive PSNR-delta sum. SSIM was worse for 192/200 image means. Edge-strength ratio error increased for all 200 identities in each seed. TBR mean-ratio error and bright-peak ratio error were worse for most identities.
- **Interpretation:** The small PSNR gain is not confined to a handful of identities, but its positive mass is concentrated and it coexists with broad structure-proxy regression. This is a descriptive proxy-target observation, not clean-SAR utility evidence or causal diagnosis. The existing aggregate report remains authoritative for means, medians, sample SDs, and source-group bootstrap intervals.
- **Boundary:** No images, arrays, predictions, test measurements, model inference, training, checkpoint/subgroup selection, or gate changes. The B1 v2 gate remains not_promoted; B2/B3 remain gated.
- **Reproduction:** [analysis.py](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/analyze.py), [report.md](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/report.md), and the audit manifest/command/environment in that directory.

## B1 v2 source-group bootstrap rederivation (2026-10-23; read-only)

- **Question:** Do the retained 95% source-group bootstrap summaries reproduce independently from the canonical paired validation rows?
- **Input:** The aggregate manifest, summary, and hash-bound `paired_per_image.csv` (600 rows; 200 validation identities × three fixed training seeds). Aggregate and summary test-access/use/enumeration flags are explicitly false.
- **Method:** Independent implementation of 10,000 paired resamples of the 15 manifest-ordered source groups using seed `20261021`; group resamples are shared across the three fixed training seeds, and training seeds are not resampled.
- **Result:** All eight means, intervals, and positive-draw probabilities match the retained summary within `2.71e-20`. In particular, ΔPSNR is `+0.001617 dB` with interval `[+0.000075, +0.003180]`; ΔSSIM is `−0.000892` with interval `[−0.001113, −0.000675]`. The gate remains `not_promoted` and does not consume these intervals.
- **Boundary:** Validation aggregate metrics and split/group metadata only; no image/tensor payload, training log, checkpoint, or test measurement was read. This estimates uncertainty over the 15 groups conditional on the three observed seeds and one fall ROI-season; it is not broader geographic, seasonal, or seed uncertainty.
- **Reproduction:** [`rederive.py`](../../experiments/diagnostics/b1_v2_source_group_bootstrap_rederivation_20261023_v1/rederive.py), [report](../../experiments/diagnostics/b1_v2_source_group_bootstrap_rederivation_20261023_v1/report.md), and the command, environment, JSON result, and access manifest in the audit directory.

## README and runtime follow-up (2026-10-23; no model experiment)

- **Change:** README Key findings now links the independent bootstrap rederivation and states its fixed-seed/single-ROI-season scope. The Limitations section reports the actual Python/PyTorch/CUDA status.
- **Check:** A local-link sweep resolved 101 README destinations. The system Python is 3.13.5 without PyTorch; `AISAR` is Python 3.13.7 with PyTorch 2.8.0+cu128, CUDA unavailable, and zero devices. `nvidia-smi -L` cannot communicate with the driver.
- **Boundary:** No dataset, metric, checkpoint, or model output was accessed. No training or inference was run. The frozen B1 gate remains `not_promoted`; B2/B3 stay gated.

## SAR400 candidate source screen (2026-10-23; no data/model experiment)

- **Question:** Does SAR400 establish a sufficiently documented real-SAR denoising reference or a task-matched optical-guided JDSR benchmark?
- **Sources:** Springer chapter metadata/abstract, the author-maintained dataset page, Tampere University publication record, and the author-uploaded presentation record.
- **Observation:** The abstract describes multiplicative, spatially correlated SAR noise with asymmetric spectrum and proposes a test set whose noise is said to resemble real SAR. The author page lists `gt` and `noi` variables plus four 100-item Rayleigh/non-Rayleigh spectral groups. The public page lists a 143 MB Matlab archive.
- **Interpretation:** The sources support a conditional denoising-candidate classification. They do not establish `gt` source/physical semantics, noise-generation parameters, sample grouping/split, or dataset-specific rights. No optical pairing or SR protocol is described.
- **Boundary:** No archive or image payload, local data, test metric, checkpoint, or model output was accessed. No experiment or gate changed.
- **Disposition:** Keep unadopted. Reconsider only after source, generator, data-domain, and rights questions are resolved; then limit any first use to an independently specified SAR-only denoising stress test. See the [audit](../../experiments/literature_audits/sar400_candidate_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sar400_candidate_20261023_v1/access_manifest.json).

## SOMA SAR–optical registration literature/code audit (2026-10-23; no local experiment)

- **Question:** Does SOMA establish prior art or evidence for misregistration-safe SAR reconstruction?
- **Sources:** AAAI record; full arXiv v1; public GitHub repository README, MIT code license, `train.py`, `test.py`, and `datasets/dataloader.py` on unpinned `main`.
- **Observation:** The paper is a dense SAR–optical registration method with FGE, coarse DINOv2, affine-flow matching, and known-warp certainty supervision. It reports registration CMR/flow errors under artificial geometry and cross-dataset matching. The current public loader uses random ±20°/±50 px/scale-1 transformations in all splits, including validation and test, whereas the manuscript's nominal regime is up to 32 px, ±5°, and scale variation 0.2. The reviewed scripts do not seed NumPy. Table 1 and Table 3 report 86.67% and 87.53% full-model SEN1-2 CMR@1px, respectively, without a reconciliation in the reviewed text.
- **Interpretation:** Registration is established prior art; it does not demonstrate SAR-output restoration, optical utility under mismatch, or safe fallback. Public code-to-paper reproducibility is unresolved for the inspected unpinned branch.
- **Boundary:** No code execution, dataset payload, split identity, local test asset, or model output. No training, inference, or gate change.
- **Disposition:** Narrow novelty claims; do not launch a local experiment. See the [audit](../../experiments/literature_audits/soma_sar_optical_registration_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/soma_sar_optical_registration_20261023_v1/access_manifest.json).


## DSRKD full-text and implementation route recheck (2026-10-23; no model experiment)

- **Question:** Did current official or author-controlled routes expose primary method details or code to resolve DSRKD's dataset/training-protocol discrepancy?
- **Sources:** Direct IEEE Xplore article page, current lead-author homepage and GitHub profile/search, DBLP record and its Unpaywall/OpenAlex pointers.
- **Observation:** IEEE returned JavaScript verification. The author homepage describes only an encoder–upscaling–decoder/multi-scale architecture and links to IEEE; no official code or full manuscript surfaced. The open-access metadata pointers were inaccessible.
- **Disposition:** Keep DSRKD as direct joint-restoration prior art; do not settle the conflict between abstract-level synthetic/SAR wording and HSFMamba's natural-image description. See the [report](../../experiments/literature_audits/dsrkd_access_route_recheck_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/dsrkd_access_route_recheck_20261023_v1/access_manifest.json).
- **Boundary:** No local data/test asset, metric, checkpoint, model, training, inference, or gate was accessed or changed.

## ISSMSAR publisher-indexed primary-text audit (2026-10-23; no model experiment)

- **Question:** What does ISSMSAR establish as joint SAR-restoration prior art, and does its protocol match the local single-SAR-plus-optical question?
- **Evidence:** Publisher-indexed article text describes two multi-temporal SAR inputs, ratio-L1 reconstruction, and 8,800 grayscale pseudo-SAR crops derived from 5 m GaoFen-1 optical imagery; it reports simulated groups and 3 m HH ALOS-2/COSMO-SkyMed examples. The direct article page returned HTTP 429 and XML/PDF were unavailable through the web reader.
- **Interpretation:** This is direct SAR-only integrated despeckling/SR prior art. Its synthetic targets are optical-derived, and the exposed text does not establish split/group integrity, exact degradation/domain, registration/timing, or a real-scene reference protocol. It is not an optical-guided method or matched local baseline. Table and prose results are author-reported and not independently reproduced.
- **Boundary:** No local dataset, test asset, metric, checkpoint, model output, training, inference, selection, or local manuscript result was accessed or changed.
- **Disposition:** Preserve the broad joint-task novelty boundary; keep the local B1 gate unchanged. See the [report](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md), [access manifest](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/access_manifest.json), and D-2026-10-23-57.

## Repository-wide project fact inventory (2026-10-23; no experiment)

- **Question:** Are the current research facts, bounded observations, hypotheses, unknowns, and unsupported claims accessible from one reviewable entry point?
- **Evidence:** Reviewed the current instructions and README, current state, claim–evidence map, current-source B1 v2 protocol/report/manifests, MMCANetV2 model and data interfaces, data-overlap/source audits, and literature matrix. The checkout remains dirty with many untracked files; the source run's hashes and manifests remain its reproduction boundary.
- **Outcome:** Added PROJECT_FACTS.md with A–E evidence categories and source links; linked it from README and the claim–evidence map. CURRENT_STATE remains the sole current-status authority.
- **Boundary:** No test image/array, held-out metric, checkpoint payload, or result-bearing manuscript section was read. No model run, gate, or historical record changed.
- **Disposition:** The inventory closes a documentation/indexing gap, not a scientific hypothesis or experiment. Continue the open P0 provenance/group-mapping work and keep B2/B3 gated.

## SEN12MS-CR-TS maintainer download-route recheck (2026-10-23; no experiment)

- **Question:** Does the author-controlled download tooling expose train-only metadata without archive retrieval, and does it clarify dataset-specific rights?
- **Evidence:** The public author-group helper routes region/modality training archives and separately named test archives. The reviewed script sections expose no standalone train-only ROI/date crosswalk. The official article copy's CC BY 4.0 notice licenses the paper and its data section points to the author website, but no distinct dataset license is stated. The author's repository root listing has no `LICENSE` file.
- **Outcome:** Recorded the route audit; the candidate remains conditional and unadopted pending dataset rights and an authorized train-only metadata route.
- **Boundary:** No archive, raster, image, metadata payload, holdout ROI identity, or local project test asset/metric was accessed. No training, inference, or gate change occurred.
- **Disposition:** Keep all 13 holdout ROIs unopened. See the [report](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/access_manifest.json).

## B1 v2 C2 failure-cause synthesis (2026-10-23; read-only)

- **Question:** Which causal explanation for the frozen C2 B1 gate failure is established by the sealed evidence?
- **Evidence:** The gate fails only the registered mean ΔPSNR and ΔSSIM thresholds; PSNR improves on 129/200 image means, SSIM worsens on 192/200, and edge-strength ratio error worsens on every identity in each seed. The training objective/selection trend makes mismatch plausible but not causal; target/degradation, nominal-only registration, and frozen-backbone capacity remain unresolved.
- **Outcome:** No single root cause is identified. Retain `not_promoted` and stop at B1; no same-cohort retune or B2/B3.
- **Boundary:** Synthesized existing reports/protocol and aggregate-manifest flags only. No raw per-image CSV, image, tensor, prediction, checkpoint payload, or test measurement was read; no training, inference, metric calculation, or gate change occurred.
- **Disposition:** Resolve source/target provenance and a valid prospective split before further model selection. See the [synthesis](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/report.md) and [access manifest](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/access_manifest.json).


## SDS-SAR literature/source audit (2026-10-23; no experiment)

The publisher-indexed abstract and official author repository were reviewed. SDS-SAR is self-supervised SAR-intensity despeckling prior art, but the visible public test/model path uses a 3×3 median-filter placeholder and does not load the supplied checkpoint. No data or weights were acquired, and no local run or metric was produced. Do not treat the placeholder as a reproduction or JDSR baseline. See the [audit report](../../experiments/literature_audits/sds_sar_primary_source_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sds_sar_primary_source_20261023_v1/access_manifest.json).


## Lattari et al. SAR despeckling literature audit (2026-10-23; no experiment)

The publisher-indexed 2019 article sections and author-institution record support a bounded account of a modified U-Net, synthetic multiplicative-speckle pretraining, and fine-tuning using simulated noise over Sentinel-1 temporal-average pseudo-references. The real-domain evaluation has no clean reference and uses ENL/visual analysis. No paper-specific code or data archive was verified, and no local run occurred. This is SAR-only denoising prior art, not a joint-SR or optical-guidance result. See the [audit](../../experiments/literature_audits/lattari_sar_despeckling_2019_primary_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/lattari_sar_despeckling_2019_primary_20261023_v1/access_manifest.json).

## SAR2SAR primary full-text and implementation route (2026-10-23; no experiment)

- **Question:** What does SAR2SAR establish, and is its paper-linked code currently reproducible as a local baseline?
- **Evidence:** Full accepted arXiv v3; author-group code listing; paper-linked GitLab page; author GitHub migration notice; related author-group `deepdespeckling` README, inference source, model source, constants, and license.
- **Outcome:** Record three-stage multitemporal SAR-only despeckling prior art. The related package provides single-image inference and a checkpoint listing, but no verified training-run/checkpoint lineage. Real paper evaluation has no clean reference and uses ENL/visual inspection, with the paper's own ENL over-smoothing caveat.
- **Boundary:** No code clone/execution, dataset or checkpoint payload download, local data/test access, or experiment. B1 and the B2/B3 stop are unchanged.
- **Disposition:** Do not adopt as a matched SAR+optical JDSR baseline; source groups, acquisitions, rights, and training-code route remain unresolved. See the [audit](../../experiments/literature_audits/sar2sar_primary_fulltext_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sar2sar_primary_fulltext_20261023_v1/access_manifest.json).


## Artifact cleanup continuation (2026-10-24; no experiment)

Removed 12 per-epoch CSV/log files from explicitly recorded failed/interrupted runs and three stale rebuildable LaTeX build logs (15 files, 118,442 bytes total), plus 178 empty directories in writable legacy output trees. The associated configs, commands, manifests, failure/interruption audits, current registered results, selected checkpoints, canonical metric records, paper sources, and PDFs remain. No data or metric payload was read; no run or gate changed. See the [cleanup report](ARTIFACT_CLEANUP_20261024.md) and [deletion record](../../experiments/artifact_cleanup_continuation_20261024_v1.json).

## Corley et al. registration benchmark audit (2026-10-24; literature only)

- **Question:** What protocol evidence from the zero-shot SAR–optical registration benchmark can inform a later mismatch study, and is its public reproduction entry bound to the paper's stated sweep?
- **Evidence:** Full arXiv v4, official CVF paper record, official repository README, and current public sweep script. The paper's SpaceNet9 results describe three labeled training scenes, and its §5.2 64-run sweep specifies 1024/1536 max-side values, two matchers, and two normalizations. The current README reproduction command invokes a mutable-main script defaulting to 1024/2048, four matchers, and percentile normalization.
- **Outcome:** Record as registration-only prior art. Protocol sensitivity motivates predeclaring geometry, scale, tiling, inlier gating, and registration metrics in a future authorized study; it does not measure SAR restoration or prove optical fallback. Exact code-to-paper run binding remains unresolved.
- **Boundary:** No code clone or execution, benchmark data, local project data/test asset, metric, checkpoint, training, inference, or gate was accessed or changed. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/corley_pretrained_matcher_fulltext_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/corley_pretrained_matcher_fulltext_20261024_v1/access_manifest.json).

## GCOANet-related patent route (2026-10-24; secondary source only)

- **Question:** Does a new source disclose GCOANet's inference inputs or a verifiable dataset lineage?
- **Evidence:** Secondary patent indexes surfaced CN120450996A and render a SAR-plus-optical conditional-diffusion reference generator, prediction-stage use, and a `QXSLAB_SAROPT` 20,000-pair 9:1:1 embodiment. The inventors overlap some GCOANet author names. The item-level official CNIPA record and exact patent-to-paper binding remain unavailable.
- **Outcome:** Treat as a diligence lead only. Do not change the GCOANet paper's abstract-level status or local dataset provenance; the local QXS-named cache has a different recorded 16k/2k/2k split.
- **Boundary:** No patent PDF, paper full text, code, data payload, local test asset/metric, model, training, inference, or gate was accessed. See the [route audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/access_manifest.json).

## OGSFlow public repository screen (2026-10-24; no experiment)

- **Question:** Does the newly surfaced OGSFlow repository establish direct method prior art or an adoptable OGSRL/OGSSR data route?
- **Evidence:** Current public GitHub landing page and README; exact-title/repository-scoped searches; attempts to open linked dataset, evaluation, training, and named implementation files. The README maps S1/S2/RGF to model files and advertises a CPU synthetic-data smoke command, but source contents remain inaccessible and commands were not run. It describes an optical-guided two-stage UAV-SAR SR system with RGF and optical-conditioned flow matching, but says the paper citation is forthcoming. Linked documents returned cache misses; no archive was inspected.
- **Outcome:** Keep as a mutable README-level prior-art lead that narrows generic RGF and flow-based SAR-SR novelty. The README presents SR rather than a documented joint despeckling/SR task. OGSRL and OGSSR remain distinct; the OGSRL package, split, and its README-stated license were not independently verified from the archive.
- **Boundary:** No code clone/execution, image/sample/test payload, external metric, local data/test asset, model, training, inference, or gate was accessed or changed. See the [audit report](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/access_manifest.json).

## Onat quality-aware SAR despeckling source-route update (2026-10-24; literature only)

- **Question:** Does the publisher-linked PLFSID repository provide the reported complete implementation and effective data/code terms?
- **Evidence:** Publisher abstract and record metadata; author repository landing page and README; visible filenames and filter bibliography; direct `main.m` and raw-file route attempts. The paper was accepted 2026-06-09 and published online 2026-06-14. README says the visible MATLAB entry is a demo, algorithm/fusion scripts are withheld, and an academic-use license is planned. The official issue TOC search listing showed Full Text/PDF links, but direct issue/PDF routes were inaccessible; the full publisher article returned 403.
- **Outcome:** Keep as adjacent single-SAR despeckling output-fusion prior art. The abstract does not establish optical input at inference, joint SAR SR, mismatch robustness, or SAR-only fallback. The repo does not yet support a reproduction or effective-rights claim.
- **Boundary:** No repository code execution, raw data/archive, paper result table, local project data/test asset, metric, model, or experiment was accessed. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/access_manifest.json).

## HSFMamba full-text audit (2026-10-24; literature only)

- **Question:** What does the full paper establish about inference-time optical guidance for joint SAR SR/denoising, and what remains open about optical quality and geometric mismatch?
- **Evidence:** IEEE publisher metadata/abstract, rendered ResearchGate full-text copy (journal pp. 16445–16461), and the current author-linked GitHub repository root/README.
- **Outcome:** Confirm direct joint-task prior art and record its CFSM/FSAM/RSSL architecture, OGSSR data/preprocessing, BM3D-reference plus bicubic/Gaussian/Gamma protocol, and author-reported ablations. The manuscript's haze example is qualitative; adaptive optical-quality perception is future work. It does not supply a Gamma equation/domain, an independently reviewable scene crosswalk, quantitative residual-registration test, or documented SAR-only fallback. The implementation and external archive/rights remain unverified.
- **Boundary:** No OGSSR or local data payload, local test asset/measurement, code bundle, or checkpoint was accessed or executed. No local training, inference, metric, or gate changed. See the [full-text report](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/access_manifest.json).

## WMoE-Net / QXS-SAROPT-SHIP source and publisher-abstract screen (2026-10-24; no experiment)

- **Question:** Does the newly surfaced Zenodo QXS-SAROPT-SHIP record establish an authorized QXS version or source-group mapping, and what method evidence does WMoE-Net provide?
- **Evidence:** Current WMoE-Net repository root/README, linked Zenodo record, original QXS-SAROPT author README, and the official ScienceDirect article record/abstract. The publisher identifies DOI `10.1016/j.jag.2026.105624`, volume 154/article 105624, and an issue date of November 2026; the direct page returned 403. Deeper configuration/code routes returned cache misses.
- **Outcome:** Correct the earlier no-paper-found result. The abstract describes oriented ship detection on MSOSD-1.0 and QXS-SAROPT-SHIP and reports author-claimed gains under low illumination/clouds/fog; no full-text protocol, numeric metrics, source-level implementation, or controlled geometric-shift result was reviewed. This is adjacent detection prior art, not SAR restoration. The Zenodo record has no visible dataset license, source crosswalk or split manifest, and the original QXS README does not link it. Local numeric-2k lineage/source groups remain unresolved.
- **Boundary:** Publisher-indexed abstract/metadata only; no full text, dataset/archive, Baidu payload, source code, sample image/label, local cache, local test asset, or metric was accessed; no code was executed. B1 remains `not_promoted`; B2/B3 remain gated. See the [updated screen report](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/access_manifest.json).

## Amieva et al. indexed published-version text update (2026-10-24; literature only)

- **Question:** What protocol and result details are visible in the institutional published-version PDF beyond the earlier abstract/opening excerpts?
- **Evidence:** Official institutional item and search-indexed PDF excerpts from sections 3.3–3.6 and 5 and Tables 2–8; official ScienceDirect metadata/abstract. Direct institutional bitstream opening timed out, publisher page access returned 403, and a bounded exact-title/DOI/GitHub search surfaced no official code repository.
- **Outcome:** Record IW→SM joint SAR SR/despeckling with a 5/2/3 image split, three full-image test scenes, 1–12 day paired acquisition gaps, a Lee-filtered SM reference, and pre-split full-image-set percentile normalization. The selected model is reported as end-to-end ESRT with SSIM loss and histogram matching. Preserve the resulting preprocessing-level held-out distribution caveat and do not treat the reported values as local evidence or a matched baseline.
- **Boundary:** Indexed paper text and author-reported test metrics only; no external image/data payload, source code, weights, local project data/test assets, or local experiment were accessed. B1 remains not_promoted; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/amieva_indexed_fulltext_update_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/amieva_indexed_fulltext_update_20261024_v1/access_manifest.json).

## Figshare QXS-derived semantic-mask package source screen (2026-10-24; metadata only)

- **Question:** Does the Figshare package establish an authorized source/version route or parent-scene crosswalk for the local numeric 2k subset or the neighboring QXS-named cache?
- **Evidence:** Current Figshare record page (version 2, posted 2026-10-16) and NLM Dataset Catalog search-indexed entry. Figshare article/version API endpoints and the direct Kaggle route were not accessible through the review reader.
- **Outcome:** The Figshare page names QXS-SAROPT as upstream and describes SAR/optical/mask directories, but exposes no itemized archive inventory or source-group map and says filename-stem matches do not establish co-registration. Its record-level CC BY 4.0 notice explicitly leaves upstream imagery governed by upstream terms. The NLM index still identifies version `.v1`, unlike the current page's v2; the catalog page was not directly accessible. Treat this as a weak third-party lineage lead, not proof of exact identity, split semantics, source groups, or reuse rights.
- **Route follow-up:** Targeted Kaggle/GitHub searches found no author-controlled processing notebook or source repository, and the linked Figshare author profile returned 403. The notebook's segmentation settings are therefore page-described only.
- **Boundary:** Public record/index metadata only; no archive, local data payload, local test asset, metrics, or model outputs were accessed. No download, experiment, selection, or gate change occurred; test flags are false. See the [audit report](../../experiments/data_audits/figshare_qxs_semantic_repackage_source_screen_20261024_v1/report.md) and [access manifest](../../experiments/data_audits/figshare_qxs_semantic_repackage_source_screen_20261024_v1/access_manifest.json).

## Fixed-test metric text exposure during cleanup-record search (2026-10-24; disclosure)

- **Event:** A recursive `rg` search intended to find cleanup totals included `experiments/` and its output surfaced stored `fixed_test_dev2k` metric CSV rows, including numeric measurements. The tool output was truncated.
- **Use:** No exposed values were copied, analyzed, or used for selection, training, inference, evaluation, or paper claims. No test image/array was opened; no result or experiment manifest was changed.
- **Consequence:** Mark this later text-search event as test-metric access and do not use the affected measurements for future model selection/tuning. This does not change the historical run-time flags or gate for registered runs. See the [disclosure report](../../experiments/data_audits/legacy_test_metric_text_exposure_20261024_v1/report.md) and [access manifest](../../experiments/data_audits/legacy_test_metric_text_exposure_20261024_v1/access_manifest.json).


## B1 v2 C2 actual-residual diagnostic (2026-10-24; validation only)

- **Question:** Does the frozen C2 checkpoint's actual gated optical residual contain target-consistent correction, and can a single global scale or the learned gate safely realize it?
- **Protocol:** Zero training; EMA inference for seeds 0/1/2; same 200 fixed validation identities and input hashes as the sealed C2 run; tested `α ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2}` plus a target-assisted pixelwise oracle and gradient-region audits. Metric policy is `b1_mean_required_median_optional_v1`; median-TBR availability is explicit (198/200 per variant/seed). Test flags are false.
- **Outcome:** Overall actual residual: ΔPSNR `+0.001621 dB`, ΔSSIM `−0.000893`, ΔEPI `+0.001311`; frozen strict selection score falls from `17.427060` to `17.379980`. Best tested PSNR alpha is `1.2` (mean ΔPSNR `+0.001805 dB`), but `α=0` remains the best strict-score choice for every seed. The target-assisted oracle gives mean ΔPSNR `+0.027348 dB` and ΔSSIM `+0.004462`; it cannot be used at inference. Per-seed classifications are D/D/A, overall D (mixed/inconclusive). This does not promote C2 or change the B1 gate.
- **Input/code binding:** The recreated 200-row validation set matched the retained B1 v2 sample names, split hash, and all nine per-sample source/target/LR/optical/mask hashes. Checkpoints record high-pass kernel/sigma `5 / 1.0`; the post-run D-2026-10-23-46 interface repair preserves those defaults. See [input binding](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/input_binding.json), [report](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/report.md), and [manifest](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/manifest.json).
- **Boundary:** `fixed_val_dev2k` is repeatedly used development validation, targets are noisy representation-domain pseudo-targets, and the oracle is explicitly target-assisted. No training, model selection, test access, gate change, or B2/B3 work occurred. Intermediate arrays and regenerated fixed payloads were pruned after verification; config, commands, manifests, figures, and per-image/aggregate metrics remain.

## B1 v2 C2 gate-to-utility alignment diagnostic (2026-10-24; validation only)

- **Question and protocol:** Does the frozen model's input-derived gate_activation_mean rank target-derived per-image actual-minus-base utility? The predeclared primary statistic is seed-specific Spearman rho against per-image selection_score; the summary averages three fixed seeds and uses 10,000 source-folder cluster-bootstrap draws over 15 folders.
- **Inputs:** The previous residual diagnostic's per_image_residual_diagnostic.csv and per_image_variants.csv, source_manifest.json, manifest.json, and config.json. All 200 validation names and three seeds were joined; all rows and groups were complete, finite, and unique.
- **Metric definition:** Per-image strict_score is PSNR + 5·SSIM − 2·|ln(edge_strength_ratio)| − |ln(tbr_mean_ratio)|, as implemented by [OGSRN/metrics.py](../../OGSRN/metrics.py#L55). This per-image score is used only for the descriptive association; the aggregate B1 gate was not recomputed or overridden.
- **Outcome:** Per-seed primary rho was +0.8675, +0.8590, and +0.8383; mean rho was +0.8549; source-folder cluster-bootstrap 95% interval was [+0.7259,+0.9107] (10,000/10,000 valid draws). This meets the predeclared bounded_support rule only for ranking the composite score on these identities. Secondary mean seed-wise correlations were +0.0551 for ΔPSNR and −0.0604 for ΔSSIM; those directions are weak/inconsistent and descriptive only.
- **Interpretation and boundary:** The composite-score association does not establish which score component drives it, all-metric utility, calibrated reliability, causal optical value, clean-SAR recovery, or generalization. The same repeatedly used validation cohort selected the checkpoints; its 15 source folders are not verified parent-scene/acquisition groups. No threshold, subgroup, checkpoint, model, or gate was selected or changed.
- **Verification and artifacts:** Six Spearman/tie/constant/percentile sanity cases passed; all 11 recorded input/output hash and size checks matched. See the [protocol](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/protocol.md), [analysis](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/analyze.py), [per-seed results](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/per_seed_correlations.csv), [summary](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/summary.json), and [manifest](../../experiments/diagnostics/b1_v2_c2_gate_utility_alignment_20261024_v1/manifest.json). Test flags are false; no training, inference, test access, or gate change occurred.

## Numeric 2k/ReFlowSET train-list membership audit (2026-10-24; split metadata only)

- **Question and protocol:** Determine whether the numeric basenames in local `fixed_train_dev2k` and `fixed_val_dev2k` occur in ReFlowSET's official QXS-SAROPT training list. Join exact basenames only; do not interpret a match as content identity.
- **Inputs:** 1,600 local train names and 200 local validation names. Each fixed split file matches its corresponding `splits/dev2k_{train,val}.txt` byte-for-byte; train and validation are disjoint. Hashes and protocol IDs are in the [audit manifest](../../experiments/data_audits/reflowset_numeric2k_train_membership_20261024_v1/access_manifest.json).
- **Outcome:** 1,310/1,600 local train names and 157/200 local validation names occur in the official ReFlowSET train list, 1,467/1,800 total. Another 333 local development names are absent from that list and remain unclassified.
- **Interpretation and boundary:** The ReFlowSET config names a model-specific frozen training list, gives 16,001 train entries, and derives SAR paths from EO paths. The reviewed `main` snapshot is not commit-pinned. Basename membership does not establish local dataset/version lineage, shared pixels, parent-scene IDs, or confirmed leakage. No local test metadata or payload, official validation/test list, image, metric, or model output was accessed; no experiment or gate changed. See the [audit report](../../experiments/data_audits/reflowset_numeric2k_train_membership_20261024_v1/report.md).

## ECDM SAR despeckling prior-art review (2026-10-24; literature only)

- **Question:** What protocol evidence does ECDM provide for SAR diffusion despeckling, and does it test optical-guided joint restoration?
- **Evidence:** Official publisher record and indexed article sections for method, synthetic MS-SAR data, real Sentinel-1 fine-tuning/evaluation, metrics, and runtime. Direct HTML returned HTTP 429; PDF/XML routes were unavailable. No paper-linked code repository surfaced in the bounded search; code availability remains unresolved.
- **Outcome:** ECDM uses SAR-only conditional DDPM inference, optical-derived synthetic targets, real multi-temporal SAR-average fine-tuning, and no-reference real-scene metrics. It is despeckling prior art, not optical-at-inference guidance or SR/joint restoration. Scene-group separation, input-date membership in the average, and GRD-to-SLC transfer details remain open.
- **Boundary:** No article code/data, local dataset/image/metric/checkpoint, test artifact, or model output was accessed. No training, inference, local evaluation, selection, or gate change occurred. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/ecdm_sar_despeckling_fulltext_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/ecdm_sar_despeckling_fulltext_audit_20261024_v1/access_manifest.json).

## Unmounted local source-volume availability probe (2026-10-24; no data/model experiment)

- **Question:** Could a currently attached local partition expose the missing SEN1-2 source archive?
- **Evidence:** Two unmounted NTFS partitions labeled `Data` and `新加卷` are visible in block metadata. The configured SEN1-2 root is absent. The sandbox cannot access the partition device nodes; a read-only mount attempt on `Data` failed because the process is not superuser. See the [audit](../../experiments/data_audits/dev2k_unmounted_ntfs_partition_probe_20261024_v1/report.md).
- **Boundary:** No partition content, directory or file names, image, tensor, split list, test identity, metric, or model output was read. No training, inference, metric calculation, model selection, data change, or gate change occurred.

## CN114359043B MLOG-related patent audit (2026-10-24; literature only)

- **Question:** Does the patent text clarify the modality path and protocol of an MLOG-related optical-guided SAR-SR method?
- **Outcome:** The patent names MLOG and describes same-scene optical use in training but a SAR-derived pseudo-optical test input; its experiment description states 800/10 SEN1-2 pairs and bicubic LR generation with zero-mean Gaussian noise of unspecified variance, after BM3D filtering/intensity normalization. It does not specify group/validation allocation or controlled residual shifts in the reviewed experimental description.
- **Interpretation:** This is patent evidence for training-time optical-guided SAR SR, not joint despeckling/SR or real optical at inference. Exact relationship to the 2022 preprint and 2025 JARS record remains unresolved. No local experiment, test data, or gate was accessed. See the [audit](../../experiments/literature_audits/mlog_related_patent_cn114359043b_20261024_v1/report.md).


## DualCDM SAR-to-optical diffusion prior-art review (2026-10-24; literature only)

- **Question:** Does the 2026 DualCDM paper establish prior art for optical-guided SAR output or joint SAR restoration?
- **Evidence:** Official MDPI indexed article text and the visible roots/READMEs of the paper-linked `dualcdm` and `s1s2` GitHub repositories. Direct publisher HTML returned HTTP 429; PMC open showed browser verification; direct GitHub source/config routes returned cache misses.
- **Outcome:** DualCDM conditions optical-target diffusion on SAR input. It adds cross-frequency/cross-channel complex Fourier modeling and adaptive SAR-feature recalibration, but its reported metrics evaluate optical output. The described SEN1-2 and S1S2 patch splits do not establish source-group separation; S1S2 SAR preprocessing includes Refined Lee filtering. The extra large-image holdout is a disjoint region within one source image.
- **Boundary:** No dataset archive, code, external weights, local project data/test artifact, checkpoint, or model output was downloaded or executed. No training, inference, local evaluation, selection, or gate change occurred. Code revision/table lineage and dataset-specific terms remain unresolved. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/dualcdm_sar_to_optical_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/dualcdm_sar_to_optical_audit_20261024_v1/access_manifest.json).

## Attention OffsetNet optical–SAR registration prior-art screen (2026-10-24; literature only)

- **Question:** Does Attention OffsetNet support optical-guided SAR restoration or residual-mismatch robustness claims?
- **Evidence:** University of Seoul institutional record and author abstract for Yu and Jung, IEEE JSTARS, DOI `10.1109/JSTARS.2026.3731425`.
- **Outcome:** The abstract describes optical-to-optical pretraining followed by optical–SAR offset-regression fine-tuning. It reports patch RMSE of 8.87 px X / 7.99 px Y and full-scene residual misregistration of about 0.8 px X / 0.2 px Y after RANSAC aggregation. The latter is a scene-level registration result, not patch-level network accuracy.
- **Boundary:** Registration only; no SAR restoration, matched SAR-only comparison, or fallback evaluation. Full paper, split/label protocol, code, and run lineage remain unavailable here. No local data, test artifact, checkpoint, or output was accessed; no experiment or gate changed. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/attention_offsetnet_author_page_screen_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/attention_offsetnet_author_page_screen_20261024_v1/access_manifest.json).

## Formal-test access and qualitative-selection interface repair (2026-10-24; no model experiment)

The test entry path now requires a promoted B1 run whose decision is bound to its aggregate manifest and summary hash. Freeze creation/verification, fixed-test preparation, main/stress/degradation runners, efficiency measurement, and predeclared sample selection carry or verify that gate before reading fixed-test metadata. The qualitative renderer verifies the freeze first, requires the pre-test selection manifest, rechecks its hash-ranked identities, and derives 48×48 crop/profile coordinates from filename SHA-256 and dimensions rather than target values. Thirty-five focused synthetic tests passed; changed Python files compiled and the two formal shell entrypoints passed syntax checks. A direct check of the current B1 v2 manifest rejected `not_promoted` before test access. No test payload or metric value was read. Historical generated reports/figures were not rewritten; treat the previous target-statistics preview as excluded. See [D-2026-10-24-30](DECISION_LOG.md#d-2026-10-24-30--require-b1-promotion-before-formal-test-inputs-and-qualitative-access).
## Formal test CLI access repair (2026-10-24; no run)

Lower-level inference and metric CLIs now require a verified formal freeze/model ID before fixed-test access and record this lineage in outputs. Main, stress, and predeclared degradation-sensitivity runners pass the freeze context; the evaluator checks prediction/freeze/input binding. Sensitivity variants use the frozen config's predeclared looks. Fifty focused synthetic tests passed; no formal run, real test payload, prediction, or metric was created or read. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-24-31.

## CGD-Net author architecture-source follow-up (2026-10-24; literature only)

- **Question:** Does the accessible author architecture note establish a SAR-restoration fallback or direct precedent for optical-utility gating in SAR restoration?
- **Evidence:** Official publisher abstract, author repository README, and author architecture note.
- **Outcome:** The reliability module combines a refined cloud prior and learned reliability inside an optical-reflectance reconstruction task. Both heads predict optical reflectance. It is cloud-mask/prior reliability, not SAR-output optical-utility estimation or a SAR-only fallback. Exact stress magnitudes/results and split identities remain unavailable; individual model/config/test files were not retrievable.
- **Boundary:** No code, dataset, local data, experiment, or gate was accessed or changed. See the [audit](../../experiments/literature_audits/cgdnet_architecture_source_followup_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/cgdnet_architecture_source_followup_20261024_v1/access_manifest.json).

## SAR-conditioned flow matching for SAR-to-optical translation (2026-10-24; literature only)

- **Question:** Does this flow-matching work establish evidence for optical-guided SAR restoration?
- **Outcome:** No. The IEEE abstract and indexed ICLR workshop manuscript describe SAR-conditioned optical generation. The workshop Table 1 shows better SSIM/LPIPS/FID but lower PSNR than its latent-diffusion comparator; the adjacent all-metric statement conflicts with that table. The workshop/IEEE version relationship is unverified.
- **Boundary:** No code, dataset, local data, test artifact, checkpoint, prediction, or metric was accessed; no local experiment or gate changed. See the [audit](../../experiments/literature_audits/sar_conditioned_flow_matching_s2o_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/sar_conditioned_flow_matching_s2o_20261024_v1/access_manifest.json).

## Camera-RAW restoration order and joint-pipeline prior art (2026-10-24; literature only)

- **Question:** Does prior work already study restoration order and separate sequential from end-to-end restoration?
- **Evidence:** CVF-hosted CVPR 2021 paper/supplement and author repository; KAUST-hosted ICCP 2022 accepted manuscript, author project/repository, and institutional/conference records.
- **Outcome:** Xing and Egiazarian compare sequential, pairwise-joint, and fully joint Bayer-camera restoration under fixed denoising→demosaicing→SR order. Qian et al. directly test camera task order and propose denoising→SR→demosaicing, with a joint denoising+SR→demosaicing intermediate-supervision design. This closes broad novelty language about order comparisons but does not answer SAR despeckling/SR order or optical-guidance utility.
- **Boundary:** Source text and repository documentation only; no code/dataset downloads or execution, local data, test material, or experiment. B1 remains not_promoted; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/access_manifest.json).

## TUM SEN1-2 public-record metadata route (2026-10-24; source audit only)

- **Question:** Does the public TUM catalogue expose the local numeric 2k subset's upstream version or scene groups?
- **Evidence:** TUM open-dataset catalogue, official ISPRS publication record, and attempted read-only access to the expected mediaTUM dataset record.
- **Outcome:** The catalogue confirms the published SEN1-2 release count and links the paper; it provides no sample-level manifest. mediaTUM returned HTTP 429. Local lineage and parent-scene mapping remain unresolved.
- **Boundary:** No archive, local data, split membership, test metadata/payload/metrics, experiment, or gate was accessed or changed. See the [audit](../../experiments/data_audits/tum_sen1_2_record_metadata_route_20261024_v1/report.md) and [access manifest](../../experiments/data_audits/tum_sen1_2_record_metadata_route_20261024_v1/access_manifest.json).

## FAF-CD prior-art audit (2026-10-24; no local experiment)

- **Question:** Does FAF-CD establish geometric-mismatch robustness or an SAR reconstruction fallback?
- **Evidence:** Full arXiv manuscript, official MONTI 2026 workshop page, and author repository README.
- **Outcome:** Adjacent change-detection prior art for feature rectification, deformable fusion, frequency-branch gating, and modality-side appearance/radiometric stress. No controlled geometric shift/warp sweep or SAR-output restoration evaluation is reported.
- **Boundary:** External published test/validation results were visible; no local data/test material, code, or dataset was accessed, and no local experiment or gate changed. See the [audit](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/access_manifest.json).

## SEORation/OpenSEP pair-curation audit (2026-10-24; no local experiment)

- **Question:** Does SEORation provide evidence about QXS source lineage, pixel-level registration, or source-group-independent validation relevant to the local numeric 2k cohort?
- **Evidence:** Search-indexed primary manuscript excerpts and the official MONTI 2026 page. Direct OpenReview PDF retrieval triggered browser verification.
- **Outcome:** The paper curates SAR–EO pairs for bidirectional retrieval using RemoteCLIP semantic deduplication and RSLIP scene-compatibility ranking. It describes QXS-SAROPT as external to OpenSEP construction. The reviewed excerpts do not report pixel-registration error or a source-scene crosswalk; source-provided held-out splits alone do not establish group independence.
- **Boundary:** Published validation and QXS retrieval results were visible. No local data, split metadata, test material, code, archive, or experiment was accessed; B1 remains `not_promoted`, and B2/B3 remain gated. See the [audit](../../experiments/literature_audits/seoration_opensep_pair_curation_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/seoration_opensep_pair_curation_20261024_v1/access_manifest.json).


## SARptical dataset provenance/task-fit screen (2026-10-25; metadata only)

- **Question:** Can SARptical serve as an authorized, source-grouped SAR–optical dataset for the project's geometry or restoration questions?
- **Evidence:** The 2018 author paper hosted by DLR, the author-linked public GitHub README, and DLR's current TerraSAR-X data-access page.
- **Outcome:** The paper describes 109 TerraSAR-X Spotlight images over Berlin (2009–2013), nine UltraCAM views at 20 cm, 32,446 selected SAR pixels, and 89,502 optical patch crops; one SAR location may have up to nine optical views. The method matches reconstructed 3D points and adjusts patches for rotation/spacing; stated correspondence accuracy is typically 1–2 m, while the resulting 2D patches are only first-approximation aligned. The paper does not report per-pair dense residuals or expose a versioned patch-to-source/view map. Its abstract/conclusion say “over 10,000 pairs,” so the release-level count requires a manifest.
- **Interpretation:** Not a matched restoration benchmark: no paired degraded/target protocol or independent clean SAR reference is described, and the corpus is one urban area with repeated acquisition/view groups. Keep it unadopted for B1; it may support a separate geometry/scatterer study if rights and lineage are cleared.
- **Boundary:** DLR states that TerraSAR-X use depends on application and that scientific use is proposal-coordinated; the reviewed author README has no dataset-specific license or release manifest, and the optical-source terms were not established. No archive, imagery, local data, test material, or payload was accessed; no local experiment or gate changed. See the [audit](../../experiments/literature_audits/sarptical_dataset_provenance_screen_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/sarptical_dataset_provenance_screen_20261025_v1/access_manifest.json).

## MultiResSAR dataset provenance/task-fit screen (2026-10-25; metadata only)

- **Question:** Can MultiResSAR supply an authorized, source-grouped optical–SAR cohort and target protocol for joint SAR despeckling/SR?
- **Evidence:** Author paper arXiv:2502.01002, Sections 4.2–4.4 and Table 5; current author repository README and file listing.
- **Outcome:** Registration benchmark only. The paper reports 10,850 pairs (including 850 Umbra pairs designated as test) and manually checked control points with claimed error within one pixel. The repository says 1,100 pairs are currently public but gives no public-subset breakdown, split/source-group manifest, or visible dataset license. No SAR restoration target or degradation protocol is described. Keep it unadopted for restoration and do not treat control-point error as dense image alignment.
- **Boundary:** No download link was followed; no external archive, local data/test payload, code, or experiment was accessed. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/multiresar_dataset_provenance_screen_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/multiresar_dataset_provenance_screen_20261025_v1/access_manifest.json).

## Gaya et al. optical-assisted SAR despeckling screen (2026-10-25; literature only)

- **Question:** What prior-art claim is supported by the 2024 IGARSS paper when only its institutional abstract and official program entry are accessible?
- **Finding:** The abstract describes optical imagery improving SAR restoration through self-supervised neural-network training. This is direct optical-assisted SAR despeckling prior art. Dataset, target/noise protocol, inference modality path, registration, metrics, and numerical results remain unverified.
- **Boundary:** Full-text routes were unavailable. No code, local data, test material, metric, or experiment was accessed or changed. No gate changed; B1 remains `not_promoted`, and B2/B3 remain gated.
- **Record:** [Audit report](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/access_manifest.json), and [D-2026-10-25-04](DECISION_LOG.md#d-2026-10-25-04--record-gaya-et-al-as-abstract-level-optical-assisted-sar-despeckling-prior-art).

## Execution environment preflight (2026-10-25; no run)

- **Check:** `AISAR` reports Python 3.13.7, PyTorch 2.8.0+cu128, and CUDA runtime 12.8, while CUDA availability is false and visible device count is zero. `nvidia-smi -L` exits 9 because the NVIDIA driver is unreachable.
- **Interpretation:** No GPU training can run in the current sandbox. Even if GPU access returns, the failed B1 gate and unresolved provenance/group constraints still prohibit B2/B3 and model selection.
- **Boundary:** Environment metadata only; no data, test material, model, checkpoint, metric, or experiment was accessed or changed. See [D-2026-10-25-05](DECISION_LOG.md#d-2026-10-25-05--defer-model-runs-under-current-gpu-and-research-gate-constraints).

## SpeckleDiff-2D prior-art screen and Gaya thesis-route follow-up (2026-10-25; literature only)

- **Question:** What does the primary-source abstract establish about SpeckleDiff-2D, and did a new official source route expose Gaya et al.'s full text?
- **Evidence:** IEEE abstract/indexed metadata and the official IGARSS program describe a complex-valued SAR despeckling diffusion network trained on optical-dataset-derived synthetic speckle; the full IEEE text was not accessible. Official Télécom Paris pages list Victor Gaya's 2025 SAR-enhancement/multimodal-fusion thesis, but provide no direct thesis document link.
- **Outcome:** Direct SAR-despeckling diffusion prior art and optical-derived synthetic-training precedent only. Inference modality, exact protocol, metrics, and physical validity remain unknown. Gaya remains abstract-level; the thesis listing is only a route lead.
- **Boundary:** No paper/thesis/code payload, local data, test payload, metric, model, checkpoint, or experiment was accessed or changed. B1 remains not_promoted; B2/B3 and local diffusion remain deferred. See [D-2026-10-25-06](DECISION_LOG.md#d-2026-10-25-06--record-specklediff-2d-and-gaya-thesis-route-boundaries) and the two linked audits.



## Amieva et al. SM subaperture enhancement (2026-10-25; literature only)

- **Question:** Does the CVPRW 2026 azimuth-subaperture study establish optical-guided utility or a local SAR-restoration baseline?
- **Evidence:** The arXiv v1 full text describes SAR-only, same-grid subaperture-to-full-aperture enhancement, with a noisy full-aperture reference and author-reported ENL/PSNR/SSIM trade-offs. The paper states whole-dataset percentile normalization before a 5/3/2 acquisition split. CVF confirms the workshop proceedings record; the linked Hugging Face page has no card and its viewer fails.
- **Outcome:** Direct SAR-only enhancement prior art; no optical benefit, controlled mismatch, fallback, or conventional ×2 SR evidence. This is distinct from the same authors' IW→SM journal study.
- **Boundary:** No supplementary payload, paper implementation code/weights, local data/test material, checkpoint, or experiment was accessed. A separate approved read-only review of the existing MMCANetV2 run-interface source made no code edit. No gate changed; B1 remains `not_promoted`; B2/B3 remain gated. See the [audit report](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/amieva_s1_sm_subaperture_enhancement_20261025_v1/access_manifest.json).


## Dong et al. complex-valued SAR SR/subaperture prior-art screen (2026-10-25; literature only)

- **Question:** Does accessible evidence establish subaperture-based complex SAR SR prior art, and does it establish joint despeckling or optical/mismatch behavior?
- **Evidence:** Amieva et al.'s primary arXiv v1 manuscript cites Dong et al. (2025, TGRS 63, 1–14) and describes subaperture/subband information for azimuth-detail SR. An Xidian University-branded research index exposes the Dong abstract: zero-padding alignment, subaperture/subband decomposition, cross-domain learning and hierarchical fusion; it reports 3.78% amplitude improvement and 4.62% phase-accuracy improvement over a competitor.
- **Outcome:** Record abstract-level prior art for complex-valued SAR SR with subaperture learning/fusion. Exact data, scale, comparator/metrics, split, speckle objective, code, and run lineage remain unknown; do not infer joint despeckling from this screen.
- **Boundary:** Publisher full text and DOI resolution were inaccessible. No paper implementation, data/code payload, local project code/data/test material, model, checkpoint, or experiment was accessed. No gate changed; B1 remains not_promoted, B2/B3 remain gated.
- **Record:** [Audit report](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/dong_complex_sar_sr_subaperture_abstract_20261025_v1/access_manifest.json), and [D-2026-10-25-18](DECISION_LOG.md#d-2026-10-25-18--classify-dong-complex-sar-sr-as-abstract-level-prior-art).


## MMCANetV2 architecture source audit (2026-10-25; no model run)

- **Question:** What does the current model source prove about missing optical input, alignment, local attention, and reliability?
- **Evidence:** Read-only review of MMCANetV2/model.py, MMCANetV2/losses.py, JointJDSR/model.py, and scripts/train_mmcanet_v2.py.
- **Outcome:** The model-level None/zero-presence path suppresses the optical residual and returns the sigmoid-bounded SAR base. Optional alignment is a bounded zero-initialized dense offset; local-window cross-attention and a structural gate feed the residual path. The optional gate target reflects target/optical edge-direction agreement, not measured residual utility.
- **Boundary:** No model forward, data, checkpoint, metric, training, validation, parameter/MAC/runtime profiling, or selection was performed. This does not establish detection/rejection of geometric or semantic mismatch, loader-level omission of optical files, or performance. B1 remains not_promoted; B2/B3 remain gated.
- **Record:** [Source audit](../../experiments/diagnostics/mmcanet_v2_architecture_source_audit_20261025_v1/report.md) and [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness).


## Pilot optical-guided despeckling prior-art integration (2026-10-25; literature only)

Optical-Driven Nonlocal (2015), GNLM (2019), and GCOANet (2025) are now cited with their distinct evidence scopes: full author article, full author manuscript, and publisher abstract respectively. The GCOANet author spelling follows the official ScienceDirect indexed record. See D-2026-10-25-54 and [claim_evidence_map.md](claim_evidence_map.md). No experiment or local result changed.

## MHFNet author repository source audit (2026-10-25; literature only)

Verified the publisher-linked author repository and README. The repository lists a code ZIP and Baidu data route, but the ZIP was not inspectable and the README does not define dataset versions/splits or a controlled mismatch protocol. This is detection-level prior art only; no local data, test payload, experiment, metric, or model selection was used. See D-2026-10-25-58 and the [audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).

## Run-interface search disclosure (2026-10-25; documentation access only)

A broad search in historical decision/state text produced a truncated response; set this session's access flag to `test_set_accessed=true`, `test_set_used=false` conservatively. No test payload was opened or text used. See D-2026-10-25-59 and the [access audit](../../experiments/data_audits/run_interface_decision_log_text_exposure_20261025_v1/access_manifest.json).

## MHFNet Related Work citation update (2026-10-25; literature only)

Added one bounded detection-prior paragraph and DOI citation to the pilot draft. It supports prior-art classification only; no local results, test payload, experiment, or gate changed. See D-2026-10-25-60 and the [source audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).

## MHFNet public issue-route follow-up (2026-10-25; literature only)

Reviewed two open user issues about dual-branch data loading and paper/code structural divergence. No maintainer reply or independently reproducible evidence was visible; record as an unresolved reproducibility risk only. No code, data, local experiment, result, or gate changed. See D-2026-10-25-61.

## MHFNet source-archive static inspection (2026-10-25; literature only)

Downloaded and statically inspected the publisher-linked public ZIP from mutable main, then deleted the temporary archive. The loader pairs visible with a sibling infrared path and formats four channels as RGB then infrared, while the inspected RGBT model YAML labels channel 0 as SAR and channels 1–3 as RGB. Record this as an apparent, unconfirmed source-interface mismatch because the exact training invocation, dataset semantics, and article-run revision are not pinned. The model path contains separate branches with DCNv4-backed blocks and multi-scale CGA feature fusion into a detector. No data, labels, test payload, metrics, model run, or gate was accessed or changed. See D-2026-10-25-62 and the [source audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).

## RQ-Net optical–SAR registration source screen (2026-10-25; literature only)

The official publisher-indexed article text identifies a registration/matching method; the direct page/PDF and a code route were not obtained. Its abstract reports three paired optical–SAR datasets, while accessible excerpts do not resolve their identities/splits or controlled residual-shift tests. This adds registration-only prior art and does not support an SAR-restoration or fallback claim. A temporary Related Work-only excerpt compiled with resolved citations and was visually reviewed; the full manuscript was not rebuilt, and preview files were removed. No local training, inference, data, test payload, metric file, checkpoint, or experiment was accessed. During separate build setup the historical pilot abstract text was surfaced, so the session flags are `test_set_accessed=true`, `test_set_used=false`; no value was used. See D-2026-10-25-63/64, the [source audit](../../experiments/literature_audits/rqnet_registration_indexed_primary_source_20261025_v1/report.md), and the [text-exposure audit](../../experiments/data_audits/draft_abstract_text_exposure_20261025_v1/report.md).
# Data-provenance audit — QXS-SAROPT official route (2026-10-25)

This was a read-only public-source check, not a model experiment. The author repository exposes a README and preview image, links downloads to an inaccessible Wenjuan form, has no GitHub Releases, and has an unanswered open request for an alternate mirror. No archive/version/source-group/split map was recovered; local cache lineage remains unresolved and no gate changed. No data were downloaded or submitted. See [D-2026-10-25-77](DECISION_LOG.md#d-2026-10-25-77--keep-local-cache-lineage-unresolved-after-qxs-author-route-recheck) and the [audit report](../../experiments/data_audits/qxs_author_data_route_recheck_20261025_v2/report.md).
# Data-source route audit — SEN12TS public hosting and tile grouping (2026-10-25)

Read-only public-source audit; no local or external image payloads were opened. The author repository and Source Cooperative list the public 1.69 TB dataset, with an AOI/modality hierarchy and CC BY-NC 4.0 terms. The official documentation describes filenames that encode tile coordinates and acquisition metadata. The per-file listing, manageable subset, tile-footprint overlaps, and independent target protocol remain unresolved. No run or gate changed. See [D-2026-10-25-78](DECISION_LOG.md#d-2026-10-25-78--advance-sen12ts-only-to-bounded-source-listing) and the [audit report](../../experiments/data_audits/sen12ts_public_listing_route_20261025_v1/report.md).
# SEN12TS Sumatra metadata census (2026-10-25; no model experiment)

Paginated Source Cooperative listings covered the complete Sumatra `s1/` and `s2/` prefixes. Each modality has 8,212 objects; all names pair exactly, with 5,827 unique tile-center coordinates and up to seven observations per tile. A compressed metadata-only manifest and summary are retained in `experiments/data_audits/sen12ts_public_listing_route_20261025_v1/`. No imagery was fetched, no model/metric/gate changed. See [D-2026-10-25-79](DECISION_LOG.md#d-2026-10-25-79--use-sen12ts-sumatra-listing-only-for-spatial-protocol-design).
# SEN12TS Sumatra center-distance pre-screen (2026-10-25; no model experiment)

Using only the compressed official filename metadata, the 5,827 Sumatra tile centers have a minimum nearest-neighbor distance of about 200 m; 2,873 center pairs fall below the nominal 3.62 km tile diagonal and 5,301 fall within 4 km. This is a coordinate-only potential-overlap screen; no GeoTIFF transform or image payload was read. It is not a split or source-scene assignment. No model or gate changed. See [D-2026-10-25-80](DECISION_LOG.md#d-2026-10-25-80--treat-sen12ts-center-distances-as-overlap-risk-only) and the [summary](../../experiments/data_audits/sen12ts_public_listing_route_20261025_v1/sumatra_center_distance_screen.json).

## SEN12TS Sumatra 2k metadata cohort and bounded header-route screen (2026-10-25; no model experiment)

- **Question:** Can a manageable Sumatra candidate cohort be formed before any image-level study, and can the public host provide bounded GeoTIFF metadata access?
- **Protocol:** One exact-name S1/S2 pair per filename tile center, selected by earliest S1 date and fixed tie-breaks; rank centers by SHA256 of a fixed seed and coordinate, greedily retain only centers >4 km apart, and take the first 2,000 for header review. The 4 km criterion is only a filename-center pre-screen. For a later conditional temporal audit, use current VV as input and average prior VV layers in linear backscatter as a noisy pseudo-reference, subject to validating actual file metadata.
- **Result:** 3,082 centers passed the greedy guard; a direct all-pairs check found their minimum center spacing is 4.087 km. The fixed 2,000-pair manifest is the header-audit candidate. The official documentation describes four SAR times 12 days apart and identifies VV layers at 1/5/9/13. Two public-proxy Range attempts returned HTTP 403 before any response body or TIFF bytes were read.
- **Boundary:** No split, TIFF metadata, pixels, local/test data, metrics, model, checkpoint, or gate changed. The candidate and temporal rule remain conditional. See [D-2026-10-25-81](DECISION_LOG.md#d-2026-10-25-81--freeze-a-2k-sen12ts-metadata-cohort-without-claiming-a-split) and the [audit](../../experiments/data_audits/sen12ts_spatial_cohort_screen_20261025_v1/).

## CDSE-S1S2-META-20261025-v1 — conditional source metadata route screen

- **Type:** Official catalog/API documentation screen; not a model experiment.
- **Question:** Is there a separate documented route to source-ID and footprint metadata for a future paired SAR–optical inventory?
- **Evidence:** CDSE documents Sentinel-1 GRD and Sentinel-2 L2A STAC collections, spatial/time search, OData footprint queries, and open Sentinel-data terms. See the [route audit](../../experiments/literature_audits/copernicus_data_space_pairing_route_20261025_v1/report.md).
- **Outcome:** Conditional fallback only. Direct live STAC and OData item queries could not be retrieved through the available reader; no candidate pair or count is established.
- **Boundary:** No local or test data, imagery, split identities, metrics, checkpoints, training, inference, or gate decisions were accessed or changed.

## gamma_sampler_moments_20261025_v1 — synthetic Gamma sampler statistics

- **Question:** Does the canonical sampler implement its declared moments at candidate looks L=1,4,8?
- **Protocol:** One million CPU float32 draws for each of intensity, representation, and amplitude modes; predeclared deterministic seed rule and 5-SE pass criterion. No SAR pixels, dataset, model, or checkpoint.
- **Result:** All nine conditions pass; maximum absolute z score is 1.542. In amplitude mode, the squared factor matches Gamma power moments while the amplitude-factor expectation is below one as analytically expected.
- **Interpretation:** Software moment contract verified only. This does not establish local physical speckle, calibration, physical ENL, or clean-reference restoration quality.
- **Artifacts:** [protocol](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/protocol.md), [script](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/audit_gamma_sampler_moments.py), [summary](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/summary.json), [manifest](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/manifest.json).


## Data protocol audit — 2026-10-25

- Audit ID: `fixed_val_sar_representation_contract_20261025_v1`
- Type: source/config audit; not an experimental run
- Result: fixed development SAR target is a log-compressed representation from 8-bit PNG; LR uses controlled representation-domain Gamma degradation, not a calibrated physical noise model
- Test access for this audit: `false`; cumulative session flags remain `test_set_accessed=true`, `test_set_used=false` per D-2026-10-25-86
- Gate/selection effect: none
- Evidence: [report](../../experiments/data_audits/fixed_val_sar_representation_contract_20261025_v1/report.md), [manifest](../../experiments/data_audits/fixed_val_sar_representation_contract_20261025_v1/access_manifest.json)


## Source-lineage audit — 2026-10-25

- Audit ID: `local_2k_source_lineage_candidate_20261025_v1`
- Type: metadata/literature cross-check; not an experiment
- Observation: local train/validation overlap geometry is QXS-compatible under one interpretation, but original source lineage and acquisition groups remain unknown
- Test access for this audit: `false`; cumulative session flags remain `test_set_accessed=true`, `test_set_used=false` per D-2026-10-25-86
- Gate/selection effect: none
- Evidence: [report](../../experiments/data_audits/local_2k_source_lineage_candidate_20261025_v1/report.md), [QXS-SAROPT paper](https://arxiv.org/abs/2103.08259), [SEN1-2 paper](https://arxiv.org/abs/1807.01569)


## Crosswalk rebind audit — 2026-10-25

- Audit: follow-up to `sen1_2_upstream_hash_crosswalk_20261019_v1`; no new upstream index scan or image payload read
- Scope: train/validation source-map, manifest, and split metadata hashes only
- Stored result: 3,600 rows, zero exact unique matches and zero non-unique hash matches against the recorded index
- Current metadata rebind: all six train/validation metadata hashes match the prior audit
- Test access/selection effect: none in this audit; no gate changed
- Evidence: [D-2026-10-25-89](DECISION_LOG.md#d-2026-10-25-89--record-zero-exact-sen1-2-source-matches-with-current-metadata-binding)


## Generated source-label correction — 2026-10-25

- Change: future split manifests and two report generators no longer assert confirmed SEN1-2 identity for the local 2k subset
- Scope: output wording/metadata only; historical artifacts were not regenerated or rewritten
- Verification: `py_compile` exit code 0 for all three modified Python sources
- Test access/selection effect: none in this change; cumulative session flags remain `true/false` per D-2026-10-25-90
- Evidence: [D-2026-10-25-91](DECISION_LOG.md#d-2026-10-25-91--label-source-as-unverified-in-future-generated-artifacts)


## QXS-named cache crosswalk rebind — 2026-10-25

- Audit: follow-up to `sar2opt_qxs_cache_crosswalk_20261019_v1`; not a new pixel-comparison run
- Stored result: 1,626 exact pixel-equivalent pairs among 1,800 fixed train/validation rows
- Revalidation: crosswalk CSV hash, current train/validation `files.csv` hashes, sibling repo HEAD, cache existence, and Git-ignore state match
- Limitation: local cache bytes, official release lineage, license, source groups, and split semantics remain unverified
- JDSR test access/selection effect: none in this follow-up; no gate changed
- Evidence: [D-2026-10-25-92](DECISION_LOG.md#d-2026-10-25-92--treat-qxs-named-cache-matches-as-local-lineage-only)

### cdse_global_aoi_group_feasibility_20261025_v1

- 类型：只读 CDSE STAC 元数据路由筛查；协议、24 个精确查询、结果、环境/access manifest 与报告均保留。
- 结果：24/24 HTTP 200、无分页；10 个 AOI 有候选并形成 10 个图分量，达到预定的 8 分量门槛。
- 解释边界：仅进入权利/校准/云/配准/目标协议审查；不是数据集、split、训练或模型结果。
- 访问边界：未取资产或像素；本审计 test flags=false/false；累计会话 flags=true/false。
- 记录：[report](../../experiments/data_audits/cdse_global_aoi_group_feasibility_20261025_v1/report.md)。

## Approved MMCANetV2 run-interface confirmation (2026-10-25; no run)

- **Interface:** The approved residual-filter configuration is already bound across trainer, model metadata, and inference under D-2026-10-23-46; no duplicate source edit or test run was needed.
- **Boundary:** A combined documentation output was truncated; conservatively record cumulative `test_set_accessed=true`, `test_set_used=false`. No test payload or result was used. No data, checkpoint, metric, run, or gate changed.
- **Record:** [disclosure report](../../experiments/data_audits/run_interface_approval_context_text_exposure_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/run_interface_approval_context_text_exposure_20261025_v1/access_manifest.json).

## Artifact cleanup inventory (2026-10-25; metadata only)

- **Disposition:** No additional files were deleted. No cache/bytecode/temp files, named smoke/scratch/failed/partial outputs, or generated array payloads remained in the inventoried output/result roots. Canonical input PNGs and historical run/checkpoint evidence were retained.
- **Boundary:** No result or checkpoint payload was opened. Cumulative session flags are `test_set_accessed=true`, `test_set_used=false` from the separate documentation-output disclosure; no gate changed.
- **Record:** [cleanup inventory](../../experiments/artifact_cleanup_current_inventory_20261025_v1.json).

## RABASAR temporal-superimage assumptions (2026-10-25; literature only)

- **Finding:** The temporal mean estimates reflectivity only under stable temporal reflectivity and sufficiently decorrelated speckle; the full RABASAR paper also requires calibrated, registered same-orbit/similar-incidence inputs and uses change-aware sample selection.
- **Boundary:** Physical-domain prior art does not validate local PNG or SEN12TS RTC gamma-zero targets. No local data, experiment, model, or gate changed. Cumulative session flags are test_set_accessed=true, test_set_used=false.
- **Record:** [audit report](../../experiments/literature_audits/rabasar_temporal_superimage_assumptions_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/rabasar_temporal_superimage_assumptions_20261025_v1/access_manifest.json).

## Active-goal document read truncation (2026-10-25; text only)

- **Disclosure:** The attempted read of lines 1501–1800 returned a truncated response; no text from that output was used. Cumulative session flags remain `test_set_accessed=true`, `test_set_used=false`.
- **Boundary:** No test payload, metric file, or checkpoint contents were opened; no run or gate changed.
- **Record:** [disclosure report](../../experiments/data_audits/goal_document_read_truncation_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/goal_document_read_truncation_20261025_v1/access_manifest.json).

## Research-state text exposure (2026-10-25; documentation only)

- **Disclosure:** Bounded current-state, next-actions, and research-risks reads surfaced historical result-bearing prose. No value was used; cumulative session flags are `test_set_accessed=true`, `test_set_used=false`.
- **Boundary:** No test payload or raw metric file was opened. No run, model, or gate changed.
- **Record:** [disclosure report](../../experiments/data_audits/research_state_read_text_exposure_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/research_state_read_text_exposure_20261025_v1/access_manifest.json).

## Temporal geometric-mean source-search disclosure (2026-10-25; public text only)

- **Disclosure:** A HAL route search response for `hal-03095153v1` was truncated and excluded from evidence; no surfaced claim or result was used.
- **Boundary:** No local data or test payload was accessed by this search. Cumulative session flags remain `test_set_accessed=true`, `test_set_used=false`.
- **Record:** [disclosure report](../../experiments/data_audits/geometric_mean_source_search_truncation_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/geometric_mean_source_search_truncation_20261025_v1/access_manifest.json).

## Temporal geometric-mean primary-route audit (2026-10-25; metadata only)

- **Finding:** The institutional author profile verifies the paper metadata and the institutional GitLab landing page identifies its associated MATLAB implementation; full text and code files were unavailable through bounded primary routes.
- **Boundary:** No paper-specific method/result or local test payload was accessed. A local development-result summary surfaced during context search but was not used. Cumulative session flags remain `test_set_accessed=true`, `test_set_used=false`; no model, run, selection, or gate changed.
- **Record:** [source-route report](../../experiments/literature_audits/temporal_geometric_mean_primary_route_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/temporal_geometric_mean_primary_route_20261025_v1/access_manifest.json).

## Project-context search disclosure (2026-10-25; text only)

- **Disclosure:** A combined search across three research records was truncated and included historical development-result prose. The full surfaced response is excluded; no value was used.
- **Boundary:** No test payload or raw metric file was opened. Cumulative session flags remain `test_set_accessed=true`, `test_set_used=false`; no run, result, or gate changed.
- **Record:** [disclosure report](../../experiments/data_audits/project_context_search_truncation_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/project_context_search_truncation_20261025_v1/access_manifest.json).

## OSdataset literature-context search disclosure (2026-10-25; text only)

- **Disclosure:** A targeted search in the long literature review was truncated and surfaced unrelated public-paper result text. No surfaced value was used.
- **Boundary:** No local test payload or raw metric file was accessed; cumulative session flags remain `test_set_accessed=true`, `test_set_used=false`.
- **Record:** [disclosure report](../../experiments/data_audits/osdataset_literature_search_truncation_20261025_v1/report.md) and [access manifest](../../experiments/data_audits/osdataset_literature_search_truncation_20261025_v1/access_manifest.json).

## OSdataset Google Earth terms review (2026-10-25; official policy pages only)

- **Finding:** Current official policy does not establish blanket dataset-scale rights; historical source and permission for the 2020 archive remain unresolved. OSdataset stays unadopted.
- **Boundary:** No archive or imagery was accessed. This is not a legal determination; audit-local test flags are false/false, cumulative session flags true/false.
- **Record:** [source report](../../experiments/literature_audits/osdataset_google_earth_terms_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/osdataset_google_earth_terms_20261025_v1/access_manifest.json).


## OPERA RTC-S1 + Sentinel-2 metadata route screen (2026-10-25; no experiment)

- **Question:** Does the fixed, globally separated 12-AOI panel contain enough repeated same-burst OPERA RTC-S1 acquisitions with nearby low-cloud Sentinel-2 L2A items to justify a separate source/target review?
- **Protocol:** [`protocol.md`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/protocol.md) and [`config.json`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/config.json).
- **Metadata result:** 36 catalog requests returned HTTP 200; 358 OPERA and 414 S2 items. Six AOIs had at least four same-burst SAR dates, but the maximum low-cloud optical pairing count was three; no four-date paired stack or candidate component met the frozen gate.
- **Decision:** Stop this fixed panel/window. Do not post hoc widen dates or relax cloud/gap rules. No assets or pixels were accessed, and no split, model, run, or gate changed.
- **Artifacts:** [`report.md`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/report.md), [`results.json`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/results.json), [`screen_summary.json`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/screen_summary.json), and [`access_manifest.json`](../../experiments/data_audits/opera_s1_sentinel2_route_screen_20261025_v1/access_manifest.json).


## B1 v2 source-group heterogeneity diagnostic (2026-10-25; read-only)

- **Question:** Were the small positive identity-weighted PSNR differences in the source-aligned B1 v2 screen broadly present across validation folders, and did they coincide with SSIM changes?
- **Input and binding:** The analysis read only `aggregate/per_group_summary.csv`, bound to the existing B1 aggregate manifest and frozen `sen12_scene_dev1800_b1_v2` protocol. Fifteen validation source folders, 200 identities, and seeds 0–2 were required; all access flags were false for test.
- **Observation:** After averaging each folder's three seed means, 10/15 folders had positive ΔPSNR and 5/15 negative ΔPSNR; all 15 had negative ΔSSIM. The identity-weighted ΔPSNR means by seed remain +0.000375, +0.001678, and +0.002799 dB. The original B1 gate remains `not_promoted` for mean ΔPSNR and mean ΔSSIM.
- **Interpretation:** The PSNR direction was not confined to only one or two source folders, but the SSIM decrease was present in every folder. This descriptive view identifies no cause and does not support a clean-SAR or general optical-benefit claim. Validation remains fall-only within one ROI-season, and fine pixel co-registration is unverified.
- **Artifacts:** [Frozen protocol](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/protocol.md), [script](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/summarize_group_heterogeneity.py), [report](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/report.md), and [access manifest](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/access_manifest.json).

## TUM SEN1-2 source-product timestamp route (2026-10-25; metadata-only)

- **Question:** Can TUM record `1436631` bind the prospective cohort's exact `ROIs1970_fall/s1_i` and `s2_i` folders to product IDs, acquisition timestamps, and spatial provenance?
- **Outcome:** Inconclusive. OAI-PMH `Identify` returned HTTP 429; the single-record request was inaccessible through the web reader and returned no metadata body. The exact mapping remains unresolved.
- **Boundary:** No local data, image, archive, product asset, test path, metric, model, or checkpoint was accessed. No experiment or gate changed.
- **Record:** [D-2026-10-25-124](DECISION_LOG.md#d-2026-10-25-124--close-the-blocked-tum-producttimestamp-metadata-route), [report](../../experiments/literature_audits/sen12_scene_product_timestamp_route_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sen12_scene_product_timestamp_route_20261025_v1/access_manifest.json).

## Sentinel-1/2 official geolocation specification audit (2026-10-25; literature only)

- **Question:** Do official product documents quantify a relative S1 GRD to S2 L1C residual bound applicable to the SEN1-2 fall mosaics?
- **Finding:** No. S2 absolute geolocation and S2-only spectral/temporal registration metrics are documented for specific processing contexts. The S1 product spec describes GRD terrain geocoding but provides no matching cross-sensor metric. Exact SEN1-2 product IDs/baselines remain unknown.
- **Interpretation:** Product-level absolute accuracy and S2-to-S2 registration do not quantify local S1-to-S2 mosaic residuals. Keep nominal georegistration conditional and fine co-registration unverified; do not derive a B2 shift level from these values.
- **Boundary:** Official public documents only; no local data, test payload, result, model, or checkpoint accessed. No gate changed.
- **Record:** [D-2026-10-25-126](DECISION_LOG.md#d-2026-10-25-126--keep-product-geolocation-specifications-separate-from-local-s1s2-registration), [report](../../experiments/literature_audits/sentinel12_official_geolocation_specs_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sentinel12_official_geolocation_specs_20261025_v1/access_manifest.json).

## Independent SAR–optical checkpoint validation precedent (2026-10-25; literature only)

- **Source:** Liao et al., *Optical and SAR Image Registration in Equatorial Cloudy Regions Guided by Automatically Point-Prompted Cloud Masks*, *Remote Sensing* 17(15), 2630 (2025), [DOI 10.3390/rs17152630](https://doi.org/10.3390/rs17152630).
- **Method evidence:** The primary article text describes a control/check separation: historical/public georeferenced datasets constrain block adjustment, while at least six distributed manual checkpoints per region measure independent positioning/edging residuals. The work reports four GF-03 with GF-07/ZY-3 equatorial regions.
- **Interpretation:** This is a useful validation-design precedent, not a SEN1-2 error estimate. Reference-source accuracy and checkpoint annotation uncertainty were not verified in the partial article review; sensors and resolutions differ.
- **Boundary:** No local imagery, validation metrics, predictions, checkpoints, or test data accessed; no training or gate change. Direct publisher HTML/PDF was inaccessible, while indexed full-text sections supplied the methods evidence.
- **Record:** [D-2026-10-25-127](DECISION_LOG.md#d-2026-10-25-127--treat-independent-saroptical-checkpoints-as-a-validation-template-only) and [audit](../../experiments/literature_audits/sar_optical_independent_tiepoint_validation_20261025_v1/report.md).

## SEN1-2 fall independent-reference availability screen (2026-10-25; literature only)

- **Question:** Do authoritative SEN1-2 source materials identify exact fall Sentinel products and an independent, accuracy-documented reference/check set?
- **Finding:** The official 2018 paper documents GEE seasonal selection, per-ROI mosaicking, and visual quality inspection, but does not publish exact fall product IDs, per-pixel mosaic provenance, independent control/check points, or local S1-to-S2 point residuals. A general Sentinel-1 geolocation statement and DEM use do not establish pair-specific residual accuracy.
- **Interpretation:** No current authoritative source satisfies the transfer criterion. Treat this as an evidence gap, not evidence that metadata never existed; the TUM record route remains blocked and was not retried.
- **Boundary:** Official literature only; no local/test data, validation metrics, model, checkpoint, training, inference, or registration estimate. B1 remains `not_promoted`; B2/B3 remain gated.
- **Record:** [D-2026-10-25-128](DECISION_LOG.md#d-2026-10-25-128--keep-sen1-2-paper-evidence-at-collection-level) and [audit](../../experiments/literature_audits/sen1_2_independent_reference_availability_20261025_v1/report.md).

## Sentinel-1A absolute geolocation calibration transfer audit (2026-10-25; literature only)

- **Question:** Does the original Sentinel-1A commissioning calibration study provide a transferable bound for SEN1-2's local S1–S2 registration residual?
- **Finding:** Schubert et al. use DGPS-surveyed corner reflectors at two Swiss sites and report corrected StripMap SLC/GRDF ALE standard deviations from 2014–2015. The paper's IW results are preliminary and unquantified; the study does not measure pairwise S1–S2 residuals.
- **Interpretation:** Treat the reported sub-sample consistency as workflow-specific absolute-geolocation context. Do not transfer it to the local 2017 GRD IW cohort or infer a pixel offset. Fine local registration remains unresolved.
- **Boundary:** Publisher-indexed primary article text only; direct page returned HTTP 429 and the PDF route was unavailable. No local/test data, model, run, shift, or gate was accessed or changed.
- **Record:** [D-2026-10-25-129](DECISION_LOG.md#d-2026-10-25-129--keep-sentinel-1a-absolute-geolocation-calibration-separate-from-local-s1s2-residuals), [report](../../experiments/literature_audits/sentinel1_geolocation_accuracy_primary_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sentinel1_geolocation_accuracy_primary_20261025_v1/access_manifest.json).

## Sentinel-1 IW absolute-geolocation follow-up (2026-10-25; literature only)

- **Question:** Does a later primary calibration study quantify IW absolute geolocation, and does it bound local S1–S2 registration?
- **Finding:** Schubert et al. (2017) report ALE time series for 194 IW SLC and 133 IW GRDH products over Swiss calibration sites through March 2017. IW SLC reports 0.15 ± 0.19 m range / 1.44 ± 1.19 m azimuth, with a proposed azimuth correction of 0.00 ± 0.55 m; GRDH reports 0.20 ± 0.24 m / 0.31 ± 1.31 m. No S1–S2 residual is measured.
- **Interpretation:** This adds mode-family absolute-geolocation context but cannot set a local pixel shift or B2 condition. Exact local product IDs/baselines and independent pair checkpoints remain missing.
- **Boundary:** Publisher-indexed primary article text; publisher XML returned HTTP 429. No local/test data, metric, model, run, or gate was accessed or changed.
- **Record:** [D-2026-10-25-130](DECISION_LOG.md#d-2026-10-25-130--record-quantified-iw-absolute-geolocation-without-transferring-it-to-local-s1s2-registration), [report](../../experiments/literature_audits/sentinel1_iw_geolocation_followup_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sentinel1_iw_geolocation_followup_20261025_v1/access_manifest.json).

## GLCNet self-supervised SAR despeckling audit (2026-10-25; literature only)

- **Question:** Does recent blind-spot self-supervised real-SAR despeckling establish a local optical-guided JDSR method or a reproducible baseline?
- **Finding:** GLCNet is single-SAR despeckling with global wavelet-shuffle and local Fourier/spatial detail branches. Publisher-indexed text describes Sentinel-1 AML-reference evaluation and separate GF-3, TerraSAR-X, and Capella-X training. It does not test optical guidance, SR, or joint restoration.
- **Reproducibility boundary:** The article says 10,000 iterations while the current author-repo config says 100,000. Its 20-image validator is annotated as test data; the displayed validation function saves predictions but does not compute metrics. This is unresolved run-role/configuration lineage, not evidence that held-out references were used for model selection. No code was run and no data assets were accessed.
- **Decision:** Record GLCNet as SAR-only self-supervised despeckling prior art, not a matched baseline. No local run, model selection, manuscript result, or B1/B2/B3 gate changed.
- **Record:** [D-2026-10-25-132](DECISION_LOG.md#d-2026-10-25-132--classify-glcnet-as-sar-only-self-supervised-despeckling-prior-art), [report](../../experiments/literature_audits/glcnet_self_supervised_sar_denoising_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/glcnet_self_supervised_sar_denoising_20261025_v1/access_manifest.json).

## SpaceNet 6 expanded candidate screen (2026-10-25; literature only)

- **Question:** Can the expanded MSAW release supply an authorized, source-grouped optical–SAR cohort for joint SAR restoration?
- **Finding:** Follow-up of the official page's release link found that the September 2020 article also says 202 strips and specifies 202 unique strips. The April 2020 arXiv v1 manuscript reports 204 individual strips; no source identifies the two-strip difference. One optical collection is dated 7–27 days after the SAR acquisitions. SAR is multilooked/log-scaled; no clean restoration target or measured SAR–optical residual is reported. Challenge test/final-scoring regions do not provide optical inputs.
- **Interpretation:** Retain it only as a conditional auxiliary source for future high-resolution geometry/structure diagnostics. It does not replace the local cohort or establish local Sentinel-1/2 registration, optical utility, or clean-target restoration.
- **Boundary:** No archive, object listing, image, local payload, or test identity was accessed. Published aggregate challenge results were visible but unused. No experiment or gate changed.
- **Record:** [D-2026-10-25-134](DECISION_LOG.md#d-2026-10-25-134--keep-spacenet-6-as-a-conditional-auxiliary-candidate), [release-count follow-up D-138](DECISION_LOG.md#d-2026-10-25-138--distinguish-spacenet-6-release-count-from-manuscript-count), [report](../../experiments/literature_audits/spacenet6_expanded_candidate_screen_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/spacenet6_expanded_candidate_screen_20261025_v1/access_manifest.json).

## SpaceNet 9 rights and split reachability (2026-10-25; literature only)

- **Question:** Does the official SpaceNet 9 release provide an authorized and independently evaluable registration-method source for this SAR restoration project?
- **Evidence:** The official challenge page states CC BY-SA 4.0 for the dataset; the winning-solutions repository separately licenses its code Apache-2.0. The official README lists three labeled development triples, public-test inputs without public tie-point labels, and a separate hidden final AOI. The published article describes the high-resolution Umbra/WorldView-2 registration task and manual tie-point workflow.
- **Outcome:** Keep the dataset as registration prior art and a possible future method-debugging reference only. The accessible training labels do not form a held-out validation set, and the source has no SAR restoration target or clean-reference SAR. Do not acquire or train on it now.
- **Boundary:** Only public paper, challenge-page, and repository text were inspected. External test filenames and published aggregate challenge findings were visible, but no test image or hidden label was accessed or used. No local data, experiment, or gate changed. See the [audit report](../../experiments/literature_audits/spacenet9_rights_split_screen_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/spacenet9_rights_split_screen_20261025_v1/access_manifest.json).



## dp_sar_pol_sar_sr_abstract_screen_20261025_v1 — literature audit only

- **Type:** Primary publisher-indexed abstract screen; no experiment or local run.
- **Protocol:** IEEE Xplore record only. Direct publisher page returned HTTP 418 in the available reader; full text, author list, code, and data were not obtained.
- **Finding:** Abstract describes scattering-aware complex PolSAR SR with point-target/main-lobe and Pauli-domain constraints. It is adjacent prior art, not an optical-guided JDSR baseline.
- **Access:** No local data, test payload, raw metrics, checkpoint, or external data payload accessed. A truncated local text search surfaced historical development-result prose; no result was used. Cumulative flags remain test_set_accessed=true, test_set_used=false.
- **Artifacts:** [report](../../experiments/literature_audits/dp_sar_pol_sar_sr_abstract_screen_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/dp_sar_pol_sar_sr_abstract_screen_20261025_v1/access_manifest.json).


## polsar_hr_sinsar_diffusion_source_route_20261025_v1 — bibliographic route screen only

- **Type:** Source-access screen; no paper-method audit and no local experiment.
- **Outcome:** IEEE record/PDF and DOI routes were not readable in the available reader. Secondary listings are retained only in the access manifest as excluded discovery routes.
- **Boundary:** No technical claim, author name, dataset, or result was adopted. No local/test data, metrics, checkpoint, or prediction was accessed.
- **Artifacts:** [report](../../experiments/literature_audits/polsar_hr_sinsar_diffusion_source_route_20261025_v1/report.md), [access manifest](../../experiments/literature_audits/polsar_hr_sinsar_diffusion_source_route_20261025_v1/access_manifest.json).


## evaluator_high_frequency_fft_center_20261025_v1 — synthetic code audit only

- **Type:** Metric implementation regression; no experiment or project-data evaluation.
- **Finding:** For even image dimensions, the prior radius mask used `(N−1)/2` after `fftshift`; the correct zero-bin center is `floor(N/2)`. An independent `fftshift(fftfreq)` expectation reproduces the corrected boundary-frequency energy.
- **Verification:** New synthetic regression failed before the fix and passed afterward. `tests/test_evaluator.py` and `tests/test_ogsrn_metric_policy.py`: 29 passed; `py_compile` passed. Full pytest collection stopped because `rasterio` is absent from AISAR.
- **Boundary:** Auxiliary high-frequency structure metric only; B1 gate and retained historical results unchanged. No project image, test payload, metric result, checkpoint, or run was accessed. Screen flags false/false; cumulative flags remain true/false. See [D-2026-10-25-139](DECISION_LOG.md#d-2026-10-25-139--center-even-sized-fft-diagnostics-on-the-shifted-zero-bin).


## dual_channel_sar_optical_registration_20261025_v1 — literature audit only

- **Type:** Primary publisher-source review; no local or external dataset experiment.
- **Finding:** Zhang and Wang (2026) report a dual-channel SAR–optical keypoint-registration method and a 200-pair SEN1-2 transfer test. The article does not report source-group IDs or an independent local tie-point reference; its geometric stress tests are synthetic.
- **Reproducibility:** Data-R uses a random pair-level split with parent-scene membership unspecified. The abstract promises public code/data while Data Availability says data are on request; the publisher HTML contains embedded figure-generation prompts. PDF, code, and dataset were not independently reviewed or downloaded.
- **Boundary:** Published aggregate tables were visible but unused. No local/test data, external test payload, metric file, checkpoint, or model was accessed. Audit flags false/false; cumulative project flags true/false. See [D-2026-10-25-140](DECISION_LOG.md#d-2026-10-25-140--treat-the-dual-channel-registration-article-as-prior-art-only) and [report](../../experiments/literature_audits/dual_channel_sar_optical_registration_20261025_v1/report.md).

## sen1_2_independent_reference_availability_20261025_v1 — official catalog route follow-up

- **Question:** Do the TUM SIPEO dataset index or German GovData catalog expose exact source products or an independent reference for the local SEN1-2 fall cohort?
- **Source and outcome:** TUM SIPEO lists SEN1-2 with a paper link but no product manifest. GovData names TUM University Library as publisher and exposes a placeholder resource plus a Turtle metadata link; the metadata export returned a cache miss and its package API was inaccessible through the available reader.
- **Interpretation:** No exact product IDs, per-pixel mosaic provenance, or independent checkpoints were recovered. This closes only the inspected catalog routes; it does not establish that metadata does not exist.
- **Boundary:** Public catalog metadata only; no data archive, imagery, metadata body, local/test payload, metric, checkpoint, model, or run was accessed. Audit flags are false/false; cumulative project flags remain true/false. See [D-2026-10-25-141](DECISION_LOG.md#d-2026-10-25-141--keep-the-official-sen1-2-catalog-route-at-metadata-only-status) and the [updated report](../../experiments/literature_audits/sen1_2_independent_reference_availability_20261025_v1/report.md).

## GCOANet same-acronym repository disambiguation (2026-10-25; no local run)

- **Question:** Is the newly surfaced GitHub `GCOANet` repository the implementation of Yang et al.'s SAR paper?
- **Evidence:** The repository README describes “GCOA-Net” as graph-regularized cross-omics modeling using gene, CpG, and miRNA data.
- **Outcome:** Exclude this same-acronym repository as a false-positive code candidate. This does not identify or rule out other official article code.
- **Boundary:** External README only; no local data, test payload, metric, model, checkpoint, experiment, or gate was accessed or changed. See [D-2026-10-25-142](DECISION_LOG.md#d-2026-10-25-142--exclude-the-same-acronym-cross-omics-repository-as-gcoanet-paper-code) and the [updated route audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md).

## Research-state text-read disclosure (2026-10-25; no experiment)

- **Scope:** Three truncated current-state/research-record outputs were excluded from evidence; no surfaced result value was used.
- **Boundary:** No test payload or raw metric file was opened. Cumulative project/session flags remain `test_set_accessed=true`, `test_set_used=false`. See the [disclosure audit](../../experiments/data_audits/research_context_read_truncation_20261025_v1/report.md).

## cdse_global_aoi_metadata_screen_20261025_v1 — external catalogue audit (2026-10-25; no model run)

- **Protocol:** Frozen eight-point, 2026-05–07 metadata-only STAC screen; 16 bounded searches; no assets or local data.
- **Outcome:** Three independent candidate components among seven complete AOIs; one paginated AOI leaves a maximum of four, below the six-component threshold. Stop this route at metadata screening.
- **Retained evidence:** Frozen protocol, exact query bodies, compact result summary/candidate metadata, access manifest, and report in `experiments/data_audits/cdse_global_aoi_metadata_screen_20261025_v1/`.
- **Boundary:** Audit test flags false/false; cumulative project flags remain true/false. No split, training, inference, selection, or gate change. See [D-2026-10-25-147](DECISION_LOG.md#d-2026-10-25-147--stop-the-global-cdse-route-at-metadata-screening).

## sen12ms_mediatum_direct_record_recheck_20261025_v1 — source-identity audit (2026-10-25; no model run)

- **Protocol:** One bounded record landing-page check and one exact official-domain search after the web reader returned HTTP 429; record-only and child APIs were not requested.
- **Outcome:** The page title was generic and the exact-domain search returned no results. Numeric node identity remains unverified; do not open the record export or children without new authoritative binding evidence.
- **Retained evidence:** Frozen protocol, status/size-only summary, access manifest, and report in `experiments/literature_audits/sen12ms_mediatum_direct_record_recheck_20261025_v1/`.
- **Boundary:** Audit test flags false/false; cumulative project flags remain true/false. No archive, split, training, inference, selection, or gate change. See [D-2026-10-25-148](DECISION_LOG.md#d-2026-10-25-148--keep-mediatum-node-identity-unverified-after-direct-page-check).

## polsar_sinsar_diffusion_ieee_direct_check_20261025_v1 — primary-record access audit (2026-10-25; no experiment)

- **Protocol:** One bounded public GET to the exact IEEE Xplore record; no PDF or link requests.
- **Outcome:** HTTP 202 with no structured title, author, DOI, or abstract metadata. Keep the bibliographic item as an unverified lead; no method/result claim entered the literature review.
- **Retained evidence:** Frozen protocol, status/field-presence summary, access manifest, and report in `experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/`.
- **Boundary:** Audit test flags false/false; cumulative project flags remain true/false. No data, test payload, model, run, selection, or gate change. See [D-2026-10-25-149](DECISION_LOG.md#d-2026-10-25-149--retain-the-polsarsinsar-diffusion-sr-item-as-an-unverified-lead).

## sen12ms_fdl_readme_direct_route_20261025_v1 — repository-route audit (2026-10-25; no experiment)

- **Protocol:** One bounded GET to the exact public raw README path; no repository tree, bucket, list, archive, or object requests.
- **Outcome:** HTTP 404; no response body read. The train-only geospatial metadata route remains unresolved.
- **Retained evidence:** Frozen request, status-only summary, access manifest, and report in `experiments/literature_audits/sen12ms_fdl_readme_direct_route_20261025_v1/`.
- **Boundary:** Audit test flags false/false; cumulative project flags remain true/false. No local/test data, split, training, inference, selection, or gate change. See [D-2026-10-25-150](DECISION_LOG.md#d-2026-10-25-150--leave-the-fdl-sen12ms-mirror-route-unresolved-after-404).

## SAR-DDPM primary-text and code audit (2026-10-25; no model experiment)

- **Question:** What does SAR-DDPM establish about diffusion for SAR despeckling, and is its public code a version-bound reproduction?
- **Protocol:** Read the complete arXiv v1 author manuscript and statically reviewed the author-linked main-branch README, training script, image loader, model, script utility, and training loop. No local data, experiment, test asset, or metric file was accessed; code was not executed.
- **Outcome:** The paper reports conditional SAR despeckling trained with synthetic L=1 Gamma-speckled optical imagery and cycle-spinning inference. Synthetic results are reported on 10 DSIFN images; the real check is two Sentinel-1 scenes with four ENL regions and no clean SAR reference. It does not study optical-at-inference, SR, or joint restoration. The mutable public source has unresolved static conditioning/configuration inconsistencies and no verified paper-run binding.
- **Interpretation:** Retain as SAR despeckling prior art; it does not justify a local diffusion implementation or alter the B1/B2/B3 decision path.
- **Retained evidence:** [protocol](../../experiments/literature_audits/sar_ddpm_fulltext_code_audit_20261025_v1/protocol.md), [report](../../experiments/literature_audits/sar_ddpm_fulltext_code_audit_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sar_ddpm_fulltext_code_audit_20261025_v1/access_manifest.json).
- **Boundary:** Audit-level test flags false/false; cumulative project flags remain true/false. No local result values were used, and no model, run, protocol, manuscript claim, or gate changed.

## ogsflow_author_repository_versioned_review_20261025_v1 — repository-source audit (2026-10-25; no experiment)

- **Protocol:** Frozen metadata/document-only check of the public OGSFlow repository; no data, sample, model, metric, or test payload access.
- **Outcome:** A root snapshot crawled yesterday lists `assets/paper` and four commits; a separate README snapshot crawled five days ago says its public paper link and final citation are forthcoming. No commit SHA or paper filename was retrieved, and repository ownership by the paper authors was not verified. Keep the paper and version unverified; no paper-level claim was added.
- **Retained evidence:** Protocol, result summary, access manifest, and report in `experiments/literature_audits/ogsflow_author_repository_versioned_review_20261025_v1/`.
- **Boundary:** Audit flags false/false; cumulative project flags true/false. No code, data, model, experiment, selection, or B1/B2/B3 gate changed. See [D-2026-10-25-151](DECISION_LOG.md#d-2026-10-25-151--keep-the-new-ogsflow-repository-signal-unverified).
## SAR–optical registration-error sensitivity literature audit (2026-10-25; no experiment)

- **Question:** What protocol and scope support Wu et al.'s simulated registration-error sensitivity claims?
- **Evidence:** Full author-uploaded publisher-formatted text plus official publisher metadata/abstract. The study uses two urban areas, manual GCP-fit alignment, deliberate 1–15 px translations, and land-cover classification across three fusion levels. Reported GCP fit RMSE is not independent checkpoint accuracy.
- **Boundary:** Literature only. No local data payload, test material, model, run, or metric file was used. Audit flags `test_set_accessed=false`, `test_set_used=false`; cumulative project flags remain true/false from prior documented text exposure. No experiment or gate changed.
- **Record:** [protocol](../../experiments/literature_audits/sar_optical_registration_error_sensitivity_fulltext_20261025_v1/protocol.md), [report](../../experiments/literature_audits/sar_optical_registration_error_sensitivity_fulltext_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sar_optical_registration_error_sensitivity_fulltext_20261025_v1/access_manifest.json).

## sen12ms_cr_candidate_20261025_v1 — source and protocol audit (2026-10-25; no experiment)

- **Question:** Can the mono-temporal SEN12MS-CR release support a future ROI-grouped train/development cohort for SAR-guided restoration?
- **Outcome:** The primary paper reports 169 nonoverlapping ROIs and states that its 114,325-patch training split can be subdivided for training and validation; the author page reports 175 ROIs. It states dataset CC BY 4.0 terms. The count/version discrepancy, train-only ROI manifest, SAR domain, acquisition timing, and residual registration remain unresolved. Keep the release conditional and distinct from SEN12MS-CR-TS and local numeric 2k.
- **Retained evidence:** [protocol](../../experiments/literature_audits/sen12ms_cr_candidate_20261025_v1/protocol.md), [report](../../experiments/literature_audits/sen12ms_cr_candidate_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sen12ms_cr_candidate_20261025_v1/access_manifest.json).
- **Boundary:** No archive, split-list contents, holdout identity, image, local/test data, metric, checkpoint, or run was accessed. Published external cloud-removal benchmark values were visible but not used. Audit-level test flags are false/false; cumulative project flags remain true/false. No experiment, model, selection, or gate changed.

## sen12ms_cr_release_lineage_followup_20261025_v1 — publication-version audit (2026-10-25; no experiment)

- **Question:** Do the 169/175 ROI and 157,521/122,218 patch counts refer to one release snapshot?
- **Evidence:** The 2020 JPRS DSen2-CR article reports 169 ROIs and 157,521 triplets with a 149/10/10 ROI split; the 2020 TGRS article reports 169 ROIs and 114,325 train + 7,893 holdout patches (122,218 total) and calls the collection the final cleaned-up version. Current author/TUM catalogue metadata lists 175 ROIs and 122,218 triplets. A revision is plausible but not explicitly documented. The JPRS paper reports S1 IW GRD σ⁰ in dB; that cannot be attributed to the later TGRS collection without a source crosswalk.
- **Boundary:** Primary-paper and official catalogue metadata only. No archive, split contents, holdout identity, image, test payload, local project data, metric value, checkpoint, or run was accessed or used. No local experiment or gate changed.
- **Record:** [protocol](../../experiments/literature_audits/sen12ms_cr_release_lineage_followup_20261025_v1/protocol.md), [report](../../experiments/literature_audits/sen12ms_cr_release_lineage_followup_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/sen12ms_cr_release_lineage_followup_20261025_v1/access_manifest.json).

## mamba_scr_sen12ms_cr_split_route_20261025_v1 — third-party split screen (2026-10-25; no experiment)

- **Question:** Does Mamba-SCR's 49-summer-ROI partition establish an authorized, version-pinned, source-disjoint train/development subset for local research?
- **Outcome:** The publisher reports a 40/4/5 ROI split and points to Zenodo for sample indices/splits/scripts. It does not provide a crosswalk to the original TGRS holdout or reconcile the current 175-ROI catalogue; keep the split metadata-only and unadopted. The accessible publisher-full-text copy describes wavelet-based SAR feature enhancement and feature-level optical–SAR fusion for cloud-free optical output. Its reviewed discussion varies cloud-cover levels and land-cover categories; no explicit geometric-shift/registration stress or SAR-only fallback was located. This is adjacent cloud-removal prior art, not SAR-output mismatch robustness evidence. A metadata-only follow-up could not retrieve the cited Zenodo record metadata through direct DOI/API opens or exact-DOI search; this does not establish global unavailability. See [D-2026-10-25-174](DECISION_LOG.md#d-2026-10-25-174--treat-mamba-scr-split-metadata-as-non-authoritative), [D-2026-10-25-175](DECISION_LOG.md#d-2026-10-25-175--keep-the-mamba-scr-zenodo-record-unverified-after-inaccessible-metadata-routes), and [D-2026-10-25-176](DECISION_LOG.md#d-2026-10-25-176--classify-mamba-scr-as-optical-output-fusion-without-explicit-geometric-stress-evidence).
- **Record:** [protocol](../../experiments/literature_audits/mamba_scr_sen12ms_cr_split_route_20261025_v1/protocol.md), [report](../../experiments/literature_audits/mamba_scr_sen12ms_cr_split_route_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/mamba_scr_sen12ms_cr_split_route_20261025_v1/access_manifest.json).
- **Boundary:** No Zenodo record metadata/file listing, file, split/index identity, third-party test payload, local test asset, checkpoint, or run was accessed. Publisher-reported performance prose surfaced in the publisher-full-text copy but was not used. Audit-level local test flags are false/false; cumulative project flags remain true/false. No local experiment, model, selection, or gate changed.

## noisy_pseudo_target_n2n_sar2sar_boundary_20261025_v1 — primary-source and local-source audit (2026-10-25; no experiment)

- **Question:** Do Noise2Noise, SAR2SAR, or multi-temporal MERLIN establish a clean-SAR interpretation for the local fixed-input target?
- **Outcome:** Noise2Noise Eq. 6 gives an indexed clean-target condition; in population form the noisy-target and clean-target squared-loss Bayes estimators align when E[Z|X]=E[S|X]. Target noise with zero conditional mean given S and independent of X given S is sufficient. The local shared-observation condition remains unidentified; see D-2026-10-25-178.
- **Record:** [audit report](../../experiments/literature_audits/noisy_pseudo_target_n2n_sar2sar_boundary_20261025_v1/report.md) and [access manifest](../../experiments/literature_audits/noisy_pseudo_target_n2n_sar2sar_boundary_20261025_v1/access_manifest.json).
- **Boundary:** No local data/result payload, split identity, test asset/metric, run, checkpoint, or calculation was accessed. Audit flags are false/false; cumulative project flags remain true/false. A truncated research-text search surfaced historical development prose that was excluded and not used. B1 remains `not_promoted`; B2/B3 remain gated.

## noise2noise_population_mean_clarification_20261025_v1 — population-risk clarification (2026-10-25; no experiment)

- **Question:** Does Noise2Noise's indexed clean-target condition mean a population target conditional mean equals each sample-specific latent signal?
- **Outcome:** Eq. 6 uses indexed-target notation. For population squared-loss estimators, noisy-target and clean-target predictors align when E[Z|X]=E[S|X]; target noise with zero conditional mean given S and independent of X given S is sufficient. The local LR and HR pseudo-target still share one source observation, and the clean-target relation remains unestablished.
- **Record:** [protocol](../../experiments/literature_audits/noise2noise_population_mean_clarification_20261025_v1/protocol.md), [report](../../experiments/literature_audits/noise2noise_population_mean_clarification_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/noise2noise_population_mean_clarification_20261025_v1/access_manifest.json).
- **Boundary:** Source-code and primary-paper theory review only. No local data/result/test payload, checkpoint, run, model, metric, or gate was accessed or changed. Audit test flags are false/false; cumulative project flags remain true/false.
## noisier2noise_known_noise_boundary_20261025_v1 — known-noise literature boundary (2026-10-25; no experiment)

- **Question:** Does Noisier2Noise justify single-observation clean-SAR recovery for the local normalized PNG pipeline?
- **Outcome:** The method requires a known, sampleable source-noise law for its synthetic augmentation and correction. Its multiplicative special case is Bernoulli masking; local source noise is not characterized, and the LR/HR pseudo-target share one SAR observation. No local use or clean-SAR interpretation is supported.
- **Record:** [protocol](../../experiments/literature_audits/noisier2noise_known_noise_boundary_20261025_v1/protocol.md), [report](../../experiments/literature_audits/noisier2noise_known_noise_boundary_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/noisier2noise_known_noise_boundary_20261025_v1/access_manifest.json).
- **Boundary:** No data/result/test payload, metric, checkpoint, run, model, or gate changed. Three truncated local-document outputs were excluded and no surfaced result was used. Audit test flags are false/false; cumulative project flags remain true/false. B1 remains `not_promoted`; B2/B3 remain gated.

## ogrn_publisher_abstract_boundary_20261025_v1 — publisher-abstract prior-art review (2026-10-25; no experiment)

- **Question:** Does OGRN demonstrate inference-time optical guidance or joint SAR despeckling/SR?
- **Outcome:** The publisher abstract says optical-derived features guide training, while test-time inference uses SAR only. It is SR-only. Sen1-2 and QXS are named, but the split, target, degradation, metrics, and code are not specified in accessible primary text.
- **Record:** [protocol](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/protocol.md), [report](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/access_manifest.json).
- **Boundary:** Publisher abstract only; full-text routes failed. No local experiment, data, result, model, or gate changed. Audit test flags are false/false; cumulative flags remain true/false.

## ogsrn_publisher_abstract_boundary_20261025_v1 — publisher-source review (2026-10-25; no experiment)

- **Question:** Does OGSRN establish inference-time optical guidance, joint SAR despeckling/SR, or residual-mismatch fallback, and is its protocol available through official full-text routes?
- **Outcome:** The SciOpen publisher abstract describes SRUN plus SORTN with optical-space feedback during training and SAR-only inference. It is optical-guided SAR SR prior art; the abstract does not specify dataset/version, target, degradation, split, metrics, implementation, or geometric robustness. SciOpen exposes a PDF-viewer prompt but no PDF body/link in the available text route; direct ScienceDirect opening returned HTTP 403. A GitHub community repository claims an OGSRN implementation, but its owner is not a listed paper author and no publisher/author link was found; see [D-2026-10-25-184](DECISION_LOG.md#d-2026-10-25-184--classify-the-ogsrn-github-repository-as-unverified-community-code).
- **Record:** [protocol](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/protocol.md), [report](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/access_manifest.json).
- **Boundary:** The abstract's qualitative performance statement and community README claims were surfaced but not used. No code, weights, or data were downloaded; no local data, test payload, metric, checkpoint, run, model, or gate changed. Audit test flags are false/false; cumulative flags remain true/false.

## oma_ssr_author_code_route_20261025_v1 — publisher and code-route screen (2026-10-25; no experiment)

- **Question:** Is there a public author- or publisher-linked OMA-SSR implementation suitable for reproduction?
- **Outcome:** The official Wiley article gives data availability as “available from the corresponding author on reasonable request.” The publisher HTML exposes no GitHub/code or supplementary link, and exact-title/DOI searches did not verify an author-linked code repository. This does not prove no code exists elsewhere; see [D-2026-10-25-185](DECISION_LOG.md#d-2026-10-25-185--record-oma-ssr-code-route-as-unverified).
- **Record:** [protocol](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/protocol.md), [report](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/report.md), and [access manifest](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/access_manifest.json).
- **Boundary:** Publisher performance tables surfaced but were not used. No code, data, test payload, metric, checkpoint, run, or model was accessed; no gate changed. Audit-level flags are false/false; cumulative flags remain true/false.

## fixed_dev2k_source_group_schema_20261025_v1 — metadata-only source-group audit (2026-10-25; no experiment)

- **Question:** Do fixed dev2k train/dev artifacts contain parent-scene, acquisition, or source-group identifiers for a prospective split?
- **Outcome:** The packs store per-file SAR/optical paths and hashes, but their CSV schemas have no source-group columns. The referenced split-manifest schema exposes pair identities and split labels plus a group-leakage threshold; it does not identify source groups. The fixed artifacts alone cannot establish group-disjointness; see [D-2026-10-25-186](DECISION_LOG.md#d-2026-10-25-186--fixed-dev2k-manifests-do-not-provide-source-groups).
- **Record:** [protocol](../../experiments/data_audits/fixed_dev2k_source_group_schema_20261025_v1/protocol.md), [report](../../experiments/data_audits/fixed_dev2k_source_group_schema_20261025_v1/report.md), and [access manifest](../../experiments/data_audits/fixed_dev2k_source_group_schema_20261025_v1/access_manifest.json).
- **Boundary:** Split-manifest membership metadata was in scope; only its schema was inspected. No test member value, payload, metric, checkpoint, run, or model was used. Audit `test_set_accessed=true` (metadata only), `test_set_used=false`; cumulative flags remain true/false.


## research_goal_text_read_exposure_20261025_v2 — text-read exposure disclosure (2026-10-25; no experiment)

- **Question:** Were truncated local-text outputs excluded from the continuing research review, and was the supplied long-form research plan read completely?
- **Outcome:** The cumulative audit records 25 grouped output events. The supplied 2,065-line plan was subsequently reread completely in six bounded, nontruncated slices; unseen spans from the other capped searches and excerpt remain excluded. No surfaced value from a truncated output was used; see [D-2026-10-25-187](DECISION_LOG.md#d-2026-10-25-187--extend-truncated-output-disclosure-after-reading-the-research-plan).
- **Record:** [report](../../experiments/data_audits/research_goal_text_read_exposure_20261025_v2/report.md), [access manifest](../../experiments/data_audits/research_goal_text_read_exposure_20261025_v2/access_manifest.json), and [decision](DECISION_LOG.md#d-2026-10-25-187--extend-truncated-output-disclosure-after-reading-the-research-plan).
- **Boundary:** Documentation-text only. No test payload, raw metric, checkpoint, run, or model was opened or used. No experiment, selection, or gate changed. Audit flags are false/false; cumulative project flags remain true/false.

## sen12ts_temporal_replicate_viability_20261025_v1 — repeated-observation metadata screen (2026-10-25; no experiment)

- **Question:** Are repeated public S1/S2 observations available at the frozen Sumatra candidate centers for a future temporal pseudo-reference study?
- **Outcome:** The listing contains 2,716 observations at 2,000 centers; 350 have repeat S1 dates with the same track/lookpass proxy. Expanding the old representative split would share acquisition/date proxies across train/development, so it cannot support temporal selection or pixel analysis.
- **Record:** [protocol](../../experiments/data_audits/sen12ts_temporal_replicate_viability_20261025_v1/protocol.md), [report](../../experiments/data_audits/sen12ts_temporal_replicate_viability_20261025_v1/report.md), [summary](../../experiments/data_audits/sen12ts_temporal_replicate_viability_20261025_v1/results/temporal_replicate_summary.json), [access manifest](../../experiments/data_audits/sen12ts_temporal_replicate_viability_20261025_v1/results/access_manifest.json), and [D-2026-10-25-189](DECISION_LOG.md#d-2026-10-25-189--keep-the-sen12ts-temporal-route-blocked-after-metadata-only-screens).
- **Boundary:** Listing metadata only. No pixels, test data, training, inference, selection, or gate change.

## sen12ts_temporal_linkage_split_20261025_v1 — prospective metadata grouping attempt (2026-10-25; no experiment)

- **Question:** Does the expanded temporal cohort yield at least 30 disjoint proxy components under the predeclared center/S1/S2 linkage rule?
- **Outcome:** Four components were found, below the 30-component threshold. No split or component assignment was generated.
- **Record:** [protocol](../../experiments/data_audits/sen12ts_temporal_linkage_split_20261025_v1/protocol.md), [report](../../experiments/data_audits/sen12ts_temporal_linkage_split_20261025_v1/report.md), [summary](../../experiments/data_audits/sen12ts_temporal_linkage_split_20261025_v1/results/summary.json), [access manifest](../../experiments/data_audits/sen12ts_temporal_linkage_split_20261025_v1/results/access_manifest.json), and [D-2026-10-25-189](DECISION_LOG.md#d-2026-10-25-189--keep-the-sen12ts-temporal-route-blocked-after-metadata-only-screens).
- **Boundary:** Observable metadata proxies are not authoritative source IDs. No pixel or test data access; B1/B2/B3 unchanged.

## sen12ts_temporal_header_source_id_screen_20261025_v1 — bounded header-tag attempt (2026-10-25; incomplete access)

- **Question:** Do the frozen public GeoTIFF headers expose candidate source/product ID tags?
- **Outcome:** All 5,432 range requests were blocked by local `PermissionError` before any bytes were received. No tag findings are available; the audit must not be described as finding no IDs.
- **Record:** [protocol](../../experiments/data_audits/sen12ts_temporal_header_source_id_screen_20261025_v1/protocol.md), [report](../../experiments/data_audits/sen12ts_temporal_header_source_id_screen_20261025_v1/report.md), [summary](../../experiments/data_audits/sen12ts_temporal_header_source_id_screen_20261025_v1/results/tag_summary.json), [access manifest](../../experiments/data_audits/sen12ts_temporal_header_source_id_screen_20261025_v1/results/access_manifest.json), and [D-2026-10-25-189](DECISION_LOG.md#d-2026-10-25-189--keep-the-sen12ts-temporal-route-blocked-after-metadata-only-screens).
- **Boundary:** No response bytes, tag values, pixels, test data, split, or model-selection activity.

## sen12ts_official_tile_group_split_20261025_v1 — documented geographic tile grouping (2026-10-25; metadata-only)

- **Question:** Can all dated SEN12TS observations at one officially documented spatial tile be kept together in a prospective train/development partition?
- **Outcome:** The official filename contract matched all 2,716 exact candidate pairs at 2,000 frozen centers. A deterministic 1,600/400 tile split has 2,156 train and 560 development rows, zero tile crossover, and minimum prior footprint clearance of 1,225.281 m. There are 79 shared S1 date/track/lookpass proxies and 53 shared S2-date proxies; acquisition independence is not established.
- **Record:** [protocol](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/protocol.md), [report](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/report.md), [group manifest](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/results/tile_group_manifest.csv), [row manifest](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/results/observation_split_manifest.csv), [summary](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/results/summary.json), and [D-2026-10-26-001](DECISION_LOG.md#d-2026-10-26-001--accept-the-documented-sen12ts-tile-group-for-exploratory-splitting).
- **Boundary:** Local public metadata only; no raster pixels, test partition, training, inference, or gate change. This does not establish acquisition-disjointness or local SEN1-2 lineage.

## sen12ts_stack_window_overlap_20261026_v1 — S1 epoch-window metadata screen (2026-10-26; no experiment)

- **Question:** Do the four documented S1 dates within each triplet and repeated triplets provide non-overlapping date windows within the frozen geographic tile split?
- **Outcome:** The 2,716 rows yield 10,864 derived epoch slots and 10,399 unique tile/track/lookpass/date proxies, leaving 465 repeated slots. Of 591 same-tile/track/lookpass row pairs, 213 share one to three dates. 209 of 239 global date/track/lookpass proxies cross train/development. These date proxies do not establish independent acquisitions or speckle.
- **Record:** [protocol](../../experiments/data_audits/sen12ts_stack_window_overlap_20261026_v1/protocol.md), [report](../../experiments/data_audits/sen12ts_stack_window_overlap_20261026_v1/report.md), [summary](../../experiments/data_audits/sen12ts_stack_window_overlap_20261026_v1/results/summary.json), [access manifest](../../experiments/data_audits/sen12ts_stack_window_overlap_20261026_v1/results/access_manifest.json), and [D-2026-10-26-002](DECISION_LOG.md#d-2026-10-26-002--treat-sen12ts-stack-windows-as-overlapping-acquisition-proxies).
- **Boundary:** Dataset documentation plus local listing and split metadata only; no pixels, test set, training, inference, selection, or gate change.

## Repository artifact cleanup review (2026-10-26)

- **Audit:** [artifact cleanup recheck](../../experiments/data_audits/artifact_cleanup_recheck_20261026_v1/report.md).
- **Disposition:** zero additional files removed; no experiment `.npy`/`.npz` result arrays remain. Runtime dependencies, completed A2 comparison checkpoints, source/configuration records, metrics, audits, and high-value negative results were retained.
- **Research status:** B1 unchanged at `not_promoted`; B2/B3 remain gated.

## Source-route audit — SEN12MS-CR author archive (2026-10-26)

- **Audit:** [author archive route screen](../../experiments/literature_audits/sen12mscr_author_archive_route_20261026_v1/report.md).
- **Finding:** Author code links seasonal TUM bundles; no train-only manifest/version crosswalk was verified. No data downloaded.
- **Use:** Candidate remains conditional; no model selection or gate change.

## ogsrn_author_group_patent_route_20261026_v1 — patent-linked prior-art review (2026-10-26; no experiment)

- **Question:** Does an author-group patent clarify the published OGSRN paper's test-time inputs, and can it be treated as the same implementation?
- **Evidence:** CN113421190B lists inventors matching the CJA OGSRN author roster and describes HR-optical-assisted training with LR/HR SAR; its test embodiment uses bicubic-upsampled, channel-expanded LR SAR in the optical branch. The sources inspected do not explicitly equate patent and article implementation. The separate 2022 JARS OGRN SSRN record lists a different fourth author.
- **Boundary:** Patent/abstract literature review only. No full article body, dataset, code, local data, test payload, metric, checkpoint, or run was accessed. No model or gate decision changed. See [D-2026-10-26-022](DECISION_LOG.md#d-2026-10-26-022--treat-cn113421190b-as-related-ogsrn-patent-evidence) and the [audit](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md).

## Source-lineage and alternate-cohort route recheck (2026-10-26; metadata only)

- **Question:** Did the accessible official release pages and neighboring cache metadata provide an authoritative source-group map for local numeric 2k data or a usable alternative cohort?
- **Sources:** Official SEN1-2 and QXS-SAROPT publications/repository, the current SEN12TP Zenodo record and publisher paper, local train/validation source-route audits, and sibling-repository metadata.
- **Outcome:** No local parent-scene/acquisition crosswalk was found. The official QXS repository exposes no file-level source map. The SEN12TP Zenodo page is reachable, but the dataset license is blank and its train/test/archive description conflicts with the paper’s spatial 80/10/10 split. No archive or scene-metadata file was downloaded or opened.
- **Boundary:** This was source/readiness review only: no model run, training, split construction, model selection, test identity access, test payload access, or gate change. B1 remains not promoted; B2/B3 remain gated.
- **Artifacts:** [Route audit](../../experiments/data_audits/source_lineage_routes_recheck_20261026_v1/report.md) and [D-2026-10-26-054](DECISION_LOG.md#d-2026-10-26-054--keep-local-lineage-unresolved-and-sen12tp-unadopted).

## Gaya et al. Fuse-MERLIN full-text audit and matrix coverage (2026-10-26)

- **Question:** What does the official GRETSI 2023 full paper establish about optical-guided SAR restoration?
- **Evidence:** The paper describes early and intermediate optical fusion for self-supervised complex-SAR MERLIN despeckling on one TerraSAR-X/Pléiades scene. It reports qualitative observations and explicitly states that quantitative validation has not been performed.
- **Outcome:** Add as a current-evidence prior-art row; do not infer joint SR, systematic mismatch robustness, or quantitative local benefit. The distinct IGARSS 2024 paper's version relationship remains unresolved.
- **Boundary:** Matrix membership remains unknown; counts are 97 total, 74 confirmed historical and 23 unresolved additions. No local data, test payload, metrics, checkpoint, or experiment was accessed or changed. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-26-116 and the [source audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md).

## B1 v2 secondary-metric source-group diagnostic (2026-10-26)

- **Question:** Do EPI, edge-amplitude, and bright-target metrics move consistently across the 15 fixed validation source groups?
- **Evidence:** A read-only aggregation of the frozen per-group summary (45 group-seed rows, seeds 0–2) found EPI correlation increased in 15/15 groups; edge-strength and bright-peak ratio error increased in 15/15; mean TBR ratio error increased in 15/15 group means (14/15 groups had that sign in every seed); bright integrated-energy ratio error and bright-region MAE increased in 12/15 and decreased in 3/15.
- **Outcome:** This adds a descriptive cross-metric trade-off to the existing PSNR/SSIM group diagnostic. It does not identify a cause or support an optical-benefit claim. The B1 v2 gate remains `not_promoted`; no B2/B3 work is authorized.
- **Boundary:** One fall-only ROI-season, fine pixel co-registration unverified, and noisy 8-bit pseudo-targets. Source group IDs are not emitted or ranked. Both run-level test flags are false; no test payload was opened. Reproduction: `experiments/diagnostics/b1_v2_multimetric_group_tradeoffs_20261026_v1/command.txt`. See the [report](../../experiments/diagnostics/b1_v2_multimetric_group_tradeoffs_20261026_v1/report.md).

## qxs_author_issue_lineage_recheck_20261026_v1 — official source-route audit (2026-10-26; no experiment)

- **Question:** Do current author-controlled QXS-SAROPT pages expose a local-cache crosswalk, file-level city/source groups, or an alternate public mirror?
- **Outcome:** No. The author repository links to a survey; issue #3 records that city labels were not mapped at the time of the 2021 reply; issue #4 says the original SAR imagery could not be shared and only authorized processed/cropped samples were released; issue #13's mirror request has no maintainer response on the page reviewed. These facts do not identify the local numeric 2k files as QXS or establish their rights.
- **Boundary:** No form submission, download, local data, test payload, checkpoint, or metric access. No experiment or gate changed.
- **Record:** [report](../../experiments/data_audits/qxs_author_issue_lineage_recheck_20261026_v1/report.md), [access manifest](../../experiments/data_audits/qxs_author_issue_lineage_recheck_20261026_v1/access_manifest.json), and [D-2026-10-26-126](DECISION_LOG.md#d-2026-10-26-126--recheck-official-qxs-saropt-author-source-routes).

## original_matrix_and_source_routes_20261026_v1 — matrix/source recovery routes (2026-10-26; no experiment)

- **Question:** Did a newly reachable local or connected source recover the original 119-row literature inventory or an authoritative local-file source crosswalk?
- **Outcome:** No. The Zotero API and Drive search routes were unavailable, official TUM SEN1-2 record/PDF opens were rate-limited, and the bounded directory-name scan did not identify local files. SMSDNet is already present with historical-membership evidence; the matrix remains at 97 rows (74 confirmed, 23 unresolved additions).
- **Boundary:** Metadata and source-route review only. No local dataset, test payload, metric, checkpoint, experiment, or gate changed. Task-level test flags are false/false; cumulative project flags remain `test_set_accessed=true`, `test_set_used=false`.
- **Record:** [route report](../../experiments/data_audits/original_matrix_and_source_routes_20261026_v1/report.md), [manifest](../../experiments/data_audits/original_matrix_and_source_routes_20261026_v1/manifest.json), and [D-2026-10-26-135](DECISION_LOG.md#d-2026-10-26-135--record-unavailable-original-matrix-and-data-lineage-routes).

## b1_v2_verification_record_hash_recheck_20261026_v1 — verifier-record binding audit (2026-10-26; metadata only)

- **Question:** Do the retained per-split verifier records match the raw hashes bound by the source-aligned B1 v2 input record?
- **Outcome:** No. The expected raw verifier hashes differ for both train and validation. The run-bound config, `files.csv`, manifest, and split hashes match their retained files; all 1,400 rebuilt validation payload hashes also match run-bound `files.csv`. The run commands state that full per-file verifier outputs were temporary and removed after hashing. This is the likely explanation for the mismatch, but cannot be confirmed without those original bytes.
- **Record:** [metadata-only audit](../../experiments/data_audits/b1_v2_verification_record_hash_recheck_20261026_v1/report.md).
- **Boundary:** No image, array, checkpoint, metric file, or test payload was opened. No run, result, split, model, selection, or gate changed.

## b1_v2_optical_candidate_head_parameter_audit_20261026_v1 — static EMA parameter audit (2026-10-26; no inference)

- **Question:** Do the trained C2 EMA checkpoints' residual and reliability projections offer a structural clue for the low paired-output sensitivity?
- **Evidence:** All three `best.pth` hashes match their run manifests. The shared config uses `MMCANetV2`, cross-attention, no alignment, and residual/output scales 1.0. The zero-initialized final residual projection has EMA weight RMS 0.00450–0.00511 across seeds; the reliability-head output projection RMS is 0.1378–0.1671.
- **Outcome:** The final residual projection remaining small is consistent with limited optical-conditioned high-pass updates, but does not establish optical independence or explain per-input reliability. No gate activations, images, metrics, or new predictions were evaluated. This is descriptive checkpoint evidence, not model-selection evidence.
- **Record:** [parameter audit](../../experiments/diagnostics/b1_v2_optical_candidate_head_parameter_audit_20261026_v1/report.md).
- **Boundary:** Read-only CPU checkpoint inspection. No training, inference, test access, selection, or gate change.

## cdse_slc_iw_aoi_group_feasibility_20261026_v1 — metadata feasibility screen (2026-10-26; no experiment)

- **Question:** Does the frozen public CDSE route expose enough complete Sentinel-1 IW SLC VV+VH metadata candidates to justify a separate protocol review with the already-screened Sentinel-2 L2A metadata?
- **Protocol:** Twelve fixed, spatially separated AOIs and the 2025-11-01 through 2025-12-31 UTC window; full AOI containment; exact `IW_SLC__1S`, IW, VV+VH; reused compact S2 metadata bound by its source-results hash; S2 cloud cover ≤20%; absolute time gap ≤24 hours; group by shared datatake ID or intersecting AOI; at least eight disconnected components permits protocol review only.
- **Outcome:** 12/12 SLC queries returned HTTP 200 with no pagination. There were 227 compact SLC items and 50 deduplicated pairs across 10 active AOIs; two AOIs had no pair. The resulting 10 metadata-defined components meet the frozen floor. No shared-datatake or AOI-overlap edges connected the active AOIs.
- **Interpretation:** Candidate-component count is not verified source-group or geographic independence, and metadata does not establish pair validity, rights, calibration, geolocation, target independence, or an SLC target protocol. Proceed only to asset-free protocol review.
- **Boundary:** No assets, sidecars, pixels, local data, split, test payload, model, training, inference, metric, selection, or gate changed. Task flags are false/false; cumulative project flags remain true/false. B1 remains `not_promoted`; B2/B3 remain gated.
- **Record:** [report](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/report.md), [protocol](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol.md), and [access manifest](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/access_manifest.json). See D-2026-10-26-146.

## cdse_slc_iw_aoi_group_feasibility_20261026_v1 — asset-free protocol review (2026-10-26; no experiment)

- **Question:** Does the candidate S1 IW SLC / S2 L2A metadata cohort support a defensible, self-supervised SLC target and product-processing route?
- **Outcome:** No. General Sentinel data rights are favorable, but the MERLIN paper explicitly rules out direct S1 TOPSAR/IW use, requires deramping, and leaves additional I/Q decorrelation preprocessing unresolved. Product annotations, orbit, calibrated L1 data, ETAD, calibration domain, burst masks, and geometry/registration handling must be fixed and validated first.
- **Interpretation:** The 50 metadata pairs across 10 AOIs justify only this asset-free review. The ≤20% S2 cloud and ≤24-hour time-gap filters do not prove cloud-free correspondence, registration, or target independence. No target is approved.
- **Boundary:** No assets, sidecars, pixels, local dataset, split, test payload, model, training, inference, metric, selection, or gate changed. Task flags false/false; cumulative project flags true/false. B1 remains `not_promoted`; B2/B3 remain gated.
- **Record:** [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md), [metadata screen](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/report.md), D-2026-10-26-147.

## b1_v2_optical_gate_activation_20261026_v1 — optical gate activation diagnostic (2026-10-26; interpretation-only)

- **Question:** Do the existing C2 EMA checkpoints change reliability-gate activation or effective high-pass residual contribution when matched optical is replaced by the frozen cross-source-group permutation?
- **Protocol:** Predeclared in [`protocol.json`](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/protocol.json). Uses 200 fixed validation identities, 15 source groups, seeds 0–2, CPU inference, strict checkpoint loading, `return_aux=True`, and the exact prior optical donor map. No targets, masks, quality metrics, training, selection, or test data enter the activation measurements.
- **Outcome:** Group-balanced reliability means were 0.4667 matched and 0.4667 permuted; paired reliability-map RMS difference was 0.00892. Effective contribution/base RMS was 0.1291% matched and 0.1290% permuted; paired contribution-map RMS difference was 8.15e-5 in normalized representation units. The SAR base was identical across conditions. This is consistent with a small residual correction, not evidence that optical is useless or safely rejected.
- **Integrity and cleanup:** 1,400 validation payload hashes matched the run-bound input records. The separate whole-file `verification.json` digest mismatch remains unresolved. The temporary staging/fixed-input tree and compile cache were removed (5,051 files; 589,510,000 bytes); the diagnostic retains only protocol, code, scalar outputs, manifests, commands, and report.
- **Boundary:** Development validation from one fall ROI-season; descriptive group summaries only, no inferential threshold. No gate or phase decision changed. B1 remains `not_promoted`; B2/B3 remain gated; task test flags are false/false.
- **Artifacts:** [report](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/report.md), [manifest](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/manifest.json), [per-identity scalar outputs](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/per_sample.csv), and [group summary](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/group_summary.csv).

## b1_v2_optical_cross_attention_ablation_20261026_v1 — interrupted interpretation-only launch (2026-10-26; cleanup 2026-10-27)

- **Question:** Does bypassing only the two cross-attention residual updates change the candidate's outputs while direct optical fusion remains active?
- **Protocol:** Frozen in [protocol.json](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261026_v1/protocol.json); 200 existing development-validation identities, three C2 EMA seeds, matched and frozen-permuted optical, full and attention-bypass forwards, CPU only. No target or mask values, quality metrics, training, selection, or test data were intended or used.
- **Outcome:** The launch did not complete. The process was absent when checked, the manifest was left `running`, and no per-sample table, source-folder summary, summary JSON, or report was produced. The exit code is unavailable. Do not interpret this attempt as a scientific result.
- **Cleanup:** Removed the owner-marked temporary source/fixed-input/bytecode tree (5,053 files; 589,524,809 logical bytes) and the empty/warning-only execution logs. Retained the protocol, command, analysis script, source staging manifest, interrupted manifest, and [cleanup record](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261026_v1/cleanup.json).
- **Boundary:** No test payload or metric file was opened or used. No model, checkpoint, gate, or phase decision changed. B1 remains `not_promoted` / `stop_at_b1`; B2/B3 remain gated. Task test flags are false/false.
- **Decision:** [D-2026-10-27-001](DECISION_LOG.md#d-2026-10-27-001--confirm-approved-run-interfaces-and-clean-interrupted-diagnostic-staging).

## b1_v2_optical_cross_attention_ablation_20261027_v1 — completed interpretation-only diagnostic (2026-10-27)

- **Question:** How much do the two optical cross-attention residual updates change fixed-checkpoint predictions when direct optical fusion remains active, and does the effect differ between matched and frozen-permuted optical?
- **Protocol:** [Versioned protocol](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261027_v1/protocol.json); 200 development-validation identities, three C2 EMA seeds, 15 recorded source folders, CPU, batch size 2. No target/mask values or quality metrics; no training, selection, or test access. The v1 incomplete attempt is separately retained as interrupted.
- **Outcome:** 1,200 per-sample rows and 30 equal-weight source-folder summaries; 200 unique identities; 400 rows per seed; 600 rows per optical condition; 15 source folders. All numeric fields were finite and all eight output hashes verified. Prediction delta RMS was 1.8959758e-05 matched and 1.8999156e-05 permuted; relative RMS 2.4610249e-05 and 2.4666665e-05. The internal LR/HR attention updates remained nonzero.
- **Interpretation boundary:** This is sensitivity to bypassing trained modules, not an image-quality comparison. Direct optical fusion remained active. The result neither proves optical utility nor absence of utility, and it does not support robustness/generalization or model selection on the overlapping validation cohort.
- **Integrity and cleanup:** Manifest records task flags false/false and no target, metric, training, or selection use. The bound whole-file `verification.json` digest mismatch remains unresolved. Removed the owner-marked temporary input/compile tree and low-value execution logs; retained scalar outputs and audit artifacts. See [cleanup record](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261027_v1/cleanup.json).
- **Decision:** [D-2026-10-27-002](DECISION_LOG.md#d-2026-10-27-002--record-interpretation-only-cross-attention-bypass-diagnostic).

## b1_v2_optical_direct_fusion_bypass_20261027_v1 — preflight failure (2026-10-27)

- **Question:** With cross-attention active, how sensitive are fixed predictions to the direct optical-HR and context channels at the fusion stem?
- **Outcome:** The analyzer failed before model inference because the command's fixed directory differed from the run-bound absolute path. No validation payload hashes were checked by the analyzer and no per-sample result was produced. This is an interface failure, not scientific evidence.
- **Cleanup:** Removed the owner-marked staging/fixed-input tree (5,053 files; 589,526,275 logical bytes), raw logs, bytecode, and duplicate attempt provenance copies. Retained the concise failed manifest and cleanup record.
- **Records:** [failure manifest](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v1/runs/attempt_001/manifest.json), [cleanup](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v1/cleanup.json), D-2026-10-27-003.

## b1_v2_optical_direct_fusion_bypass_20261027_v2 — completed interpretation-only diagnostic (2026-10-27)

- **Question:** With both trained cross-attention updates active, how much do the direct optical-HR and optical-context fusion-stem inputs change fixed-checkpoint predictions, and does the effect differ under matched versus frozen-permuted optical?
- **Protocol:** [Predeclared v2](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/protocol.json); 200 validation identities, seeds 0–2, 15 recorded source folders, CPU inference. The fusion-stem pre-hook zeroed only optical-HR/context channel groups; SAR-HR/LR-up and both cross-attention updates remained active. No targets/masks, quality metrics, training, selection, gate tuning, or test data.
- **Outcome:** 1,200 per-sample rows, 30 folder summaries, 400 rows per seed, 600 per optical condition; all values finite and all nine output hashes matched. Folder-equal prediction delta RMS was 1.86920e-4 matched and 1.86906e-4 permuted; relative RMS was 0.0002428 (0.02428%) for both. SAR-base delta was zero. Fusion feature relative RMS was about 0.365 under both pairings. These are out-of-training-distribution sensitivity measures, not image-quality or optical-utility evidence. The separate direct-channel and attention interventions are not additive.
- **Integrity/cleanup:** Expected validation payload hashes were checked. The whole-file `verification.json` digest mismatch remains unresolved. Removed the temporary tree (5,053 files; 589,526,177 logical bytes), raw logs (449 bytes), and 38 bytecode cache entries; retained scalar outputs and audit artifacts.
- **Boundary:** B1 remains `not_promoted` / `stop_at_b1`; B2/B3 remain gated; task flags are false/false. Future model-forward runs should use GPU after a successful availability check, per the user's instruction.
- **Records:** [report](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/report.md), [manifest](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/manifest.json), [per-sample scalars](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/per_sample.csv), [folder summaries](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/source_folder_summary.csv), [cleanup](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/cleanup.json), D-2026-10-27-004.

## Related-work matrix Git object recovery (2026-10-27; metadata-only audit)

Checked the current Git index/history and all four unreachable blob objects for a prior 12-column matrix. The matrix path is untracked, has zero reachable path revisions, and no blob matched its header/row shape. No original inventory was recovered and no matrix row, experiment, metric, or gate changed. The 99-row current inventory remains incomplete (74 historical-membership confirmed; 25 additions unresolved). See D-2026-10-27-005 and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).



## Related-work matrix membership reconciliation (2026-10-27; documentation audit)

Two existing rows were reclassified from unresolved to confirmed historical membership using pre-corruption inclusion decisions: GCOANet (D-2026-10-18-08 and D-2026-10-23-26) and Gaya et al.'s IGARSS 2024 paper (D-2026-10-25-04 and D-2026-10-25-65). The current matrix remains 99 rows and 12 columns; the preserved 46-row prefix is byte-identical. It now has 76 confirmed members and 23 unresolved additions. The pre-corruption 119-row read predates six additions by D-2026-10-25-46, so 125 is the known lower bound; the exact final count is unknown. At least 26 original entries remain unaccounted for. See D-2026-10-27-006 and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).

## Literature screen — dual-channel SAR–optical registration (2026-10-27)

- **Decision:** [D-2026-10-27-008](DECISION_LOG.md#d-2026-10-27-008--add-dual-channel-sar-optical-registration-as-current-evidence).
- **Source:** Zhang and Wang, AIP Advances 16(10), 105202 (2026), DOI [10.1063/5.0343146](https://doi.org/10.1063/5.0343146).
- **Evidence:** Publisher-text screen backed by the existing [audit](../../experiments/literature_audits/dual_channel_sar_optical_registration_20261025_v1/report.md); no model run or local data access.
- **Interpretation:** Adjacent learned registration prior art only. Parent groups and transfer reference identities are unspecified; no restoration error-propagation, fallback, or local mismatch-threshold evidence. Historical matrix membership remains unknown.


## Literature screen — SAG-CR and CGD-Net matrix completion (2026-10-27)

- **Type:** bounded publisher/author-source review and matrix maintenance; no model run.
- **Outcome:** appended two reverse-task current-evidence rows to the 12-column related-work matrix. SAG-CR reconstructs optical cloud-free imagery with supervised align-guided fusion; CGD-Net reconstructs optical reflectance with cloud-mask/prior reliability gating and reports registration/mismatch stress categories.
- **Interpretation:** adjacent prior art only; neither establishes SAR-output restoration fidelity, inference-time optical utility for SAR restoration, or SAR-only fallback.
- **Boundary:** no local data, checkpoint, metric file, test payload, or experiment was accessed. No author performance values were copied. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit](../../experiments/literature_audits/related_work_matrix_sagcr_cgdnet_completion_20261027_v1/report.md).


## Native SOTANet cache partition vs frozen v2 cohort (2026-10-27; metadata audit)

- **Type:** read-only train/val filename/group metadata audit; no model run.
- **Finding:** all 30 prefix groups occur in both native cache partitions; no exact basename is duplicated across them. The frozen v2 cohort uses a separate 15/15 group allocation, with all 1,800 selected filenames mapping to their recorded group IDs; its existing official-source audit verifies selected SAR/optical pairs.
- **Boundary:** filename-group separation does not prove acquisition/geographic independence; v2 validation is fall-only. This does not authenticate the full native cache or resolve the separate numeric-name 2k subset. B1 remains `not_promoted`; B2/B3 remain gated. See the [audit](../../experiments/data_audits/sen12_native_partition_group_overlap_20261027_v1/report.md).

## Literature-only audit — reverse-task cloud-removal alignment (2026-10-27)

- **Record:** `reverse_task_cloud_removal_alignment_review_20261027_v1`
- **Type/status:** Primary-source literature review; no local run.
- **Sources:** Publisher-indexed SAG-CR/CGD-Net text; official CGD-Net README/architecture/training entry point; publisher-linked SAG-CR repository.
- **Outcome boundary:** Both sources reconstruct optical imagery. No local performance, B1 promotion, or B2/B3 evidence. Task test flags: `accessed=false`, `used=false`; cumulative project flags remain governed by the exposure audit.
- **Audit:** [report](../../experiments/literature_audits/reverse_task_cloud_removal_alignment_review_20261027_v1/report.md) and [access manifest](../../experiments/literature_audits/reverse_task_cloud_removal_alignment_review_20261027_v1/access_manifest.json).


## 2026-10-27 — Direct-fusion device records

- `b1_v2_optical_direct_fusion_bypass_20261027_v2`: complete under its predeclared CPU protocol; test flags false/false. Preserve its run outputs and manifest as historical evidence.
- `b1_v2_optical_direct_fusion_bypass_20261027_v3`: completed on `cuda:0` in the approved elevated profile; exact frozen input/checkpoint bindings match v2, test flags false/false, nine output hashes verified, and temporary staging/logs removed. See D-2026-10-27-014.

## b1_v2_optical_direct_fusion_bypass_20261027_v3 — CUDA hardware repeat (2026-10-27)

- **Protocol:** [Frozen CUDA-only definition](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/protocol.json); same 200 development-validation identities, three C2 EMA seeds, 15 recorded source folders, matched and frozen-permuted optical, and direct optical-HR/context bypass while both attention updates stayed active. Training budget zero; no target/mask values or quality metrics.
- **Environment:** Approved elevated AISAR execution profile; RTX 3050 Laptop GPU (4 GiB), PyTorch 2.8.0+cu128, CUDA 12.8; manifest duration 125.88 s.
- **Outcome:** 1,200 finite scalar rows and 30 folder summaries. Folder-equal prediction delta RMS was 1.8691972e-4 matched and 1.8690553e-4 permuted; relative RMS was 0.00024281485 and 0.00024282937 (about 0.02428% in each condition). Fusion-feature relative RMS was 0.3650986 and 0.3658403; SAR-base delta was zero. On the same paired rows, CPU-v2 versus GPU-v3 maximum absolute difference in prediction relative RMS was 6.49e-9; maximum folder-summary difference was 6.03e-10. No tolerance was predeclared for cross-device parity, so these are descriptive comparisons only.
- **Interpretation/boundary:** Confirms this fixed-checkpoint sensitivity readout is numerically very close across these two execution devices. It is not an independent replicate, image-quality or optical-utility result, mismatch robustness, or generalization evidence. No model selection, gate tuning, or promotion; B1 remains `not_promoted`, B2/B3 gated. Test flags false/false.
- **Integrity/cleanup:** Input-binding, fixed-manifest, source-manifest, source-protocol, optical-mapping and checkpoint records match v2; nine retained output hashes verified. Removed the owner-marked staging tree (5,053 files; 589,527,168 logical bytes) and two raw run logs after verification. The whole-file `verification.json` digest mismatch remains unresolved even though the expected validation payload hashes passed.
- **Records:** [report](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/report.md), [manifest](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/manifest.json), [per-sample scalars](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/per_sample.csv), [folder summaries](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/source_folder_summary.csv), [cleanup record](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/cleanup.json), and D-2026-10-27-014.

## IGARSS 2025 physics-inspired SAR diffusion source-route recheck (2026-10-27; literature only)

- **Type/status:** Primary-source metadata and publisher-route screen; no local experiment.
- **Verified:** Official IGARSS paper/session pages confirm title, authors, and oral slot. The page shows a manuscript label but no accessible link/body in the current reader.
- **Unverified lead:** A secondary record exposed a DOI candidate; IEEE/Crossref was inaccessible and secondary author rendering was inconsistent. No DOI, method, data, degradation, results, or code was promoted into the matrix/bibliography.
- **Boundary:** Existence-level SAR despeckling diffusion prior art only. No local data/test access, model run, gate change, or claim about optical-guided joint restoration.
- **Audit:** [route report](../../experiments/literature_audits/igarss25_physinspired_sar_diffusion_route_recheck_20261027_v1/report.md) and [access manifest](../../experiments/literature_audits/igarss25_physinspired_sar_diffusion_route_recheck_20261027_v1/access_manifest.json).

## Failed patch-oracle attempt artifact pruning (2026-10-27)

- **Runs:** `mmcanet_v2_phase1_t_patch_oracle_v2` and `v3`, both recorded as failed. The retained manifests preserve the v2 `KeyError` and v3 missing memory-preflight failure; both record `test_set_used=false`.
- **Cleanup:** Removed their four generated per-image/summary CSV result files; retained configs, commands, manifests, preflight records, and the v3 report. No CSV values were opened.
- **Record:** [cleanup report](../../experiments/data_audits/artifact_cleanup_failed_t_patch_oracle_20261027_v1/report.md) and manifest.

## sen12_dev1800_parent_roi_capacity_20261027_v1 — source-folder and top-level directory capacity

- **Question/protocol:** Summarize source-folder and top-level directory-bucket capacities from only `assignment`, `group_id`, and `roi_season` in the retained 1,800-row train/validation verification table.
- **Observation:** Train/validation have 15 source-folder groups each with no overlap. The shared top-level directory is `ROIs1970_fall`; selected capacities are `ROIs1868_summer=106` and `ROIs1970_fall=1,694`. The official supporting document defines the full directory labels as random ROI-distribution/season folders and second-level `s1_i` / `s2_i` folders as source scenes.
- **Boundary:** The audit does not contain individual-ROI or geospatial membership and cannot establish individual-ROI overlap, geographic separation, or acquisition independence. D-2026-10-27-025 corrects the original interpretation; no row counts, source assignments, metrics, or gates changed. Task flags false/false; CPU metadata only, no model forward.
- **Records:** [corrected report](../../experiments/data_audits/sen12_dev1800_parent_roi_capacity_20261027_v1/report.md), [script](../../experiments/data_audits/sen12_dev1800_parent_roi_capacity_20261027_v1/audit.py), [official supporting document](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt), and D-2026-10-27-025.

## sen1_2_scene_folder_taxonomy_20261027_v1 — source-scene folder semantics

- **Question/protocol:** Reconcile the `ROIs1970_fall` and `s1_i`/`s2_i` labels with the primary dataset paper, official supporting document, and frozen group assignment; read train/validation metadata only.
- **Observation:** The primary paper labels 1970 as the fall sampling seed and reports 78 globally distributed scenes in that distribution. The supporting document defines second-level folders as source scenes. The frozen assignment has 15 train and 15 validation source-scene folder IDs with no intersection; train contains 14 fall and one summer folder, while validation contains 15 fall folders.
- **Boundary:** The cohort does not retain per-folder coordinates, footprint geometry, acquisition IDs, or dates. This supports a source-scene-folder-disjoint development split, not a geographic-separation or acquisition-independence claim. No model forward pass, image, test identity/payload, or metric file was accessed; task flags false/false. B1 remains `not_promoted`; B2/B3 remain gated.
- **Records:** [audit report](../../experiments/data_audits/sen1_2_scene_folder_taxonomy_20261027_v1/report.md), [manifest](../../experiments/data_audits/sen1_2_scene_folder_taxonomy_20261027_v1/manifest.json), [group assignment](../../protocols/sen12_scene_dev1800_v2/group_assignment.csv), and [primary paper](https://doi.org/10.5194/isprs-annals-IV-1-141-2018).


## Frequency-enhanced joint-SAR diffusion lead — open-location/rights metadata (2026-10-28; index-only)

OpenAlex reports the IOP PDF as the only location and no repository full text, tagging it CC BY-NC-ND. Crossref's license URI redirects to IOP's general copyright/permissions page rather than an article-specific license. The direct article route remains challenge-blocked, so exact reuse terms cannot be verified. Treat this as a locator/rights risk only; no article text was used. See D-2026-10-28-016 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).

## 2026-10-28 CUDA-only Early-Fusion baseline interfaces

- **Scope:** Reusable Cascade/SFT training and fixed-inference CLIs in `Early-FusionCascadeJDSR/`.
- **Change:** `cascade_baseline.py`, `sft_baseline.py`, `train_sft_protocol.py`, `cascade_fixed_infer.py`, and `sft_fixed_infer.py` now accept only `--device cuda` and fail closed if CUDA is unavailable. Formal SFT training performs the device preflight before creating the run directory or writing a `running` manifest.
- **Verification:** Python compilation passed for all five files; a bounded source scan found no remaining `cpu`/`auto` CLI choices or CPU fallback in these interfaces. No tests, model forward pass, training, inference, data, metric, or checkpoint payload was run or accessed.
- **Boundary:** Historical CPU diagnostic scripts and sealed experiment protocols are unchanged. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-28-023.

## 2026-10-28 CUDA-only JointJDSR/OGSRN/SwinIR-Joint interfaces

- **Scope:** Reusable training/inference entrypoints in `scripts/`; protocol-specific evaluators, benchmarks, and matrices were not changed.
- **Change:** `train_joint_jdsr.py`, `infer_joint_jdsr.py`, `train_swinir_joint.py`, `infer_swinir_joint.py`, `train_ogsrn_srun.py`, `train_ogsrn_sortn.py`, and `infer_ogsrn.py` accept only `--device cuda`. Each checks CUDA before creating a run/output directory, and no code path selects CPU when CUDA is unavailable.
- **Verification:** `python -m py_compile` and a direct trailing-whitespace scan passed for all seven files; a bounded source scan found no CPU/auto CLI options or CPU fallback in these interfaces. The files are untracked, so `git diff --check` did not cover them. No test, training, inference, data, test payload, metric, or checkpoint was accessed.
- **Boundary:** The completed CPU v2 direct-fusion diagnostic is retained as historical provenance; its CUDA v3 repeat is already complete. No run was active or started in this task. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-28-024.

## 2026-10-28 QXS-SAROPT source-crosswalk route check

- **Question:** Does the author-maintained QXS-SAROPT README or primary paper expose a file-level source map that identifies the local numeric-name PNG subset?
- **Evidence:** The repository README links a questionnaire-based download route and the primary paper; it does not expose a versioned per-file manifest or numeric-name-to-scene crosswalk. The paper describes QXS-SAROPT as 20,000 GF-3/Google Earth SAR-optical patch pairs from three cities and explains its patch construction, but it does not map this project's local files to those records. See the [author repository](https://github.com/yaoxu008/QXS-SAROPT/blob/main/README.md) and [primary paper](https://arxiv.org/abs/2103.08259).
- **Boundary:** This route check does not establish QXS-SAROPT identity for the local subset. No form was submitted, archive downloaded, local image opened, or test payload accessed. No split, selection, experiment, or gate changed; B1 remains `not_promoted`, and B2/B3 remain gated.
- **Decision:** Keep the source mapping unresolved; do not repeat this route unless a new authoritative manifest or endpoint appears. See D-2026-10-28-025.

## 2026-10-28 Explicit CUDA in reusable Cascade/SFT launch commands

- **Scope:** `scripts/run_cascade_baseline.sh`, `scripts/run_sft_baseline.sh`, and `scripts/run_sft_seed_matrix.py`.
- **Change:** The two shell launchers now pass `--device cuda`; the SFT seed runner appends it after matrix-provided arguments so generated command records show and enforce CUDA selection.
- **Verification:** `python -m py_compile` passed for the runner; `bash -n` passed for both shell files; static inspection confirmed the argument placement. No launch or model/data execution occurred.
- **Boundary:** Existing helper scripts are untracked in the current checkout and remain unstaged. No test workflow, experiment protocol, result, or gate changed. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-28-026.

## 2026-10-28 Canonical degradation implementation checks (synthetic-only)

- **Scope:** Existing `tests/test_degradation_and_model.py` operator and binding cases; synthetic tensors only.
- **Verification:** Fourteen selected normalization, FFT, blur, and Gamma-statistics tests passed; the separate MMCANet/Cascade canonical-binding test and pre-speckle snapshot replay test passed. The ordinary shell has no PyTorch; AISAR has PyTorch 2.8.0+cu128 but CUDA is currently unavailable. No model forward pass was executed.
- **Interpretation:** These checks support implementation-level mathematical invariants and wrapper bindings. They do not validate calibrated SAR units, physical sensor fidelity, the local PNG source/target semantics, or model quality. No protocol, checkpoint, metric, or gate changed; B1 remains `not_promoted` and B2/B3 remain gated. See D-2026-10-28-027.

## 2026-10-28 Google Drive source-record route recheck

- **Scope:** Three metadata-only Drive searches for SEN1-2 source groups, numeric-filename mapping, and the JDSR subset-generation manifest.
- **Result:** All three connector calls failed at the backend with `Unknown tool`; zero file metadata was returned. This does not imply no matching file exists.
- **Boundary:** No Drive content, local/test image, metric file, dataset archive, model, split, or gate was accessed or changed. Keep lineage unresolved and do not retry until the connector backend changes. See D-2026-10-28-028.

## Gaya et al. IGARSS 2024 IEEE full-text route recheck (2026-10-28; literature only)

- **Question:** Did direct official IEEE Xplore routes expose the full paper after the prior DOI/abstract screen?
- **Result:** The IEEE document and stamp/PDF URLs were inaccessible through the reader. The author-associated HAL CV index points to the previously known `hal-04676452v1` record, but its record/document endpoints remain access-protected. No full text or new technical evidence was obtained.
- **Disposition:** Keep the paper at abstract-level prior-art evidence; do not infer protocol details or version continuity with Fuse-MERLIN.
- **Boundary:** No paper text, metrics, code, data, local test asset, model, or experiment was accessed; no gate changed. See D-2026-10-28-029 and the [route audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md#addendum--official-ieee-full-text-route-recheck-2026-10-28).

## Lian et al. structure-confidence registration audit — 2026-10-28 (literature only; no experiment)

- **Question:** Does this work provide relevant prior art for confidence-aware optical–SAR correspondence matching, and does it address restoration-time optical reliability?
- **Evidence:** The official publisher-indexed text describes RNW-PC structure-confidence weighting and WSC coarse-to-fine matching, with a 60-pair evaluation across four groups. The reported gains are author claims. Direct HTML returned HTTP 429; PDF/XML routes were inaccessible, and pair-level grouping, reference-point independence, and code/run binding remain unverified.
- **Decision:** Add as registration prior art with unresolved historical matrix membership. Do not infer SAR-restoration fidelity, local residual-shift robustness, optical utility, or a SAR-only fallback. B1 remains `not_promoted`; B2/B3 remain gated.
- **Boundary:** No local or test data, metric, checkpoint, model, experiment, or gate was accessed or changed; task flags are `false/false`.
- **Records:** D-2026-10-28-030, [audit](../../experiments/literature_audits/registration_structure_confidence_phase_congruency_20261028_v1/report.md), [access manifest](../../experiments/literature_audits/registration_structure_confidence_phase_congruency_20261028_v1/access_manifest.json), and [matrix](related_work_matrix.csv).

## Lian et al. full-text route recheck — 2026-10-28 (literature only; no experiment)

- **Question:** Did a new author-controlled or alternate open-access route expose the full paper?
- **Result:** The previous author/institutional search output was clipped and excluded. The OpenAlex endpoint was inaccessible through the reader; an exact-title arXiv search returned no results. No usable author copy was obtained.
- **Disposition:** Keep the paper at publisher-indexed abstract/selected-text level. Reopen only if access state changes; source-group split, reference independence, controlled warp protocol, and code/run lineage remain unresolved.
- **Boundary:** No local data, project test payload, metric file, checkpoint, model, experiment, or gate was accessed or changed. Test flags for the overall continuation remain `true/false` because of earlier group 473; no result value was used. See D-2026-10-28-032 and exposure groups 475–479.

## HSFMamba author-linked implementation route recheck — 2026-10-28 (literature only; no experiment)

- **Question:** Does the currently linked author repository expose a versioned HSFMamba implementation?
- **Result:** The current GitHub index's OGSSR section links the dataset share; its Toolkit/Results fields and Algorithm row are placeholders, and a targeted search for `HSFMamba` returned no match. The author catalog URL was inaccessible through the reader.
- **Disposition:** No implementation was obtained through the checked routes. Keep implementation availability unverified outside this index; HSFMamba remains prior art, not a reproducible local baseline.
- **Boundary:** No data/code share, local project data, test payload, metric, checkpoint, model, experiment, or gate was accessed. See D-2026-10-28-034 and the [full-text audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md).

## 2026-10-28 Manuscript eligibility and exposure review (no experiment)

- **Finding:** The formal 2k preview is archived after the train/validation overlap audit; the separate July pilot is explicitly an internal engineering report with no validation. Neither is a current submission draft.
- **Disclosure:** Group 473 surfaced historical test-metric prose in the pilot source. No value was used and no test image, payload, or metric file was opened. Current flags are `test_set_accessed=true`, `test_set_used=false`; cumulative flags remain `true/false`.
- **Disposition:** Keep both artifacts as historical records, exclude the pilot's 100 identities from future selection/tuning, and do not refresh claims until eligible source-group evidence supports a promotion-eligible result. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-28-031.


## DSRKD ResearchGate availability recheck — 2026-10-28 (literature only; no experiment)

- **Question:** Did the matrix-linked ResearchGate route provide DSRKD's full manuscript?
- **Result:** The page states that no full text is available and exposes a request-to-authors action; only abstract and bibliographic/reference material was returned.
- **Disposition:** Keep protocol details unresolved and do not use DSRKD as a reproducible task-matched baseline. No author request was made.
- **Boundary:** No local data, test asset/metric, code, checkpoint, model, experiment, or gate was accessed or changed. See D-2026-10-28-035 and the [route audit](../../experiments/literature_audits/dsrkd_access_route_recheck_20261023_v1/report.md).


## MONet optical-derived synthetic supervision — 2026-10-28 (literature only; no experiment)

- **Question:** Does MONet use paired optical imagery as an inference-time guide for SAR despeckling?
- **Result:** The full author manuscript describes optical grayscale images as synthetic training references and SAR-only inference. The 2022 follow-up is reviewed at institutional-abstract and linked-repository-README level only.
- **Disposition:** Record as training-time optical-derived prior art. It does not establish local inference-time optical utility, joint SR, misregistration tolerance, or fallback; no experiment or gate changed.
- **Boundary:** No local data/test payload, metric, checkpoint, or model was accessed or used. No code/data was downloaded or executed. See D-2026-10-28-049 and the [audit](../../experiments/literature_audits/monet_optical_training_reference_20261028_v1/report.md).

## 2026-10-28 CUDA-only B1F model execution interfaces

- **Scope:** B1F training, evaluation, and optical-pairing diagnostic CLI entrypoints.
- **Change:** Each requires explicit `--device cuda`, selects `cuda:0` after a fresh availability/device-count check, and fails before reading dataset inputs or creating run outputs if CUDA is unavailable. Training and evaluation manifests record the resolved device. Historical runs and manifests are unchanged.
- **Verification:** `python -m py_compile scripts/train_phase_b1f.py scripts/evaluate_phase_b1f.py scripts/diagnose_phase_b1f_optical_pairing.py` passed. Static inspection found no `auto`/`cpu` options or CPU fallback in these three entrypoints. No training, inference, test, or GPU launch occurred.
- **Boundary:** This interface change does not authorize a run or alter any gate. The direct-fusion v2 CPU diagnostic and v3 CUDA repeat are both already complete; B1 remains `not_promoted` / `stop_at_b1`, and B2/B3 remain gated. See D-2026-10-28-050.

## 2026-10-28 OMA-SSR implementation/correction route recheck

- **Question:** Did a current primary-source route expose OMA-SSR's author/publisher implementation or an official correction resolving the test-loss wording?
- **Evidence:** An exact DOI/title GitHub search surfaced no author- or publisher-linked implementation. A focused DOI correction/corrigendum/erratum search surfaced the Wiley article record but no separate correction item; the bounded Wiley page extraction exposed no correction or supplementary-code link. The prior direct Crossmark route remains inaccessible. Search output surfaced publisher-reported performance text, which was excluded.
- **Disposition:** No new route was verified, and this bounded result does not prove no code or correction exists elsewhere. Keep OMA-SSR as optical-guided SAR SR prior art; its actual training data flow remains unknown. Do not use its reported test metrics as independent comparison evidence or contact authors without authorization.
- **Boundary:** Literature route check only. No local test payload, metric file, checkpoint, model, or dataset was accessed; no performance value was used and no experiment or gate changed. See D-2026-10-28-051 and the [audit addenda](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/report.md#addendum--2026-10-28-exact-doicodecorrection-recheck).

## MTDN adjacent SAR multitask despeckling prior art — 2026-10-28 (literature only; no experiment)

- **Question:** Does the 2025 MTDN paper establish inference-time optical utility or joint SAR super-resolution?
- **Evidence:** Publisher-indexed article text and the WHU author profile confirm bibliographic details and describe dual-temporal dual-polarization SAR despeckling jointly guided by SAR change detection. The author repository README describes two noisy temporal acquisitions as input, a third as the denoising target, and a change label. The repository's current DSDN/training sources use change-derived features and an unchanged-area mask. The reported ENL/EPD values are author claims.
- **Disposition:** Add as adjacent SAR-only multi-task despeckling prior art. It has no optical input or spatial SR and does not establish local H1, optical reliability, safe fallback, or a task-matched baseline. The matrix now has 119 rows (78 historically confirmed; 41 unresolved historical membership).
- **Boundary:** No local/test payload, metric, checkpoint, dataset, model, source code, or experiment was accessed or executed. B1 remains not_promoted; B2/B3 remain gated. See D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).

## ISSMSAR SAR task-order literature update — 2026-10-29 (literature only; no experiment)

- **Question:** Does ISSMSAR report a matched experiment comparing SAR despeckling/SR task order and training coupling?
- **Evidence:** Publisher-indexed primary text explicitly distinguishes stepwise despeckling→SR from SR-first methods with despeckling as an auxiliary task, and qualitatively argues that the stepwise strategy can lose detail and amplify errors. The reported baselines are Frost, SAR-NLM, FFDNet, AGSDNet, and DBWAM; the reviewed text does not establish an order-reversed, matched joint-versus-cascade experiment. Direct publisher access remains HTTP 429.
- **Disposition:** Record this as SAR-specific strategy framing, not measured cascade evidence. H8 remains prospective and gated on valid source/target provenance and a verified development cohort. No matrix row/count, protocol, model, run, or gate changed.
- **Boundary:** Literature only. A bounded local H8 search surfaced an existing development-result summary and unrelated historical result prose; both were excluded and no value informed the decision. No test/confirmation payload or metric, checkpoint, model, code, or experiment was accessed or used; no test ran. Current-task test flags are false/false; cumulative project flags remain `true/false`. See D-2026-10-29-025, [exposure group 516](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md#group-516--issmsar-task-order-literature-search-2026-10-29), and the [audit addendum](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md#addendum-2026-10-29-sar-task-order-framing).

## H8 bounded SAR task-order source review — 2026-10-29 (literature only; no experiment)

- **Question:** Does accessible primary text verify a matched SAR comparison across restoration order and training coupling?
- **Result:** Muzeau et al. compare SR supervision targets, not reversed restoration order; DSRKD is confirmed as joint-task prior art, with full protocol inaccessible. A journal-indexed cascade-comparison claim could not be verified because the publisher article/PDF routes returned 403.
- **Disposition:** H8 remains open and data-gated. Do not infer a general joint-versus-cascade ranking. No protocol, model, run, or gate changed.
- **Boundary:** No project/test/confirmation data or metrics were accessed or used; no test or model run occurred. Current task flags are false/false; cumulative project flags remain true/false. See D-2026-10-29-028 and the [source review](../../experiments/literature_audits/sar_task_order_source_review_20261029_v1/report.md).

## 文献审计记录：LIT-2026-10-30-PHYSICS-01

- **类型：** 文献核查，不是模型实验或数据实验。
- **问题：** 物理约束扩散及硬约束后验测度研究能否支撑当前 SAR/optical 研究主张？
- **结果：** CoCoGen 与 PISD 是 PDE 科学计算先例；CoCoS 为尚未确认正式发表状态的 arXiv 预印本。均无 SAR/JDSR 直接验证。
- **产物：** [核查报告](../../experiments/literature_audits/physics_constrained_generation_20261030_v1/report.md)、[访问清单](../../experiments/literature_audits/physics_constrained_generation_20261030_v1/access_manifest.json)。
- **本轮测试集标记：** 图像访问=否；指标访问=否；指标使用=否。项目累计图像访问=是、指标使用=否。
- **对阶段的影响：** 无；B1 未晋级，B2/B3 未运行。

## 文献审计登记：LIT-2026-10-30-RESTORE-01

- **类型：** 一手来源文献对照，不是模型实验。
- **问题：** 直接 SAR 联合恢复、光学条件可靠性和少步生成有哪些先例与证据边界？
- **结果：** 确认 DSRKD、ISSMSAR、HSFMamba、OGSRN、Ada-RefSR 及通用少步工作；OGSFlow 目前仅到作者 README。区域专用扩散 SAR JDSR 的出版方正文未核实。
- **产物：** [前沿文献对照报告](../../experiments/literature_audits/optical_sar_generative_frontier_20261030_v1/report.md)、[访问清单](../../experiments/literature_audits/optical_sar_generative_frontier_20261030_v1/access_manifest.json)。
- **本轮测试标记：** 图像访问=否，指标访问=否，指标使用=否；未运行代码/模型。
- **阶段影响：** 无；B1 未晋级，C2 执行验收未完成，B2/B3 未运行。


## 只读数据审计登记：C1-TDV-META-20261030-01

- **类型：** 冻结划分/来源组元数据核验；不是训练、推理或模型实验。
- **协议：** `mmcanet_phase_c1_v2_sen1_2_source_grouped_20261028`。
- **范围：** 仅读协议 manifest、训练/开发身份、来源组清单及既有块扫描摘要；未读取影像。
- **观察：** train/dev 为 1,600/200 个身份、92/19 个来源组，组交集 0。已有 128×128、步长 64 块扫描未发现匹配签名；其覆盖不足以排除窄条或任意位移重叠。影像未下载且临时载荷未保留，地理/采集交叉字段缺失。
- **结果状态：** 精细像素扫描未运行；不用于晋级、模型选择或独立泛化主张。
- **本次访问标记：** `test_set_accessed=false`、`test_set_used=false`、`confirmation_set_accessed=false`、`confirmation_set_used=false`。本次训练/推理/项目测试均未运行；累计测试标记仍为 `true/false`。
- **证据：** [审计报告](../../experiments/data_audits/mmcanet_phase_c1_train_dev_metadata_boundary_20261030_v1/report.md)及同目录访问清单。


## 总纲采纳登记：CHARTER-ADOPTION-20261030-V1.6

- **类型：** 用户批准的研究治理/文档转接，不是模型实验或结果。
- **正式文件：** `docs/research/RESEARCH_CHARTER.md`，版本 v1.6，日期 2026-10-30。
- **同步文件：** `AGENTS.md`、`README.md`、`MASTER_RESEARCH_PLAN.md`、`CURRENT_STATE.md`、决策与待办记录。
- **实验状态：** 本次没有运行训练、推理或测试；没有访问确认集/测试集影像或指标；B1/B2/B3 和 C1/C2 门槛未改变。

## 2026-10-31 实验记录：本地 2k 全量训练误启动

- **运行编号：** `local2k_train_v4_all_20261031_v1`
- **状态：** 中断；非方法结果。
- **数据：** `data/sar/` 与 `data/opt/` 的 2,000 对数字 PNG；误用全部数据训练，验证数为 0。
- **进度与产物：** 中断于 307/2000；无 `train_log.csv`、无 checkpoint。
- **数据边界：** 旧 `test_100` 名单中 13 个样本进入优化，`test_set_accessed=true`、`test_set_used=true`；不得再用该旧名单作独立测试。C1 v2 官方确认影像未访问。
- **有效性：** 不报告指标、不选模型、不用于阶段晋级。C1 v2 与本地 2k 没有验证过的身份映射。
- **审计：** [C1 v2 与本地 2k 对齐记录](../../experiments/diagnostics/c1_v2_local2k_alignment_20261031_v1/report.md)。


## 非实验性文献审查 — 2026-11-07

- **记录：** FlowSGS / FMPS 一手全文审查。
- **证据：** [审查报告](../../experiments/literature_audits/flow_inverse_methods_update_20261107_v1/report.md)、[FlowSGS 原文](https://arxiv.org/html/2609.20769v3)、[FMPS 原文](https://arxiv.org/html/2411.07625v3)。
- **边界：** 未运行代码或模型，不产生实验指标，不访问本地数据载荷；不改变阶段 gate。

## `c3a_full_resolution_cpu_smoke_20261107_v1` — 固定尺寸合成 CPU 软件核验

- **类型/状态：** 无 GPU 的合成软件诊断；通过，不是模型性能实验。
- **问题：** C3-A 现有 CPU 优化步只覆盖 32×36；检查冻结协议实际批量 4、256×256 尺寸的前向、反向、参数更新和推理采样链。
- **协议/实现：** `protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json`，SHA-256 `147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`；使用 `SarConditionalFlow(base_channels=16,time_dimension=64)`、纯合成张量、单次 Adam 更新及 8 步 Heun。
- **结果：** 输入和输出均为 `[4,1,256,256]`；loss/梯度/采样输出有限，参数更新发生，CUDA 不可见且未初始化；耗时 4.627 秒。
- **载荷：** train、development、confirmation、test、optical 均未读取；未加载 checkpoint。
- **解释：** 仅证明该张量尺寸的软件链可运行；不证明收敛、图像质量、采样稳定性或物理一致性，不改变 C1/C2/C3 gate。
- **产物：** `experiments/diagnostics/c3a_full_resolution_cpu_smoke_20261107_v1/` 下的中文报告、执行命令、运行清单及可重放脚本。


## 2026-11-07 纯合成噪声分布诊断

- **编号：** `multiplicative_noise_distribution_cpu_20261107_v1`。
- **目的：** 在均值、方差匹配时比较 C2 `Gamma(8, rate=8)` 乘子与 SDE 对应的均值保持对数正态乘子。
- **输入与配置：** 纯合成随机变量；每种分布 500,000 个样本；seed `20261107`；目标均值 1、方差 0.125。
- **观察：** 两者模拟均值/方差符合目标，但偏度约 0.713/1.098、99 分位数约 2.005/2.093。
- **产物：** [脚本](../../experiments/diagnostics/multiplicative_noise_distribution_cpu_20261107_v1/compare_noise_models.py)、[结果](../../experiments/diagnostics/multiplicative_noise_distribution_cpu_20261107_v1/results.json)、[报告](../../experiments/diagnostics/multiplicative_noise_distribution_cpu_20261107_v1/report.md)。
- **边界：** 未访问影像、模型、checkpoint、真实逐样本结果或确认/测试载荷；无 GPU、训练或模型评价；不改变 gate。


## 2026-11-07 纯合成 log-Yeo–Johnson Gamma 分布诊断

- **编号：** `log_yj_gamma_normality_20261107_v1`。
- **类型：** 纯合成 CPU 分布诊断；不是 SAR 影像实验、模型训练、项目评价或阶段 gate。
- **冻结协议：** `N ~ Gamma(L, scale=1/L)`，`L∈{1,2,4,8}`；每个 L 使用 100,000 个拟合样本和 5 组各 100,000 个独立评估样本。Yeo–Johnson `lambda`、两种变换的均值与标准差均只由拟合样本决定；评估指标为标准正态两侧经验 KS 距离 `D`。
- **观察：** `ΔD=D(log-Yeo–Johnson)−D(log-only)` 等权平均为 `−0.040896`；四种 L 各自的 5/5 个评估重复均为负。此结果只适用于该随机变量、样本量和标准化协议。
- **产物：** [协议](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/protocol.md)、[脚本](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/run_cpu_experiment.py)、[逐组结果](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/results.json)、[报告](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/report.md)、[运行清单](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/manifest.json)。
- **数据边界：** 只生成合成随机数；未访问项目 train/development/confirmation/test 载荷、模型、checkpoint 或项目指标；未导入 PyTorch/CUDA，无训练、推理、模型评价或项目测试；不改变冻结协议和研究阶段 gate。


## 2026-11-07 固定 `L=8` 乘性观测的共享 log-Yeo–Johnson 诊断

- **编号：** `log_yj_multiplicative_observation_levels_20261107_v1`。
- **类型：** 纯合成 CPU 分布诊断；不是 SAR 影像实验或模型结果。
- **协议：** `Y=sN`、`N~Gamma(8, scale=1/8)`；五档信号强度等权参与共享 `lambda` 拟合；每档另有独立校准样本及 5 组留出评估样本。评估集不用于拟合 `lambda` 或标准化量。
- **观察：** 共享 `lambda=1.2217133`；所有信号档的 25 个留出配对结果平均 `ΔD=−0.00634053`，每档 5/5 重复均为负，但改善幅度随信号强度变化。
- **产物：** [协议](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/protocol.md)、[脚本](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/run_cpu_experiment.py)、[结果](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/results.json)、[报告](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/report.md)、[运行清单](../../experiments/synthetic_cpu/log_yj_multiplicative_observation_levels_20261107_v1/manifest.json)。
- **边界：** 不代表真实 SAR、项目表示域或 C3-A 效果；不据此迁移变换。未读取项目影像/指标或 confirmation/test 载荷，无 GPU、训练、模型评价或项目测试。

## 非实验性元数据路线审查 — SEN12MS 作者相关维护仓库（2026-11-07）

- **记录：** 检查公开仓库入口、README 工具说明与目录名称，评估 ROI 地图 Notebook 是否能支持 C1 train/development 地理交叉。
- **观察：** 尚未建立角色隔离的一对一场景—坐标映射；未打开 split 清单内容、Notebook 内容/输出或 GeoTIFF。
- **证据：** [路线审查报告](../../experiments/data_audits/sen12ms_author_repo_crosswalk_route_20261107_v1/report.md)、[维护仓库](https://github.com/schmitt-muc/SEN12MS)。
- **边界：** 没有坐标、影像、测量结果、确认/测试载荷访问；不改变 C1/C2/C3 gate。


## SAR 指标已知样例 CPU 核验（2026-11-07）

- **编号：** `metric_known_case_cpu_20261107_v1`。
- **问题：** 项目常用指标在完全匹配、已知偏置/幅值变化、TBR 中位数为零及 C3-A 输出裁剪条件下，是否符合其实现公式？
- **协议：** 仅由脚本生成 64×64 与 4×6 浮点数组；固定 `[0,1]` data range、ratio epsilon `1e-8`、ENL epsilon `1e-12`；25 条预设断言；CPU/NumPy/OpenCV。
- **结果：** 25 条通过。EPI 对正幅值缩放不变；边缘强度比检测到 0.5 缩放。ratio 的约 `4.05e-8` 偏差与 denominator epsilon 一致。C3-A 裁剪后手算 PSNR 与公共函数一致。只将 SSIM 小图的 `ValueError` 提示改为中文；指标数值实现未改。
- **隔离：** 未访问项目载荷、模型/检查点、confirmation/test 或既有结果；未使用 GPU、未训练。
- **产物：** [报告](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/report.md)、[协议](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/protocol.md)、[源码](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/run_cpu_audit.py)、[结果](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/results.json)、[清单](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/manifest.json)。
- **解释边界：** 合成算术合同核验，不验证指标的物理含义或 ROI 适用性，不推进 C1/C2/C3 gate。


## 输出裁剪边界偏差合成诊断（2026-11-07）

- **编号：** `clip_boundary_bias_cpu_20261107_v1`。
- **问题：** `[0,1]` 区间投影如何影响预测误差、PSNR、裁剪率和均值偏差？
- **协议：** 7 个目标亮度×3 档高斯噪声；每档 500,000 个固定种子样本；共 21 个组合和 45 次断言；NumPy CPU。
- **结果：** 对所有组合，裁剪后的平方误差不大于裁剪前。边界目标在 `σ=0.05` 时裁剪约一半样本，PSNR 上升约 3 dB，但引入约 ±0.02 的有方向偏差。
- **隔离：** 只用脚本生成的标量样本；未访问项目数据、模型/检查点、confirmation/test 或既有结果；未训练或使用 GPU。
- **产物：** [报告](../../experiments/synthetic_cpu/clip_boundary_bias_cpu_20261107_v1/report.md)、[协议](../../experiments/synthetic_cpu/clip_boundary_bias_cpu_20261107_v1/protocol.md)、[源码](../../experiments/synthetic_cpu/clip_boundary_bias_cpu_20261107_v1/run_cpu_experiment.py)、[manifest](../../experiments/synthetic_cpu/clip_boundary_bias_cpu_20261107_v1/manifest.json)。
- **边界：** 标量合成诊断，不可外推为 SAR 噪声或模型表现；不改变 C3-A 冻结裁剪规则或任何 gate。


## RSDM/LSIR 作者稿与官方来源路径补查（2026-11-07）

- **问题：** IEEE/DOI 之外是否存在新的官方作者预印本路径？
- **结果：** 项目既有记录已审阅 ResearchGate 作者稿；本轮 arXiv 官方域名搜索未命中精确标题，直接 Search/API 页面请求受限。没有新增一手正文。
- **结论：** 作者稿方法记录保留为作者报告级先例；正式 IEEE 版本与 DOI 仍开放。搜索未命中不作为不存在预印本的证据。
- **隔离：** 未重开作者稿、未访问项目/confirmation/test 数据、没有训练或 GPU。详见[来源审计](../../experiments/literature_audits/region_specific_sar_jdsr_source_recheck_20261107_v1/report.md)。


## ICSP 2024 SAR 扩散超分出版方摘要与 PDF 路线审计（2026-11-07）

- **问题：** IEEE 摘要可支持哪些 SAR 扩散超分先例事实？能否取得全文核对实验协议？
- **来源：** IEEE Xplore 官方摘要；官方 PDF `stamp.jsp` 路线无法读取。
- **结果：** 摘要描述残差预测、MSTAR/SARBake、传统指标与下游分割；全文和数据/训练协议仍未审阅。
- **边界：** SAR 单模态超分摘要级先例，不是联合去斑、optical 引导或可复现本地基线。未访问项目/confirmation/test 数据，无 GPU。
- **记录：** [来源审计](../../experiments/literature_audits/icsp_sar_diffusion_sr_abstract_pdf_route_20261107_v1/report.md)。


## 解析双峰条件流求解器 CPU 探索（2026-11-07）

- **主诊断：** `c3a_bimodal_flow_solver_spike_20261107_v1`。在解析双峰目标上用同一组 100,000 个初值比较 16 NFE 的 Euler16、Heun8、中点法8、RK4-4，并测 Heun 2–64 步收敛。Heun8 强分离中央谷概率为理论值约 11.6 倍；Heun32/64 接近理论与有限样本参考。无门槛、无 gate 变更。
- **分离度补充：** `c3a_bimodal_separation_sweep_cpu_20261107_v1`。固定 `τ=0.2` 扫描 `μ/τ={1,2,4,7.5}`，比较 Heun8/32；该扫描在主结果后追加，标记为适应性探索。
- **时间网格补充：** `c3a_bimodal_time_grid_cpu_20261107_v1`。同一强分离场、同为 8 步/16 NFE，四种简单非均匀网格均未优于均匀网格。
- **表示尺度补充：** `c3a_representation_scale_bimodal_cpu_20261107_v1`。标量条件中心 0.5、源噪声标准差 0.1、目标模式 0.4/0.6；Heun8 中央窗质量约为理论值 2.17 倍，Heun32 接近。解析场经仓库 `sample_heun` CPU 路径与 NumPy 输出最大绝对差 `2.220e-16`；5 项定向采样器测试通过。
- **产物：** 各实验目录的[主报告](../../experiments/synthetic_cpu/c3a_bimodal_flow_solver_spike_20261107_v1/report.md)、[分离度报告](../../experiments/synthetic_cpu/c3a_bimodal_separation_sweep_cpu_20261107_v1/report.md)、[时间网格报告](../../experiments/synthetic_cpu/c3a_bimodal_time_grid_cpu_20261107_v1/report.md)、[表示尺度报告](../../experiments/synthetic_cpu/c3a_representation_scale_bimodal_cpu_20261107_v1/report.md)及各自 protocol、JSON、stdout、环境和 SHA256 manifest。
- **边界：** 仅解析合成标量场，不包括图像结构、神经向量场误差或真实 SAR 分布；不改变 C3-A 的冻结 Heun8、输出裁剪、协议或阶段门槛。未访问项目/confirmation/test 影像、模型、checkpoint、结果；无 GPU、无训练。


## C1 v2 train/development 来源路径与派生输入交叉核验（2026-11-07）

- **编号：** `c1_v2_train_development_source_crosswalk_20261107_v1`。
- **问题：** 冻结 C1 v2 的 train/development sample 是否能仅凭已登记元数据连接到原始相对路径及派生 x2_light 输入来源？
- **方法：** 仅提取 train/development 的 C1 `sample_manifest`、`group_assignment`、staging CSV、payload file manifest 对应行，以及派生 x2_light train/development `files.csv`；检查 SHA-256 绑定、`source_group_id + patch_id` 唯一性、路径父目录、SHA-512 和文件字节数。整份 split CSV 只作 SHA-256 完整性核对，其他角色行不解析路径/哈希。
- **结果：** 1,600/200 对、92/19 组全部连接；SAR/optical 来源 SHA-512 完全匹配；3200/400 个原始模态路径的已登记官方/观测摘要一致；train/development 来源组、路径、派生 sample ID、来源哈希交集均为 0；48,739 条断言通过。
- **解释边界：** 元数据交叉连接，不是本轮重读 PNG 后的哈希复算；不证明地理/采集独立或数据集/传感器属性，也没有把本地 `data/sar`、`data/opt` 数字文件名逐一映射到原始目录。不得读取未知角色载荷补齐映射。
- **产物：** [审计报告](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/report.md)、[协议](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/protocol.md)、[脚本](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/audit_source_crosswalk.py)、[结果](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/results.json)、[清单](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/manifest.json)。
- **隔离：** 本次只访问 split/哈希元数据，未打开图像、数组、checkpoint 或 confirmation/test 载荷；无训练/GPU，C1–C6 gate 不变。
