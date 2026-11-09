# 总研究计划

## 2026-11-09 一手生成式先例复核

已核验 CODSR（CVPR 2026）与条件 Flow 贝叶斯后验论文（AISTATS 2026）。研究主张须排除通用单步扩散超分、区域先验激活和后验 Flow 采样的宽泛新颖性；任务特定问题仍为 optical 对 SAR 联合恢复的增量价值、SAR 输出分布质量、观测相容性与错配回退。此文献更新不改变 C1–C6 顺序、任何晋级门槛或训练状态。详见[来源审查](../../experiments/literature_audits/flow_and_one_step_sr_prior_art_20261109_v1/审查报告.md)。

## 2026-11-09 已知后验统计评价与 Flow 控制先例

补查 ICLR 2026 的精确后验统计基准及 ICML 2026 MPC-Flow。前者提供以精确后验样本检查分布质量的通用方法，后者说明 Flow 逆问题推理控制及超分已有通用先例；MPC-Flow 本轮仅有出版方摘要级证据。新增六维稀疏高斯混合 CPU 诊断后，精确后验在闭式 CRPS 与边际覆盖检查中优于预设的欠/过分散与均值塌缩预测。该实验属于评价方法验证，不是 SAR 模型结果，也不改变阶段顺序、资格或 gate。C3-A 真实模型质量、SAR 表示域观测似然与光学增量效用仍待各自合规证据。见[文献审查](../../experiments/literature_audits/posterior_sampling_statistical_benchmark_20261109_v1/报告.md)及[CPU 诊断](../../experiments/synthetic_cpu/posterior_score_calibration_cpu_20261109_v1r1/报告.md)。

## 2026-11-09 局部亮点阈值解析敏感性

独立 Gamma 解析敏感性计算显示，固定阈值 0.25 的目标命中率随目标均值/Gamma 形状从接近零至接近一，整图误报概率也高度依赖分布形状和独立像素假设。因此局部亮点量只保留为未启用候选，不存在可由该构造支持的通用阈值；真实表示域阈值、掩膜和空间误报处理仍待定义。该诊断不改变 C3-A v3、阶段 gate 或研究路线。详见[解析诊断报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_threshold_sensitivity_cpu_20261109_v1/报告.md)。

## 2026-11-09 稀疏目标与图幅敏感性的合成补充诊断

在预设 64×64 稀疏亮点构造中，三项全图评分可区分压暗、漏检、错位和伪亮点；在单亮点 64×64 至 256×256 尺度扫描中，漏检造成的绝对分差随目标密度降低而约按面积缩小。保留全图评分并把局部亮点统计作为未启用的补充设计候选；跨图幅不直接比较绝对分差。此证据仅属合成方法诊断，不改变 C3-A v3、阶段 gate、训练/数据权限或总研究路线。详见[评分敏感性报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_metric_sensitivity_cpu_20261109_v1/报告.md)与[图幅敏感性报告](../../experiments/synthetic_cpu/c3a_bright_scatterer_resolution_sweep_cpu_20261109_v1/报告.md)。

## 2026-11-09 C1 train-only 官方清单交叉路线状态

本次官方作者 train_list 正文未能通过可用只读接口取得；三个 train-only 精确文件名搜索也没有形成可核验的交叉项。此结果不能作为“跨版本无匹配”证据。保持 C1 v2 的来源场景文件夹级结论，不把地理属性从 SEN12MS 迁移到 C1 PNG；不重试同一受限读取和搜索，继续其他可独立推进的非 GPU 工作。详见[获取尝试报告](../../experiments/data_audits/c1_sen12ms_trainlist_access_attempt_20261109_v1/report.md)。

## 2026-11-09 C1 地理元数据的 SEN12MS 候选路线

官方 SEN12MS 作者论文给出 georeferenced GeoTIFF 和 ROI/patch 命名，但未核实这些文件与冻结 C1 v2 SEN1-2 PNG 的逐文件、逐场景对应。README 中的分类 `train_list` / `test_list` 也不是 C1 冻结划分。本轮没有读取影像或 test/hold-out 列表，精确样本映射仍开放；不提升地理或采集独立性主张。仅在获得仅限 train/development 的直接身份交叉表时继续。见[来源复核](../../experiments/data_audits/c1_sen12ms_georeference_crosswalk_recheck_20261109_v1/report.md)。

## 2026-11-09 变差分数的横纵相关错配诊断

在横纵 AR(1) 相关为 0.75/0.35 的两个互为轴交换真值方向中，当前逐边平方误差实现的 h=1 与四尺度变差分数均达到预登记门槛；逐像素 CRPS 与整图能量分数未能区分。该结果只验证一个平稳可分离 logistic-normal 各向异性案例，不能替代真实 SAR、非平稳纹理或散射点验证，也不支持修改现行评分或 gate。见[合成诊断](../../experiments/synthetic_cpu/c3a_spatial_variogram_anisotropy_cpu_20261109_v1/report.md)。

## 2026-11-09 多尺度变差分数相关强度敏感性诊断

在匹配 lag=1 高斯相关的 AR(1)/RBF logistic-normal 场上，将 ρ 扩展至 0.4、0.6、0.8、0.9。多尺度等权分数在 8 个相关强度/真值条件中有 7 个同时达到预登记的均值差区间及正确排序门槛；ρ=0.4、RBF 真值下排序率为 0.600，低于 0.65。该失败条件限制了“多尺度普遍有效”的解释。结果只支持保留评分为未启用设计候选，不移植到真实评价协议，也不改变 C3-A v3 或 gate。见[敏感性实验](../../experiments/synthetic_cpu/c3a_spatial_variogram_correlation_sweep_cpu_20261109_v1/report.md)。

## 2026-11-09 多尺度变差分数识别长程依赖的合成诊断

在边际与一阶相关匹配的 AR(1)/RBF logistic-normal 场上，单尺度 `h=1` 变差分数没有区分两者；`h={1,2,4,8}` 等权平均在两个真值方向均更常选中正确协方差模型。该观察仅支持保留多距离评分作为后续候选，并不授权将其并入 C3-A v3 或改任何 gate。真实 SAR 输出的尺度与权重须在未来补充 protocol 中另行确定。详见[合成实验](../../experiments/synthetic_cpu/c3a_spatial_variogram_multiscale_cpu_20261109_v1/report.md)及[未启用设计](C3A_DISTRIBUTION_EVALUATION_ADDENDUM_DESIGN_20261109.md)。

## 2026-11-09 IOP SAR JDSR 作者仓储路线补查

使用精确题名检索 `edu.cn` 页面、机构仓储及 arXiv，没有发现新的一手全文或可验证作者稿；现有可见内容仍只有 ResearchGate 摘要级二手条目。具体方法、数据协议、代码与结果不作事实认定。该有界阴性检索不改变 SAR JDSR 先例待核实状态；仅在出现新的可访问一手来源时再恢复。详见[来源审查补充](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## 2026-11-09 C3-A CPU development 验证依赖权重

只读检查没有识别到 C3-A 正式训练 run 或其 `best.pt`，故当前不能启动有意义的 development CPU 验证；不得以随机权重或不同协议 checkpoint 代替。训练依用户要求暂停。恢复训练后，先按冻结 v3 产生带完整 manifest 的 run，再调用现有评价入口。此预检不改变 C3-A gate；模型质量仍未知。详见[预检报告](../../experiments/diagnostics/c3a_cpu_evaluation_checkpoint_preflight_20261109_v1/report.md)。

## 2026-11-09 PCSDD 复数 SAR 去斑先例补核

PCSDD 的作者公开全文补充了一项相邻先例：复数 SLC 域各向异性 PDE 漂移与学习型 score 修正结合，并以 blind-spot 自监督和 Gamma 强度似然进行去斑。它说明不能把一般性的物理引导扩散或复数 SAR 扩散去斑作为独占贡献；但论文没有超分观测模型、光学条件、错配回退或本地 8-bit PNG/noisy pseudo-target 证据。正文的 blind-spot 掩码符号存在内部不一致，不能据此确认无泄漏实现。C2/C3/C4 gate、C3-A v3 与训练暂停不变。详见[全文审查](../../experiments/literature_audits/pcsdd_complex_diffusion_despeckling_fulltext_20261109_v1/report.md)。

## 2026-11-09 候选多样本评分量 CPU 成本初测

以 `[16,256,256]` 合成预测集合测量候选 CRPS、整图能量和邻接变差分数的单图 CPU 时间，中位数分别为 16.700、7.183 和 14.568 毫秒（每项 7 次）。这只是当前主机、单图幅实现级测量，不包含预测生成、文件读写或批处理，不外推完整开发集/GPU 成本，也不决定真实样本数或协议。C3-A v3、阶段 gate 与训练暂停状态保持不变。完整记录见[耗时诊断](../../experiments/synthetic_cpu/c3a_distribution_metric_runtime_cpu_20261109_v1/report.md)。

## 2026-11-09 RSDM/LSIR 一手全文来源仍未补齐

对作者两家机构的官方域名进行精确题名/作者检索，未取得机构全文或可核验作者页；公开索引只重复出现已记录的 ResearchGate 作者稿。当前维持其作者稿级 SAR-only 联合恢复先例记录，不将新检索当作独立复核，不更新正式出版版本对应或代码/协议事实。若没有新的官方全文或仓储入口，转做其他可独立推进的非 GPU 工作。

## 2026-11-09 候选多样本评分量纯 CPU 原型核验

为后续可能的多样本分布补充评价建立隔离工具 `scripts/distribution_metrics.py`，实现逐像素 CRPS、归一化整图能量分数与水平/垂直邻接变差分数。定向测试在 `[0,1]` 合成输入上覆盖手算案例和主要拒绝条件，10 项通过；脚本不依赖 PyTorch，也没有接入冻结 C3-A v3。该工作只证明候选公式的 NumPy 实现符合已知案例，不证明真实生成模型的适用性或评价效力。

SEN12MS 作者仓库 issue #5 的公开搜索索引显示，ROI 地图 Notebook 在不同环境下曾绘出不一致的位置；由于 issue 页面无法直接打开，不扩写其原因或修复状态。现有来源不能给出 C1 v2 train/development 的坐标交叉表，地理与 patch footprint 独立性保持未知。训练/GPU 暂停，C3-A/C2 gate 和任何阶段资格均不改变。

## 2026-11-09 已知分布上的多样本评价设计诊断

按 v1.8 的单次 draw、均值点估计和样本分布分开评价要求，完成三项预登记的 NumPy CPU 合成诊断。已知双峰例子显示，均值正确的塌缩点和均值/协方差匹配的单峰分布都可能偏离真实分布；增加集合样本数可降低该例子中能量分数比较的重复间波动。归一化图像域例子还显示，边缘分布相同但空间依赖错误时，逐像素 CRPS 不敏感，而整图能量和邻接变差分数可发现差异。结果只核验这些评价量在受控已知案例中的行为，不验证本地模型或 SAR 物理表示。

不据此调整 C3-A v3、训练设置、draw 数、阶段 gate 或模型方法。已核验 CRPS、能量分数与变差分数的一手理论来源；正式文献审查记录在[评分规则来源核验](../../experiments/literature_audits/proper_scoring_rules_multisample_eval_20261109_v1/report.md)。已准备未启用的[多样本分布评价补充设计](C3A_DISTRIBUTION_EVALUATION_ADDENDUM_DESIGN_20261109.md)；训练依用户要求暂停，真实分布评价需待恢复后另行冻结协议。详见[二维评价量诊断](../../experiments/synthetic_cpu/c3a_distribution_metric_sanity_cpu_20261109_v1/report.md)、[样本数诊断](../../experiments/synthetic_cpu/c3a_distribution_draw_count_cpu_20261109_v1/report.md)与[图像空间依赖指标诊断](../../experiments/synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/report.md)。

## 2026-11-09 研究总纲 v1.8 获批并正式生效

用户明确批准 v1.8-proposed 并授权同步。正式总纲、`AGENTS.md` 与 `README.md` 已更新；逐项差异覆盖首页版本信息、第 1 节先例边界、第 7 节随机生成评价、第 8.3 节运行记录、第 9 节相关工作、第 12 节论文问题及第 15 节版本记录。差异稿同时记录对历史实验、未来协议和论文主张的影响。

v1.8 将 MONTI SAR 两阶段潜扩散超分列为已核验先例，并要求单次 draw、像素均值点估计和样本分布分开评价；不改变阶段 gate、历史结果、数据资格或训练授权。v1.8 生效不恢复训练；训练仍按用户指示暂停，不启动 GPU。

## 2026-11-07 ICSP 2024 全文作者主页路线仍受限

作者主页上的论文条目明确没有全文；另行精确搜索 NUDT 官方域名未找到可核验作者稿。未向作者发送请求。当前仍只能将 ICSP 2024 作为摘要级 SAR 单模态超分先例，不扩写数据或方法事实。没有阶段或 gate 变化。若未来出现公开可读的一手全文再复核；训练暂停期间继续处理其他独立的无 GPU 任务。见[来源路线补充记录](../../experiments/literature_audits/icsp_sar_diffusion_sr_abstract_pdf_route_20261107_v1/report.md)。

## 2026-11-07 SAR-only 直接联合恢复先例全文补充核实

Amieva 等人的 2026 年论文全文已从机构仓储直接读取。它是 IW→SM 跨模式、非生成式 SAR 联合去斑/超分先例；完整影像集合的百分位归一化先于场景拆分，构成保留影像分布进入预处理参数的风险，但未证明测试标签训练或 test 指标选模。该工作不检验 optical 效用或本地协议，不能替代匹配 baseline。综述与矩阵已更新，历史索引摘录记录保留。训练暂停期间继续只做有新证据支撑的无 GPU 任务；不改 gate、阶段资格或正式总纲。详见[全文补充审查](../../experiments/literature_audits/amieva_direct_fulltext_followup_20261107_v1/report.md)。

## 2026-11-07 新发现的 SAR 联合扩散论文先例仍待一手正文

有一条 IOP DOI `10.1088/1361-6501/ae8686` 的题名/摘要级二手线索直接涉及 SAR 联合去斑与超分；出版方和 DOI 正文入口在本轮不可读，作者稿检索未找到可验证全文。将它列为开放来源核验，宽泛的“没有 SAR 生成式 JDSR 先例”主张暂不成立；其具体方法与结果尚不能写成事实。C1–C6 与任何 gate 不变。见[来源审查](../../experiments/literature_audits/frequency_enhanced_sar_jdsr_source_check_20261107_v1/report.md)。

## 2026-11-07 相关 draw 下的像素均值收益敏感性已核验

使用共同 Gamma 分量构造保持单 draw 边际分布不变、控制 draw 相关系数；45 个 `L×ρ×K` 纯合成条件通过预声明核验。K=8 的 PSNR 均值收益从独立 draw 的约 9.03 dB 降至 `ρ=0.5` 的约 2.49 dB、`ρ=0.9` 的约 0.40 dB，完全相关时为零。该结果只揭示理论/模拟敏感性，不是模型行为证据。报告：[相关样本 CPU 诊断](../../experiments/synthetic_cpu/c3a_correlated_draw_mean_cpu_20261107_v1/report.md)。

同期只读核对 C3-A 多样本 protocol v3 和 evaluator，确认逐 draw 标量指标及像素标准差与协议一致；当前没有均值图/分布/依赖估计。协议未要求这些统计，故不改源码。任何后续扩展均须另行预登记且不得回写 v3。

## 2026-11-07 v1.8-proposed 文本候选完成，未生效

基于 MONTI 2026 SAR 潜扩散超分全文和独立 Gamma 多样本均值 CPU 诊断，准备了[候选总纲](RESEARCH_CHARTER_REVISION_PROPOSAL_20261107.md)与[逐节差异](RESEARCH_CHARTER_CHANGELOG_20261107.md)。候选仅明确 SAR 潜扩散相邻先例，并区分单次生成 draw、像素均值点估计和生成分布评价。v1.7 仍为正式版本；不改变 C1–C6、任何 gate、数据资格、阶段状态或训练授权。待用户审阅前不改 `AGENTS.md`、`README.md` 或正式总纲。

## 2026-11-07 SAR 生成样本评价边界与新全文先例

OpenReview 作者稿全文确认存在 SAR-only“结构超分→潜扩散散斑增强”先例，缩小通用生成式 SAR 超分主张范围；它不检验光学引导、配准错配或 SAR 观测一致性。纯合成 Gamma CPU 诊断表明，多 draw 像素均值会把散斑方差缩为约 `1/K`，同时提高均值点估计 PSNR；这不是单次生成样本。当前 C3-A 逐 draw 评价已符合该边界。该证据用于完善评价设计，不改变 C2/C3 gate、C3-A 协议或 v1.7 路线。详细来源见[文献审查](../../experiments/literature_audits/monti_two_stage_sar_sr_20261107_v2/report.md)及[CPU 诊断](../../experiments/synthetic_cpu/c3a_sample_mean_variance_cpu_20261107_v1/report.md)。

## 2026-11-07 无 GPU 正式执行预检修正

将级联、SFT、正式测试套件的冻结授权检查，以及 MMCANetV2 validation checkpoint 绑定检查，前置到随机状态初始化和 CUDA 选择之前；实际模型执行仍要求 CUDA。临时合成测试核验访问顺序，不改变数据划分、研究协议、gate 或模型结论。72/73 项定向测试通过；单项旧 B1 源码哈希完整性失败保留，见[核验报告](FORMAL_PREFLIGHT_CPU_AUDIT_20261107.md)。训练继续暂停，继续推进不依赖 GPU 的协议和软件工作。

## 2026-11-07 C2 全 development CPU 描述诊断不改变阶段资格

已按 v5 冻结协议对四条件全部 200 个 development 样本执行零训练 CPU 推理，完成 matched/permuted/missing/SAR-only 的表示域目标—背景对比误差复核与来源组等权汇总。结果方向依退化条件变化；development 已重复使用，因此只作描述，不更新 H1/Q1，不改变 C2 gate。训练暂停期间继续处理一手文献与独立软件问题；C3-A 质量判断仍需后续模型实验。

## 2026-11-07 SAR 条件光学生成 Flow 先例补入方法路线

SAR-to-optical 条件 flow 已有 2026 workshop/IEEE 同题名先例。后续不得把通用 SAR 条件 flow 或泛称结构保持列作本项目独占贡献；本项目任务边界仍是 SAR 观测条件下 SAR 输出联合去斑/超分，以及经验证的 optical 增量效用。该文献不改变总纲 v1.7 或 C2/C3 阶段门槛。

## 2026-11-07 C1 数字 PNG 精确哈希映射仍未建立

将冻结 C1 train/development 的 3,600 个来源 SHA-512 与仓库历史上游哈希表交叉核对，未出现精确命中。该路线不能把本地数字 PNG 名称连到 C1 源路径；零命中不排除重编码像素相同，也不证明数据集身份或空间独立。继续保持 C1 来源映射开放，只在有新的安全 train/development 元数据时推进；不读取未知角色 PNG 或 confirmation/test 载荷。训练仍按用户要求暂停。详见[审计报告](../../experiments/data_audits/c1_v2_numeric_png_source_crosswalk_20261107_v1/report.md)。

## 2026-11-07 13:51（中国标准时间）区域专用扩散 SAR JDSR 先例保持开放核验

DOAJ 可见题名和摘要线索指向一项可能直接涉及 SAR-only 生成式联合超分/去斑的 2026 论文；当前 IEEE Xplore、DOI 与 Crossref 路线未提供可读正式正文。因此它只作为高优先级开放先例，不纳入正式已核实结论，不支撑“没有 SAR 生成式 JDSR”或宽泛首次主张。继续文献与协议工作，不启动训练或 GPU，也不重复同一失败入口。

## 2026-11-07 13:33（中国标准时间）C1 作者地图路线仍无可用交叉表

作者 README 确认有绘制 ROI 世界分布的概览 notebook，但网页正文和原始文件入口均未提供可读取的内容；因此不能判断是否含可复核坐标，也不能将图示升级为 C1 source-group/patch footprint 对照。此前 TUM 发布条目限流/超时不重复请求。C1 地理独立性继续未知；训练暂停期间改做不依赖该交叉表且不触碰封存角色的无 GPU 工作。

## 2026-11-07 13:27（中国标准时间）等 RHS 预算求解器诊断及 v1.7 阶段边界

解析条件高斯速度场上的 16 次/样本调用对照显示，中点法与 RK4 的相对误差依目标/源方差比变化；此结果限于数值离散，不决定学习型速度场的求解器，也不改变 C3-A 冻结的 Heun 8 步协议。用户更新的 AGENTS.md 确认研究总纲 v1.7，并规定 C1 confirmation 已访问后封存、不得用于晋级，且 C1 v2 没有独立 test 角色。继续训练暂停期内的 C1 一手元数据查找和可独立完成的 CPU 工作。

## 2026-11-07 10:22（中国标准时间）解析条件高斯流的求解误差诊断

新增一项纯 NumPy CPU 诊断：在解析最优速度场已知的一维高斯条件流上，对比 8、16、32、64 步 Heun。12 组结果显示误差随步数增加而下降，8 步误差随目标/源标准差比而变。该结论限于数值离散，不代表模型逼近误差或 SAR 效果；C3-A 冻结步数和 gate 不变。遵守用户的训练暂停，C1 来源交叉表与 C2/C3-A 模型证据仍未解决。

## 2026-11-07 10:01（中国标准时间）无 GPU 前置审查收尾

C3-A 续训路径已修正并通过 16 项定向 CPU 回归；方法目标、Heun、续训状态与取数角色静态检查没有其他已证实代码缺口。C1 地理对应仍未知，C2 gate 与 C3-A 质量待模型评价/训练。遵守当前训练暂停，不改变 C1→C6 顺序、冻结协议或门槛；整体目标未完成。

## 2026-11-07 09:51（中国标准时间）确认 C3-A 载荷角色边界

C3-A runner 只构造 train/development，SAR-only 数据包装器只读取 LR 与 pseudo-target；confirmation、test、optical 标记在续训合同中被 fail-closed 校验。来源清单角色字段仅用于 split-integrity 元数据检查，不载入封存载荷。未发现代码缺口，不改协议与阶段门槛。见 D-2026-11-07-034。

## 2026-11-07 09:47（中国标准时间）修正 C3-A 续训绑定路径

C3-A `_resume_code_hashes` 原登记路径不存在，导致在绑定构建阶段失败；现已指向实际导入的 `MMCANetV2/phase_c2_matched.py`。针对性纯合成 CPU 续训回归 16 项通过，两个 Python 文件编译通过。续训其余模型、optimizer、RNG 和 epoch/best 状态合同静态核对一致。训练继续暂停，不读取 checkpoint、不改冻结协议。见 D-2026-11-07-033。

## 2026-11-07 09:37（中国标准时间）补充 C3-A 流匹配实现核对

冻结 C3-A v1 的源分布、插值路径、速度目标、未加权均值 MSE、逐样本噪声和 Heun 采样均与代码一致；不因形式一致而推断训练有效。训练保持暂停，C2/C3-A gate 与阶段顺序不变。下一项无 GPU 工作转为 checkpoint 与 RNG 状态恢复合同静态审查。详见 D-2026-11-07-032。

## 2026-11-07 09:27（中国标准时间）补充 C1 路径格式观察与 C2 写入审查

- C1 train/development 的文件名模式与 SEN12MS 作者仓库示例相容，但这不证明目录编号或 patch 网格可直接跨版本映射；地理/采集独立仍属未知。见 D-2026-11-07-031。
- C2 screen 常规 runner 在模型评价前检查冻结输入记录与新输出路径，结果文件独占创建；未发现需要修改的实现缺口。C2 gate 仍未判定。见 D-2026-11-07-030。
- 训练继续暂停。下一项仅做 C3-A flow-matching/Heun 源码与冻结公式的无 GPU 静态比对；若没有具体偏差，不新增实现或重复测试。

## 2026-11-07 09:05（中国标准时间）补充共享变换诊断及 C1 元数据路线复核

- 固定 `L=8` 的纯合成 `Y=sN` 对照使用五档训练混合拟合共享 log-Yeo–Johnson `lambda`，并以独立校准/留出样本评估。KS 距离改善幅度依赖信号强度，故不将单独拟合 Gamma 乘子的结果推广为观测级或本地变换主张。详见 D-2026-11-07-028。
- C3-A 当前变换/退化数据链与冻结协议相符；没有确认需要修改的代码合同缺口。
- SEN12MS 作者相关维护仓库的 ROI 概览是候选路线，但尚未取得可安全使用的 train/development 专属坐标映射。confirmation/test 清单与内容继续不访问；地理/采集/patch footprint 独立性未知。详见 D-2026-11-07-029。
- 训练继续暂停；C1→C6 阶段顺序、C2 gate、C3-A 协议与 H1–H6 不变。

## 2026-11-07 08:51（中国标准时间）补充 log-Yeo–Johnson 合成分布诊断

独立评估的纯合成对照显示，在 `Gamma(L, scale=1/L)`、`L∈{1,2,4,8}` 下，训练样本拟合的 log-Yeo–Johnson 变换在五组留出随机流中均降低了标准正态 KS 距离；20 个配对结果的等权平均 `ΔD=−0.040896`。这只验证指定抽象分布下的变换行为，不证明本地 SAR 噪声或 PNG/noisy pseudo-target 的迁移效果。C3-A 训练仍暂停，C2/C3-A gate、H1–H6 和 C1→C6 顺序不变。见 D-2026-11-07-027 与[实验报告](../../experiments/synthetic_cpu/log_yj_gamma_normality_20261107_v1/report.md)。

## 2026-11-07 08:29（中国标准时间）纳入 DiffusionSAR 一手全文边界

DiffusionSAR 全文确认已有以 SAR 观测为条件的扩散去斑、Gamma speckle 合成与 log-Yeo–Johnson 变换；没有 SAR 超分或推理时 optical 条件。相关工作矩阵及扩散可行性记录据此收窄宽泛新颖性表述。当前 C3-A 为 8-bit 表示域 noisy pseudo-target，不能未经验证移植论文预处理；训练暂停，H1–H6 和 C2/C3-A gate 不变。见 D-2026-11-07-026 与[审查报告](../../experiments/literature_audits/diffusionsar_conditional_sar_despeckling_fulltext_20261107_v1/report.md)。

## 2026-11-07 08:20（中国标准时间）历史 confirmation 汇总文字终端暴露隔离

一次跨文档检索的输出显示了历史 confirmation 汇总文字，已按 D-2026-11-07-025 隔离，不复制、复核或用于研究决策。未打开底层 confirmation/test 数据或逐样本结果；阶段资格、C1 候选元数据路线及用户要求的训练暂停不变。后续研究文档读取改为明确路径。

## 2026-11-07 08:09（中国标准时间）新增 C1 地理元数据候选来源

一手 SEN12MS 论文称沿用 SEN1-2 的同一批随机 ROI 和季节定义，且提供 GeoTIFF 地理参考，为 C1 来源场景元数据交叉核验提供候选路径。论文未声明 ROI 子目录编号逐一一致，且两版数据独立筛选与生成；不得由相同编号直接推断坐标或空间独立。后续只核验 train/development 来源组身份，confirmation/test 继续封存。该文献发现不改变 C1 结论、C2 gate、C3-A 训练暂停或 C4–C6 阶段顺序。见 D-2026-11-07-024 与[路线审查](../../experiments/data_audits/c1_geospatial_crosswalk_route_20261107_v1/report.md)。

## 2026-11-07 07:02（中国标准时间）补足 C3-A 固定训练尺寸 CPU 软件核验

- 按冻结协议中的批量 4、256×256、`base_channels=16`、`time_dimension=64`，在 CPU 纯合成输入上完成一次优化步与 8 步 Heun 采样；有限性、形状和参数更新检查通过。
- 该检查仅确认软件能运行在实际训练张量尺寸，不支持质量、收敛或 GPU 训练可行性结论。协议、训练数据和 gate 均未改变；恢复训练前仍须按原协议执行资源预检。
- 运行报告：[固定尺寸 CPU 核验](../../experiments/diagnostics/c3a_full_resolution_cpu_smoke_20261107_v1/report.md)。

## 2026-11-07 06:01（中国标准时间）修复 C3-A v3 bootstrap 配置读取

C3-A v3 把 repeats/seed 存在 `metrics` 中，评价器却在多次采样后访问缺失的 `statistics` 键；per-metric seed offset 也未写入协议。现于协议加载阶段验证配置，按冻结唯一 seed 汇总并检查 19 组覆盖。31 项 C3-A 多次采样 CPU 合成测试通过，协议 JSON 未改，模型训练/评价仍暂停。见 D-2026-11-07-018 与[报告](../../experiments/diagnostics/c3a_v3_bootstrap_contract_cpu_20261107_v1/report.md)。

## 2026-11-07 05:49（中国标准时间）confirmation bootstrap 统计合同 fail-closed

一次性 confirmation 协议校验现绑定来源组单位、20 组、组内平均后组间等权、10,000 次 bootstrap、seed `20261104` 和 paired 95 percentile 区间。六项合成统计漂移均被拒绝，confirmation/screen 定向 CPU 回归共 46 项通过，相关代码编译通过。旧协议源码哈希与当前实现不匹配，且实现哈希覆盖组件已变化；不重跑确认集，不改变 C2 gate。见 D-2026-11-07-017 与[核验报告](../../experiments/diagnostics/c2_confirmation_statistics_contract_20261107_v1/report.md)。

## 2026-11-07 05:33（中国标准时间）C2 来源组 bootstrap 统计语义已绑定

按冻结 screen 协议，来源组 bootstrap 以 `source_group_id` 为单位、19 组等权，先在组内对 patch 求均值，采用 10,000 次 percentile bootstrap 与 seed `20261030`。代码先前忽略了部分语义字段与组数；现在对不支持的统计配置和组数失配 fail-closed。纯合成 CPU 验证区分出来源组等权与 patch 等权结果，C2 screen 两份测试共 24 项通过、相关 Python 编译通过。无模型效能证据，不改变 C2 gate；训练按用户要求暂停。见 D-2026-11-07-016 与[合成核验报告](../../experiments/diagnostics/c2_source_group_bootstrap_synthetic_20261107_v1/report.md)。

## 2026-11-07 04:06（中国标准时间）C2 固定输入生成协议错配 fail-closed 修复

静态核对固定退化链与 C2 v2 合同，发现 `--seed` 未要求等于冻结的 20261032，代码中的 scale/Gamma 参数亦未绑定协议字段。修复后生成器在 CUDA、输出目录和图像访问前校验 seed 与四个条件映射；两个错误请求的 CPU 合成回归先失败后通过，source-manifest 保护一起共 3 项通过，相关 Python 编译通过。原协议、退化、gate 不变；无项目载荷访问或 GPU 初始化。见 D-2026-11-07-010 与[C2 核查报告](C2_FIXED_GENERATION_PROTOCOL_BINDING_20261107.md)。

## 2026-11-07 03:46（中国标准时间）更正 C1 v2 官方来源链范围

冻结 C1 v2 train/development 已有可审计的官方文件来源链：MediaTUM 条目 1436631、官方 SHA-512 文件索引及 2026-10-31 暂存记录，覆盖 3,600 个文件。旧 `fixed_train_dev2k` / `fixed_val_dev2k` 哈希零匹配结论只适用于其 3,600 个历史文件，不适用于 C1 v2。`./data/` 数字命名子集仍未映射；坐标、日期、轨道和 patch footprint 未知，地理/采集独立性仍开放。该更正不改变 C2/C3 阶段条件或暂停训练状态。见 D-2026-11-07-009 与[C1 来源链核查](C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)。

## 2026-11-07 03:32（中国标准时间）官方来源核查未取得本地数据交叉表

> 历史范围说明：后续依据留存暂存记录确认 C1 v2 train/development 官方文件来源链；本节“未取得交叉表”特指地理/采集坐标映射，不再表示 train/development 的发布文件身份未知。见 D-2026-11-07-009。

原始 SEN1-2 论文确认发布版本的传感器、季节 ROI 目录与 patch 方案，但未列逐 ROI/patch 坐标/日期/产品映射；TUM 官方记录和 OAI-PMH 访问返回 HTTP 429。不得由目录名推断本地数据来源或地理独立。若无新归档元数据，按当前证据边界推进；训练仍按用户要求暂停。见 D-2026-11-07-008 和[C1 来源核查](C1_OFFICIAL_SOURCE_METADATA_AUDIT_20261107.md)。

## 2026-11-07 03:25（中国标准时间）完成 C3-A 输入/输出数值域合同核对

SAR LR 高于 1 而 noisy pseudo-target 限于 `[0,1]` 与冻结路径一致：输入保留原值、条件流预测无界速度、积分不裁剪、评价时才裁剪并记录 raw 越界。现无直接证据支持修改模型或协议；真实模型效果仍未知，训练按用户要求暂停。见 D-2026-11-07-007 和[C3-A 合同审计](C3A_NUMERIC_DOMAIN_CONTRACT_AUDIT_20261107.md)。

## 2026-11-07 03:17（中国标准时间）完成旧 pilot 论文主张与当前阶段静态核对

旧英文稿记录的是 2026 年 7 月的 1,900/100 文件级 pilot，不是现行冻结 C1 v2 的 1,600/200/200 来源组划分，也不为当前 C2/C3-A 提供晋级或方法效果证据。C2 gate 未判定，C3-A 训练继续暂停；H1/Q1 与路线顺序不变。稿件未跟踪且保持原样。定位时被动看到稿件内旧 test 汇总表，未读取底层载荷/逐图表或用于决策；已披露并禁止用该旧集合继续选择。见 D-2026-11-07-006 和[静态审计](PAPER_CLAIM_EVIDENCE_AUDIT_20261107.md)。下一步检查 C3-A 数值域与输出参数化合同，只做静态代码分析和合成 CPU 验证。

## 2026-11-07 02:59（中国标准时间）完成 C2 零背景对比差 CPU 描述诊断

新增的零安全目标—背景对比差只在原 C2 v2 fail-closed 的 6 条 development 记录上应用。24 次 CPU 前向和逐图/配对输出通过输入、权重、掩膜与协议身份核验。观察只描述由原指标失败选出的 2 个来源组；不更新 H1/Q1、C2 gate 或晋级状态。CUDA 未初始化，训练与正式评价仍按用户要求暂停。详情见 D-2026-11-07-005 与[登记](EXPERIMENT_REGISTRY.md)。

## 2026-11-07 02:13（中国标准时间） 完成 C3-A 表示域范围审计并更新未运行的多次采样诊断

按冻结输入对 train/development 的 SAR LR、实际上采样条件和 noisy pseudo-target 做 CPU 描述统计。条件像素大于 1 的来源组等权均值约为 14.94%/15.29%，target 恰为 1 的比例约为 1.23%/1.28%；这只描述归一化 PNG 表示域。为区分未来 Flow/baseline 输出的低端与高端越界，保留补充协议 v2 并新增未运行的 v3，既有总裁剪比例和质量/选模规则不变。合成测试 30 项通过；不改变 C2/B1/B2/B3 gate。用户要求训练继续暂停，详细证据见 D-2026-11-07-004、[审计报告](../../experiments/diagnostics/c3a_representation_range_audit_20261107_v1/report.md)和[协议更新记录](../../experiments/diagnostics/c3a_representation_range_audit_20261107_v1/protocol_update.md)。

## 2026-11-07 01:42（中国标准时间）完成 C1 v2 身份键元数据复核；模型训练保持暂停

按 split-integrity audit 只读冻结 C1 v2 成员元数据，确认 train/development/confirmation 的 source group 两两互斥；裸 `scene_id`/`patch_id` 数值复用是跨 season root 的非全局编号，关联的 source group、文件路径和官方哈希不同。该结果不能证明地理/采集独立或排除空间重叠。与此同时，新增的 MS-Flow 与 SAR 回波扩散后验文献审查收窄方法新颖性边界。用户仍要求暂缓训练；本轮无模型/性能证据，C3-A 待恢复，其他 gate 不变。见 D-2026-11-07-002/003 及两份审计报告。

## 2026-11-07 01:21（中国标准时间）完成无 GPU 的生成逆问题与 SAR 成像先例核查

核对了 ICML 2026 MS-Flow 与 UAV SAR 扩散后验成像预印本，更新相关工作矩阵、方法边界和决策记录。该工作排除了通用流式逆问题及生成先验结合 SAR 回波观测模型的宽泛新颖性主张；没有增加项目实验或本地性能证据。用户要求暂缓训练，因此 C3-A 训练与 development 评价仍未启动；C2/B1/B2/B3 gate 状态不变，C4/C5/C6 不提前执行。详见 D-2026-11-07-002 与[审计报告](../../experiments/literature_audits/sar_echo_and_flow_trajectory_prior_art_20261107_v1/report.md)。

## 2026-11-07 00:26（中国标准时间） C3-A 续训状态合同完成；GPU 训练暂停

补齐 C3-A runner 的显式 epoch 边界续训和 checkpoint v2 绑定；修复 `best.pt` 已更新而 `last.pt` 尚未更新时的原子恢复路径。CPU 合成连续/恢复等价测试及 C3-A Flow、评价器、等待器定向回归共 67 项通过；相关文件 `py_compile` 通过。没有运行真实训练、等待器或正式评价，没有访问封存载荷。协议与阶段 gate 不变；当前状态和证据见[当前状态](CURRENT_STATE.md)及[实验登记](EXPERIMENT_REGISTRY.md)。

## 2026-11-06 23:37 C3-A CPU 合同与数据预检；GPU 训练暂停

用户要求暂缓训练并先做无 GPU 工作。v8 等待器已手动停止；C3-A 训练/评价均未启动。修正了条件流训练对和 Heun 采样的 dtype、时间区间及速度场 shape/device/dtype 合同；多次采样评价器现会在读取 development 载荷前核验训练 manifest 与 input manifest。相关 CPU 定向测试共 51 项通过；合成张量 Adam 单步、Heun 二阶收敛和逐样本噪声固定规则均通过。随后按冻结 C3-A `x2_light` 输入在 CPU 上核验 train/development 的 3,600 个 SAR LR 与 target 数组文件，样本数和来源组数分别为 1,600/92 与 200/19；哈希、shape、dtype、有限值和值域合同均通过。预检未初始化 CUDA，未读取 optical、confirmation 或 test 载荷。项目分区当前约有 19.67 GiB 可用；本轮没有可安全清理的大型产物。软件与数据完整性结果不代表模型重建效果，也不推进 C3/C2/B1 gate。接下来审查可由 CPU/合成输入验证的 C3-A 训练、随机性和 checkpoint 合同；训练保持暂停，不启动 GPU 等待器。

## 2026-11-06 20:54 v1.7 正式采纳；C3-A 资源等待继续

用户已明确批准 v1.7；正式总纲、代理规范和 README 已同步。该版本落实冻结 C1 v2 使用边界及 confirmation 永久封存，补充 FIRM/Twist Flow 的方法边界；不改变历史 gate。C3-A v8 仍等待 GPU（空闲 1146/4096 MiB、3 个其他进程），训练尚未启动；分区可用约 15.06 GiB，没有本轮新增的大型结果可清理。满足冻结资源门槛后继续已预登记训练与 development 评价。

## 2026-11-06 18:35 C3-A 等待继续、磁盘空间回升

v8 GPU 快照仍为空闲 1,453/4,096 MiB、2 个其他 CUDA 进程，训练未启动。磁盘约四分钟内从 4.88 GiB 回升至 13.47–13.611 GiB，项目内没有对应的大文件增删；来源仍未知。已清理本轮验证新生成的可重建缓存，原始/派生数据、历史实验和 checkpoint 均保留。

## 2026-11-06 18:31 C3-A GPU/磁盘复核

v8 等待仍未满足 GPU 门槛（空闲 1,229/4,096 MiB，3 个其他 CUDA 进程）。项目分区可用空间约 11 分钟从 12.135 GiB 降到 4.88–4.909 GiB；当天仓库未增加大文件，空间变化原因未知。只清理了本轮验证生成的 26,850 字节可重建缓存；历史数据、实验和 checkpoint 均保留，等待器维持既有 2 GiB 最低余量。

## 2026-11-06 C3-A 等待恢复与 v1.7 文献校订

沿用正式生效的 v1.6 与已冻结 C3-A v1 协议，不重划 C1 v2、不改 C2 gate。v7 等待进程已不存在且训练未启动；等待入口因固定 v7 输出目录而不能复用，已补唯一 `--run-id` 并以 v8 重新等待。GPU 尚未满足 0 个其他 CUDA 进程及至少 3,072 MiB 空闲的要求。磁盘独立读数短时不一致且波动明显；未发现本任务新产生的大型结果，不清理来源不明文件。v1.7 候选仍待用户批准，FIRM v2 题名校订不改变阶段路线。

## C3-A GPU 与磁盘双重等待 — 2026-11-06 14:30（中国标准时间）

1. 等待记录 v4 在 2026-11-06 02:22:20 UTC 后停止更新；句柄失效且未发现对应进程。训练/评价输出目录不存在，退出原因不明，旧记录保留。
2. v5、v6 均在训练启动前安全中断；当前唯一等待会话为 v7（`87305`），每分钟检查 GPU 与磁盘。06:28:49 UTC GPU 空闲 1,070/4,096 MiB、3 个其他 CUDA 计算进程，未达到 GPU 门槛。
3. 分区 06:28 UTC 可用约 16.429 GiB；根据 C3-A 输出构成设置的 2 GiB 磁盘门槛继续生效。近 20 分钟仓库无大文件增长、主要目录大小不变、无可安全清理的大型本任务产物；保留数据与历史结果，实验协议与 GPU 门槛不变。

## C3-A 跨日等待状态 — 2026-11-06 10:11（中国标准时间）

1. 工具会话 `23990` 仍在运行，使用唯一等待记录 v4；训练尚未启动，manifest 每分钟记录 GPU 资源。
2. 最近快照为 2026-11-06 02:11:19 UTC：RTX 3050 空闲 1,153/4,096 MiB、3 个其他 CUDA 计算进程、利用率 100%，未满足门槛。
3. 分区可用空间约 17.74 GiB；C3-A 训练和评价输出目录尚不存在。继续按冻结 C3-A v1 训练、v2 development 诊断的顺序等待，不降低并发或显存门槛。


## C3-A 最新执行计划 — 2026-11-05 21:32（中国标准时间）

1. 当前多次采样补充协议采用 v2（SHA-256 `8d886f26023f9d929265b3f70ad9f51314f94788b1e4a01f269af5c1f0a589fc`），在原四次采样基础上加入固定 edge mask、EPI、边缘强度比和 mask manifest；旧 v1 未访问载荷即被替代。
2. 等待记录 v3 意外终止且其空 manifest 已保留；当前等待记录 v4 每 60 秒轮询，见 [`manifest`](../../experiments/diagnostics/c3a_gpu_wait_20261105_v4/manifest.json)。需同时满足零其他 CUDA 计算进程与至少 3,072 MiB 空闲后，才按原 C3-A v1 训练协议训练。
3. 训练完成后按同一 GPU 门槛运行 v2 development 诊断；只报告描述性表示域指标，不改变模型选轮规则或 C2/B1 gate。
4. 完成全量评价、失败分析与阶段决定后，按用户要求提交并推送目标相关文件供 ChatGPT 审查。

## C3-A 当前执行步骤 — 2026-11-05 16:18（中国标准时间）

1. 等待 GPU 同时满足无其他 CUDA 计算进程和至少 3 GiB 空闲显存。
2. 完整运行已冻结的 C3-A v1 SAR-only Flow 筛选，不访问 confirmation/test/optical。
3. 训练成功后，在同一条件下运行预登记的 development 四次采样补充诊断，并报告配对质量、采样波动、时间和峰值显存。
4. 按原 C3-A v1 继续条件决定是否追加 seed；补充诊断不得回写原选轮、继续规则或 C2/B1 gate。
5. 根据结果进入附件规定的研究决策审查，再决定后续候选。

## 本轮方法推进 — 2026-11-05

已完成一项不改旧门槛的代码方法改进：为目标区/背景区增加零背景可计算的均值与中位数对比差，并仅在分母为正时报告表示域 TBR。合成边界与 C2 回归共 43 项通过；尚未对真实开发集模型输出运行新分析。设计见[C2 零背景对比差诊断](C2_ZERO_BACKGROUND_CONTRAST_DIAGNOSTIC.md)。

C3-A 已按预登记入口再启动一次，GPU 对进程可见，但 4 个其他 CUDA 计算进程触发冻结并发保护；训练器在读图前退出。详见[预检 v7](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v7/manifest.json)。GPU 资源变化后沿用同一 C1 v2 `x2_light` train/development 清单训练；不重新切分、不使用 confirmation/test。

## 当前研究位置 — 2026-11-05（总纲 v1.6 已采纳）

用户明确要求沿用项目原先冻结的 C1 v2 来源场景文件夹划分；没有新划分数据。train 1,600/92 组和 development 200/19 组已完成 C2 v2 八组训练与开发集选择；confirmation 200/20 组随后用于一次性复核。确认结果为 `inconclusive`：matched 相对 SAR-only 的四条件 PSNR 均未达到 +0.05 dB 且 95% bootstrap 下界大于零的联合门槛，故不支持稳定 optical 增益主张。

确认数据载荷访问后，执行流程发现置换路径和决策阈值字段映射问题，做了两次只改执行器读取映射的修订，并用相同冻结数据、checkpoint、指标和阈值完成技术恢复运行。第一次修订前的推理已完成但未落盘、未查看指标值。最终结果明确属于协议偏差下的描述性复核，不作为 C2 v2 gate 或 B1 晋级证据；confirmation 不得再用于调参或模型选择。原 C2 v2 fail-closed 失败状态和 B1 `not_promoted / stop_at_b1` 均保持。详见[决策记录](DECISION_LOG.md)、[实验登记](EXPERIMENT_REGISTRY.md)和[确认评价报告](../../experiments/phase_c2_confirmation_c1_v2_20261105_amended_v2/report.md)。

下一项工作回到 train/development 上做探索性单因素研究；development 已被用于选模，不再称为独立 holdout。C3-A SAR-only 条件流匹配筛选的协议与代码已准备，实际 GPU 训练受共享显存占用阻塞；资源可用后直接用既有 C1 v2 `x2_light` train/development 固定输入运行。该小筛选不改变 C2 原 gate 或正式 C3 晋级状态。C1 来源组的地理/采集独立性和精细空间重叠仍未知。

2026-11-05 另完成 v1.7-proposed 的候选审阅包（候选、逐节差异、数据审查和文献支持审查）；v1.6 继续生效，候选仍待用户决定。候选审查未访问任何 confirmation/test 载荷，test 相关文字暴露单独披露于[当前状态](CURRENT_STATE.md)。

## C3-A SAR-only 条件流匹配筛选 — 2026-11-05（预登记；训练待 GPU）

已冻结[协议](../../protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json)，直接使用 C1 v2 的 `x2_light` train 1,600/92 组和 development 200/19 组固定输入，不新建划分；confirmation/test 不读。方法为像素域 SAR-only conditional rectified flow，固定 SAR 上采样条件、输入中心化高斯源、velocity MSE 和 8 步 Heun 推断。评价使用来源组等权 PSNR/SSIM、逐身份配对差值及来源组 bootstrap，并以已完成的 C2 SAR-only checkpoint 作锚点。

实现与 CPU 单样本通路检查已通过；GPU 预检先报告空闲 1.06/3.67 GiB，后续检测到 3 个其他 CUDA 计算进程，训练尚未开始。该筛选只是单 seed 方法可行性探索，不回答 optical utility、观测一致性或稳健回退问题，不构成 C3 阶段 gate 结果。若达到预登记的开发集探索信号，只追加另一个 SAR-only seed；否则停止该方法方向的扩展。输入中心化 flow source 已有自然图像超分先例 RFMSR，故不把通用 flow 或该参数化本身主张为新颖贡献。

本轮按用户澄清直接使用冻结 C1 v2，不需要额外数据路径或重新切分。通过提权在 `AISAR` 环境运行同一预登记入口后，连续多次新鲜预检仍发现 3 个其他 CUDA 计算进程，并在读取影像前退出；最新记录见[预检 v6](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。不并发启动训练；待外部 CUDA 进程退出并通过显存门槛后复用原命令和原固定输入。

## 历史研究位置 — 2026-10-31（本节状态已由上文更新）

用户已批准研究总纲 v1.6，正式文本见[RESEARCH_CHARTER.md](RESEARCH_CHARTER.md)。本轮直接沿用冻结的 C1 v2 来源场景文件夹划分，没有重新切分：train/development 各 1,600/200 个样本、92/19 个来源组；confirmation 200 个样本未下载、未访问。C2 四条件的固定输入和八组 ×30 epoch 训练均已完成。冻结 paired screen 按原协议因开发集中两条身份的背景中位数为零而 fail-closed，故 C2 v2 gate 未判定；另完成保留全部开发样本的非门槛诊断。matched optical 相对 SAR-only 的来源组 PSNR 增益为 −0.0055 至 +0.0061 dB，四个 95% 区间均跨零且点估计低于 +0.05 dB 预设继续阈值；不能据此支持 optical 增益，也不构成正式 gate 结果。详见[当前状态](CURRENT_STATE.md)、[实验登记](EXPERIMENT_REGISTRY.md)和[非门槛诊断报告](../../experiments/phase_c2_c1_v2_20261031_v1/non_gate_diagnostic_20261031_v1/diagnostic_report.md)。B1 仍为 `not_promoted / stop_at_b1`，B2/B3 保持关闭。

## C1 v2 固定划分上的 C2 v2 方法筛查 — 2026-10-31（训练完成；正式 gate 未判定）

沿用项目已冻结的来源场景文件夹划分，不重新随机切分：train 1,600、development 200、confirmation 200。只使用前两部分；confirmation 未下载、未访问。按冻结 C2 v2 生成四条件固定输入，并训练每条件 `sar_only` 与 `optical_guided` 两臂，共八组、每组 30 epoch、batch=8。原 paired screen 因两条开发身份的 target 背景中位数为零，按冻结的“任一必需指标未定义则整次失败”规则 fail-closed；不得把它改记为通过或停止。另完成单独的非门槛诊断，完整保留 200 个 development 身份，只将两项未定义的 median TBR 误差标为不可用。全部结果和证据边界见[实验登记](EXPERIMENT_REGISTRY.md)。

源组仅保证文件夹级分离；地理位置、采集时间和精细空间重叠仍未完全核验。该筛查只能给出单 seed、19 个开发来源组上的探索性结果，不构成独立泛化或阶段晋级证据。

非门槛诊断中 matched optical 相对 SAR-only 的四条件 PSNR 增益依次为 −0.0055、+0.0042、+0.0061、−0.0024 dB，各来源组 bootstrap 95% 区间均跨零；相对 permuted optical 的增益为 +0.0099 至 +0.0152 dB。观察支持模型对 optical 配对关系有小幅响应，但没有显示超过 SAR-only 的明确开发集效用。结果为单 seed、用于选 epoch 的同一 development 集上的描述性观察，不能作为 gate、稳定增益或独立泛化证据。当前下一步是单独审查零背景下 TBR 指标定义并冻结新版本协议；不得改写 C2 v2。

## C2 配对屏幕代码与资源复核 — 2026-10-31（载荷取得前的历史验收记录）

新增 `scripts/evaluate_phase_c2_screen.py`，核验完整八组矩阵与 best checkpoint/初始化哈希，并只从四条件 development 输入推理 matched、跨来源组 permuted、missing optical 和 SAR-only。逐图结果记录固定掩膜阈值、像素数、覆盖率和 SHA-256；配对摘要按 19 个来源组等权 bootstrap，并按冻结规则输出继续、停止或不确定。9 个 C2 定向测试文件共 59 项通过；冻结模型合成 CUDA 推理预检通过。

此项仍是代码/资源验证，不是 optical 方法实验。一次冻结 batch=8 合成训练步预检已通过，但预检内仅余 52.9 MiB，真实训练启动时仍需复核共享显存。用户指出的 `data/` 布局与 C1 v2 清单不匹配，角色受限暂存已在复制前失败；真实训练和 screen 未开始。不得降低 batch、改动阈值或把合成检查当作正式结果。见 D-2026-10-31-001/003/004 与[C2 代码/资源审计](../../experiments/diagnostics/c2_execution_acceptance_20261031_v1/report.md)。

## C2 来源与指标代码首轮验收 — 2026-10-30

C2 固定输入生成器/读取器现在核对冻结协议 ID、哈希及 C1 清单 pin；指标 wrapper 显式传入 `eps=1e-8`，对未定义/非有限必需值整项失败，并提供掩膜记录结构。新增四条件身份、来源、target 和 optical 哈希 parity 审计，以及共同初始化、seeded DataLoader、group-balanced PSNR 选模和单 epoch 训练/开发 helper。五个 C2 定向测试文件为 29 项通过、3 项 CUDA 跳过；协议和 C1 清单哈希与冻结值一致。该工作只验证合成 manifest 与测试模型，不构成 C2 方法结果。八次训练/恢复编排、paired screen、正式结果 writer、C1 载荷/来源交叉表和可用 CUDA 仍未完成。


## 流匹配后验采样前沿 — 2026-10-30

两篇 2026 年一手论文分别研究离线条件流后验摊销，以及预训练流先验上的训练外似然引导／源空间后验采样。其中 D-Flow SGLD 已于 2026-09-04 在线发表于 *Journal of Computational Physics*；贝叶斯流匹配论文当前核验到 arXiv 版本。它们使通用流后验采样本身不能作为新颖性，但都不覆盖 SAR 图像、联合去斑与超分或光学条件；贝叶斯流匹配还报告强集中后验和先验预测尾部观测的近似误差风险。新增两条矩阵记录后共 144 行（78 行已确认属于旧清单，66 行归属未确认）。候选更新为 v1.4-proposed，仍待批准；C2、B1、B2/B3 状态不变。本次无本地数据／测试访问，无项目实验或测试。见 D-2026-10-30-007 和[来源审查](../../experiments/literature_audits/flow_posterior_sampling_frontier_20261030_v1/report.md)。

## OGSFlow README 新增架构细节 — 2026-10-30

作者 README 将 OGSFlow 描述为先估计光学引导结构、再由条件流进行生成式重建，并称可靠性门控引导融合同时用于两阶段。该信息进一步收窄候选新颖性边界，但尚无可核验源码、提交版本、论文引用、协议或结果。候选更新为 v1.3-proposed，仍待批准；相关工作矩阵不变，研究假设、阶段门槛、协议和实验资格不变。C2 执行验收未完成，B1 仍停在原门槛，B2/B3 继续受门控。本次未访问本地数据或测试／确认材料，也未运行实验或测试。见 D-2026-10-30-006 和[审查记录](../../experiments/literature_audits/ogsflow_readme_details_20261030_v1/report.md)。

## DiSpeckle 书目线索核验 — 2026-10-30

ORCID 作品记录提供该论文的题名、作者、年份、期刊和 DOI 元数据，记录来源标为 Crossref；本次未取得 IEEE 正文或作者稿。二手网页上的摘要未用作技术证据。该线索可能与 SAR 扩散去斑相邻，但模型、物理假设、数据和实验均未核实，因此未写入相关工作矩阵，也未修改候选 v1.2。当前候选仍待批准，B1 停在原门槛，C2 执行验收未完成，B2/B3 继续受门控。本任务未访问测试／确认数据或指标，测试标记为 `false/false`；项目累计标记仍为 `true/false`。见 D-2026-10-30-005 和[来源审查](../../experiments/literature_audits/dispeckle_metadata_route_20261030_v1/report.md)。

## 自适应参考引导相关工作 — 2026-10-30

Ada-RefSR 是自然图像扩散式参考超分辨率中自适应门控、无关参考控制及仿射/模糊压力测试的直接先例。STMP-DiT 增加了阶段感知多模态先验超分辨率的相关证据，并由出版方索引文本报告一个语义先验失效案例。这些工作缩小了通用门控和置信度条件引导的可主张新颖性；两者均未测试 光学/SAR 配准、SAR 输出恢复、JDSR 或 SAR 回退。C5 仍应作为前瞻假设，并以匹配 仅 SAR 基线上的 SAR 输出损害降低为检验目标。矩阵现有 142 行（78 行确认属于旧清单，64 行归属尚未确认）。没有改变假设、实验或门控。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。

## 条件流逆问题相关工作审查 — 2026-10-30

VFM 提供了通用图像条件逆问题与 ×4 超分辨率先例，评估设置为 ImageNet、已知算子和加性高斯噪声；BFM 仅提供通用生成效率证据。两者都不验证本地 SAR 观测模型或光学引导 SAR 恢复。基于冻结 C2 表示域退化链推导出的 Gamma 密度仍是候选解析式，完整变分目标与有限精度行为尚未验证。该次审查新增两条文献记录，矩阵共 140 行（78 行确认属于旧清单，62 行归属未确认）。这不改变研究假设、实验资格或历史门槛。见 D-2026-10-30-001/002/003 和[来源审查](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md)。

## Required paper review text-exposure record — 2026-10-29 (second review)

The source brief's required read of the internal pilot draft surfaced historical test-associated prose and outcomes; a bounded formal-paper README search surfaced a test-results artifact reference. No test image, fixed input, prediction, metric file, checkpoint, or test manifest was opened, and no value informed research decisions. Current-task flags are `test_set_accessed=true`, `test_set_used=false`; cumulative flags remain `true/false`. Exclude the surfaced values and mini identities from future design and selection. See D-2026-10-29-030 and [exposure group 518](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md#group-518--source-brief-review-reread-historical-pilot-test-prose).

## Historical optical-gate report review — 2026-10-29

The required historical reports preserve their own failed/not-promoted/stopped states: legacy B1, B1R R0, B1F inner-dev, current-source B1 v2, and Phase 1-R/S/T. The inactive candidate §14 already preserves those records, the B1 stop, and the contaminated development boundaries; no candidate or gate change is warranted. These historical validation and target-assisted oracle findings do not count as current C2 or generalization evidence. See D-2026-10-29-031 and its report links.

## Required manuscript review text-exposure record — 2026-10-29

The source brief's required manuscript/document review surfaced historical test-associated prose in project text. Record current-task flags as `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no result informed selection. Exclude the surfaced prose from future method design/selection. This does not change any result, gate, or charter approval status. See D-2026-10-29-027 and [exposure group 517](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md).

## C2 metric-contract acceptance detail — 2026-10-29

C2 v2 remains frozen and execution acceptance remains incomplete. The helper's effective ratio epsilon is currently inherited from source (`1e-8`), its reported edge-correlation metric can be null and is passed through by the C2 wrapper, and the mask helper does not return coverage. Any future authorized runner must bind the evaluator revision/effective epsilon, save per-mask coverage with thresholds and condition provenance, and enforce `fail_entire_evaluation` for undefined required metrics before C2 data generation. This source audit changes no protocol, result, or gate; see D-2026-10-29-026 and the [acceptance inventory](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md#c2-metric-formula-and-mask-coverage-contract-review-2026-10-29).

## C2 source/protocol binding and execution acceptance — 2026-10-29

The C2 v2 JSON pins an exact C1 v2 sample-manifest hash, but the current fixed-input generator and reader only compare against a manifest supplied by the caller. The available staging utility is bound to an older train/val cohort. No end-to-end C2 trainer/checkpoint selector, cross-condition array-parity audit, complete paired-control evaluator/result writer, or runner-level CUDA/resource preflight is present in the scoped inventory. Keep C2 closed until a C1-v2-specific role-limited staging path and protocol-bound runner implement and pass those acceptance checks; preserve the protocol hash and historical gates. This is static evidence only, not an authorization to stage or run. See D-2026-10-29-029 and the [acceptance audit](../../protocols/mmcanet_phase_c2/execution_acceptance_audit_20261029_v1.md).

## H8 literature boundary includes SAR-specific task-order framing — 2026-10-29

Bu et al.'s publisher-indexed ISSMSAR text distinguishes a stepwise despeckling→SR strategy from SR-first methods with despeckling as an auxiliary task, and qualitatively criticizes simple despeckling→SR for detail loss and amplified restoration errors. The reported method comparisons do not isolate task order or training coupling against a matched direct-joint control. This narrows broad novelty framing but is not a measured joint-versus-cascade result. Keep H8 prospective and gated on authoritative source/target provenance and an independently verified development cohort. No matrix, protocol, model, run, or gate changes; see D-2026-10-29-025 and the [audit addendum](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md#addendum-2026-10-29-sar-task-order-framing).

The 2026-10-29 bounded source review confirms SAR joint-restoration prior art and supervision-target comparisons but does not verify a matched `D→SR`/`SR→D` and stage-wise/end-to-end/direct-joint study. A journal-indexed cascade comparison remains a lead because the publisher article is inaccessible. Preserve H8 as an open, data-gated question; see D-2026-10-29-028.

## Physical-consistency title boundary for the current cohort — 2026-10-29

Official SEN1-2 documentation identifies stored SAR patches as 8-bit sigma-nought dB, but the retained record has no byte-to-dB mapping or raw calibrated values. C1 v2's selected PNGs are byte-matched to official source; this verifies lineage and storage type, not calibration. The separate earlier B1 v2 operator is a controlled representation-domain stress degradation of a noisy pseudo-target. Do not describe PNG-based work as physically consistent or infer a calibrated SAR likelihood. The charter's title remains conditional; use representation-domain observation-consistency only when an explicit operator is validated. Reopen the physical route only with authoritative measurement-domain data and a validated forward/noise model. See D-2026-10-29-024.

## CN122694669A patent route check — 2026-10-29

The official CNIPA publication-announcement page opened but no item-specific record or claims were retrieved; the linked PSS route returned HTTP 412 and the official publication portal timed out. Secondary-index text remains a provisional diligence lead, not a verified scholarly or efficacy source, and supports no formal change to the candidate's novelty boundary. No candidate, matrix count, experiment, or gate changed. See D-2026-10-29-023 and the [route audit](../../experiments/literature_audits/cn122694669a_official_route_20261029_v1/report.md).

## C1 official catalog route check — 2026-10-29

The official ISPRS release announcement points to the TUM mediaTUM SEN1-2 record, but the record returned HTTP 429 and focused official searches exposed no coordinate/source-product/acquisition crosswalk. No archive or image payload was downloaded. Keep the C1 claim at source-scene-folder-disjoint; do not call it geographically or acquisition independent. C1/C2 eligibility, confirmation boundary, B1 stop, and B2/B3 gates are unchanged. See D-2026-10-29-022.

## Reliability-aware SAR–optical classification prior art — 2026-10-29

Added Vulli et al.'s publisher-indexed 2026 article on a learned SAR/optical reliability gate for patch classification under synthetic optical degradation. It narrows generic reliability-gate novelty, but its output is categorical classification and the indexed discussion reports heavy-noise instability and a one-region/season data limitation. It does not test SAR restoration, JDSR, geometric mismatch, or SAR-only reconstruction fallback. The matrix now has 138 rows (78 confirmed historical members; 60 unresolved additions). Evidence is limited to publisher-indexed excerpts because direct article/PDF/XML access returned HTTP 429 or was unavailable. No experiment or H1/gate change follows. See D-2026-10-29-021.

## Candidate requirement coverage clarified — 2026-10-29

Candidate v1.2 explicitly names expected academic contributions while retaining the prior conditional framing: contributions remain candidates until supported, and a bounded null/negative result can be a valid outcome. This clarifies the source-brief crosswalk without changing the scientific direction or gates. The candidate remains pending approval. See D-2026-10-29-020.

## Candidate novelty boundary updated after OGSFlow review — 2026-10-29

The inactive C1–C6 charter candidate now treats optical-guided flow SAR super-resolution and learned reliability-gated fusion as repository-level OGSFlow prior art, not standalone novelty candidates. The README does not establish a verified paper, task-matched joint despeckling/SR result, observation likelihood, controlled mismatch behavior, or SAR-only fallback. The candidate and supporting audit are updated; approval remains pending and no protected instruction, gate, protocol, or experiment changed. Matrix remains 137 rows. See D-2026-10-29-019.

## GLCdiffcr narrows consistency-constrained diffusion novelty — 2026-10-29

Cao et al.'s GLCdiffcr uses SAR-guided conditional diffusion to reconstruct cloud-free optical images, with an abstract-described pixel/SSIM consistency constraint. This is not SAR-output restoration or a SAR observation-consistency likelihood. The publisher full-text route returned HTTP 403; split and implementation remain open. Matrix: 137 rows (78 historical members, 59 unresolved additions). It changes no local hypothesis, gate, or run. See D-2026-10-29-018 and the [audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).

## DMDiff adds reverse-task SAR-guided optical diffusion prior art — 2026-10-29

Zhang et al.'s DMDiff uses SAR-conditioned diffusion, cloud-aware optical gating, and cross-modal attention to reconstruct cloud-free optical output. This narrows generic multimodal conditional-diffusion and degraded-optical gating novelty, but does not test SAR-output restoration, JDSR, or geometric mismatch/fallback. Publisher-indexed text reports three datasets; the exact real-data ROI allocation is unclear, and the paper-linked repository currently contains only a README. Matrix: 136 rows (78 historical members, 58 unresolved additions). No hypothesis, gate, or run changes. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).

## CDD adds adjacent optical–SAR diffusion clustering prior art — 2026-10-29

Liu and Chang's abstract describes conditional diffusion that generates optical images from SAR for multimodal clustering. This narrows generic optical–SAR diffusion-fusion novelty, but does not test optical input to SAR-output restoration, joint despeckling/SR, or controlled mismatch/fallback. Full text and author-linked implementation remain unverified. The matrix now has 135 rows (78 historical members, 57 unresolved additions). No research gate, model, or experiment changes. See D-2026-10-29-016 and the [audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).

## Optical–SAR diffusion fusion prior art — 2026-10-29

Zhang et al.'s SDIF adds adjacent optical–SAR image-fusion prior art: SAR-BM3D preprocessing, a variational diffusion fusion objective, and semantic classification guidance. It produces a fused image rather than a restored SAR output and does not study joint despeckling/SR or controlled optical mismatch. The inference-time availability of the semantic labels used in reverse sampling is unclear. This narrows broad fusion-method novelty without changing the task-specific question, inactive charter status, or any gate. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).

## SAR split-Gibbs diffusion posterior prior art — 2026-10-29

Gao et al.'s full arXiv v2 manuscript combines a complex Fourier measurement likelihood with a learned SAR diffusion prior using split-Gibbs updates. This narrows broad novelty around SAR measurement-consistent diffusion, but the operator/noise model is not the local image-space normalized-PNG task and uses no optical guide. The reported real-scene protocol has a 192-versus-24 sampling-count discrepancy and omits source-product/split details. Keep its claims author-attributed and the related IEEE ICASSP record separate until a version link is verified. No C3/C4 eligibility, gate, or experiment changes; diffusion remains deferred. See D-2026-10-29-014 and the [full-text audit](../../experiments/literature_audits/sar_split_gibbs_posterior_imaging_fulltext_20261029_v1/report.md).

## Flow posterior-sampling literature review — 2026-10-29

FlowDPS is an inference solver for a learned flow prior, while conditional flow training is a separate modeling choice. A 2026 posterior-transport preprint derives exact source reweighting for deterministic diffeomorphic flow priors and gives a posterior-bias bound for approximate correction fields; its controlled 2D study reports mode-weight errors for a FlowDPS-style guidance comparison. Direct source weighting has severe high-dimensional ESS limits, and its image-scale solver is explicitly heuristic. These results motivate measuring posterior fidelity if this route is later authorized, but they do not establish SAR likelihood validity, optical utility, or behavior of conditional SAR flows. No C3/C4 gate or execution status changes. See D-2026-10-29-013 and the [audit](../../experiments/literature_audits/flow_posterior_transport_source_review_20261029_v1/report.md).

## C1 v2 scene-split provenance review — 2026-10-29

The SEN1-2 primary paper describes ROI/scene-level grouping, nearby-point exclusion intended to avoid overlapping scenes, and scene-wise splits as a way to create independent sets. C1 v2's whole-folder role assignment matches that source-scene unit. Local coordinates/footprints and Sentinel product/acquisition identifiers are unavailable, so this supports the split unit without establishing geographic-footprint or acquisition independence for the specific allocation. Keep C1 v2 at its existing source-scene-folder-disjoint exploratory C2 scope; no gate changes. See D-2026-10-29-012 and the [metadata review](../../experiments/data_audits/sen1_2_scene_split_metadata_review_20261029_v1/report.md).

## Root manuscript status — 2026-10-29

The July root paper is archived internal pilot material, not current submission evidence. Its mini-set test-associated results are not a valid independent confirmation basis under current provenance and group-split requirements. Keep the manuscript unchanged while the charter candidate awaits approval; a future formal version must be rebuilt from eligible, approved runs. See D-2026-10-29-011.

## SAR-to-EO flow-matching literature update — 2026-10-29

ContraFM-S2O and ReFlowSET add recent flow-matching prior art for SAR-conditioned EO generation. Their direction and output do not answer optical marginal utility for SAR-output JDSR or mismatch-safe SAR restoration. ReFlowSET's mention of local misalignment is not a controlled tolerance result. Do not make broad flow-matching novelty claims; no change to the inactive candidate, B1, or C2 prerequisites follows. See D-2026-10-29-010 and the [source audit](../../experiments/literature_audits/sar_to_eo_flow_matching_preprints_20261029_v1/report.md).

## Primary-source literature update — 2026-10-29

The γ-Bridge preprint adds direct SAR-only prior art for a look-indexed Gamma–Lévy diffusion bridge and synthetic-to-real despeckling. It does not test optical guidance or joint SR, and its reported PNG/JPG real-SAR evaluation does not validate the local observation model. Keep diffusion deferred behind the data and optical-utility gates; include the paper as a SAR-only control candidate only if later stages are authorized. See D-2026-10-29-008.

## Research-charter revision proposal — 2026-10-29

A review-only candidate charter, sectionwise change log, and supporting literature/data-protocol audit are available at [`RESEARCH_CHARTER_REVISION_PROPOSAL.md`](RESEARCH_CHARTER_REVISION_PROPOSAL.md), [`RESEARCH_CHARTER_CHANGELOG.md`](RESEARCH_CHARTER_CHANGELOG.md), and [`RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md`](RESEARCH_CHARTER_SUPPORTING_AUDIT_20261029.md). The proposed “Physics-Consistent Generative Joint SAR Despeckling and Super-Resolution with Optical Guidance” direction remains a hypothesis pending explicit user approval and evidence; it changes no historical gate. The full source brief and candidate were cross-checked in D-2026-10-29-009; C1 now explicitly prioritizes source-verifiable official SEN1-2 data. D-2026-10-29-007 records the static C2 readiness inventory: protocol v2 freezes the design, component helpers exist, but the C2 staging entrypoint and end-to-end runner/acceptance evidence remain incomplete. Do not stage data or train until the frozen acceptance conditions are met.

The C2 matched model, scale-aware train/development reader, CUDA-only fixed-input generator, and screen-analysis helpers are implemented. Synthetic ×2/×4 model/data contracts and all four candidate fixed-degradation conditions passed on CUDA; no C1 inputs or model-quality run exists yet. The generator consumes an already staged root; no C2 staging entrypoint or train/evaluation runner appears in the scoped inventory. Protocol v2 freezes the design as an artifact, not execution acceptance; no C1 staging or training has occurred. See D-2026-10-29-003/004/007.

A scale-aware, parameter-matched SAR-only/HR-optical C2 control, train/development reader and CUDA fixed-input generator passed synthetic CUDA contracts at ×2 and ×4. The four candidate degradation settings and full screen design are frozen in protocol v2. The end-to-end runner and execution acceptance checks remain before comparative training; status synchronization is recorded in D-2026-10-29-004. See D-2026-10-29-001/002/003/004.

C1 v2 is accepted for a prospective, source-scene-folder-disjoint exploratory C2 development screen with explicit remaining independence limitations. The SEN1-2 paper supports this split unit, but local geographic footprints and acquisition/product IDs remain unavailable. A versioned ×4 candidate path and frozen protocol-v2 design now exist; complete the runner/acceptance path before any model-quality run. The B1/B2/B3 gates are unchanged. See D-2026-10-28-075, D-2026-10-29-012, and D-2026-10-29-003/004.

C1 v2 narrow-overlap screening completed with no candidate translations; 115 optical windows hit an excluded repeated signature. Assess exclusions and source-provenance limits before model selection. See D-2026-10-28-074.

C1 v2 remains a frozen candidate, with formal selection gated pending a train/development narrow and arbitrary-shift overlap audit; the prior 128×128 window screen cannot detect the narrower leakage geometry found historically. See D-2026-10-28-072.

The C1 v2 train/development payload audit completed with matching hashes and schema and no duplicates in its declared fixed-grid screen. Assess remaining geographic/acquisition, arbitrary-offset and registration risks before deciding C1 eligibility. See D-2026-10-28-071.

## Phase C1 v2 metadata allocation — 2026-10-28

Official recursive metadata listings and the retained per-group SHA-512 inventory expose 131 candidate scene folders in `ROIs1158_spring` and `ROIs2017_winter`, with 142,506 matched SAR/optical patch indices. A deterministic group-first allocation now freezes 92/19/20 whole groups and 1,600/200/200 train/development/confirmation pairs at the metadata level. Exact candidate cross-group file duplicates and exact overlap with historical B1 v2 inputs are zero. Folder-level difference still does not establish geographic/acquisition independence or fine registration. Before model selection, validate train/development payload bytes and PNG schema, representation, exact decoded-pixel duplicates, and the declared bounded spatial-window overlap. Keep confirmation payload untouched and B1 at `not_promoted / stop_at_b1`. Existing representation, noisy pseudo-target, degradation, and metric findings remain normalized representation-domain evidence. See D-2026-10-28-069 and the [C1 v2 report](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/report.md).

The earlier C1 v1 inventory correctly recorded that its then-inspected cohort had no unused confirmation group and its first unauthenticated rsync request failed. That status is historical and is superseded only for candidate group availability by the later official listing; it did not create a split or authorize model selection. See D-2026-10-28-067 and the [C1 v1 protocol](../../protocols/mmcanet_phase_c1/protocol_report.md).

## Source-lineage connector remains unavailable — 2026-10-28

Although Google Drive search/list actions appear in the current tool catalog, three metadata-only searches and one `list_drives` call fail with backend `Unknown tool` and return no metadata. Keep the 2k local source map unresolved and selection paused; retry only after an actual connector response or a new authoritative source artifact. Preserve B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-066.

## CloudLULC cloud-degraded fusion prior art — 2026-10-28

The arXiv abstract and official project README describe reliability-guided SAR–optical fusion for semantic LULC mapping under cloud contamination. This narrows generic degraded-optical reliability-fusion novelty, while leaving restoration-time optical utility, SAR joint SR/despeckling, geometric-mismatch tolerance, and SAR-only restoration fallback untested. The matrix is 121 rows (78 confirmed historical members; 43 unresolved additions), with at least four entries still unaccounted for against the historical lower bound of 125. Keep B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated. Full text/code and journal status remain unverified. See D-2026-10-28-065 and the [audit](../../experiments/literature_audits/cloudlulc_cloud_degraded_semantic_fusion_20261028_v1/report.md).

## Direction-aware SAR–optical matching metadata lead — 2026-10-28

The search-indexed publisher listing and official citation record verify a new SAR–optical image-matching paper's metadata, but its article text is not accessible. Keep it as a registration prior-art lead only; its title does not establish an algorithm, controlled mismatch robustness, restoration utility, or local error threshold. The current matrix is 120 rows (78 historically confirmed, 42 unresolved); at least five records remain unaccounted for against the known 125-row lower bound. Keep B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-064 and the [metadata audit](../../experiments/literature_audits/direction_aware_sar_optical_matching_metadata_20261028_v1/report.md).

## MTDN adjacent multitask SAR prior art — 2026-10-28

MTDN adds SAR-only dual-temporal despeckling/change-detection prior art. Its SAR-derived change cue may be conceptually relevant to identifying unreliable temporal observations, but it is not optical guidance or spatial SR and provides no evidence for H1. The matrix has 119 rows (78 historically confirmed, 41 unresolved additions); at least 6 items from the known minimum 125-row historical inventory remain unaccounted for. B1 remains not_promoted / stop_at_b1, and B2/B3 remain gated. See D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).

## Related-work matrix reconciliation — 2026-10-28

The 2026 Ma et al. flood-mapping preprint row was restored from the explicit contemporaneous D-2026-10-18-09 inclusion record. At that decision point the matrix had 118 rows (78 historically confirmed, 40 unresolved additions). Its ±2-pixel loss handles segmentation-label offset; it does not evaluate optical-guided SAR intensity restoration or justify a shift-tolerant image loss without fixed-coordinate evaluation. No research gate or experiment changed. See D-2026-10-28-061 and the [restoration audit](../../experiments/literature_audits/flood_mapping_shift_invariant_row_restore_20261028_v1/report.md).

## PolSAR/SinSAR diffusion SR lead — access boundary update (2026-10-28)

A new search surfaced a candidate DOI and IEEE Xplore document URL, but the direct IEEE page still requires JavaScript/robot verification, the DOI route is inaccessible, and the ORCID search-index page marked Crossref as the source but could not be opened. Aggregator method/performance claims remain excluded. Keep this as an unverified lead outside the formal matrix until direct publisher/author evidence is readable; it does not change the optical-guidance hypothesis, diffusion route decision, or B1 gate. See D-2026-10-28-053 and the [route addendum](../../experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/report.md#addendum--source-route-recheck-2026-10-28).

## ICSP 2024 SAR diffusion SR metadata verification — 2026-10-28

The IEEE-copyrighted proceedings table of contents verifies the author list and starting page for Chen, Zhang, and Xiong’s SAR diffusion SR paper; the IEEE Xplore abstract supports only a high-level method summary. Full data, degradation, split, evaluator, code, and run details remain unknown. This adds no evidence for optical guidance or joint despeckling/SR and does not change the model-family decision or B1 gate. See D-2026-10-28-052 and the [metadata audit](../../experiments/literature_audits/icsp2024_sar_diffusion_sr_metadata_20261028_v1/report.md).

## RGSNet reliability-guided fusion prior art — 2026-10-28

A newly surfaced SSRN abstract and linked repository identify RGSNet as reliability-guided SAR–optical semantic segmentation under heterogeneous degradation. Its frequency-evidence proxy narrows generic fusion novelty, but the evidence is abstract-only, code is pending acceptance, and the output task is segmentation. It does not resolve the restoration-time optical-utility question or establish mismatch-safe SAR-output restoration. Keep the study as adjacent prior art; do not change the local hypothesis or B1 gate. See D-2026-10-28-045 and the [audit](../../experiments/literature_audits/rgsnet_reliability_guided_sar_optical_segmentation_20261028_v1/report.md).

## Bibliography-to-matrix source crosswalk — 2026-10-28

The current literature crosswalk restored two general misalignment-aware filtering/reconstruction records that an earlier audit documented but the current CSV lacked, and added one SAR–optical matching paper from the formal bibliography. These close a current coverage inconsistency and narrow generic novelty claims; they do not test H1 or permit model selection. The historical inventory remains incomplete and B1 remains `not_promoted` / `stop_at_b1`. See D-2026-10-28-043 and the [audit](../../experiments/literature_audits/bibliography_matrix_crosswalk_20261028_v1/report.md).

## PLISA publisher-indexed registration evidence — 2026-10-28

PLISA adds current optical–SAR registration prior art. Its keypoint pseudo-labeling assumes pre-registered pairs, and its reviewed SEN1-2 split does not establish source-group independence or independent geodetic checkpoints. It therefore does not resolve local natural residual registration or justify a mismatch threshold for SAR restoration. Keep local source mapping and independent reference checks as P0 prerequisites; see D-2026-10-28-058 and the [audit](../../experiments/literature_audits/plisa_optical_sar_registration_publisher_indexed_20261028_v1/report.md).

## OSR-Net metadata correction — 2026-10-28

The official MDPI index exposes the complete nine-author roster, while its cited arXiv DOI is a separate MultiResSAR review rather than an OSR-Net manuscript. This clarifies a dataset/review pointer but leaves the OSR-Net subset, split, independent-reference protocol, and code/run lineage unresolved. It remains registration prior art and does not change the B1 gate. See D-2026-10-28-059 and the [audit](../../experiments/literature_audits/osr_net_publisher_indexed_text_update_20261028_v1/report.md).

## DPS1-SAR dataset source screen — 2026-10-28

The official IEEE DataPort record lists a dual-polarization SAR despeckling dataset with multitemporal mean/median references, but file-level grouping and whether the noisy input is excluded from each reference remain unresolved. It is SAR-only and contains no SR target, so it does not supply a task-matched baseline or evidence for optical utility. Local model selection remains paused for source-group provenance; B1 stays not_promoted / stop_at_b1. See D-2026-10-28-042 and the [source audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).

## B1F run-device enforcement — 2026-10-28

The B1F training, evaluation, and optical-pairing diagnostic interfaces now require explicit CUDA and reject CPU/auto fallback before data or output staging. The completed CPU v2 direct-fusion diagnostic remains historical; its CUDA v3 repeat is complete. This is an execution-interface change only: no new run, model result, gate, or phase transition. B1 remains `not_promoted` / `stop_at_b1`; B2/B3 remain gated. See D-2026-10-28-050.

## OMA-SSR protocol eligibility — 2026-10-28

Keep OMA-SSR as optical-guided SAR SR prior art. The paper text, as written, describes test-set loss in training; the author's actual implementation data flow remains unknown. The latest bounded DOI/GitHub/Wiley check found no separate correction or author/publisher-linked code release, while the Crossmark route remains inaccessible. Exclude its reported test metrics from independent comparison unless a primary correction or versioned implementation clarifies the data flow. Do not contact authors without authorization. This does not alter local hypotheses or B1/B2/B3 gates; see D-2026-10-28-051.

## Direct baseline source-route status — 2026-10-28

The targeted ResearchGate route for DSRKD states that no full text is available and offers a request to the authors; no contact was made, so its data/target/degradation/split/training protocol remains unresolved. HSFMamba has a completed full-text audit, but its current author-linked index exposes a data share and no model-code entry; it is prior art, not a reproducible baseline through the checked route. Neither update changes the current B1 `not_promoted` / `stop_at_b1` gate. See D-2026-10-28-034/035.

## Manuscript eligibility and text-exposure boundary — 2026-10-28

The visible MMCANet manuscripts are archival/internal artifacts: the formal 2k preview was withdrawn from current claim use after the source/target overlap audit, and the July pilot explicitly lacks validation and submission status. Keep their historical records intact; do not present them as the current paper or refresh their claims from the failed B1 screen. Rebuild the submission manuscript only after an eligible source-group split and a promotion-eligible result. Group 473 records historical test-metric prose surfaced in the pilot source; no value was used (`test_set_accessed=true`, `test_set_used=false`), and its 100 identities remain excluded from selection/tuning. See D-2026-10-28-031.

## Structure-confidence SAR–optical registration prior art — 2026-10-28

The publisher-indexed Lian et al. paper adds registration prior art for structure-confidence-weighted phase congruency and coarse-to-fine matching. Its 60-pair evaluation and registration gains remain author-reported; the accessible evidence does not establish SAR-output fidelity, restoration-time optical utility, or a mismatch-safe fallback. The related-work matrix now has 110 rows (77 confirmed historical members, 33 additions with unresolved historical membership); against the 125-row lower bound, at least 15 original entries remain unaccounted for, with a 15–48 row gap if the inventory had exactly 125 rows. See D-2026-10-28-030 and the [bounded audit](../../experiments/literature_audits/registration_structure_confidence_phase_congruency_20261028_v1/report.md).

## Sim2Real SAR joint-restoration prior art — 2026-10-28

De Paepe et al.'s full-text-reviewed Sim2Real paper jointly targets SAR despeckling and sidelobe reduction using MOCEM simulations and acquisition metadata. Its reflectivity target is an effective reference rather than a physically measurable quantity, and its inserted CAD objects—not contextual backgrounds—receive electromagnetic scattering modeling. Real Umbra/Capella evaluation reports ENL and visual results but no quantitative sidelobe metric. This is adjacent SAR-only prior art, not optical-guided restoration or spatial super-resolution, and it does not alter the local B1 gate. See D-2026-10-28-040 and the [full-text audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).

## Literature-review-to-matrix coverage crosswalk — 2026-10-28

A bounded crosswalk added Capella SAR SR, 3MOS, SOMA-1M, and conditional SAR400 records to the matrix. Their tasks and incomplete source-group/split evidence do not establish optical-guided SAR restoration utility or a matched local baseline. The matrix now contains 110 rows (77 confirmed historical members and 33 additions with unresolved historical membership); at least 15 original records remain unaccounted for against the 125-row lower bound. Keep B1 at `not_promoted` and selection stopped pending authoritative source-group mapping. See D-2026-10-28-041 and the [crosswalk audit](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).

## ADAM as adjacent joint-processing evidence — 2026-10-28

The full published text for ADAM adds a SAR despeckling/compression method with an author-reported comparison to MERLIN→JPEG2000. Its 21-acquisition Hamburg pseudo-reference and different task make the result non-transferable to local joint SR or optical utility. Index it as adjacent prior art only; do not change the B1 gate or use it as the local SAR-only joint baseline. See D-2026-10-28-036 and the [audit](../../experiments/literature_audits/adam_joint_compression_despeckling_20261028_v1/report.md).

## Muzeau paper-to-code lineage remains unresolved — 2026-10-28

The author-linked repository history contains 14 reachable commits; its seven config-bearing revisions use 10 epochs and no versioned 100-epoch setting was found, while the reviewed manuscript reports 100 SR epochs. No run-bound artifact links those records. Preserve this discrepancy as a reproduction limitation, not evidence about the paper's actual run; it does not change the local B1 gate. See D-2026-10-28-038 and the [history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).

## Canonical degradation operator verification — 2026-10-28

Sixteen synthetic-only checks passed for normalization, FFT filtering, anisotropic blur, Gamma statistics, pre-speckle replay, and MMCANet/Cascade bindings to the canonical implementation. This supports code-level mathematical contracts, not physical fidelity to calibrated SAR or the local PNG acquisition process. Keep the representation-domain speckle pipeline described as controlled synthetic degradation; source-domain and target semantics remain open. No model result or gate changed. See D-2026-10-28-027.

## CUDA-only model execution interfaces — 2026-10-28

Reusable Cascade/SFT and JointJDSR/OGSRN/SwinIR-Joint train/inference launchers accept only `--device cuda` and check CUDA before creating run/output directories. The remaining model-forward entrypoints and wrapper matrices under `scripts/` now use the same CUDA-only contract; they reject `cpu`/`auto`, preflight before reading run inputs or creating outputs, and pass `--device cuda` to child commands. Python compilation and static fallback scans passed. No model or data was run, no result was produced, and no phase gate changed. The previously completed direct-fusion v2 diagnostic used CPU because its frozen protocol and runner explicitly selected CPU, and the runner aborts if CUDA is visible; successful completion therefore implies that process did not see CUDA. Its manifest records no CUDA preflight explaining why. The same-input v3 CUDA hardware repeat and scalar comparison are already complete. Preserve both records. Future model-forward work requires a fresh preflight in the approved GPU profile. See D-2026-10-28-023/024/026/062.

## MHFNet prior-art boundary refinement — 2026-10-28

The official publisher index identifies MHFNet's uncertainty term as training-example weighting for cross-modal box-label shifts, within a detector whose feature and decision stages handle alignment and detection-output disagreement. This narrows generic alignment-aware multimodal-fusion novelty but does not establish optical reliability for SAR reconstruction. Exact dataset partitions and controlled residual-shift protocol remain unverified. No B1/B2/B3 gate or experiment changed; see D-2026-10-28-021.

## OMA-SSR test-set loss wording and evidence eligibility — 2026-10-28

The current Wiley full text explicitly describes the entire test set as the source of the loss used for overall training, while its dataset section defines distinct training and test splits. Treat the published protocol as test-contaminated for evidence eligibility: do not use its reported test scores as independent evaluation. Actual code execution and whether the sentence is a wording/translation error remain unknown. This narrows the prior-art evidence boundary only; it does not change the B1 `not_promoted` / `stop_at_b1` gate or authorize B2/B3. See D-2026-10-28-017 and the [protocol recheck](../../experiments/literature_audits/oma_ssr_test_loss_protocol_recheck_20261028_v1/report.md).

## Nullable-metric consumer and device-interface correction — 2026-10-28

Source-level follow-through now handles undefined ratio-residual/edge correlations in the Phase-A2, joint-vs-cascade, and formal aggregators, preserving metric coverage. The joint-vs-cascade model audit requires CUDA. The approved elevated AISAR profile passed a CUDA preflight on the RTX 3050 Laptop GPU on 2026-10-28; the ordinary restricted shell remains unable to access the driver. No project model run occurred in that check, and no research gate changed. Future model-forward work still requires a fresh CUDA preflight. See D-2026-10-28-001 and D-2026-10-28-003.

## SEN1-2 source-group metadata route — 2026-10-28

The official GovData record and the MediaTUM OAI-PMH and JSON item-export routes are identified. The item export supports CC BY 4.0, but the OAI rights tag remains `restrictedAccess`; its precise meaning is unresolved. Official guest child exports are empty. An authorized rsync `--list-only` request confirms the four season roots and an upstream SHA-512 index. The retained index audit compared 3,600 fixed train/validation local inputs against all 564,768 upstream image hashes and found zero exact byte matches. This excludes only byte-identical mapping for those inputs; transformed copies remain possible. Bounded public GitHub searches did not surface a source crosswalk, and the third-party task derivative was excluded. The local numeric 2k cohort remains unidentified, so a prospective source-group split cannot yet be frozen. Continue source-independent work; do not infer provenance from filenames or patch geometry. See D-2026-10-28-002, D-2026-10-28-005, D-2026-10-28-008/010, and the [MediaTUM route audit](../../experiments/data_audits/sen1_2_mediatum_oai_metadata_recheck_20261028_v1/report.md).

### QXS-SAROPT author-source recheck — 2026-10-28

The author-maintained repository README and primary dataset paper expose no local numeric-filename-to-source-scene crosswalk. Their general dataset-description details do not identify this project's local PNGs. Keep source attribution unresolved and selection paused; reopen only when a direct versioned source manifest or per-file/group map appears. See D-2026-10-28-025.

### Google Drive source-record route recheck — 2026-10-28

The newly listed Drive search function still fails in the backend (`Unknown tool`) for focused metadata-only queries. No files were returned, so this does not establish source-record absence. Keep local subset lineage unresolved and do not retry the same route until the backend changes. See D-2026-10-28-028.

## Joint-SAR diffusion literature lead — bibliographic/access boundary (2026-10-28)

Crossref confirms the article's DOI and bibliographic record, but official IOP HTML/PDF routes return a bot-manager challenge and no author manuscript or repository copy has surfaced. OpenAlex lists only the IOP PDF and an unverified CC BY-NC-ND tag; Crossref's license URI resolves to general IOP copyright/permissions guidance rather than an article-specific license. Keep technical claims and article-specific reuse terms unverified; do not add a technical matrix row or use secondary abstract text as established prior art. This source-route issue does not alter the current B1 `not_promoted` / `stop_at_b1` gate or authorize model selection. See D-2026-10-28-014/015/016 and the [route audit](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).

## Related-work claim-map extension — 2026-10-27

The source-only claim map now includes current-evidence boundaries for learned SAR–optical registration, SpaceNet 9 tie-point registration, SAR-only self-supervised despeckling, and reverse-task SAR-guided optical reconstruction. This strengthens the novelty audit but does not change B1 `not_promoted` / `stop_at_b1`, authorize B2/B3, or substantiate any local effectiveness claim. The local contribution-to-eligible-run map remains open pending admissible prospective evidence. See D-2026-10-27-054.

## CPU/CUDA reproducibility of the direct-fusion diagnostic — 2026-10-27

The retained v2 CPU and v3 CUDA runs use the same fixed-input manifest, input binding, and three checkpoint hashes; a hash-bound comparison paired 1,200 scalar rows. Prediction-delta fields differed by at most `6.49e-9` absolute, fusion-feature delta RMS by at most `2.64e-6`, and the SAR-base delta was identical. This is a technical repeatability check only. It neither establishes optical utility nor changes B1 `not_promoted` / `stop_at_b1`; future model work still requires an approved CUDA preflight and explicit CUDA selection. See D-2026-10-27-047.

## Source-folder and top-level directory capacity — 2026-10-27 correction

The official source schema defines second-level `s1_i` / `s2_i` folders as scenes. The primary paper identifies `1970` as the fall random-sampling seed and reports 78 globally distributed fall scenes. The frozen 1,600/200 cohort has 15 distinct source-scene folders per partition (14 fall plus one summer in train; 15 fall in validation). The group IDs support a source-scene-folder-disjoint development split; retained records do not map those IDs to coordinates, product IDs, or acquisition dates, so no geographic or acquisition-independence claim follows. The separate numeric-name 2,000-pair subset remains unmapped. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-27-027 and the [scene taxonomy audit](../../experiments/data_audits/sen1_2_scene_folder_taxonomy_20261027_v1/report.md).


## Related-work inventory recovery — Git object route (2026-10-27)

**Fact:** The current 103-row matrix is not tracked and has no reachable Git path revisions. Four unreachable blobs were checked against its exact 12-column CSV schema; none matched. A pre-corruption read counted 119 rows before D-2026-10-25-46 added six, so the known post-update lower bound is 125; the exact final count remains unknown.

**Decision:** Keep the matrix incomplete. GCOANet and Gaya et al.’s IGARSS 2024 paper are confirmed by D-2026-10-18-08/D-2026-10-23-26 and D-2026-10-25-04/D-2026-10-25-65. A bounded decision-log reconciliation also recovered the program-level *Physics-Inspired Diffusion Model for SAR Image Despeckling* row from D-2026-10-19-29; the official IGARSS page verifies title, authors, and session only. The matrix now has 77 historically confirmed rows and 26 additions of unresolved membership. Given the 125-row lower bound, at least 22 original entries are absent from the current inventory; the exact gap remains unknown. Continue recovery only from a user-provided original or a materially new authoritative source. The Google Drive search tool reappeared in the 2026-10-27 catalog, but three metadata-only queries failed with backend `Unknown tool` and returned no files (D-2026-10-27-017); do not retry until backend state changes. Do not infer membership from DOI candidates or repeat unavailable routes without changed state. No model gate or scientific conclusion changes; see D-2026-10-27-016 and D-2026-10-27-017.


## Current-evidence registration prior art — 2026-10-27

Added Zhang and Wang's AIP Advances dual-channel SAR–optical registration paper as current evidence, with original matrix membership unresolved. It narrows the registration-prior-art boundary but does not test residual errors in SAR restoration, SAR-only fallback, or a local mismatch threshold. Keep B1 `not_promoted`; B2/B3 remain gated. See D-2026-10-27-008 and the existing publisher-text audit.

## Direct optical fusion-input sensitivity (2026-10-27; interpretation-only)

A zero-training fixed-checkpoint intervention zeroed the optical-HR and optical-context channel groups at the fusion-stem input while retaining LR/HR cross-attention. Across 200 repeatedly used development identities, three seeds, and 15 recorded source folders, the final prediction changed by about 0.0243% relative RMS under both matched and cross-folder-permuted optical. The v3 same-input hardware repeat completed on `cuda:0` and closely reproduced the scalar outputs; no cross-device tolerance was predeclared. This remains a small, out-of-training-distribution sensitivity measure; the interventions are not additive and do not identify image-quality benefit or optical utility. The run does not alter the B1 gate or authorize B2/B3. Preserve the pathway question as unresolved. Future model-forward diagnostics should use an approved GPU profile after a fresh preflight and explicit CUDA selection; defer them if unavailable, without CPU/`auto` fallback. See D-2026-10-27-014 and the [CUDA report](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v3/runs/attempt_001/report.md).

## CDSE Sentinel-1 IW SLC route — metadata-only review boundary (2026-10-26)

The frozen 12-AOI metadata screen met its eight-component route threshold with 10 active metadata-defined components. The asset-free protocol review is complete under D-2026-10-26-147; it found no approved IW SLC target because deramping, calibration/geometry handling, I/Q independence, and local S1–S2 registration remain unresolved, and authorized sidecar access is unavailable. Keep product access, split creation, training, and inference closed until authorized access and a frozen product-bound target protocol exist. Do not treat raw IW I/Q as independent MERLIN targets. B1 remains `not_promoted`; B2/B3 remain gated. See the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md) and [screen report](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/report.md).

## B1 v2 prediction-difference norms (2026-10-26; interpretation-only)

A zero-training, streaming follow-up found mean relative output RMS differences of 0.0105% for matched versus cross-source-group permuted optical, and about 0.129% for matched versus missing optical or the SAR-only checkpoint. This shows small but nonzero output sensitivity to optical pairing on the same 200-identity, 15-group, three-seed validation cohort; it does not establish no optical value, safe fallback, or generalization. B1 remains not_promoted, with B2/B3 gated. See the [diagnostic report](../../experiments/diagnostics/b1_v2_prediction_difference_norms_20261026_v1/report.md) and D-2026-10-26-134.

## Historical pilot test-text exposure (2026-10-26)

A capped search in the internal pilot manuscript surfaced historical metric text associated with its 100-identity test subset. The values are excluded from current evidence and were not used. Record task flags as test-set accessed/used = true/false and stop all future selection or tuning on those 100 identities. This does not rewrite the recorded B1 v2 gate or authorize B2/B3; lineage overlap between the pilot subset and other cohorts remains unresolved. See D-2026-10-26-138.


## Published dataset lineage crosswalk (2026-10-26)

Compare the local patch-spacing observation with the primary SEN1-2 and QXS-SAROPT construction papers. The geometry weakly conflicts with preserving the canonical SEN1-2 grid and is compatible with QXS-SAROPT's reported overlap ratio, but cannot identify the local files. Keep the subset project-labelled; continue seeking direct source/acquisition records and precise registration evidence. No gate change. See the [crosswalk audit](../../experiments/data_audits/dataset_lineage_source_paper_crosswalk_20261026_v1/report.md).

The author-hosted SEN1-2 manuscript further documents the collection's GRD IW VV σ⁰ dB input, orbit/DEM orthorectification, no speckle filtering, Sentinel-2 RGB preprocessing, and manual patch curation. This improves the publication-level processing description but does not identify the local PNG lineage or fine registration. Keep the model-selection pause and B1/B2/B3 boundary unchanged; see D-2026-10-26-136.

The paper calls the released pairs co-registered and says scene- or season-based splits can be independent; it also identifies the then-current Sentinel-2 data as L1C/TOA. Treat these as collection-level author statements, not local validation. The exact file/scene/acquisition crosswalk and independent local registration checkpoints are still missing (D-2026-10-26-137).

The pinned SEN12MS author utility notebook obtains region coordinates from GeoTIFF CRS/affine metadata and generates a shapefile at runtime. It does not provide a version-pinned crosswalk to SEN1-2 source folders or the local numeric 2k subset. An unresolved community issue reports discrepancies in the displayed region locations, so the generated map is not independent geospatial truth. Keep source grouping and fine registration unresolved; do not use matching region strings or this plot as evidence of cross-release identity. See the [utility-route audit](../../experiments/literature_audits/sen12ms_geospatial_utility_route_20261026_v1/report.md) and [issue #5](https://github.com/schmitt-muc/SEN12MS/issues/5).

## PNG container metadata lineage audit (2026-10-26)

A read-only header scan of the current SAR/optical PNG subset found no embedded text or ancillary source metadata. This does not resolve dataset lineage, acquisition pairing, or fine co-registration, and it does not change the B1 decision or unlock B2/B3. Continue source/product and registration evidence work before treating the local subset as a named release. See the [audit report](../../experiments/data_audits/png_header_lineage_20261026_v1/report.md).

## Interpretation-only optical-pairing diagnostic (2026-10-26; no gate change)

The existing B1 v2 optical candidate was compared under matched, cross-source-group permuted, and missing optical inputs, alongside its SAR-only checkpoint. The 200-sample, 15-group, three-seed validation diagnostic found near-zero metric differences between matched and permuted optical, while matched versus missing/SAR-only retained the known small PSNR/SSIM and radiometric trade-off. Treat this as one-cohort development evidence only. B1 remains `not_promoted`; B2/B3 remain gated. Full record: [diagnostic report](../../experiments/diagnostics/b1_v2_optical_pairing_sensitivity_20261026_v1/report.md).


## Cohort provenance checkpoint (2026-10-26)

The local numeric 2k source remains unresolved: exact pixel matches to a QXS-named neighboring cache do not identify its upstream release or parent scenes, and the official QXS repository offers no file-level crosswalk. SEN12TP’s official record page is reachable but does not disclose a dataset license and conflicts with the paper on the archive split; its target is NDVI rather than clean SAR. Keep the local model-selection pause and B1 `not_promoted` status. Resume cohort construction only from an authorized, version-pinned train/validation source manifest with verified parent groups and spatial separation. See the [source-route audit](../../experiments/data_audits/source_lineage_routes_recheck_20261026_v1/report.md) and D-2026-10-26-054.

## Related-work matrix recovery priority (2026-10-26)

The last confirmed pre-corruption inventory has 119 data rows and 12 columns. The current incomplete matrix has 99 rows: the preserved 46-row prefix, 28 historically confirmed reconstructions, and 25 current evidence additions whose historical membership is unresolved. The original row-level inventory remains unavailable, leaving 20–45 original rows unaccounted for. SARGANet and Muzeau et al. are current prior-art evidence, not recovered historical membership. Earlier dated-decision and explicit-phrase checks remain bounded and do not prove completeness. A further check of the local Zotero API, connected Drive search, official SEN1-2 record/PDF route, and bounded local directory names recovered neither the original row-level inventory nor a local data-source crosswalk; do not repeat those routes without a new source (D-2026-10-26-135). Continue only source-backed recovery, keep original-membership uncertainty explicit, and do not treat the DOI locator index as a row inventory. See D-2026-10-26-097 and the [matrix recovery report](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).

## 2026-10-26 — OGSRN full-text and local source-manifest route boundaries

The official SciOpen record labels OGSRN open access but the available extracted page exposes only the abstract and metadata, without a usable PDF body/link; the official ScienceDirect article and guessed PDF routes are inaccessible through the current reader. Keep its dataset, target, degradation, split, metrics, and implementation unresolved. The author-maintained SEN1-2 patch-matching repository documents SIFT matching annotations, not a restoration source manifest, and no exact mapping to local numeric IDs exists. Keep the local provenance blocker, B1 `not_promoted`, and B2/B3 gate unchanged; pursue only an authoritative source manifest or exact local crosswalk. See D-2026-10-26-023 and D-2026-10-26-024.

## B1 development uncertainty follow-up (2026-10-26)

A predeclared source-folder bootstrap on the hash-bound B1 aggregate gives a tiny positive group-equal PSNR mean and a negative SSIM mean. It adds uncertainty context only; B1 remains `not_promoted`. The current route still requires authorized non-test source/parent-ROI metadata for a prospective cohort before new model selection. See [D-2026-10-26-015](DECISION_LOG.md#d-2026-10-26-015--record-conditional-source-folder-bootstrap-uncertainty).

Clarify the grouping unit when designing that cohort: official SEN1-2 top-level roots combine an ROI-sampling batch and a season, while second-level folders are source scenes. The current 15-folder validation is scene-folder-disjoint within one fall root; it still lacks scene coordinates/acquisition identifiers and does not establish cross-season generalization. See [D-2026-10-26-016](DECISION_LOG.md#d-2026-10-26-016--clarify-sen1-2-root-and-scene-folder-units).

## SEN12TS temporal repeats and source-group prerequisite (2026-10-26)

The expanded acquisition-proxy grouping still yields only four connected components, so the earlier no-split result remains unchanged for acquisition-disjoint analysis. The official SEN12TS documentation defines filename coordinates as tile centers and allows multiple dated pairs per tile. A separate predeclared tile-only split now assigns 1,600 groups to train and 400 to development, with no tile overlap and at least 1,225.281 m prior footprint clearance. The partition shares 79 S1 date/track/lookpass proxies and 53 S2-date proxies across splits; it supports only future exploratory tile-held-out analysis and does not establish acquisition independence or local SEN1-2 lineage. A further metadata-only screen derives four epoch dates per stack and finds 465 repeated within-tile date slots, 213 overlapping same-tile row pairs, and 209 of 239 global acquisition-date proxies shared across train/development. Treat multi-date rows as dependent until pixel-level correlation and stability checks support otherwise. The existing 43-tile pilot plus 38-tile supplement already completed the predeclared train/development screen and stopped further sampling for this question under [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit); these metadata audits do not reopen it. The GeoTIFF tag route remains blocked with no findings. B1 remains not_promoted; B2/B3 remain gated. See [D-2026-10-25-189](DECISION_LOG.md#d-2026-10-25-189--keep-the-sen12ts-temporal-route-blocked-after-metadata-only-screens), [D-2026-10-26-001](DECISION_LOG.md#d-2026-10-26-001--accept-the-documented-sen12ts-tile-group-for-exploratory-splitting), and the [tile split audit](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/report.md).

Updated: 2026-10-26 (Asia/Shanghai)

## Keep noisy-pseudo-target claims within their conditional-mean assumptions (2026-10-25)

Noise2Noise Eq. 6 states an indexed clean-target condition. In population form, unrestricted squared-loss estimators match when E[Z|X]=E[S|X]; target noise with zero conditional mean given S and independent of X given S is sufficient, and this population conditional mean need not equal a sample-specific S. SAR2SAR and multi-temporal MERLIN use independent or explicitly excluded observations and handle SAR-specific temporal/spatial dependence. The local fixed-input path derives LR and HR pseudo-target from the same preprocessed SAR observation, and neither its latent target nor the population target-residual condition is identified. Treat local metrics as pseudo-target consistency; do not infer clean-SAR recovery. This source review changes no run or gate. See D-2026-10-25-178.

## Keep local synthetic degradation claims in the representation domain (2026-10-25)

The canonical LR synthesis chain is shared across MMCANet training/inference and the legacy baseline. Its default Gamma multiplier operates on normalized PNG representation values and on an already noisy pseudo-target, so it is a controlled composite-noise stress protocol rather than a calibrated physical ENL model. Focused synthetic tests verify code contracts, sampler moments, optional-blur kernel normalization, axis rotation, and current zero-padding behavior, not local SAR fidelity. If anisotropic blur is used in a future protocol, declare and justify image-axis orientation and boundary behavior. See [D-2026-10-25-169](DECISION_LOG.md#d-2026-10-25-169--characterize-optional-blur-with-synthetic-tests) and [D-2026-10-25-168](DECISION_LOG.md#d-2026-10-25-168--bound-the-canonical-degradation-to-representation-domain-evidence).

## Keep both legacy paper directories outside the active manuscript (2026-10-25)

`paper/MMCANet/` is an internal July engineering pilot with historical held-out result prose. `paper/MMCANet_formal/` is an archived 2k development-subset preview; its fixed train/validation partitions have known target overlap and do not establish independent generalization. Preserve both as audit records, but do not use their old results for current model selection, promotion, or submission claims. The root README now labels both directories accordingly. Any new manuscript must follow current provenance, split, pseudo-target, and gate constraints and use only admissible prospective evidence. This does not change B1/B2/B3 status; see [D-2026-10-25-167](DECISION_LOG.md#d-2026-10-25-167--classify-both-paper-directories-as-historical-archives) and [D-2026-10-25-166](DECISION_LOG.md#d-2026-10-25-166--retain-the-july-pilot-draft-as-historical-only).

## SOTANet-named checkout is not a source manifest (2026-10-25)

The neighboring checkout and the bounded public repository-page snapshot identify CycleGAN/pix2pix, not a versioned SAR–optical release. Neither provides a patch-to-source crosswalk, acquisition groups, rights, or split semantics for the local numeric 2k subset. Close this repository route; keep source lineage unresolved and model selection paused until an authoritative manifest or independently supported group mapping is available. This does not change the B1 stop or B2/B3 gates. See [D-2026-10-25-165](DECISION_LOG.md#d-2026-10-25-165--close-the-sotanetwork-named-checkout-as-a-data-provenance-route).

## DSRKD institutional source-route status (2026-10-25)

The corresponding author's Tsinghua Department of Electronic Engineering profile route did not provide a current DSRKD manuscript, protocol, or code. Its displayed publication list ends in 2015, the linked academic-papers subpage timed out, and the personal homepage was inaccessible in this review. Keep the dataset/training-protocol conflict unresolved and do not treat this route's silence as evidence that a manuscript or implementation does not exist elsewhere. This changes no phase decision or local experiment; see [D-2026-10-25-160](DECISION_LOG.md#d-2026-10-25-160--keep-dsrkd-protocol-unresolved-after-the-tsinghua-institutional-route).

## Optical–SAR registration prior-art boundary (2026-10-25)

Lan et al.'s full-text registration study adds a high-resolution HongTu-1/Google Earth registration pipeline, but its manually annotated control-point residual lacks a stated separation from coordinate refinement and its 16-pair affine test uses synthetic transforms. The commercially licensed HT imagery is not public, and the paper evaluates registration rather than SAR restoration. This narrows the prior-art map without resolving local SEN1-2 residual error or authorizing a local B2/B3 study. Keep the exact local product/baseline crosswalk and independently referenced spatial checkpoints as prerequisites; see [D-2026-10-25-158](DECISION_LOG.md#d-2026-10-25-158--classify-lan-et-al-as-registration-only-prior-art).

## SAR-DDPM prior-art boundary (2026-10-25)

Full author-manuscript review confirms conditional diffusion for SAR despeckling, with optical imagery used only to synthesize training pairs and no optical input at inference. Its real-scene evidence is no-reference ENL on two Sentinel-1 images. The mutable author repository is not bound to the paper's tables and has unresolved static interface inconsistencies. This narrows broad diffusion-despeckling novelty but does not address optical-guided joint restoration or justify a local diffusion run. Keep local diffusion deferred under the unchanged B1 and source/target-provenance gates. See [D-2026-10-25-156](DECISION_LOG.md#d-2026-10-25-156--record-sar-ddpm-scope-and-keep-local-diffusion-deferred).

## So2Sat LCZ42 candidate source boundary (2026-10-25)

The author-associated v4 train metadata shows no cross-city overlap within a shared EPSG code, but a buffered common-CRS envelope screen found one candidate pair among 835 different-EPSG city pairs. The exact patch review did not yield a valid count: 9,986 transformed boundaries were non-convex, invalidating the attempted convex clipper; its provisional output was removed. Treat city labels as unverified source groups and do not use them for a disjoint split. The classification benchmark also lacks a clean SAR restoration target, exact product/acquisition lineage, and measured S1/S2 residual registration. Keep it outside model selection unless those task-fit and provenance gaps are resolved. The incidental test-file LFS pointer metadata exposure is disclosed; no test identity or payload was read or used. B1 and B2/B3 statuses are unchanged. See [D-2026-10-25-154](DECISION_LOG.md#d-2026-10-25-154--keep-so2sat-city-labels-out-of-disjoint-splits).

## Noisy pseudo-target risk boundary (2026-10-25)

For the population squared-error Bayes estimators, `T=X+E` gives `ΔR_T=ΔR_X+E||δ_E||²+2E⟨δ_X,δ_E⟩` when comparing `Y` with `(Y,O)`. This distribution-free identity leaves the proxy-target versus latent-signal ordering open unless the conditional-mean change in residual `E` is controlled. The local LR and target share the same preprocessed noisy SAR tensor, so no such independence/zero-increment condition is established. Treat historical proxy-target metrics as consistency observations; do not interpret them as clean-SAR optical utility. This theoretical clarification authorizes no experiment and leaves the source-group/target/registration work and B1/B2/B3 statuses unchanged. See [D-2026-10-25-152](DECISION_LOG.md#d-2026-10-25-152--record-the-exact-noisy-pseudo-target-bayes-risk-decomposition).

## Source-route status after metadata screens (2026-10-25)

The new global CDSE AOI screen found three independent candidate groups among seven complete AOIs; its only incomplete AOI could raise the maximum to four, below the frozen six-group threshold. Stop that CDSE route without assets or further pagination. The mediaTUM candidate node `1474000` remains unverified: the direct page had a generic title and the frozen exact-domain search returned no result, so no record or child export was requested. Continue the existing source/target and local-registration investigation; model selection remains paused, B1 remains `not_promoted`, and B2/B3 stay gated. See [D-2026-10-25-147](DECISION_LOG.md#d-2026-10-25-147--stop-the-global-cdse-route-at-metadata-screening), [D-2026-10-25-148](DECISION_LOG.md#d-2026-10-25-148--keep-mediatum-node-identity-unverified-after-direct-page-check), and [CURRENT_STATE](CURRENT_STATE.md).

The FDL SEN12MS mirror's exact raw README request returned HTTP 404, so the earlier indexed bulk-sync excerpt remains the only mirror evidence; it still supplies no train-only geospatial crosswalk. Do not guess a new path or list the bucket. Reopen only with an identified, authorized selective metadata route. See [D-2026-10-25-150](DECISION_LOG.md#d-2026-10-25-150--leave-the-fdl-sen12ms-mirror-route-unresolved-after-404).

The candidate PolSAR/SinSAR diffusion-SR citation remains outside the formal literature synthesis: its single direct IEEE record request returned HTTP 202 without structured title, author, DOI, or abstract metadata. Reopen only on a readable official or author-controlled source; the lead does not affect the diffusion feasibility gate. See [D-2026-10-25-149](DECISION_LOG.md#d-2026-10-25-149--retain-the-polsarsinsar-diffusion-sr-item-as-an-unverified-lead).

OGSFlow source status (2026-10-25): a root-page snapshot crawled yesterday lists `assets/paper` and four commits; a separate README snapshot crawled five days ago says the public paper link and final citation are forthcoming. No commit SHA binds the snapshots, and the paper filename remains unavailable. The versioned folder/history routes could not be retrieved, so retain only the existing mutable-README prior-art boundary. This does not change the literature gate, B1 status, or B2/B3 restrictions; see [D-2026-10-25-151](DECISION_LOG.md#d-2026-10-25-151--keep-the-new-ogsflow-repository-signal-unverified).

Mamba-SCR source-route status (2026-10-25): the publisher reports a 49-summer-ROI SEN12MS-CR subset and links a Zenodo metadata release, but the third-party split is not crosswalked to the original TGRS holdout or current catalogue version. Keep it out of local cohort construction and model selection; continue requiring an authorized, version-pinned source-group manifest. See [D-2026-10-25-174](DECISION_LOG.md#d-2026-10-25-174--treat-mamba-scr-split-metadata-as-non-authoritative).

## SpaceNet 6 auxiliary candidate boundary (2026-10-25)

The official expanded MSAW page and its linked September 2020 release article agree on 202 SAR strips; the article specifies 202 unique strips. The April 2020 arXiv v1 manuscript reports 204, and no source maps the difference to specific strips. The optical scene is 7–27 days later than the SAR strips, the SAR is airborne X-band log intensity after multilooking, and no clean restoration target or measured SAR–optical residual is supplied. The challenge's final test area withholds optical data. Keep it as a conditional auxiliary candidate for future geometry/structure diagnostics only; do not use it as the primary B1 cohort or as evidence of local Sentinel-1/2 alignment. Reopen only after an authorized, versioned strip manifest reconciles the count, a train/development-only source-strip and footprint graph is built, and an explicit noisy-target protocol is frozen. See [D-2026-10-25-138](DECISION_LOG.md#d-2026-10-25-138--distinguish-spacenet-6-release-count-from-manuscript-count) and the [screen](../../experiments/literature_audits/spacenet6_expanded_candidate_screen_20261025_v1/report.md).

## Quantitative Sentinel-1/2 registration precedent (2026-10-25)

Ye et al.'s six-pair study provides direct evidence of measurable Sentinel-1/2 offsets and terrain-dependent held-out checkpoint error. Treat its correspondences as an evaluation-design template, not external geodetic truth or a local SEN1-2 measurement. Keep exact product/baseline mapping and independently referenced local checkpoints as prerequisites for any local registration claim. This does not promote B1 or authorize B2/B3; see [D-2026-10-25-133](DECISION_LOG.md#d-2026-10-25-133--use-sentinel12-registration-checkpoints-as-a-method-template-only).

CDSE route update (2026-10-25): the frozen 12-AOI screen advanced only to product-protocol review. The 80-item metadata inventory and rights review show a viable sidecar route for calibration/noise and S2 scaling metadata, but no product files or pixels were fetched. All 12 screened AOIs have same-orbit S1 repeat groups with 3+ acquisitions over 47–59 days; temporal stability and a clean reference remain unverified. Resolve mixed S1 product-type metadata and read only the required XML sidecars before deciding whether a prospective group-disjoint split is supportable. Do not split, train, or change B1/B2/B3 yet; see [D-2026-10-25-104](DECISION_LOG.md#d-2026-10-25-104--bound-cdse-asset-metadata-and-repeat-stack-evidence) and the [inventory report](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/report.md).

CDSE access update (2026-10-25): the selected S1 sidecar hrefs use the `s3://eodata` scheme, while CDSE's official S3/OData routes require a registered credential. No credential variable or S3 client is available in the current environment, and no sidecar bytes were fetched. Keep the five-item/11-sidecar selection frozen; resume only after authorized credentials are provided through a secure local source. Continue source-backed work that does not depend on the XML; see [D-2026-10-25-105](DECISION_LOG.md#d-2026-10-25-105--pause-cdse-sidecar-fetch-until-authenticated-access-is-configured).

Provenance-route follow-up (2026-10-25): the official QXS-SAROPT issue requesting a non-Baidu mirror remains open without maintainer response. No versioned source archive or patch-to-group mapping is available through that issue, so local data provenance and source-disjoint development remain unresolved; model selection stays paused. A separate SOTANet page-read response was truncated and its text is excluded from evidence. See D-2026-10-25-93/94 and their linked audits.

Execution update (2026-10-25): Official band mapping was verified, and a frozen train/development-only temporal screen ran on 43 pilot tiles plus a second hash-ranked tile for 38 eligible components (26 train, 12 development; five components were singletons). Schema and mask-coverage checks passed. Median local-mean changes were similar across cohorts, but component-level differences varied; normalized-residual correlations remained positive, so temporal independence and a clean SAR reference are not established. Per D-2026-10-25-98, stop external sampling for this question and keep the temporal power mean labeled a noisy pseudo-reference. No local test payload, training, inference, model selection, or B1/B2/B3 gate change occurred. See the [pilot report](../../experiments/data_audits/sen12ts_temporal_reference_pilot_20261025_v1/report.md), [paired supplement](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/report.md), and [decision](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit).

Prior-art update (2026-10-25): Gaya et al.'s official GRETSI 2023 full text describes early/intermediate optical fusion for complex-SAR MERLIN despeckling and reports qualitative one-scene evidence without quantitative validation; its preliminary two-pixel registration statement is not a systematic mismatch study. The same authors' distinct-title IGARSS 2024 paper remains abstract-level and its version relationship to the 2023 paper is unverified. This narrows broad optical-guided despeckling novelty but does not add joint-SR, local optical-utility, or quantitative fallback evidence. No experiment or phase gate changed; see [D-2026-10-25-66](DECISION_LOG.md#d-2026-10-25-66--classify-fuse-merlin-as-single-scene-qualitative-despeckling-prior-art) and the [full-text audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md).

2026-10-25 cohort feasibility update: the authenticated v2 cohort is source-folder-disjoint but not parent-ROI/season-disjoint. Its only summer component is 106 train pairs in one folder; holding out fall would leave that as the training set, while holding out summer produces one validation cluster. Do not use the spent v2 validation for further model selection or cross-ROI claims. Continue provenance work for an authorized, multi-ROI prospective cohort; keep B1 not_promoted and B2/B3 gated. See [D-2026-10-25-13](DECISION_LOG.md#d-2026-10-25-13--do-not-derive-a-parent-roi-holdout-from-the-v2-cohort).

2026-10-25 update: A source-backed moment derivation confirms that the B1 v2 Gamma condition is representation-domain only; it supplies no physical-speckle validation and changes no phase gate. Keep the prospective B1 decision unchanged. See [D-2026-10-25-12](DECISION_LOG.md#d-2026-10-25-12--bound-the-inverse-moments-of-representation-domain-gamma-noise) and the [derivation audit](representation_domain_gamma_composition_20261025.md).

Prior-art update (2026-10-25): MD-Glow adds direct single-modality SAR multi-task despeckling/SR prior art. Its ablation suggests that auxiliary SR/multiscale training can affect despeckling differently by resolution and metric, but its tables do not quantify SR restoration and do not compare matched joint versus cascaded outputs. This narrows generic joint-task novelty; it does not change the optical-utility question, data gate, or B1/B2/B3 status. See [D-2026-10-25-08](DECISION_LOG.md#d-2026-10-25-08--record-md-glow-as-sar-multi-task-despeckling-sr-prior-art).

Prior-art update (2026-10-25): the 2026 JSTARS MuChaPro-based InSAR article adds abstract-level prior art for spatially regularized phase/coherence restoration from despeckled projections. Its phase/coherence task and unavailable detailed protocol do not validate the local intensity-SAR forward model or optical-guided joint SR. H9 remains blocked on source/product/domain provenance; no local experiment or gate changed. See [D-2026-10-25-07](DECISION_LOG.md#d-2026-10-25-07--bound-muchapro-insar-spatial-regularization-prior-art).

## Research objective

Determine when and how optical imagery can improve SAR joint despeckling and super-resolution without transferring unsupported optical texture or degrading SAR structure. Develop a defensible method only if the evidence supports one, then validate and write a reproducible paper. The working title, MMCANet-v2, SAR-dominant residual fusion, reliability factors, and diffusion are candidate framing or routes, not fixed outcomes.

## Core questions

1. Does matched optical provide stable value over a matched SAR-only joint model under paired, full-metric evaluation?
2. Which optical information (geometry, edge, gradient, frequency band, texture, semantic cues) helps, for which images and regions, and under what data conditions?
3. How do optical utility and negative transfer change with speckle/degradation strength and with despeckling-only, SR-only, or joint objectives?
4. Can wrong alignment or stale content be detected and rejected while preserving valid optical utility?
5. Which model family is justified: a calibrated discriminative fusion model, a generative conditional model, or a physics-constrained inverse method?

Questions are falsifiable. The research can conclude that a given approach is not justified by this dataset/protocol.

## Candidate research matrix (questions only)

This matrix records possible distinctions to evaluate after the literature, data, and current-paper lineage audits. It does not authorize an experiment, expand an existing gate, or make a candidate a contribution. The failed B1 gate remains unchanged; B2/B3, stale-optical, diffusion, and new fusion training remain gated.

| Axis | Candidate conditions | Evidence status and constraint |
|---|---|---|
| Speckle/degradation strength | Low, protocol-matched, and high controlled representation-domain noise; the historical formal protocol labels looks settings `L∈{1,2,4,8}`. | Verify the authoritative config/operator and parameterization before reuse. Local PNG semantics are unknown, so these are not calibrated physical-speckle levels. No change to the B1 protocol. |
| SR scale | ×2, ×4, ×8. | The active aligned-optical B1 question is ×2. ×4/×8 would require new fixed inputs, matched baselines, and a prospective protocol; prior-art coverage does not establish local benefit. |
| Task decomposition | Despeckling/restoration only, SR only, and joint despeckling+SR. | Compare each to a task-matched SAR-only baseline and preserve target/degradation meaning. Joint vs. cascade is a separate controlled comparison, not an inference from Phase-A2. |
| Optical representation | RGB appearance, edges/gradients, multi-scale structure or frequency cues, and learned optical residual features. | Every cue must be available at inference and assessed against proxy-target, structure, and bright-region guardrails. The current pairing observation does not select a cue. |
| Joint vs. cascade | SAR-only joint backbone; sequential SAR despeckling→SR; optical-guided joint restoration; training-time optical guidance. | Camera-RAW prior art already compares task orders and sequential versus end-to-end processing (Qian et al., ICCP 2022) and fixed-order partial/full joint solutions (Xing and Egiazarian, CVPR 2021), so broad order-novelty claims are unsupported. Existing methods remain non-interchangeable baselines. Lee and Son's MONTI 2026 study compares ordered structure-restoration→speckle-synthesis with single-stage SAR SR, but targeted text does not report reversed task order or a matched joint baseline. A future H8 audit must cross task order (`D→SR`, `SR→D`) with training coupling (stage-wise, end-to-end), then compare both orders and the direct-joint control under matched LR tensor, target, split, evaluator, and training provenance. See the [operator-level comparison design](restoration_order_analysis.md); preserve Phase-A2 as reference only. |
| Reliability/fallback | Geometric correspondence confidence, local residual mismatch, and a target-free local usefulness/fallback signal. | No local cue is currently supported for restoration benefit. First specify a falsifiable inference-time rule and validation-only diagnostic; do not begin B2/B3 or stale-content training from this matrix. |
| Model family | Discriminative fusion, conditional diffusion, or physics-constrained/unrolled inverse methods. | These are route hypotheses. Review task-matched primary sources, target/data assumptions, compute, and failure risks before choosing one. No diffusion/unrolling implementation or training is implied. |

Any future expansion should change one axis at a time and state its matching baseline, data split, fixed degradation, metrics, seed/compute budget, and stop rule before execution.

The 2026 ProSR full-text audit adds discrete-token SAR ×4 SR prior art and a task-specific scattering-peak concern; its Umbra SLC protocol does not alter the optical-guided joint-restoration questions or B1/B2/B3 gates. See [D-2026-10-24-36](DECISION_LOG.md#d-2026-10-24-36--bound-prosr-as-discrete-sar-sr-prior-art).

## Evidence standard

- An existing checkpoint or one-seed diagnostic can establish behavior for that artifact and split, not general performance.
- Promotion requires the phase-specific predeclared gate, multiple seeds where specified, full fixed validation membership, per-image paired metrics, all guardrail metrics, and a complete manifest.
- Development validation is not a holdout. Do not read test imagery or measurements for design or selection.
- The local fixed development partitions are additionally known to share exact source/target pixels across train/validation and fit/inner-dev. Their historical results remain unchanged but are not independent generalization evidence; no future model selection or promotion may use them. A preferred v2 frozen 1,600/200 cohort with 15 train and 15 fall-only validation source folders has now passed official per-pair authentication and source-folder/declared 128-grid checks; it supports only a new prospective screen after the task protocol, group-aware uncertainty, and resources are frozen. See the [local overlap audit](sen1_2_train_val_spatial_overlap_audit.md) and [prospective cohort audit](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v2/report.md).
- Current local images are normalized 8-bit PNG and targets are noisy pseudo-targets. Metric gains are proxy-target consistency, not proof of clean, calibrated SAR recovery.
- Preserve all historical results and gates. A new protocol can be proposed only prospectively.

## Phases and present status

Maintenance-only update (2026-10-24): the historical MMCANetV2 validation evaluator now checks checkpoint test-access flags and binds its validation split/manifest hashes before inference. This changes no research phase, model, result, or promotion gate; see [D-2026-10-24-43](DECISION_LOG.md#d-2026-10-24-43--bind-historical-mmcanetv2-validation-to-checkpoint-input-metadata).

Provenance-route update (2026-10-24): a newly surfaced Figshare QXS-derived deposit does not publish a local file-to-source-group crosswalk, and its page keeps upstream imagery subject to upstream terms. It does not reopen data adoption or model selection; see [D-2026-10-24-44](DECISION_LOG.md#d-2026-10-24-44--classify-the-figshare-qxs-derived-deposit-as-a-non-resolving-provenance-route).

Candidate-screen update (2026-10-24): EarthMiss offers multi-city paired SAR–optical segmentation data, but the public terms prohibit secondary development and the release supplies no restoration-specific noisy/clean target protocol. Do not use it for the restoration track without clarified rights and task-fit; see [D-2026-10-24-45](DECISION_LOG.md#d-2026-10-24-45--keep-earthmiss-unadopted-pending-restoration-use-permission-and-target-fit).

Candidate-screen update (2026-10-25): SARptical supplies a potentially useful urban scatterer/geometry pairing context, but the primary paper only establishes center-point 3D correspondence and approximate patch alignment. The author-linked release provides no reviewed dataset-specific reuse license or patch-to-acquisition manifest, while DLR describes TerraSAR-X usage as application-dependent. Keep it unadopted and outside the B1 benchmark unless rights, source groups, and a suitable restoration target are verified; see [D-2026-10-25-01](DECISION_LOG.md#d-2026-10-25-01--keep-sarptical-unadopted-pending-rights-and-source-lineage).

Candidate-screen update (2026-10-25): MultiResSAR is also registration-only: its author paper describes manually checked sparse control points and an 850-pair Umbra test subset, while the live repository exposes only 1,100 pairs without a subset/source-group map or visible data license. It provides no SAR restoration target, so it does not unblock B1 source provenance or task-matched validation; see [D-2026-10-25-03](DECISION_LOG.md#d-2026-10-25-03--keep-multiresar-unadopted-for-sar-restoration).

The separate named-cache crosswalk is now complete: zero exact modality or paired matches for 1,700 eligible local train/validation pairs versus 14,400 candidate composites. This closes that exact mapping route without recovering local source groups; see the [audit report](../../experiments/data_audits/sen1_2_sotanet_named_cache_crosswalk_20261019_v1/report.md).

The 2026-10-22 bounded literature continuation reviewed alignment-aware SAR-guided optical reconstruction prior art. SAG-CR tests injected SAR shifts but reconstructs optical imagery; CGD-Net reports registration, temporal-mismatch, and mask perturbation tests in a multi-temporal optical cloud-removal task. A 2026-10-24 author architecture note clarifies that CGD-Net's gate combines cloud-mask/prior reliability for optical-reflectance output, not SAR-output fallback. These works narrow general alignment/fusion novelty claims but do not answer the project's SAR-output fidelity/fallback question. A separate DA-PhysDiff bibliography lead is visible only in secondary indexes, so it remains outside the evidence-backed matrix pending official verification. B1 remains failed, B2/B3 remain gated, and the search is not exhaustive; see the [literature review](literature_review.md) and [CGD-Net source audit](../../experiments/literature_audits/cgdnet_architecture_source_followup_20261024_v1/report.md).

A 2026-10-24 targeted institutional search surfaced published-version PDF text from sections 3.3–3.6 and 5 and Tables 2–8 for Amieva et al. The details support recording its IW→SM joint SR/despeckling protocol, three full-image test scenes, author-reported comparisons, and pre-split full-image-set percentile normalization. Direct PDF opening still times out; exact statistic pooling, independent geographic grouping, registration residuals, and official code remain unresolved. This is SAR-only prior art, not optical-guidance evidence or a matched local baseline. See the [indexed-text audit](../../experiments/literature_audits/amieva_indexed_fulltext_update_20261024_v1/report.md).

The historical `paper/MMCANet_formal/` preview is retained but is not the current submission source after the local split-overlap and provenance audits. Its existing results remain historical development records and must not be used for new selection or promotion. A 2026-10-22 text-search exposure of a test-labelled generated table is disclosed separately; no surfaced value was used, and `test_set_accessed=true` for that turn.

The historical B1 artifact's source-hash equality test was re-run on 2026-10-22 and still fails for four current untracked source files. Keep the original manifest and results unchanged; source-level authentication requires an authorized exact archive. This does not alter the failed B1 gate or validate the old result against current code.

The recorded commit equals current HEAD, but the four mismatching files are absent from its tree and have no path history in local reachable Git refs. A read-only GitHub API check found only the same `main` commit and no per-path commit history for those files. Unreachable Git objects were not scanned because they may contain experiment payloads. An authorized external source snapshot is still needed.

The official Mendeley v1 record for the 2026 multitemporal SAR-only candidate was also rechecked on 2026-10-22: `Download All` is visible, but the extracted `Files` section remains empty. No files were retrieved, so the candidate's input-to-reference stack membership and pixel semantics remain unaudited and it is not adopted.

A metadata-only `load_frozen_context()` call revalidated the actual prospective B1 protocol context: 1,600/200 identities, 15 validation groups, registration-addendum binding, and false test flags. The fixed payloads were not opened and remain pruned, so this confirms metadata consistency only; rebuild inputs before a future run.

The related-work matrix was re-parsed after that update; a pre-existing malformed SORF evidence field was corrected, and all 85 records now match the declared 12-column schema.

The initial official-source probe authenticated one candidate pair and found all 30 filename groups crossing the cache’s native train/val partitions; those native partitions remain unsuitable for independent validation. A 24/6 cohort was then authenticated. Before any model metrics, a capacity-aware v2 allocation increased validation to 15 disjoint fall source folders while retaining 1,600/200 samples; all selected pairs and the declared overlap checks pass. Use v2 for prospective model work, not as a final test or proof of acquisition/geographic independence. See the [initial probe](../../experiments/data_audits/sen1_2_named_cache_official_pair_probe_20261021_v1/report.md), [v1 audit](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v2/report.md), and [preferred v2 audit](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md).

The 2026-10-21/22 artifact cleanup changed no research phase, hypothesis, or promotion gate; it removed rebuildable legacy and experiment pixel arrays, named smoke/error artifacts, generated caches/builds, and four checkpoints from an explicitly invalidated train/test-overlap run. Necessary records and the original pilot manifest remain; the manifest describes the pre-prune archive and does not match the reduced tree. Duplicate verification read the legacy mini test subset and matching source PNGs; exclude those 100 identities from future selection. Follow-up membership metadata places all 100 in fixed train and none in validation; no fixed-test manifest was read. See the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).

| Phase | Objective and exit condition | Status |
|---|---|---|
| 0. Project audit and infrastructure | Consolidate source of truth, register evidence, expose gaps and reproduce a bounded diagnostic. | In progress; core docs and one diagnostic are recorded. Full source, data-provenance, and manuscript audits remain open. |
| 1. Literature and problem formulation | Read task-matched primary work; distinguish prior joint and optical-guided methods; define candidate gap and hypotheses. | First structured review now includes full-text review of SpaceNet 9 as tie-point-grounded registration prior art, a standalone HSFMamba author-manuscript audit, and a WMoE-Net source/data screen. A publisher-indexed WMoE-Net article record and abstract are now verified for oriented ship detection; its full text and online-publication date remain unavailable, and the QXS-SAROPT-SHIP record's lineage/rights remain unresolved. It is adjacent detection prior art, not SAR restoration evidence. HSFMamba confirms direct inference-time optical-guided joint SAR SR/denoising prior art; its BM3D-reference/noise protocol, missing Gamma parameterization, and lack of quantitative residual-registration stress evidence remain explicit limits. The current OGSFlow README also says its Stage-2 source-noise scale is calibrated on validation after S1 selection, but the estimator and paper citation remain unavailable; retain it as repository-level SAR-SR prior art. Full-text checks and systematic search coverage remain incomplete. The 2026-10-23 DSRKD publisher/author/GitHub recheck yielded no full article or official code, so its dataset/training discrepancy remains open. SpaceNet 9 is not restoration evidence. See the [HSFMamba audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md) and [WMoE-Net screen](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md). |
| 2. Data and evaluation audit | Confirm local pairing, representation, split membership, target, degradation, metrics, and leakage limits. | Exact spatial/target overlap is confirmed: 580 train–validation and 440 B1F fit–inner-dev geometries. A train-only basename audit found 1,467/1,800 local numeric development names in the current, unpinned ReFlowSET QXS train list, but this does not resolve dataset/version identity or source groups; 333 names absent from that list remain unclassified. Original numeric-2k parent-scene/patch mapping remains unavailable. For the official 1,800-pair cohort, the source paper documents same-season mosaicking but no pairwise product IDs, timestamps, time-gap rule, or residual alignment; see the 2026-10-23 audit. Historical metrics/gates are preserved but the current partitions cannot support independent validation or future selection. The 63-bin cutoff is legacy-only: source-aligned B1 v2's bound train/validation inputs explicitly use 129 bins as frozen, although the CLI default remains 64 for replay compatibility; see the [2026-10-24 cross-check audit](../../experiments/data_audits/sen12_v2_degradation_contract_crosscheck_20261024_v1/report.md). The representation-domain multiplier remains unclipped above 1. |
| 3. Optical information utility | Diagnose matched/permuted/missing/SAR-only under paired validation and guardrail metrics. | The current-source, source-folder-disjoint B1 v2 six-run matrix is complete; the unchanged gate returned `not_promoted` because mean ΔPSNR was `+0.001617 dB` (<`+0.05`) and mean ΔSSIM was `−0.000892` (<0). Stop at B1; B2/B3 remain gated. The former four sealed runs are retained but excluded because four source hashes differ. The result is limited to this fall-only development cohort and nominal registration. |
| 4. Candidate architecture exploration | Test one falsifiable route after utility evidence, with an inference-available rule and matched SAR-only baseline. | No candidate selected. Diffusion and new reliability training deferred. |
| 5. Method selection | Compare only justified candidates; lock a single method, gates, and ablations. | Not started. |
| 6. Formal validation | Multi-seed, task-matched comparisons, sensitivity and robustness, efficiency, paired uncertainty, reproducible reports. | Not started for a promoted method. Existing historical formal dev2k results are context only. |
| 7. Paper finalization | Write claims directly from final artifacts; build and visually inspect the paper. | Drafts exist; current research claim is not established. |
| 8. Independent review | Audit novelty, method, data, fairness, reproducibility, limitations, and claims. | Not started for a final method. |

## Immediate decision path

**SOTANet helper source check (2026-10-25):** The only adjacent `prepare_sar2opt_data.py` found locally is Git-ignored and absent from commit history. Tracked history contains a `sar2opt` visualization example path whose referenced PNG is absent from current HEAD, but no QXS/SEN1-2 source crosswalk. The helper's hard-coded SEN1-2 source mount/root is currently absent, is not a versioned source record, and does not bind the separately named QXS cache or local 2k subset. Keep those sources unresolved; use only an authorized archive/version manifest or authoritative source-group crosswalk to restore lineage. See [D-2026-10-25-163](DECISION_LOG.md#d-2026-10-25-163--do-not-treat-the-ignored-sotanet-helper-as-source-lineage).

**Conditional alternate data route (2026-10-25):** Standard SEN12MS has a train-list metadata inventory of 226 provisional ROI/scene keys across four season branches, but the groups have not been checked for spatial overlap, its task-specific reuse terms remain unresolved, and its dual-pol dB SAR / cloud-adaptive optical mosaics require a new representation-domain noisy-pseudo-target protocol. Treat it as a candidate for a prospective group-disjoint development split only. Do not infer local numeric-2k provenance, inspect candidate holdout/test lists, or change the B1/B2/B3 gates. See the [train-side audit](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/report.md).

The follow-up filename audit found that the numeric scene/data token is reused across branches (78 tokens occur in multiple seasons), so a numeric token alone cannot serve as the cross-season key. The branch-qualified inventory still does not establish geographic independence; a train-only spatial/source crosswalk remains a prerequisite for any prospective split. See the [namespace audit](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/report.md).

1. Resolve the source chain for the located QXS-named cache and local subset. The JDSR workspace still contains only the project-labeled 2,000-pair set, but a sibling Git-ignored `SOTANet/datasets/sar2opt-QXS` cache contains 20,000 composites (16,000/2,000/2,000). Exact panel comparison maps 1,626 of 1,800 fixed train/validation pairs to same-name cache train/val composites; 174 names appear only in the cache's test directory by filename. This strongly links the local pairs to that cache, but not to an official QXS-SAROPT release: archive/version, license, split semantics, and parent-scene metadata remain unresolved. The QXS maintainer says the public release lacks a city-to-patch map and the original SAR scenes cannot be shared without source-center consent ([issue #3](https://github.com/yaoxu008/QXS-SAROPT/issues/3), [issue #4](https://github.com/yaoxu008/QXS-SAROPT/issues/4)); a 2026 request for a non-Baidu mirror has no visible reply ([issue #13](https://github.com/yaoxu008/QXS-SAROPT/issues/13)). The paper's “stride 52” wording also conflicts with conventional step-size arithmetic for its claimed 20% overlap, so local overlap geometry is not a source fingerprint. See the [crosswalk audit](../../experiments/data_audits/sar2opt_qxs_cache_crosswalk_20261019_v1/report.md) and [primary QXS paper](https://arxiv.org/html/2103.08259). Recover an authorized archive/version and verified source mapping if available; otherwise do not create guessed source groups. Pixel-overlap components and cache folder labels are not source groups. Do not use current contaminated validation or inner-dev for model selection/promotion.
   An alternative, [OSdataset](literature_review.md#osdataset-alternative-sar-optical-source-candidate-2020), has author-reported 20 GF-3/Google Earth parent scene pairs, but its distributed SAR is 8-bit display-adjusted amplitude and the reviewed materials expose no crop-to-scene mapping or acquisition dates. Google's [Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/) allow attributed research/education use while the [Maps/Earth Additional Terms](https://www.google.com/intl/en-US/help/terms_maps/) also restrict redistribution and bulk downloading. Do not download or adopt this candidate until its archive-level rights are established; then audit group mapping and registration before considering a representation-domain study.
2. Preserve all failed B1/B1R/B1F and Phase-1 R/S/T gates and saved values as historical evidence.
3. The preferred 15/15 source-folder-disjoint cohort and prospective B1 protocol were frozen and used for the source-aligned 2026-10-23 six-run matrix. The earlier v1 seed-0 validation failure remains historical. The current-source matrix completed under protocol v2 with all six runs sealed and 200 canonical validation rows each; the frozen gate returned `not_promoted` because mean ΔPSNR `+0.001617 dB` missed `+0.05 dB` and mean ΔSSIM `−0.000892` missed zero. All seed ΔPSNR values were positive and other declared guardrails passed. The cohort remains fall-only, nominally source-matched with fine alignment unverified, and uses normalized PNG-derived noisy pseudo-targets. Inputs and per-epoch intermediates were pruned after aggregation; manifests, source/input hashes, selected checkpoints, logs, and canonical outputs remain. This does not rewrite historical B1 and does not establish that optical information is intrinsically useless. Stop at B1; do not run B2/B3 unless a future authorized B1 gate passes under a prospective protocol.
4. Treat the B1F pairing and entropy results as bounded observations conditional on a contaminated validation partition: pair assignment affects this checkpoint's output, metric tradeoffs persist, and the tested entropy cue does not consistently rank the eight utility outcomes.
5. The descriptive per-image/fixed-region analysis and matcher-entropy cue screen are complete; do not repeat them or use them to tune a checkpoint or gate.
6. Continue primary-source literature and local data/target provenance review. The new v2 protocol uses a predeclared 129-bin support and unclipped representation-domain inputs; label it controlled composite noise, not calibrated physical speckle. The legacy 63-bin cutoff remains a historical limitation and must not be silently substituted.

## Formal-test interface checkpoint (2026-10-24)

Formal test access is now fail-closed on a hash-bound promoted B1 decision. The freeze and fixed-input workflow records post-promotion preparation separately from metric use; test runners verify the freeze before reading test metadata. Qualitative identities and spatial display locations are filename-hash based and bound to a pre-test selection manifest. This code safeguard does not alter the scientific gate: current B1 remains `not_promoted`, B2/B3 remain gated, and the historical target-selected preview is excluded until an eligible future run regenerates it. See [D-2026-10-24-30](DECISION_LOG.md#d-2026-10-24-30--require-b1-promotion-before-formal-test-inputs-and-qualitative-access).
7. Before training, recheck the accelerator profile and regenerate fixed inputs from the retained v2 manifests. The native-run sealer validates trainer/config/checkpoint selection, fixed-input and test-boundary metadata, and creates the aggregator's selected-epoch validation records; synthetic six-run integration passes. The previous CPU rebuild verified source, file-list, split, and protocol hashes. The seeds, metrics, resource estimate, stop rule, and falsifiable explanation are already recorded. If suitable compute is unavailable, defer training; if evidence or capacity does not justify training, record that decision and change the research question.
8. Do not begin B2/B3 or stale-optical training unless the new source-valid B1 screen passes and a prospective plan is frozen. No test measurements may enter that decision.
   Registration-error literature from Wu et al. (2022) is classification-only context; its injected shifts do not define local restoration perturbations.

## Resource tiers

- **Tier 0:** source review, manifest audits, deterministic analysis of saved validation rows, metric-definition checks. Available now on CPU.
- **Tier 1:** small inference or data-analysis diagnostic on validation. Available if it uses fixed inputs, a unique output directory, and no gate changes.
- **Tier 2:** short multi-seed training screen. Requires available GPU, fixed train/validation inputs, and a new approved research protocol with a stop rule.
- **Tier 3:** formal multi-seed training and full stress matrix. Requires evidence from Tier 2 and sufficient GPU/data resources.
- **Tier 4:** expanded dataset or untouched confirmation. Optional and later; 20k pairs are not a prerequisite for exploratory work.

## Required durable records

`CURRENT_STATE.md` controls current status. Each material run is in `EXPERIMENT_REGISTRY.md`; decisions go to `DECISION_LOG.md`; next work and risks go to their dedicated files. Maintain falsifiable, evidence-labeled hypotheses in [`research_hypotheses.md`](research_hypotheses.md) and update them when results challenge assumptions. Only update manuscript claims after their supporting evidence is final and traceable.

## B1 v2 execution checkpoint — 2026-10-22

Historical checkpoint superseded by the current-source matrix outcome in Phase 3 and D-2026-10-23-34. The updates below record the state at their timestamps; the current gate result is `not_promoted` and B2/B3 remain gated.

The registered source-folder-disjoint B1 v2 screen has started under its unique output root. Its fixed train/validation file manifests were recreated byte-for-byte against the frozen input references. SAR-only seed 0 completed and was sealed; the other five runs and the six-run aggregate remain pending. This is progress within Phase 3 only; it does not establish optical utility, B1 promotion, or submission readiness. The historical fixed-test text-exposure disclosure is tracked separately and is not used for this v2 screen.

Execution update (2026-10-22): the seed-0 and seed-1 SAR-only/C2 pairs are sealed; only seed-2 and aggregation remain. The checkpoint loader accepts the baseline's fixed-input checkpoint under the candidate's variable-size crop contract while retaining strict weight and lineage checks. No B1 gate has been evaluated; B2/B3 remain conditional on the complete three-seed aggregate.

Execution update (2026-10-23): CUDA is unavailable, four of six runs are sealed, and seed-2 plus aggregation remain pending. Rebuildable fixed-input payloads were removed after metadata verification and can be reconstructed from retained commands after a fresh preflight. The calibration CLI and API now require an explicit validation directory after a prior omitted-argument attempt fell back to the legacy manifest; no model run or gate changed.

Execution update (2026-10-23): both prospective B1 trainers now support strict epoch-boundary resume from their same-run `last.pth`, with full optimizer/model/scheduler/scaler/RNG/DataLoader state and checkpoint-bound epoch records. The synthetic resume tests and downstream sealer/aggregator fixtures pass. The archived seed-2 attempt remains non-resumable because its partial checkpoint was pruned; after CUDA preflight and fixed-input reconstruction, start that run from scratch. No B1 gate or research stage changed.



Execution update (2026-10-22): the first seed-1 SAR-only attempt disappeared after epoch 19/25 with unknown exit cause; no OOM attribution is supported. Its partial checkpoints/epoch CSVs were pruned while the train log and interruption audit remain. The fixed train/validation payloads were rebuilt and verified against the exact retained metadata hashes. The same seed-1 command completed 25/25 epochs after GPU preflight and sealed at epoch 15. Train-only residual calibration then succeeded at `lambda_residual=0.8083524623361836` after an invocation retry with the explicit frozen validation path. The matching optical candidate then completed and sealed at epoch 1 (strict score 17.439021). Four of six runs are sealed; finish the seed-2 pair while keeping the frozen v2 protocol and promotion gate unchanged.


Execution update (2026-10-22): four of six prospective B1 v2 runs are sealed. Seed-2 SAR-only stopped at epoch 19/25 when GPU access disappeared; no OOM or traceback evidence identifies the cause. The partial attempt was archived with logs/config/provenance and pruned of partial weights and per-epoch CSVs. The registered screen is paused for GPU restoration; restart the seed-2 baseline from scratch, then calibrate and train its paired C2 candidate before aggregation. The gate remains unevaluated; B2/B3 remain gated.


Execution update (2026-10-23): the RTX driver remains unavailable. To honor the artifact cleanup request, the reproducible train/validation tensors and staged source-panel copies were removed after exact metadata matches were confirmed against retained records. Four of six prospective B1 v2 runs remain sealed; after CUDA restoration, regenerate and verify the fixed inputs before restarting the seed-2 pair. The gate remains unevaluated; B2/B3 remain gated.

Execution update (2026-10-23): local QXS-named cache generator/config review found no invocation-bound source lineage. The tracked SOTANet splitter is generic, unseeded, and per-file; the ignored SEN1-2 preparation script points to another output path; ignored trainer options record only the QXS-named cache path. Keep archive/version, license, partition semantics, and source groups unresolved; see the [audit](../../experiments/data_audits/sar2opt_qxs_local_generator_audit_20261023_v1/report.md). No model experiment or gate changed.

Follow-up (2026-10-25): the local SOTANet cache remains Git-ignored and unbound to its current repository commit; its root has no version/license manifest. The current official QXS repository and maintainer issues still supply no patch-to-source crosswalk or public alternate archive. A bounded exact-name public search found no path-specific cache generator. Keep this provenance route unresolved and do not repeat it unless an authorized archive, source map, or new preparation record appears; see the [follow-up audit](../../experiments/literature_audits/qxs_sotanet_cache_route_followup_20261025_v1/report.md) and D-2026-10-25-22.

Execution update (2026-10-23): indexed OpenReview text for the MONTI 2026 SAR-SR paper confirms ordered structure restoration followed by diffusion speckle synthesis, with FWD comparisons against single-stage SR backbones and a separate synthetic paired PSNR/SSIM table. This narrows broad ordered-cascade novelty but does not answer H8's matched reverse-order or joint-restoration question. Direct PDF/code access remains unresolved; no experiment or gate changed.

Execution update (2026-10-23): a bounded public-code search for the MONTI paper across exact-title, author, paper-ID, GitHub, and SI Analytics-domain routes surfaced no official repository. Code availability remains unresolved because direct PDF access is blocked and indexed search may omit links; no experiment or gate changed.

Execution update (2026-10-23): the third-party SEN1-2 SIFT derivative's linked code route resolves to RRL-Net training/testing code using prepared archives; the dataset repository exposes precomputed lists but no label-generation script. The observed zero-offset positive labels remain annotation-schema evidence, not independent registration measurements. Keep fine alignment and acquisition metadata open; no model experiment, gate, or test access changed.

Execution update (2026-10-23): primary-source review found SEN12TP as a conditional source for a future separate protocol. Its paper/Zenodo record describe scene timestamps, ROI IDs, and spatial splits, but the released ~90/10 train/test archive mapping to the paper's 80/10/10 split is unverified, the visible dataset license field is blank, and the task supplies no clean SAR restoration target. Archive listing also conflicts with the prose about the number of train parts. A paper-level aggregate test-metric text surfaced in search output and was not used; no dataset test assets or local project test data/metrics were opened. No data were downloaded and no experiment/gate changed; see the [candidate audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/access_manifest.json).

Execution update (2026-10-23): official terms for SEN12TP's listed source components indicate plausible attributed reuse routes (Sentinel open terms, WorldCover CC BY 4.0, AW3D30 under JAXA terms), but do not resolve rights for the compiled archive or curated metadata. The official Zenodo metadata API/export route was inaccessible and its JSON preview said “Cannot preview file”; no metadata or archive was downloaded. Keep package rights and train/validation membership unresolved; no experiment, local test access, or gate changed. If using provider feeds instead, require product IDs, processing versions, and credits in a new manifest.

Execution update (2026-10-23): a public-rights metadata recheck found OpenAIRE's merged SEN12TP entry displays `CC BY`, while Zenodo's version record still renders Rights/License blank. Zenodo's generic developer docs say dataset defaults are `cc-zero`, whereas its Help page says CC BY 4.0; these generic defaults do not resolve the selected license for this 2023 record. The visible record inventory lists one metadata JSON, five train archives, and one test archive, while the record prose says four train archives. No metadata contents or archive data were accessed, and no test identity/image/measurement or local project test data/metrics were read. Rights and split membership remain unresolved; no experiment or gate changed. See the [updated audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md) and [access manifest](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/access_manifest.json).

Execution update (2026-10-25): static inspection of the author loader found that each raster patch exposes bounding box, CRS, and affine transform, creating a possible train-only spatial-overlap audit without reading the all-scenes metadata JSON. The default 256-pixel patches use stride 249 and overlap by 7 pixels; any prospective split must group dated scene directories by parent `roi_id`. The branch remains mutable and unpinned, and the route is conditional on package rights and train-archive mapping. No dataset payload or test identity was accessed; no experiment or gate changed. See D-2026-10-25-70 and the [updated audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).

Execution update (2026-10-23): a read-only direction audit of the sealed current-source B1 v2 paired validation rows found positive three-seed image-mean PSNR deltas for 129/200 identities, while SSIM worsened for 192/200 and edge-strength ratio error worsened for all 200 identities in each seed. The top 40 identities contributed 61.4% of positive PSNR-delta mass. This refines the observed proxy-target tradeoff, but does not explain its cause, change the B1 gate, or justify B2/B3. No images or test measurements were read; see the [audit](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/report.md).

Execution update (2026-10-23): screened SEN12MS-CR-TS as a possible source of ROI grouping and acquisition-date metadata. Its 53-ROI train/holdout split and date-level S1/S2 pairing are potentially useful, but the dataset targets optical cloud removal, has no clean SAR target, is approximately 2 TB, and dataset-specific terms remain unresolved. Keep it conditional; no data were downloaded, no holdout payload was accessed, and no experiment or gate changed. See the [candidate audit](../../experiments/literature_audits/sen12ms_crts_candidate_20261023_v1/report.md).

Execution update (2026-10-23): a targeted search across manuscript TeX/Bib sources surfaced aggregate held-out result text from the legacy pilot. No PDF or raw test asset/metric file was opened and the exposed text was not used for selection, tuning, paper updates, or gate decisions. The exposure is disclosed in the current state and a dedicated manifest; preserve historical run flags and exclude that sample set from future selection. The current B1 v2 result and B2/B3 gate are unchanged.

Execution update (2026-10-23): a read-only C2 objective/code audit found that reconstruction plus high-frequency residual loss can fall while validation SSIM and the frozen selection score decline; the SAR backbone was frozen and the learned optical correction grew across epochs. Treat this as objective/metric tension, not a proven cause. The B1 run's original 26 source hashes matched at sealing; D-2026-10-23-46 later changed the current model/trainer files, now documented without altering historical run hashes. Preserve `not_promoted`, do not retune on this cohort, and keep B2/B3 gated. Continue provenance and source/target-independence work.

Execution update (2026-10-23): scanned PNG chunk headers for the local dev2k train/validation SAR and optical files. All 3,600 contain only `IHDR/IDAT/IEND`; no embedded provenance metadata exists, and pixel/test payloads were not read. This closes that route but leaves source, acquisition, calibration, registration, and target-generation lineage unresolved. Continue only through authoritative source records or a rights-cleared, better-documented dataset; preserve the B1 stop.

Execution update (2026-10-23): checked the source path hard-coded by the neighboring SOTANet preparation helper. It is absent in the current namespace, and bounded local mount/path checks found no alternate source archive. This does not rule out unmounted/external copies or resolve the numeric 2k lineage. No payload, test data, experiment, or gate was accessed; see the [path-availability audit](../../experiments/data_audits/dev2k_configured_source_root_probe_20261023_v1/report.md).

Execution update (2026-10-23): independently rederived all eight B1 v2 source-group bootstrap summaries from the hash-bound 600-row validation CSV. Every mean, 95% interval, and positive-draw probability matched within `2.71e-20`. The bootstrap conditions on the three observed training seeds and 15 groups within one fall ROI-season; it does not change the failed B1 gate or justify B2/B3. See the [rederivation audit](../../experiments/diagnostics/b1_v2_source_group_bootstrap_rederivation_20261023_v1/report.md).

Execution update (2026-10-23): synchronized README Key findings with the independent interval rederivation and checked 101 local Markdown destinations; none are missing. Runtime inspection found PyTorch 2.8.0+cu128 in `AISAR`, but CUDA is unavailable and the default system Python has no PyTorch. Keep model execution stopped at B1 and continue only evidence-backed work that does not repeat closed source routes.

Execution update (2026-10-24): full-text audit of Corley et al. (CVPRW 2026) confirms a registration-only benchmark with three labeled SpaceNet9 training scenes and a descriptive, non-held-out primary result. The current official repository's mutable-main sweep defaults differ from the paper's §5.2 parameter grid; exact code-to-paper run binding remains unresolved. Record as protocol-control context only, not SAR restoration or fallback evidence. No local data, test asset, run, or gate changed; B1 remains `not_promoted`, and B2/B3 stay gated. See the [audit](../../experiments/literature_audits/corley_pretrained_matcher_fulltext_20261024_v1/report.md).

## Conditional SAR400 dataset route (2026-10-23)

SAR400 may be a supplementary SAR-only denoising stress set, but the accessible sources do not establish the `gt` source/domain, noise generator, grouping/split, or dataset-specific rights. It does not address optical utility, joint SR, or misregistration robustness. Do not acquire or incorporate it unless these metadata are resolved; any approved future use must remain a separately specified controlled-denoising evaluation and cannot alter the frozen B1 gate.

## SOMA prior-art boundary (2026-10-23)

SOMA adds gradient-enhanced dense SAR–optical registration and coarse-to-fine affine-flow matching as established prior art. It does not answer whether such alignment improves SAR-output fidelity or enables safe optical rejection. Keep registration as a possible future tool, not a current restoration module; do not reopen model work until the existing B1 and source/target gates permit it. Its public unpinned implementation does not yet provide a verified paper reproduction.

Execution update (2026-10-23): official publisher-indexed text for ISSMSAR identifies a two-temporal-SAR joint-restoration model trained on grayscale pseudo-SAR targets derived from GaoFen-1 optical imagery. This strengthens the boundary against broad “first joint SAR despeckling and SR” claims, while its optical-derived target construction and unresolved split/degradation/reference details do not supply a matched baseline for the local optical-guided question. No local data, model run, or gate changed; see the [audit](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md).

Execution update (2026-10-23): added a repository-wide A–E fact classification index linked from README. It consolidates existing evidence without replacing CURRENT_STATE or promoting any method. The current B1 v2 gate remains not_promoted; the next valid model-selection path is still blocked on authoritative source provenance and parent-scene grouping for the numeric 2k subset.

Execution update (2026-10-23): SEN12MS-CR-TS author download-route recheck. The maintainer helper exposes region/modality archives and separately named test archives but no standalone train-only metadata crosswalk in the reviewed sections. The official paper copy licenses the article under CC BY 4.0 and points to the dataset website without stating a separate dataset license. Keep the candidate unadopted and all 13 holdout ROIs unopened; revisit only if official dataset terms and an authorized train-only metadata route appear. See the [route audit](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/report.md).

Execution update (2026-10-23): synthesized the sealed B1 v2 C2 gate, per-image, and objective evidence. The frozen result fails the mean PSNR and SSIM thresholds, with the other guardrails passing; the objective/selection trend is plausible but not causal, and geometry, target/degradation, and capacity explanations remain unresolved. Preserve `not_promoted`, do not retune the same cohort or enter B2/B3, and keep provenance resolution ahead of future model selection. See the [failure-cause synthesis](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/report.md).


## SAR-only self-supervised despeckling prior art (2026-10-23)

SDS-SAR narrows the generic no-clean-reference SAR-denoising gap, but the inspected author repository currently exposes a placeholder median-filter test path rather than a reproducible implementation. GLCNet adds a recent self-supervised real-SAR blind-spot method with global frequency and local detail branches. Its publisher-indexed text and current author repository leave iteration count and validation/test role unresolved, so do not treat it as a reproduced or matched baseline. Both are denoising-only and neither replaces the task-matched SAR-only joint SR baseline or answers optical utility. Any future reproduction depends on a paper-matched implementation, verified data/reference domain and reuse terms, and a valid prospective split. The B1 stop and test boundary remain unchanged; see the [GLCNet audit](../../experiments/literature_audits/glcnet_self_supervised_sar_denoising_20261025_v1/report.md).


## Temporal-average SAR pseudo-reference prior art (2026-10-23)

Lattari et al. (2019) provide SAR-only despeckling prior art with synthetic pretraining and real-domain fine-tuning on temporally averaged Sentinel-1 references. This establishes a domain-adaptation precedent, not clean-reference ground truth or optical-guided joint restoration. Any use of temporal averages must preserve stack membership and source-group provenance and must report real-domain limits when clean SAR is unavailable.

## SAR2SAR prior-art and code boundary (2026-10-23)

Full accepted-text review confirms SAR2SAR as a three-stage, multitemporal SAR-only despeckling method with synthetic speckle warm-start and change-compensated self-supervision. Its real imagery has no clean reference; ENL and visual inspection are the stated evaluation, with an explicit ENL over-smoothing warning. The author-linked training repository was not inspectable through the current route. A related author-group inference package and checkpoint listing exist but do not establish a paper-faithful training run. It is not a matched optical-guided joint SR baseline and does not resolve the local data/provenance gap. No model run or B1/B2/B3 gate changed; see the [audit](../../experiments/literature_audits/sar2sar_primary_fulltext_20261023_v1/report.md).

Execution update (2026-10-24): secondary patent indexes surfaced a possible GCOANet-related disclosure (CN120450996A) that describes an SAR-plus-optical prediction-stage reference path and a QXSLAB_SAROPT split. The official item-level patent source and paper relationship remain unverified, and the local QXS-named cache has a different recorded split. Preserve the clue as a diligence lead only; keep the journal article's inference inputs and local source lineage open. No data, test asset, run, or gate changed. See the [audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md).

Execution update (2026-10-24): a new public OGSFlow repository README adds a code-first lead for reliability-gated optical guidance and flow-matching UAV-SAR SR. Its paper citation and detailed protocol are unavailable in the current reader; the README does not establish joint despeckling/SR, geometric robustness, or fallback. A follow-up describes its Stage-2 source-noise scale as a perception–distortion control and names semantic downstream evaluations; whether calibration `val` is separate from later selection/reporting remains unknown, not a leakage finding. Keep it as prior-art context, with OGSRL/OGSSR identities and terms separate. No local data, test asset, experiment, or gate changed. See the [audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md) and D-2026-10-24-18.

Execution update (2026-10-24): the publisher-linked PLFSID landing page for Onat's 2026 SAR despeckling fusion paper is now readable. Its README says the visible `main.m` is demo-only and the algorithm scripts are withheld; full article and source routes remain inaccessible, and its post-acceptance license text is stale relative to the paper's June publication. Record this as single-SAR output-fusion prior art only. Keep implementation, protocol, and reuse terms open; no local experiment or B1/B2/B3 gate changed. See the [route audit](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/report.md).

Execution update (2026-10-24): a newly surfaced Figshare v2 package names QXS-SAROPT as its upstream source, but the accessible record has no archive inventory or source-group crosswalk and warns that filename pairing does not prove alignment. Its CC BY 4.0 record license does not supersede upstream imagery terms. Keep it as an unverified lineage lead; no archive was downloaded and local source identity, rights, and groups remain unresolved. See the [screen audit](../../experiments/data_audits/figshare_qxs_semantic_repackage_source_screen_20261024_v1/report.md).

Follow-up (2026-10-24): targeted Kaggle/GitHub searches did not surface the referenced processing notebook or an author-controlled source repository; the linked Figshare author profile returned 403. Close this route until direct access or a new metadata/source crosswalk becomes available; preserve the unresolved local-QXS lineage and do not infer source groups.


Execution update (2026-10-24): the permitted zero-training C2 residual diagnostic completed for all three frozen seeds on the 200-row validation split. It found small positive actual-residual PSNR deltas, negative SSIM deltas, no strict-score improvement from the tested global alpha sweep, and a larger target-assisted oracle gap. Treat this as a mechanism observation only; keep `not_promoted`, do not enter B2/B3, and prioritize authoritative source/parent-scene provenance before further model selection. See the [diagnostic report](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/report.md).

Execution update (2026-10-24): both B1 training CLIs now fail before data access if `--protocol-id` is supplied without `--validation-dir`; non-B1 calls preserve the documented legacy validation default. Parser regression checks and syntax compilation passed. No run or gate changed; one full-suite failure remains tied to historical source hashes that intentionally do not match the current worktree.

Execution update (2026-10-24): the University of Seoul author abstract for Attention OffsetNet adds geometric optical–SAR registration prior art. It reports patch-offset RMSE separately from full-scene RANSAC residuals; neither is evidence for SAR restoration quality or local-cohort alignment. Full protocol and code lineage remain open. No local experiment or gate changed. See D-2026-10-24-24 and the [audit](../../experiments/literature_audits/attention_offsetnet_author_page_screen_20261024_v1/report.md).

Synthesis follow-up (2026-10-24): updated [`research_gaps.md`](research_gaps.md) to distinguish registration accuracy from restoration robustness and preserve the local mismatch/fallback question as unresolved. No method or experiment decision changed.

Disclosure update (2026-10-24): a broad manuscript claim search surfaced historical pilot test-result text. The access is recorded and the values are excluded from future tuning/selection; defer the result-bearing manuscript audit and continue source-independent work. See [D-2026-10-24-25](DECISION_LOG.md#d-2026-10-24-25--disclose-historical-test-result-text-exposure).

Execution update (2026-10-24): made the MMCANetV2 test-inference path require an explicit fixed-set directory and corrected per-mode model-input metadata. The loader's continued reading of target/optical files for zero/missing modes is recorded as a remaining interface limitation. Four focused CPU interface tests passed; no project data or gate changed. See D-2026-10-24-26.
Execution update (2026-10-24): source review found direct lower-level inference/evaluation CLIs that did not inherit the high-level freeze gate. They now require a verified freeze/model identity for test-marked fixed inputs, and formal main/stress/sensitivity runners pass it through. The evaluator binds the prediction manifest to the same model and inputs. Speckle-sensitivity variants derive allowed looks from the frozen predeclared matrix config. Fifty focused synthetic tests and Python compilation passed; no real data was accessed. This does not change the B1 decision. See D-2026-10-24-31.

## Prior-art boundary update (2026-10-24)

Ryu et al.'s SAR-conditioned flow-matching work generates optical imagery from SAR. Keep its structure-preservation claims and optical-domain metrics separate from the project's optical-to-SAR utility question. The workshop version's PSNR table conflicts with its all-metric prose, and its relationship to the IEEE early-access version remains unverified. No local experiment or gate decision follows; see the [bounded audit](../../experiments/literature_audits/sar_conditioned_flow_matching_s2o_20261024_v1/report.md).

Literature update (2026-10-24): Qian et al. (ICCP 2022) provide direct camera-pipeline prior art for task-order ablations and intermediate-stage end-to-end supervision; Xing and Egiazarian (CVPR 2021) compare sequential, pairwise-joint, and fully joint camera restoration under a fixed order. H8 remains a task-matched SAR question. No experiment or B1/B2 gate changed; see the [audit](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/report.md).

Source-provenance update (2026-10-24): TUM’s current dataset catalogue confirms the official SEN1-2 paper and published count but exposes no sample-level manifest; the expected mediaTUM record returned HTTP 429. The local 2k source/version and source groups remain unresolved, so contaminated partitions remain unavailable for selection. No run or gate changed; see the [route audit](../../experiments/data_audits/tum_sen1_2_record_metadata_route_20261024_v1/report.md).

Prior-art update (2026-10-24): FAF-CD establishes adjacent feature rectification, deformable EO–SAR fusion, frequency-branch gating, and appearance/radiometric modality-side stress testing for change detection. The reviewed protocol contains no controlled geometric offset sweep or SAR-output reconstruction metrics. Keep the project question narrow: test optical utility and fallback only with task-matched SAR-output evidence under a valid prospective protocol. B1 remains not promoted; B2/B3 remain gated. See [D-2026-10-24-40](DECISION_LOG.md#d-2026-10-24-40--bound-faf-cd-as-adjacent-change-detection-prior-art).

Prior-art/data update (2026-10-24): SEORation adds SAR–EO semantic deduplication and scene-compatibility curation for retrieval. Its QXS-SAROPT results are retrieval results on an external dataset, not pixel-registration or SAR-restoration evidence; source-provided held-out splits do not supply a local source-group crosswalk. Preserve local numeric 2k lineage as unresolved. No phase gate or experiment decision changed; see [D-2026-10-24-41](DECISION_LOG.md#d-2026-10-24-41--classify-seoration-as-pair-curation-and-retrieval-prior-art).

Repository-source update (2026-10-24): OGSFlow's README maps S1/S2/RGF to model files and advertises unit-test and CPU synthetic-data smoke commands, but those files remain inaccessible and no command was executed. This is release metadata only; it adds no verified implementation or experiment evidence and does not change the prior-art boundary or B1/B2/B3 state. See [D-2026-10-24-42](DECISION_LOG.md#d-2026-10-24-42--treat-ogsflow-source-map-and-smoke-command-as-readme-metadata).
Prior-art update (2026-10-25): Gaya et al.'s IGARSS 2024 abstract describes optical-assisted SAR restoration with self-supervised neural-network training. Record it as direct optical-assisted despeckling prior art, while keeping its data, target, inference path, registration, metrics, and results unknown because the full text was unavailable. It does not answer joint SAR SR or controlled mismatch/fallback questions. No local experiment or gate changed; B1 remains `not_promoted`, B2/B3 remain gated. See [D-2026-10-25-04](DECISION_LOG.md#d-2026-10-25-04--record-gaya-et-al-as-abstract-level-optical-assisted-sar-despeckling-prior-art).

Resource update (2026-10-25): the current `AISAR` runtime exposes no CUDA device and `nvidia-smi -L` cannot contact the driver. Keep training deferred. GPU availability would not override the failed B1 decision or the unresolved source/target provenance requirement; continue only source-independent work until both resource and scientific gates are satisfied. See [D-2026-10-25-05](DECISION_LOG.md#d-2026-10-25-05--defer-model-runs-under-current-gpu-and-research-gate-constraints).
Prior-art update (2026-10-25): Gaya et al.'s IGARSS 2024 abstract describes optical-assisted SAR restoration with self-supervised neural-network training. Record it as direct optical-assisted despeckling prior art, while keeping its data, target, inference path, registration, metrics, and results unknown because the full text was unavailable. It does not answer joint SAR SR or controlled mismatch/fallback questions. No local experiment or gate changed; B1 remains `not_promoted`, B2/B3 remain gated. See [D-2026-10-25-04](DECISION_LOG.md#d-2026-10-25-04--record-gaya-et-al-as-abstract-level-optical-assisted-sar-despeckling-prior-art).

Prior-art update (2026-10-25): SpeckleDiff-2D's IEEE abstract adds complex-valued diffusion for SAR despeckling, trained on synthetic speckle generated from optical imagery. The accessible sources do not identify inference inputs or establish joint SAR SR, same-scene optical guidance, mismatch handling, or fallback. Official Télécom Paris pages newly expose a Gaya thesis listing but no direct text; keep that route at lead status. No local experiment or gate changed; local diffusion remains deferred and B1 remains not_promoted. See [D-2026-10-25-06](DECISION_LOG.md#d-2026-10-25-06--record-specklediff-2d-and-gaya-thesis-route-boundaries).



Prior-art update (2026-10-25): A second, distinct Amieva et al. CVPRW paper uses azimuth subapertures for SAR-only same-grid enhancement and reports an ENL-versus-structure-consistency trade-off. Its full-aperture target is noisy, and its whole-dataset percentile normalization is described before splitting. It narrows broad SAR-only joint enhancement novelty but supplies no optical-utility, mismatch, or fallback evidence and does not change the failed B1 gate. See [D-2026-10-25-17](DECISION_LOG.md#d-2026-10-25-17--classify-amieva-sm-subaperture-enhancement-as-sar-only-prior-art).


Prior-art update (2026-10-25): Dong et al. (2025) add abstract-level prior art for complex-valued SAR SR using subaperture/subband learning and fusion. Keep broad novelty claims about that mechanism narrow; the accessible evidence does not establish a joint despeckling objective, optical guidance, or mismatch-safe fallback. Publisher full text and exact protocol remain unavailable. No phase gate changed; see [D-2026-10-25-18](DECISION_LOG.md#d-2026-10-25-18--classify-dong-complex-sar-sr-as-abstract-level-prior-art).


Architecture update (2026-10-25): distinguish a code-level optical-absence fallback to the SAR base from the untested claim that the model detects/rejects geometric or semantic mismatch. The optional alignment field and edge-consistency gate are not calibrated utility evidence; B2 remains gated. See [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness).

## Target/input coupling interpretation update (2026-10-25)

The frozen B1 v2 input manifests bind the development comparison to a pipeline in which the normalized SAR tensor is both the HR pseudo-target and the source for the degraded LR observation. The LR adds a representation-domain Gamma perturbation after filtering and downsampling. This creates a known input-target construction relationship, but the native target residual, calibrated signal domain, and residual covariance are unknown. The one-mode Gaussian analysis in `optical_information_analysis.md` makes clear that a proxy-target optical gain and a clean-signal gain are different estimands. This is an interpretation refinement only: no gate changes, no new model selection, and no B2/B3 work until a valid prospective protocol is available.

## Multivariate target-risk ordering update (2026-10-25)

A vector linear-Gaussian extension shows that optical risk reduction against the shared-input pseudo-target can be smaller or larger than reduction against the latent signal, depending on operator/residual-covariance geometry. The scalar attenuation ratio is not a general correction. Keep local claims at pseudo-target consistency until target residuals and an independent reference are characterized; this theory does not reopen model selection or B2/B3.

## Pilot manuscript citation coverage update (2026-10-25)

Metadata for the seven citations currently used by the pilot draft is consistent with available publisher, author, or institutional records. The draft still omits direct optical-guided SAR SR and joint-restoration prior art from its citation keys. Before manuscript reuse, update Related Work and map novelty statements to verified sources; quantitative local claims remain gated by valid prospective evidence.

Execution update (2026-10-25): the pilot Related Work and bibliography now cite the official OMA-SSR and HSFMamba records and the CVPR SFT formulation used by the baseline. The paragraph distinguishes SR-only OMA-SSR from optical-guided joint restoration and states that the reviewed protocols do not establish the project's matched noisy-LR result or mismatch/fallback claims. This is citation integration only; the claim-to-source map and broad Related Work review remain open, no result-bearing section changed, and the historical pilot cohort remains excluded from selection/claims after the separately logged text exposure. See the [integration audit](../../experiments/literature_audits/related_work_integration_20261025_v1/report.md) and D-2026-10-25-39.

Related-work update (2026-10-25): DSRKD, ISSMSAR, MD-Glow, and MambaR-ESRGAN are now cited as SAR-only joint-restoration prior art with their non-matched protocols and unresolved details stated. This narrows broad novelty framing but does not supply a matched baseline or optical-utility evidence; C9 coverage remains non-exhaustive. No phase gate or experiment changed; see [D-2026-10-25-53](DECISION_LOG.md#d-2026-10-25-53--integrate-sar-only-joint-restoration-prior-art-into-pilot-related-work).

MHFNet remains bounded as misaligned SAR–optical ship-detection prior art. Static inspection of its mutable public archive reveals the dual-branch DCNv4/CGA detection path and an unresolved loader/YAML channel-order question; neither the exact paper-run version nor a controlled restoration-mismatch protocol is established. This does not change the research phase or gate; see D-2026-10-25-62.

## RQ-Net registration prior-art update (2026-10-25)

The publisher-indexed RQ-Net article adds optical–SAR feature matching and registration as direct prior art for alignment mechanisms. Its task is registration rather than SAR intensity restoration, and the partial source view does not establish the dataset/split or controlled-error protocol. Keep the optical-guided SAR restoration question open; no model candidate, B1/B2/B3 gate, or experiment plan changes. See D-2026-10-25-63.



Related-work update (2026-10-25): the pilot now also cites optical-guided despeckling/predictor-rejection prior art (Optical-Driven Nonlocal, GNLM, and GCOANet). Their evidence does not support joint-SR fallback or controlled-shift claims; GCOANet's inference inputs remain unresolved. This is a literature-only update, with no experiment or gate change; see [D-2026-10-25-54](DECISION_LOG.md#d-2026-10-25-54--integrate-optical-guided-sar-despeckling-and-predictor-rejection-prior-art).

Execution update (2026-10-25): clarified that the LR input is degraded from the same SAR observation saved as the noisy HR pseudo-target. The pilot Method and Limitations now state that reference-based scores measure agreement with a correlated proxy and cannot establish clean-SAR recovery. This is a source-linked interpretation clarification, not a new metric or experiment; the same-observation code trace does not identify residual-noise covariance. See D-2026-10-25-55. Session-level qualitative held-out-result text exposure is recorded as `test_set_accessed=true`, `test_set_used=false`; the historical cohort remains excluded.

Execution update (2026-10-25): the pilot degradation equation now specifies the sampler's shape=`L`, rate=`L` convention and unit-mean, variance-`1/L` multiplier from source code. Synthetic moment tests cover intensity, representation (`L=2`), and amplitude-square cases; this does not establish physical speckle fidelity. A pooled LR input-range summary surfaced with unspecified split scope during the bounded manuscript read and was not used; retain session flags `test_set_accessed=true`, `test_set_used=false`. See D-2026-10-25-56 and the access audit.

Related-work source update (2026-10-25): MHFNet's publisher-linked author repository is now verified as a public code/data route, but its listed archive was not inspectable and the README does not establish dataset/split or controlled-mismatch details. Classify it as downstream detection prior art only; it does not change the restoration question, phase gates, or claim support. See D-2026-10-25-58.

Related-work execution update (2026-10-25): added MHFNet as adjacent misalignment-aware SAR–optical ship-detection prior art, explicitly bounded to detection outputs. This narrows broad misalignment claims but does not establish restoration mismatch robustness or a safe fallback. Broader citation coverage remains open; see D-2026-10-25-60.

Execution update (2026-10-25): the HSFMamba paper-linked public repository index lists OGSSR as a data resource but has no algorithm/code entry. The linked dataset share was not opened, and the model was not reproduced. This narrows code availability only for the inspected route; it does not change the literature classification, local protocol, or B1/B2/B3 gates. See D-2026-10-25-69.

## SEN12TS conditional source route (2026-10-25)

The initial source/task-fit screen correctly withheld adoption because the full collection is 1.69 TB, its general split is unspecified, and it provides no independent clean SAR target. Subsequent work established a manageable 2,000-pair Sumatra cohort with matching grids, a conservative spatial gap, and a prospective metadata-linkage train/development assignment. Official channel mapping and an 81-tile train/development screen now pass basic schema and mask coverage, but the 43 component keys remain proxies rather than reviewed parent-scene IDs. Positive residual correlations and component-level temporal variation leave target independence, radiometry, and fine registration unresolved. Per D-2026-10-25-98, stop further sampling for this question; keep SEN12TS conditional and do not change B1/B2/B3. See D-2026-10-25-73/75/95/96/98 and the linked audits.

The temporal-reference literature and the pilot/supplement screen confirm that a leave-input-date-out average is only a candidate noisy pseudo-target. Local-mean change and normalized-residual dependence were measured descriptively on selected train/development tiles; positive correlations persisted, and the coarse coherence layers still cannot certify 10 m independence. The bounded screen is closed without training or model selection; see [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit), the [literature audit](../../experiments/literature_audits/sen12ts_temporal_reference_feasibility_20261025_v1/report.md), and the [paired screen](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/report.md).

## Conditional CDSE metadata route (2026-10-25)

The direct STAC v1 endpoint returns live metadata. A complete two-month fixed-point query produced a low-cloud same-day S1/S2 pair whose product footprints cover an approximate 2.56 km square probe. A frozen four-anchor extension found candidate pairs at all four locations, but shared datatakes cross anchors. The follow-up metadata graph retrieved 29 candidate Item records: 19 pairing edges, two same-datatake edges, and 141 positive-area product-footprint overlaps. All four anchors merged into one connected component at 1 m², 100 m², and 1 km² thresholds; without spatial-overlap edges there are nine components. Reject this four-anchor product pool as a group-disjoint train/development cohort under the frozen footprint rule. This does not reject CDSE for different AOIs because product footprints may conservatively merge small future crops. Before any new screen, freeze AOIs and grouping rules; before any asset download, establish release-specific terms and define calibration, cloud masks, and a valid representation-domain target/noise protocol. S1 GRD is not clean SAR truth; no pixel alignment, calibration, temporal independence, or model performance has been established. Do not alter B1/B2/B3; see D-2026-10-25-100/101/102 and the [graph audit](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/report.md).

## Gamma sampler implementation check (2026-10-25)

The canonical unit-mean Gamma multiplier passes a predeclared moment audit at candidate L=1,4,8 in intensity, representation, and amplitude modes. This closes only the sampler-statistics check. Local normalized PNGs are still not calibrated amplitude/intensity data, so representation-domain L is not a physical ENL. Do not expand severity experiments until an authorized source and target-domain protocol support the interpretation; see D-2026-10-25-85.


## 2026-10-25 — Data-domain evidence update

The fixed-development preparation contract is bound to code/config: 8-bit SAR PNG `/255`, log normalization (`log_scale=20`), 128×128 LR generation, and representation-domain Gamma multiplication (`L=2`). The pseudo-target is not an independent clean SAR reference; original radiometric semantics remain unknown. This is a documentation-only audit: no experiment/gate change, test payload, or new physical claim. See D-2026-10-25-87 and its audit report.


## 2026-10-25 — Preserve source uncertainty

Local train/validation overlap geometry is consistent with the QXS-SAROPT stated 20% overlap under one reading, but this does not identify the local 2k subset or recover acquisition groups. Keep source lineage unresolved and make no dataset-name or sensor claim until a versioned manifest/crosswalk or independent source metadata is verified. This does not change B1/B1R/B1F gates or reopen B2/B3. See D-2026-10-25-88.


## 2026-10-25 — Exact-byte evidence narrows, but does not settle, lineage

The stored SEN1-2 upstream checksum crosswalk found zero exact matches for the 3,600 train/validation source rows. Its current split/source-mapping metadata hashes still match, while current image bytes have not been rehashed. Treat this as evidence against direct byte identity only; preserve the unresolved-source gate and do not promote QXS or another source hypothesis. See D-2026-10-25-89.


## 2026-10-25 — Keep source labels evidence-bound

Future split manifests and generated reports now describe the local 2k SAR/optical PNG subset as project-recorded with upstream source/version unverified. Historical result artifacts remain immutable. Revisit the source label only after a versioned archive or authoritative crosswalk is recovered. See D-2026-10-25-91.


## 2026-10-25 — Separate local-cache identity from official source attribution

The 1,626-pair pixel crosswalk links most fixed train/validation rows to a sibling Git-ignored `sar2opt-QXS` cache, but no official archive/version, license, or source groups are available. Treat this as a local derivative relationship only. The sibling cache's partition names do not establish independent groups or test behavior. See D-2026-10-25-92.

## 2026-10-25 — external candidate route screen

CDSE global-AOI metadata screen 达到预声明的 8-component 入口，当前只允许进入数据权利、校准、云/配准与目标协议审查。候选 AOI 图不是 scene-level split，也不构成模型晋级证据；任何 prospective split 仍须基于可核验来源组并通过空间泄漏审计。详见 [screen report](../../experiments/data_audits/cdse_global_aoi_group_feasibility_20261025_v1/report.md)。

## Temporal pseudo-reference assumption update (2026-10-25)

The full RABASAR primary paper separates temporal reflectivity stability from temporal speckle decorrelation and requires calibrated, accurately registered, same-orbit/similar-incidence inputs for its temporal-mean superimage. This strengthens the current rule that an averaged stack is only a noisy pseudo-reference unless both assumptions and source groups are justified. The prior SEN12TS pilot remains stopped; no new sampling, split, model selection, or gate change is authorized by this literature review. See D-2026-10-25-113 and the [audit](../../experiments/literature_audits/rabasar_temporal_superimage_assumptions_20261025_v1/report.md).

## General misregistration prior-art boundary (2026-10-25)

Generic misalignment-aware cross-modal filtering and joint reconstruction-registration are established prior art. Keep research claims focused on the unverified SAR-specific question: optical utility and negative-transfer control for speckled SAR joint restoration, evaluated against a matching SAR-only baseline on a valid prospective split. This literature update changes no B1/B2/B3 gate and authorizes no model selection. See [D-2026-10-25-107](DECISION_LOG.md#d-2026-10-25-107--bound-general-misalignment-aware-restoration-prior-art).
## OSdataset rights route update (2026-10-25)

The official current Google Earth terms do not establish blanket rights to copy, bulk-download, or redistribute an imagery-derived patch archive; research/education display permissions do not by themselves clear dataset-scale reuse. These 2025 terms do not establish the historical rights or source-provider permissions for the 2020 OSdataset material. Keep the candidate unadopted until an authoritative, version-specific rights route is available. No archive or imagery was accessed. See D-2026-10-25-118 and the [source audit](../../experiments/literature_audits/osdataset_google_earth_terms_20261025_v1/report.md).

Execution update (2026-10-25): official SpaceNet 9 dataset terms are now verified at the challenge-page level (CC BY-SA 4.0), separately from Apache-2.0 winning-solution code. The repository lists three labeled development triples; public-test labels and the separate-AOI final test remain hidden. Keep the source registration-only and do not treat training scenes as held-out validation. No assets were accessed, no run was made, and no phase gate changed. See [D-2026-10-25-135](DECISION_LOG.md#d-2026-10-25-135--keep-spacenet-9-as-registration-prior-art-not-held-out-validation).



Execution update (2026-10-25): the DP-SAR IEEE abstract adds adjacent complex-PolSAR, scattering-fidelity SR prior art. Keep its point-target and Pauli-domain evaluation separate from the local 8-bit single-channel cohort; author metadata/full text remain unresolved. No method, run, or phase gate changed. See D-2026-10-25-136 and the abstract-level audit.


Execution update (2026-10-25): a potentially relevant PolSAR/SinSAR diffusion SR citation remains a source lead only because no readable publisher abstract or author manuscript was found. It does not alter the evidence-based literature synthesis or experiment gates. See D-2026-10-25-137.

## SEN1-2 official catalog-route follow-up (2026-10-25)

The TUM SIPEO and GovData catalog pages do not provide an exact `ROIs1970_fall` product manifest; the advertised GovData Turtle export and package API were inaccessible through the current reader. Keep source-product timing and independent fall registration unresolved, and do not reopen B2/B3. See [D-2026-10-25-141](DECISION_LOG.md#d-2026-10-25-141--keep-the-official-sen1-2-catalog-route-at-metadata-only-status) and the [updated audit](../../experiments/literature_audits/sen1_2_independent_reference_availability_20261025_v1/report.md).
Execution update (2026-10-25): reviewed Noisier2Noise's single-realization derivation. It assumes a known, sampleable source-noise distribution; its non-additive example is multiplicative Bernoulli masking. The local normalized SAR PNG source noise remains uncharacterized, so no clean-SAR route or new run follows. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-25-179.

Execution update (2026-10-25): publisher-abstract reviews of OGRN and OGSRN confirm existing training-time optical-guidance routes for SAR SR with SAR-only inference. OGSRN's SciOpen abstract is readable, but its PDF body is not exposed through the available official routes, leaving its dataset and protocol unverified. OMA-SSR provides direct inference-time optical SAR-SR prior art, but is not joint denoising/SR; its data are available on request and no author-linked code route was verified. A schema audit of the fixed dev2k packs found no parent-scene/acquisition/source-group IDs. These findings do not change the local optical-utility hypothesis or justify model selection. Keep source/target and registration prerequisites ahead of model selection; B1 remains `not_promoted`, B2/B3 remain gated. See [D-2026-10-25-181](DECISION_LOG.md#d-2026-10-25-181--classify-ogrn-as-training-time-optical-guidance), [D-2026-10-25-183](DECISION_LOG.md#d-2026-10-25-183--keep-ogsrn-at-the-publisher-abstract-boundary), [D-2026-10-25-185](DECISION_LOG.md#d-2026-10-25-185--record-oma-ssr-code-route-as-unverified), and [D-2026-10-25-186](DECISION_LOG.md#d-2026-10-25-186--fixed-dev2k-manifests-do-not-provide-source-groups).

## 2026-10-26 — SEN12MS-CR archive-route boundary

Author-maintained code exposes TUM season-level download links, but the current route does not provide a versioned train-only ROI/source manifest. Treat this as a potentially accessible archive, not an admissible training/development cohort. Do not download a bundle unless train-only isolation and target validity can be established first. See D-2026-10-26-010. Follow-up source review identifies the underlying dataset paper as Ebel et al. (2021), DOI `10.1109/TGRS.2020.3024744`; the author page reports 175 ROIs and the paper 169, so release lineage remains unresolved. The dataset predicts optical Sentinel-2 and does not establish local 2k provenance; the conditional-use boundary is unchanged. See [D-2026-10-26-048](DECISION_LOG.md#d-2026-10-26-048--verify-sen12ms-cr-paper-identity-and-keep-the-source-conditional).

## 2026-10-26 literature execution update — OGSRN patent boundary

A primary patent route with an inventor roster matching the CJA OGSRN article adds a specific test-time implementation: LR SAR is bicubic-upsampled and channel-expanded to feed the optical branch. The article-patent mapping remains unconfirmed, and the separate JARS residual-learning OGRN record remains distinct. Continue to separate training-time optical supervision from direct inference-time optical utility; no protocol or promotion gate changes. See D-2026-10-26-022 and the [audit addendum](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md#addendum--2026-10-26-author-group-patent-route).

## Prior-art update — 2026-10-26 optical-guided despeckling

**Fact:** The 2015 Optical-Driven Nonlocal method gates optical use with a homogeneity classifier; the 2019 Guided patchwise nonlocal method uses an SAR-domain test to reject risky predictors; Gaya et al.'s IGARSS 2024 abstract describes self-supervised optical-assisted SAR despeckling. **Inference:** These sources establish prior art for generic optical guidance and predictor safeguards, so those mechanisms alone cannot support a novelty claim. **Open:** The reviewed records do not establish task-matched joint SAR SR or controlled geometric-mismatch robustness; those questions remain open under the project's frozen evidence gates.

### Prior-art update — 2026-10-26 multitemporal SAR filtering

Meraoumia et al. (2023) adds self-supervised multitemporal SAR despeckling prior art using SLC SAR and TerraSAR-X stacks. It is SAR-only and has no joint SR result in the reviewed abstract; it is not a matched B1 baseline. Do not bind it to SEN12TS without source evidence. See [D-2026-10-26-047](DECISION_LOG.md#d-2026-10-26-047--record-multitemporal-sar-despeckling-without-binding-it-to-sen12ts).


## Related-work matrix recovery update (2026-10-26)

Seven additional historical entries are confirmed from pre-corruption decisions D-2026-10-18-07 through D-2026-10-23-24: two existing optical-guided despeckling rows and five restored rows for Ka-SAR, CN122694669A, GFS-Net, SpaceNet 9, and MEOW. MONTI was already present in the captured prefix. A separate current-evidence registration-validation row was added with historical membership unresolved. Current inventory is 81 rows (74 historically confirmed; seven unresolved); 38–45 entries remain unaccounted for against the last confirmed 119-row count. See D-2026-10-26-051 and D-2026-10-26-053.


### Captured-prefix integrity follow-up (2026-10-26)

The 46-row partial prefix now matches the preserved snapshot byte-for-byte; one HSFMamba description edit was reverted in the matrix and retained in the literature review/integration audit. Historical reconstruction and unresolved inventory counts are unchanged. See [D-2026-10-26-050](DECISION_LOG.md#d-2026-10-26-050--restore-the-captured-46-row-matrix-prefix-byte-for-byte).


### Claim-evidence map update — 2026-10-26 optical-guided SAR SR

OS_SR is now mapped alongside OMA-SSR as optical-guided SAR-SR prior art. The OS_SR entry remains abstract-level: a 2026-10-26 SSRN re-open returned HTTP 403, and dataset, degradation, target, split, metrics, results, and code remain unverified. Keep the two records separate because their version relationship is unresolved. This does not support joint despeckling/SR, mismatch rejection, or fallback claims and does not change local gates. See [D-2026-10-26-105](DECISION_LOG.md#d-2026-10-26-105--map-os_sr-as-abstract-level-optical-sar-sr-prior-art).


### Related-work matrix update — 2026-10-26 Mamba-SCR

Added Mamba-SCR as a current-evidence row for adjacent SAR-assisted optical cloud removal, based on the existing publisher-text audit. The output is optical, the paper does not establish SAR-output mismatch rejection, and its SEN12MS-CR split is not crosswalked to the original release holdout. Historical matrix membership remains unresolved. Current matrix counts are 95 total rows, 74 confirmed historical memberships, and 21 current additions of unknown membership. See [D-2026-10-26-107](DECISION_LOG.md#d-2026-10-26-107--add-mamba-scr-as-adjacent-reverse-task-prior-art).

### Related-work matrix update — 2026-10-26 Muzeau et al.

Added Muzeau et al., “On the impact of despeckling for supervised SAR super-resolution,” as current-evidence prior art. The author manuscript describes noisy-LR to MERLIN-denoised-HR supervision for ×2 SR using a single ONERA SETHI scene and a patch-level 80/20 split; it does not compare cascade order or use optical guidance. The VDE route redirected to an inaccessible TIB page, and the paper-to-current-repository configuration mapping remains unresolved. Historical matrix membership is unknown; counts are now 96 rows, 74 confirmed memberships, and 22 unresolved additions. See [D-2026-10-26-110](DECISION_LOG.md#d-2026-10-26-110--add-muzeau-et-al-as-current-sar-restoration-prior-art).

### R-DDPM full-text follow-up — 2026-10-26

Reviewed the arXiv v1 manuscript behind the existing R-DDPM row. It specifies Sentinel-1 VV+VH near Shanghai, Gaussian-noise synthetic observations, and no-reference ENL/EPI for real images; sample counts and explicit split allocation are absent, and the regional parameter statement is inconsistent. The row lies in the preserved original 46-row prefix, so it was not edited and matrix counts did not change. The result narrows diffusion feasibility context but does not support optical-guided joint restoration. See [D-2026-10-26-112](DECISION_LOG.md#d-2026-10-26-112--complete-r-ddpm-full-text-audit-without-changing-the-preserved-matrix-prefix).

### Related-work matrix update — 2026-10-26 Gaya et al. GRETSI 2023

Added the full-text-audited Fuse-MERLIN paper as current optical-guided self-supervised complex-SAR despeckling prior art. It is one-scene qualitative evidence, explicitly lacks quantitative validation, and does not address joint SR. Keep the distinct 2024 IGARSS record separate pending evidence of version continuity. Matrix counts are 97 total (74 confirmed historical; 23 unresolved), with 22–45 original rows unaccounted for. Historical membership remains unknown; the preserved prefix and reconstruction ledger are unchanged. See D-2026-10-26-116.

### B1 v2 multi-metric group diagnosis (2026-10-26)

A validation-only analysis of the frozen 15 source groups found EPI correlation improved in every group while edge-strength and bright-peak ratio errors increased in every group; TBR error increased across group means, and bright-energy/region-MAE errors increased in 12/15 groups. This sharpens the observed metric trade-off but does not identify its cause. Keep the source-aligned B1 v2 gate `not_promoted` and B2/B3 closed. Fine registration and the one-ROI-season scope remain limiting. See D-2026-10-26-123 and the [diagnostic report](../../experiments/diagnostics/b1_v2_multimetric_group_tradeoffs_20261026_v1/report.md).

## Official QXS-SAROPT source-route follow-up (2026-10-26)

The author repository and issue history still provide no file-level city/source-group map or public alternate mirror. The owner's dated explanation says original GF-3 SAR could only be released as authorized processed/cropped data; this does not establish rights for the neighboring cache. Keep local source lineage, parent groups, and use rights unresolved. See [D-2026-10-26-126](DECISION_LOG.md#d-2026-10-26-126--recheck-official-qxs-saropt-author-source-routes) and the [route audit](../../experiments/data_audits/qxs_author_issue_lineage_recheck_20261026_v1/report.md).

## Sentinel-1 IW SLC route boundary (2026-10-26)

The 12-AOI metadata screen passed its component-count threshold for a protocol review only. That review is complete and does not authorize data access or experiments: MERLIN is not directly applicable to S1 IW/TOPS, deramping is required, and I/Q decorrelation remains unresolved. Freeze product-bound processing, calibration, geometry, registration, and target validity before any authorized asset request; then establish real source groups and a prospective disjoint split before model selection. B1 remains `not_promoted` / `stop_at_b1`, and B2/B3 remain gated. See D-2026-10-26-147 and the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).

## Optical-use mechanism update — 2026-10-26

A predeclared interpretation-only audit found that the B1 v2 C2 reliability gate was not near zero on the current validation cohort, while the effective gated correction was small relative to the SAR base. This narrows one explanation for the low paired-output sensitivity but does not establish optical utility, safe rejection, or causality. Treat it as development-cohort mechanism evidence only; keep B1 `not_promoted`, B2/B3 closed, and continue source/product provenance and precise registration work. See D-2026-10-26-154 and the [activation report](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/report.md).


## 2026-10-27 follow-up — cross-attention mechanism evidence

**Fact:** The approved analyzer now supports a fresh attempt output directory while preserving its frozen definition artifacts; v1 remains interrupted and v2 completed with verified scalar-output hashes.

**Observation:** On the fixed development cohort, bypassing both optical cross-attention residual updates changed predictions by about 1.90e-05 RMS (about 2.46e-05 relative RMS) under both matched and permuted optical. Internal attention updates were nonzero; direct optical fusion stayed active.

**Open question:** The intervention does not identify why prediction sensitivity is small or establish the value of the optical branch. The cohort remains unsuitable for model selection or promotion. B1 remains `not_promoted`; B2/B3 remain gated. See D-2026-10-27-002.


## 2026-10-27 related-work matrix completion

SAG-CR and CGD-Net are indexed as current-evidence reverse-task prior art. Their optical-output tasks narrow broad alignment/reliability novelty claims but do not change the B1 stop condition or establish SAR-output utility/fallback. The matrix is 102 rows (76 confirmed historical members, 26 additions with unresolved historical membership); at least 25 original entries remain unaccounted for. See the [source-backed matrix audit](../../experiments/literature_audits/related_work_matrix_sagcr_cgdnet_completion_20261027_v1/report.md).


## 2026-10-27 source-group audit update

The sibling cache's native `train`/`val` folders share all 30 filename-prefix groups. The separate prospective `sen12_scene_dev1800_v2` split assigns source groups 15/15 and its selected pairs are covered by the official-source pixel audit. Preserve this as development-split evidence only; it does not change B1 `not_promoted`, authorize B2/B3, or resolve the separate numeric-name 2k subset. See [audit](../../experiments/data_audits/sen12_native_partition_group_overlap_20261027_v1/report.md).

## 2026-10-27 literature boundary update

SAG-CR and CGD-Net are now indexed as reverse-task SAR-assisted optical cloud-removal sources. Their alignment/reliability mechanisms narrow broad multimodal-fusion novelty claims but do not establish optical benefit for SAR output. This is a literature clarification only; the B1 gate and phase order are unchanged. See the [source audit](../../experiments/literature_audits/reverse_task_cloud_removal_alignment_review_20261027_v1/report.md).


## 2026-10-27 execution note

For future model-forward diagnostics, require explicit CUDA after a successful environment preflight and prohibit CPU/auto fallback. The direct-fusion v2 CPU run remains a completed historical run under its own frozen protocol; v3 completed as a same-input CUDA hardware repeat in the approved elevated profile. Its near-identical sensitivity deltas do not promote B1 or authorize B2/B3. See D-2026-10-27-014.


## 2026-10-27 prior-art resource note

HSFMamba remains direct prior art for optical-guided joint SAR SR/denoising. Its author-linked catalog currently exposes OGSSR as a download route but no model-code link or archive manifest; repository-level research-only terms do not settle archive-component or upstream imagery rights. Preserve the current B1 stop and do not infer that the local numeric-name subset is OGSSR. The resource audit is recorded at `experiments/literature_audits/hsfmamba_resource_license_code_screen_20261027_v1/`.


## MONet training-time optical-reference distinction — 2026-10-28

MONet adds prior art for generating synthetic SAR despeckling supervision from grayscale optical images, while the model uses SAR alone at inference. This does not answer the project's matched inference-time optical-utility question or change the B1 stop. Keep these evidence categories separate in novelty and method claims; see D-2026-10-28-049 and the [audit](../../experiments/literature_audits/monet_optical_training_reference_20261028_v1/report.md).

## 2026-10-30 文献补充与阶段边界

C4 候选证据已补充 PDE 物理约束生成与后验测度研究。CoCoGen、PISD 支持“可在 PDE 任务中把物理/观测约束并入生成过程”这一有限事实；CoCoS 预印本提示精确硬约束下还需审查目标测度。它们没有验证 SAR 表示域或 optical 引导。候选章程为 v1.5-proposed，未获批准前不同步到受保护的项目规范。此文献更新不构成实验晋级，不改变 B1 停止点；B2/B3 继续受原门槛约束。

## 2026-10-30 文献边界补充：联合 SAR 恢复、可靠性与效率

候选总纲 v1.6-proposed 增加已核实的直接先例：SAR-only 联合恢复、光学引导 SAR JDSR、自然图像参考自适应扩散，以及通用少步流/扩散逆问题求解。新增证据收窄“首次联合”“通用可靠性门控”和“少步生成”主张，但不证明这些方法适用于本地 SAR/光学协议。一个区域专用 SAR 扩散 JDSR DOI 线索仍需出版社正文核验。此更新不改变 B1/C2/B2/B3 阶段门槛。

## 2026-10-31 执行状态补记：C1 v2 数据入口

C1 v2 的训练/开发/确认身份已冻结，但仓库 `./data/` 目前无法映射到该清单。C1 v2 的 C2 实验尚不能在本地 2k 上开跑；恢复条件为可信身份交叉表或通过冻结校验的 C1 v2 训练/开发载荷。误启动的全量训练已中断并登记为无效，不改变阶段门槛。详见[误启动与对应关系审计](../../experiments/diagnostics/c1_v2_local2k_alignment_20261031_v1/report.md)。


## 2026-11-07 文献更新记录

新增 FlowSGS 与 FMPS 一手全文审查，收窄通用 flow 逆问题和后验条件化的新颖性空间。未改变 C1→C2→C3 的阶段顺序与门槛；训练仍按用户要求暂停，C3-A 真实模型效果未验证。


## 2026-11-07 无 GPU 方法进展

训练暂停期间完成了乘性噪声 SDE 一手文献审查，并用纯合成 CPU 实验比较方差匹配的 Gamma 与对数正态乘子。该工作用于明确生成模型先例和噪声假设边界；不修改 C1–C6 顺序、冻结协议、gate 或晋级资格。详见[文献审查](../../experiments/literature_audits/multiplicative_noise_sde_fulltext_20261107_v1/report.md)与[合成实验](../../experiments/diagnostics/multiplicative_noise_distribution_cpu_20261107_v1/report.md)。


## 2026-11-07 双峰流求解器 CPU 方法探索

解析双峰场的 16 NFE 求解器对照发现，Heun8 在强分离条件下的中央谷质量高于理论分布；Heun32/64 在这些合成设置上接近目标。四种简单非均匀 8 步网格没有改善该强分离场。使用 C3-A `noise_sigma=0.1` 量级的 `[0,1]` 标量场后，Heun8 仍显示中央质量偏差；仓库 `sample_heun` 与独立 NumPy Heun 实现逐点一致。以上仅是求解器方法筛查，不修改冻结 C3-A 配置，不构成阶段晋级，也不替代真实条件模型验证。材料见[双峰主报告](../../experiments/synthetic_cpu/c3a_bimodal_flow_solver_spike_20261107_v1/report.md)、[时间网格诊断](../../experiments/synthetic_cpu/c3a_bimodal_time_grid_cpu_20261107_v1/report.md)和[表示尺度/实现核验](../../experiments/synthetic_cpu/c3a_representation_scale_bimodal_cpu_20261107_v1/report.md)。


## 2026-11-07 C1 train/development 来源元数据连接

冻结 C1 v2 train/development 的 source group 与原始相对路径、SHA-512、派生 x2_light 文件清单现已按 `(source_group_id, patch_id)` 完整交叉核对。结果支持派生训练/开发输入的清单级来源可追溯和组间互斥，不证明地理/采集独立或官方数据集/传感器属性；原始 `data/sar`、`data/opt` 数字文件名仍没有安全的 train/development-only 内容映射。不得读取未知角色载荷填补此项。详见[审计记录](../../experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/report.md)。


## 评价指标的纯 CPU 已知样例核验（2026-11-07）

已用手工构造数组核对公共评价函数的基本算术合同，并对照 C3-A 推理评分中的 `[0,1]` 裁剪及 PSNR 计算。25 条断言通过。EPI 相关对正幅值缩放不变，因此必须结合边缘强度、亮回波/TBR 与辐射误差解释；ratio epsilon 的偏差已按源码公式计入。未发现覆盖范围内的实现错误，不改变正式评价定义或阶段门槛。详见[实验报告](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/report.md)。该结果不证明指标的物理有效性。


## 输出裁剪边界偏差的合成诊断（2026-11-07）

对 7 个 `[0,1]` 目标亮度、3 档加性高斯误差（每档 500,000 个值）进行纯 CPU 对照。裁剪作为区间投影不会增加目标在区间内时的逐点平方误差，但在 0/1 边界附近可提升 PSNR 并引入指向区间内部的均值偏差；因此 PSNR 应与裁剪率及强度/亮回波诊断共同报告。此结果不代表模型输出分布或 SAR 噪声，不改冻结裁剪协议。见[报告](../../experiments/synthetic_cpu/clip_boundary_bias_cpu_20261107_v1/report.md)。


## SAR-only 联合扩散先例的来源等级更新（2026-11-07）

项目既有作者稿审阅记录支持将 RSDM/LSIR 记为作者报告级 SAR-only 联合超分/去斑扩散先例；本次 IEEE/DOI 复核和 arXiv 有界检索均未取得新的正式全文。不得把正式 DOI/出版版本关系写成已核实，也不得声称无此类方法。来源和方法边界见[审计记录](../../experiments/literature_audits/region_specific_sar_jdsr_source_recheck_20261107_v1/report.md)。


## SAR 单模态扩散超分摘要证据边界（2026-11-07）

IEEE 官方摘要为一篇 2024 ICSP SAR 单模态扩散超分论文提供了题名、残差预测、MSTAR/SARBake 与下游分割任务线索；正式 PDF 直连不可访问，协议细节仍未知。它不证明联合去斑或光学引导效用。详见[审计](../../experiments/literature_audits/icsp_sar_diffusion_sr_abstract_pdf_route_20261107_v1/report.md)。


## 指标错误消息的中文一致性（2026-11-07）

纯 CPU 指标已知样例发现 SSIM 窗口尺寸异常仍为英文；已用 TDD 将该单条用户可见 `ValueError` 提示改为中文。25 条合成断言及既有评价器测试 26 项通过。只改错误消息，不改公式、协议或 gate。见[核验记录](../../experiments/diagnostics/metric_known_case_cpu_20261107_v1/report.md)。
