# 下一步行动

## 当前优先行动 — 2026-11-07 00:59（中国标准时间）

1. **维持训练暂停：** 用户要求暂缓训练；不要启动 C3-A 训练、GPU 等待器或正式评价。
2. **CPU 合同工作已完成：** train/development SAR 数据预检、Flow 张量/数值合同、评价器输入清单保护、checkpoint v2 续训绑定及 epoch 中断边界均有直接 CPU 验证；相关定向回归 67 项通过，`py_compile` 通过。
3. **当前研究边界：** 不读取 optical、confirmation 或 test 载荷；CPU 合成测试不构成模型效果或晋级证据，不改变 C2/B1/B2/B3 gate。
4. **后续依赖：** 只有用户明确恢复训练后，才按冻结协议进行资源门槛核验并启动训练及 development 评价；训练暂停期间不再遗留本轮已识别的 C3-A CPU 续训实现项。
5. **文档语言：** 本次直接引用的 5 份历史审计文档已改为中文；专门的中文报告测试通过。
6. **交付与空间：** 清理本轮生成的 14 个 `/tmp` 诊断日志和 15 个相关 Python 缓存后，分区可用约 21.57 GiB；pytest 既有缓存、项目数据、历史实验和 checkpoint 均保留。全部授权工作和验证完成后，审查精确提交范围并提交、推送。

## 当前优先行动 — 2026-11-06 23:37（中国标准时间）

1. **继续 CPU 合同审查：** C3-A train/development 的 3,600 个 SAR LR/target 数组已完成 CPU 完整性预检；下一步审查训练入口的配置、随机性、checkpoint 保存/恢复合同，发现问题时用合成输入和定向测试验证。
2. **保持训练暂停：** 用户要求暂缓训练；v8 等待器 `42473` 已停止。不要自动重启等待器、模型训练或正式评价；后续训练须等用户明确恢复指示。
3. **保持研究边界：** 不读取 optical、confirmation 或 test 载荷；CPU 软件与数据校验不构成模型性能证据，不改变 C2/B1/B2/B3 gate。
4. **磁盘与交付：** 23:37 UTC 项目分区可用 21,117,218,816 字节（约 19.67 GiB）。仅清理本任务确认产生且可重建的缓存；全部目标工作及适用验证完成后，检查提交范围并提交、推送。

## 用户暂停 GPU 任务前的优先行动 — 2026-11-06 20:54（中国标准时间）

1. **已完成总纲批准同步：** 用户已批准 v1.7；正式总纲、AGENTS.md、README.md 和状态记录已更新，不再等待审批。
2. **继续 C3-A 资源等待：** v8 会话 `42473` 最近 GPU 空闲 1146/4096 MiB、3 个其他 CUDA 进程；达到零并发且至少 3,072 MiB 空闲后，按原训练协议运行，再做冻结 development 多次采样 v2 评价。
3. **监视磁盘：** 当前可用约 15.06 GiB。近期项目内无新增大文件；只清理本任务产生且确认可重建的缓存，保留原始/派生数据、历史实验与 checkpoint。低于 2 GiB 时等待器继续等待。
4. **完成实验后交付：** 继续更新当前状态、实验登记和决策记录；全部授权工作与验证完成后，核对目标文件并提交、推送供用户审查。

## 当前优先行动 — 2026-11-06 18:35（中国标准时间）

1. **继续 C3-A 等待：** 10:34:26 UTC GPU 空闲 1,453/4,096 MiB、2 个其他 CUDA 进程；训练尚未启动，门槛未满足。
2. **继续监视磁盘：** 可用空间约四分钟从 4.88 GiB 回升至 13.47–13.611 GiB，仓库无对应大文件变动。变化来源未确认；不删除历史/他人文件，等待器低于 2 GiB 时继续等待。
3. **记录与交付：** GPU/磁盘门槛满足后沿原协议训练并评价；完成总纲审批及授权范围内工作后，核对并仅提交本目标文件、推送给用户审查。

## 当前优先行动 — 2026-11-06 18:31（中国标准时间）

1. **保持 C3-A 等待：** v8 最新 GPU 快照为空闲 1,229/4,096 MiB、3 个其他 CUDA 计算进程；只有零个其他进程且空闲至少 3,072 MiB 才可启动。
2. **监视磁盘：** 当前可用约 4.88–4.909 GiB，约 11 分钟减少 7.23 GiB；仓库没有对应新增大文件，原因未知。低于既定 2 GiB 门槛时等待器会阻止启动；不清理来源不明内容或历史数据。
3. **续跑与记录：** GPU 和磁盘门槛同时满足后，由 v8 按原 C3-A v1 训练并接续冻结的 development 多次采样 v2 诊断；不重新划分、不访问 confirmation/test/optical。
4. **总纲：** 等待用户对 v1.7 候选的答复；在明确批准前继续以 v1.6 为正式版本。
5. **交付：** 目标范围内工作、实验与适用验证完成后，核对提交范围并推送。

## 当前优先行动 — 2026-11-06 18:20（中国标准时间）

1. **等待并运行 C3-A：** v8 等待会话 `42473` 使用唯一目录每分钟检查 GPU/磁盘；最近快照为 1,453/4,096 MiB 空闲、2 个其他计算进程，未达到零并发且至少 3,072 MiB 空闲的门槛。达到冻结门槛后依次运行原 C3-A v1 训练和多次采样 v2 development 诊断。
2. **守住数据与空间边界：** 当前训练/评价目录不存在，train/development 载荷尚未读取。当天新增大于 10 MiB 项目文件为零；没有确认属于本任务的可清理大型结果。继续保留源数据、派生数据、历史实验、checkpoint 与其他项目文件；仅清理本任务新生成且可重建的缓存。
3. **完成总纲审阅：** 核对 v1.7 候选与逐节差异报告；当前只校正 FIRM arXiv v2 题名，v1.6 继续生效。候选批准前不改正式总纲、AGENTS.md、README.md 的研究方向。
4. **完成后交付：** 完成授权范围内实验、验证和研究记录后，审查并仅提交本目标改动，再推送供用户审阅。

## 当前优先行动 — 2026-11-06 14:30（中国标准时间）

1. **持续等待 C3-A：** v5、v6 均在训练启动前安全停止；唯一活动会话为 `87305`，记录见[v7 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261106_v7/manifest.json)。06:28:49 UTC 快照有 3 个其他 CUDA 进程、空闲显存 1,070 MiB；不启动并发训练。
2. **资源满足后训练：** 按原 C3-A v1 协议使用冻结 C1 v2 train/development SAR-only 输入；训练成功后再按冻结多次采样 v2 运行 development 诊断。
3. **监视磁盘和数据边界：** 分区 06:28 UTC 可用约 16.429 GiB。v7 在训练/评价前要求至少 2 GiB，低于门槛时等待。仓库没有可安全清理的大型本任务产物；仅清理了本轮测试生成的缓存。保留原始数据、历史实验和其他项目文件。confirmation/test/optical 保持关闭。
4. **完成后交付：** 完成目标范围内的实验、验证和研究记录后，只提交目标文件并推送，供用户审查。

## 当前优先行动 — 2026-11-06 10:11（中国标准时间）

1. **持续等待 C3-A：** 工具会话 `23990` 使用等待记录 v4 每分钟轮询；最近门槛未满足，继续等待，不启动并发训练。
2. **资源满足后训练：** 按原 C3-A v1 协议使用冻结 C1 v2 train/development SAR-only 输入；训练成功后再按冻结多次采样 v2 运行 development 诊断。
3. **控制空间与访问：** 启动前持续核对分区空间；只清理可再生缓存，不删原始数据、历史实验或其他项目文件。confirmation/test/optical 保持关闭。
4. **完成后交付：** 完成目标范围内的实验、验证和研究记录后，只提交目标文件并推送，供用户审查。


## 当前优先行动 — 2026-11-05 21:32（中国标准时间）

1. **完成 C3-A 训练：** 新等待记录 v4 每 60 秒轮询，见[等待 manifest](../../experiments/diagnostics/c3a_gpu_wait_20261105_v4/manifest.json)；只有 GPU 无其他 CUDA 计算进程且空闲显存至少 3,072 MiB 时，才启动原 C3-A v1 训练。等待期间不读影像、不操作其他进程。
2. **运行补充评价：** 训练成功后等待同一资源门槛，按 v2 协议在固定 development 身份上执行四次采样及固定 edge-mask 结构评价；不访问 confirmation/test/optical。
3. **核对并作阶段决定：** 检查完整身份/来源组覆盖、失败样本、逐图与 bootstrap 统计、时间和显存；原 C3-A v1 规则决定是否追加 seed，补充诊断不改 gate。
4. **完成后交付：** 目标工作与验证全部完成后，只提交本目标文件并推送，供用户进行 ChatGPT 审查；收到用户下一步计划后再开展后续工作。


## 当前优先行动 — 2026-11-05 16:18（中国标准时间）

1. **等待并执行 C3-A v1：** 等待会话 `28461` 正在轮询；只在零个其他 CUDA 计算进程且空闲显存至少 3,072 MiB 时运行原训练入口；保留原预登记配置与新输出目录。
2. **执行四次采样诊断：** C3-A 完成后继续等待同一 GPU 门槛，再运行冻结的 development-only 补充协议。
3. **审查证据：** 核对完整样本数/来源组、同身份 baseline、逐图结果、bootstrap 区间、运行时间、显存和失败案例。
4. **作阶段决定：** 原 C3-A v1 规则决定是否追加 seed；补充稳定性结果用于解释，不修改既有 gate。之后继续附件中的研究计划。

## 活动等待与路线续接 — 2026-11-05 15:07（中国标准时间）

1. **继续 C3-A：** 活动安全等待会话 `25767` 仅检查资源，不读取数据；达到无并发 CUDA 进程且空闲显存至少 3 GiB 时自动按原 C1 v2 `x2_light` train/development 协议训练。不要启动第二个等待器或并发训练。
2. **在等待期间推进：** 已更新 README 的获批路线与证据状态，已完成 C3-A 参数核算。C3-A 训练前仍可继续做与数据访问无关的实现审查；不得更改冻结 protocol，也不得读取 confirmation/test。
3. **完成验证后再提交推送：** 当前目标仍未完成。待原训练、development 配对分析和阶段决策完成后，审查提交范围，提交并推送目标相关文件供用户审查。

## 最新续记 — 2026-11-05 12:01（中国标准时间）

1. **完成 C3-A 实验：** 最新预检 v8 检测到 5 个其他 CUDA 计算进程，尚未训练。GPU 资源变化后继续运行原协议、原命令和冻结 C1 v2 train/development；不重新划分、不访问 confirmation/test。不得把该阻塞当作整个目标已完成。
2. **保持已完成的代码改进：** 零背景对比差诊断及 43 项测试通过；若在真实 development 上应用，先独立登记描述性协议，结果只能作探索观察。
3. **完成后再提交推送：** 当前目标完成并验证后，审查提交清单，只提交目标相关文件，再提交并推送；不把大批既存未跟踪文件或密钥/数据纳入提交。

## 本轮更新 — 2026-11-05

1. **继续 C3-A 训练：** 当前 GPU 对训练器可见，但 4 个其他 CUDA 进程导致预检按协议停止。待资源状态变化后，再以原命令和原固定 C1 v2 `x2_light` train/development 输入启动；不重新切分、不并发训练、不访问 confirmation/test。见[预检 v7](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v7/manifest.json)。
2. **完成零背景指标的真实开发集诊断设计：** 新增均值/中位数对比差 helper 已通过 43 项合成及回归测试，但还没有在真实模型输出上运行。若继续用于模型比较，先登记独立的 development 描述性协议，并明确复用 development、固定掩膜和来源组统计；不得回写 C2 v2 gate。
3. **保持研究阶段边界：** 正式总纲为获批 v1.6；C2 原 gate 仍未判定，B1 仍 `not_promoted / stop_at_b1`，B2/B3 关闭。confirmation 已封存，test 不用于当前实验。

## 当前待办 — 2026-11-05（总纲 v1.6 已采纳）

1. **封存 C1 v2 confirmation 的方法选择用途：** 200 个确认身份已在访问后修订的协议下完成一次技术恢复复核，结果为 `inconclusive`。保留原协议、两份修订、失败记录和恢复运行；confirmation 不再用于训练、选模、调参或重复筛选。
2. **保留 C2 v2 与 B1 原门槛：** 原 C2 paired screen 因零背景 median TBR fail-closed，仍为“未判定”。本次 confirmation 描述性复核不回写原 gate，不晋级 B1；B2/B3 继续关闭。
3. **运行 C3-A SAR-only Flow 筛选：** 协议、实现和 CPU 通路验收已完成；用户已明确直接复用冻结 C1 v2 的 `x2_light` train/development 固定输入，不重新划分。最新 `AISAR` 预检仍发现 3 个其他 CUDA 计算进程，训练在读图前停止；不再重复轮询。待外部 CUDA 任务退出后做一次新鲜资源预检，达到 3 GiB 空闲门槛时运行 `scripts/run_phase_c3a_sar_flow.py`。按预登记规则与 C2 SAR-only 基线做来源组等权配对比较。development 已用于选模，只能报告探索性观察。详见[预检 v6](../../experiments/diagnostics/c3a_gpu_preflight_20261105_v6/manifest.json)。
4. **限制 optical 结论：** 四条件 matched−SAR-only PSNR 均未达到 +0.05 dB 且 bootstrap 下界大于零的预设条件；不得声称稳定增益。Q1–Q6 仍开放。
5. **保留来源限制：** C1 v2 仅证实 source-folder-disjoint；地理/采集交叉表和精细空间重叠仍待核验。不能将 confirmation 结果推广为地理泛化证据。
6. **报告执行偏差：** 两次执行器字段映射修正发生在 confirmation 载荷访问之后；结果仅作非晋级描述性复核。后续代码修复须先用合成样例验证完整后处理链。
7. **审阅 v1.7-proposed 候选：** 候选、逐节差异、支持审查和 C1 v2 数据协议审查已就绪；v1.6 继续生效，只有用户明确批准后才更新正式总纲及受保护文件。该审阅不阻塞使用现有 C1 v2 train/development 的预登记研究。

## 历史待办快照 — 2026-10-31（已由上方状态更新）

1. **保留 C2 v2 原始状态：** 八组训练已完成；冻结 paired screen 因两条开发身份的 target 背景中位数为零而 fail-closed，gate 状态是“未判定”。保留失败 manifest、checkpoint 和已完成的非门槛诊断，不覆盖原目录、不将补充结果写成 gate 结论。
2. **审查指标定义并冻结新版本：** 针对零背景下 median TBR 不可定义的问题，先用已知样例明确 TBR 及替代统计量的数学定义、适用域、无效值处理和全样本汇总规则；新规则不得回写或重跑成 C2 v2 结果。覆盖零背景、接近零背景、均匀区域和结构边缘的定向测试后，再决定是否需要新的开发集诊断。
3. **限制方法结论：** 当前非门槛结果显示 matched optical 相对 SAR-only 的 PSNR 点估计很小、四个区间均跨零且低于 +0.05 dB 继续阈值；不启动三 seed follow-up，不声称已证实 optical 增益。matched 与 permuted 的差值仅作为同一 development 集上的配对敏感性观察。
4. **保护确认数据：** C1 v2 confirmation 200 个样本继续封存；本轮没有访问 confirmation/test 载荷。development 已用于选 epoch 和本次诊断，后续不得把它重新称为独立 holdout。
5. **保留来源边界：** 来源组—地理/采集交叉表、精细空间重叠尚缺；在补足前只报告 source-folder-disjoint，不声称地理或采集独立。`./data/` 数字命名子集与 C1 v2 身份映射仍未核实。
6. **保持历史门槛：** 不修改 B1 `not_promoted / stop_at_b1`，不解锁 B2/B3。任何新方法筛查须基于版本化协议和新输出目录，并继续只用允许的 development 材料。


## 流匹配后验采样前沿 — 2026-10-30

- [x] 审阅 D-Flow SGLD 与贝叶斯流匹配后验采样两篇 2026 年一手全文，并将其加入相关工作矩阵；矩阵共 144 行，78 行旧清单归属已确认、66 行未确认。见 D-2026-10-30-007 和[审查报告](../../experiments/literature_audits/flow_posterior_sampling_frontier_20261030_v1/report.md)。
- [x] 收窄候选 v1.4 的方法新颖性边界：通用流后验、源空间后验、似然引导和少步生成本身均不构成独立贡献；不改变 SAR/JDSR 任务门槛。
- [x] 通过 PubMed 核验 D-Flow SGLD 的 *Journal of Computational Physics* DOI 与 2026-09-04 在线发表日期；卷期日期为 2026-12-15。见[审查报告](../../experiments/literature_audits/flow_posterior_sampling_frontier_20261030_v1/report.md)。
- [ ] 如 D-Flow SGLD 作者代码或贝叶斯流匹配论文正式版本／评审信息变得可访问，再核验实现与发表状态；不得把通用 PDE／湍流结果当作本地 SAR 证据。
- [ ] 候选 v1.4 仍待批准。C2 执行验收未完成；B1 仍为 `not_promoted / stop_at_b1`，B2/B3 仍受门控。

## OGSFlow 作者仓库后续审查 — 2026-10-30

- [x] 阅读作者仓库 README，并将两阶段结构估计／条件流生成及两阶段可靠性门控作为仓库说明层先例，收窄候选 v1.3 的 §9。见 D-2026-10-30-006 和[审查记录](../../experiments/literature_audits/ogsflow_readme_details_20261030_v1/report.md)。
- [ ] 只有在具体源码、固定提交或正式论文变得可访问时，才复核其实现、协议与相关工作对应关系；当前不得将 README 描述扩展为已验证的代码行为或效果。
- [ ] 候选 v1.3 仍待批准。C2 执行验收未完成；B1 仍为 `not_promoted / stop_at_b1`，B2/B3 仍受门控。

## DiSpeckle 书目来源路线 — 2026-10-30

- [x] 核验 ORCID 中 Crossref 来源的题名、作者、期刊、年份和 DOI 元数据；尝试出版方入口但未取得全文。未使用二手摘要，未加入矩阵或修改候选 v1.2。见 D-2026-10-30-005 和[审计记录](../../experiments/literature_audits/dispeckle_metadata_route_20261030_v1/report.md)。
- [ ] 仅当出版方全文、作者稿或可核实的作者实现可访问时，审查其随机微分方程、散斑模型、数据／退化和评价协议；不得用二手摘要作技术依据。

## 自适应参考引导文献 — 2026-10-30

- [x] 审查 Ada-RefSR 与 STMP-DiT 的自适应参考/多模态超分辨率相关工作；新增两条当前证据矩阵记录并收窄 C5 新颖性边界。矩阵现有 142 行（78 行确认属于旧清单，64 行归属未确认）。见 D-2026-10-30-004 和[审查记录](../../experiments/literature_audits/adaptive_reference_guidance_prior_art_20261030_v1/report.md)。
- [ ] 候选 v1.2 获批前继续暂停实现和实验。C2 执行验收未完成；B1 仍为 not_promoted / stop_at_b1，B2/B3 仍受门控。

## 条件流逆问题文献 — 2026-10-30

- [x] 审阅 ICML 2026 VFM 论文记录与 NeurIPS 2025 BFM 记录。将 VFM 记录为条件图像逆问题/超分辨率相关先例，将 BFM 记录为后续通用效率比较候选；该次审查后的矩阵计数为 140 行（78 行确认属于旧清单，62 行归属未确认）。见 D-2026-10-30-001 和[审查记录](../../experiments/literature_audits/flow_inverse_generation_prior_art_20261030_v1/report.md)。
- [x] 将 VFM 假设与已核验目标及冻结 C2 算子交叉核对。本地目标是已含噪的归一化伪目标，不能直接套用 VFM 的高斯似然。见 D-2026-10-30-002。
- [x] 根据冻结的逐元素 Gamma 退化链解析候选 p(y|T)：退化前的零像素对应零点原子；非零像素服从 Gamma 缩放密度；大于 1 的值仍合法。该结果仅为解析推导，未用数组或实现验证。见 D-2026-10-30-003。
- [ ] 获批且 C2 执行验收完成后，推导该候选似然下完整的 VFM 变分目标，明确 float32 固定输入的有限精度观测语义，并在已知合成样例上验证目标。该模型针对伪目标 T，不代表干净 SAR；不得作物理主张。
- [ ] 在批准的研究方向、数据/目标/似然依据、光学效用资格及相关协议门槛满足前，继续暂缓 C3/C4 实现。不得根据 ImageNet 结果推断本地 SAR 效果。
- [ ] 候选 v1.2 获批前继续只读审查章程；不得修改受保护的 AGENTS.md/README.md 方向，也不得在缺少各自前置条件时启动 C2/B2/B3 工作。

## Required manuscript review exposure boundary — 2026-10-29

- [x] Record text-only exposure of historical test-associated prose during the required manuscript/document review as `test_set_accessed=true`, `test_set_used=false`; no metric payload or test image was opened, and no value informed a decision. See D-2026-10-29-027 and [group 517](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md).
- [x] Record the second required read of the pilot draft and the bounded formal-paper README status search as text-only exposure; exclude all surfaced pilot test values and mini identities from future method design or selection. See D-2026-10-29-030 and [group 518](../../experiments/data_audits/research_text_read_exposure_20261029_v1/report.md#group-518--source-brief-review-reread-historical-pilot-test-prose).
- [ ] Continue charter and literature planning without using surfaced test-associated values for design, tuning, or model selection. No run was started; no manifest exists to update.

## Historical gate review for the charter candidate — 2026-10-29

- [x] Read the legacy B1/B1R/B1F reports, the current-source B1 v2 aggregate, and the selected Phase 1-R/S/T reports. Candidate §14 preserves their recorded states and split limits; no historical gate or proposal text needs changing. See D-2026-10-29-031.
- [ ] Keep those validation/inner-dev and target-assisted oracle findings out of future selection and promotion. Preserve B1 `not_promoted / stop_at_b1` and keep B2/B3 gated.

## C2 metric-contract acceptance — 2026-10-29

- [x] Compare frozen v2 ratio and mask requirements with helper behavior. The shared helper inherits `eps=1e-8`, can return null for the reported edge correlation, and the mask builder does not return coverage; see D-2026-10-29-026.
- [ ] Before C2 generation/training, complete an authorized runner that binds evaluator revision and effective epsilon, writes each mask's positive-pixel count and coverage fraction with thresholds/condition provenance, and rejects any undefined required metric under the frozen whole-evaluation policy. Add focused runner-level acceptance tests when implementation is authorized.
- [ ] Keep C2 data generation and training gated until this runner acceptance, staging-path review, and the required fresh GPU/input preflight are complete. Do not alter frozen protocol v2 in this audit.

## H8 SAR task-order literature boundary — 2026-10-29

- [x] Recheck the publisher-indexed ISSMSAR primary text for SAR-specific task-order framing. It distinguishes despeckling→SR from SR-first auxiliary despeckling and qualitatively critiques the former, but its reported baselines do not form a matched order-reversal or joint-versus-cascade comparison. See D-2026-10-29-025 and the [audit addendum](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md#addendum-2026-10-29-sar-task-order-framing).
- [x] Record that the bounded H8 note search surfaced unrelated historical development-result prose and exclude it from the literature decision; no test/confirmation payload or metric was opened and no value informed the decision. See [exposure group 516](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md#group-516--issmsar-task-order-literature-search-2026-10-29).
- [x] Complete the bounded primary-source search. Muzeau et al. compare SAR-SR supervision targets but not task order; DSRKD confirms joint-task prior art but its full protocol remains inaccessible. A separate journal-indexed cascade comparison is only an unverified lead because the publisher article/PDF returned 403. See D-2026-10-29-028 and the [source review](../../experiments/literature_audits/sar_task_order_source_review_20261029_v1/report.md).
- [ ] Reopen this search only if a publisher/author full-text route or versioned implementation becomes accessible. Keep H8 gated on valid source/target provenance and an eligible development split; do not run a model comparison before those prerequisites.

## Physical-consistency title evidence — 2026-10-29

- [x] Resolve the current-data boundary: official SEN1-2 documentation describes stored SAR as 8-bit sigma-nought dB; the selected C1 v2 PNGs match official source, while the separate earlier B1 v2 protocol uses representation-domain synthetic noise and a noisy pseudo-target. No byte-to-dB mapping or raw calibrated values are retained. Keep “physics-consistent” conditional and use representation-domain wording for PNG-based work. See D-2026-10-29-024.
- [ ] Reopen physical-model work only if authoritative measurement-domain data, calibration/transfer metadata, and a forward/noise model suitable for independent validation become available.

## CN122694669A official publication route — 2026-10-29

- [x] Check the official CNIPA homepage and publication-announcement search route for the provisional patent lead. The official search page opened but no item-specific record was retrieved; PSS returned HTTP 412 and the publication portal timed out. See D-2026-10-29-023 and the [route audit](../../experiments/literature_audits/cn122694669a_official_route_20261029_v1/report.md).
- [ ] Resume only if an item-specific official publication or complete patent document becomes accessible; preserve the provisional status and do not use the secondary index as efficacy evidence.

## SEN1-2 official catalog metadata route — 2026-10-29

- [x] Verify the official ISPRS release announcement's mediaTUM record pointer. Direct record access returned HTTP 429; exact-ID/site-scoped official searches did not surface a scene-coordinate/product/acquisition crosswalk. See D-2026-10-29-022 and the [route audit](../../experiments/data_audits/sen1_2_official_catalog_metadata_route_20261029_v1/report.md).
- [ ] Resume metadata-only review only if the official catalog becomes accessible or a versioned authoritative crosswalk/sidecar appears. Preserve the source-scene-folder-disjoint limit and keep confirmation payload closed.

## Reliability-aware SAR–optical classification prior art — 2026-10-29

- [x] Review the publisher-indexed Vulli et al. article and version notes; record its sample-dependent SAR/optical classification gate, synthetic optical-degradation protocol, author-reported multi-seed/spatial evaluation, and stated one-region/season and heavy-noise limitations. Direct MDPI HTML/PDF/XML retrieval was unavailable; keep the evidence level bounded. See D-2026-10-29-021 and the [audit](../../experiments/literature_audits/reliability_aware_sar_optical_classification_20261029_v1/report.md).
- [x] Add the article to the related-work matrix as adjacent classification prior art; the matrix now contains 138 rows (78 confirmed historical members; 60 with unresolved original membership).
- [ ] Keep a generic reliability gate out of standalone novelty claims. Reopen full-text/supplementary review only if direct publisher access changes; do not use classification evidence as restoration efficacy or H1 evidence.

## Charter requirement coverage update — 2026-10-29

- [x] Make the candidate's expected academic-contribution deliverable explicit while preserving its conditional, evidence-dependent framing. Candidate is now `v1.2-proposed`; see D-2026-10-29-020 and the [change report](RESEARCH_CHARTER_CHANGELOG.md#8-explicit-expected-contribution-deliverable--2026-10-29).
- [ ] Review and explicitly approve or revise candidate v1.2 before adopting it or editing protected `AGENTS.md`/`README.md` research-direction text.
- [ ] Keep C2 execution acceptance, source eligibility, CUDA preflight, and the existing B1/B2/B3 gates as the independent prerequisites for any experiment; candidate approval alone authorizes no run.

## Charter novelty candidate narrowed against OGSFlow — 2026-10-29

- [x] Add OGSFlow's repository-level flow SAR-SR and reliability-gate evidence to the candidate's prior-art boundary; do not present generic flow matching, optical guidance for SAR-SR, or a learned gate as standalone novelty.
- [x] Keep it out of the scholarly matrix because no paper identity or bibliographic record is verified; update the candidate, change report, and supporting audit. See D-2026-10-29-019.
- [x] Update the approval request to identify candidate v1.2-proposed; approval remains pending.
- [ ] If a versioned OGSFlow paper or inspectable implementation/protocol becomes available, review it before any future method comparison. Do not repeat inaccessible routes without a material access change.

## GLCdiffcr conditional-consistency diffusion audit — 2026-10-29

- [x] Verify the official publisher metadata/abstract and classify the method as SAR-guided optical cloud removal with optical pixel/SSIM consistency.
- [x] Add the reverse-task abstract-level row and sync the research records. Matrix: 137 rows, 78 historical-membership confirmed, 59 unresolved. See D-2026-10-29-018 and the [audit](../../experiments/literature_audits/glcdiffcr_sar_guided_optical_cloud_consistency_diffusion_abstract_20261029_v1/report.md).
- [ ] Keep the paper's optical-output consistency separate from SAR observation consistency. Reopen the full protocol only if the publisher or author-linked manuscript becomes accessible; the abstract says data are available on request, and no request was sent.

## DMDiff SAR-guided optical diffusion audit — 2026-10-29

- [x] Review publisher-indexed method/data text and the paper-linked repository. Classify DMDiff as SAR-to-optical cloud removal; code is not present in the inspected repository page.
- [x] Add it as reverse-task diffusion/gating prior art and synchronize records. Matrix: 136 rows, 78 historical-membership confirmed, 58 unresolved. See D-2026-10-29-017 and the [audit](../../experiments/literature_audits/dmdiff_optical_cloud_removal_diffusion_fulltext_excerpt_20261029_v1/report.md).
- [ ] Keep cloud-mask robustness separate from geometric mismatch and SAR-output fallback; revisit only if runnable author code or an identified source-group split becomes available and materially affects a method comparison.

## CDD optical–SAR diffusion-clustering abstract audit — 2026-10-29

- [x] Verify the official institutional abstract and bibliographic record; add the method at abstract-only evidence level. The full-text link returned HTTP 403 and no author-linked code route was established.
- [x] Add the adjacent-prior-art row and sync the research records. Matrix: 135 rows, 78 historical-membership confirmed, 57 unresolved. See D-2026-10-29-016 and the [audit](../../experiments/literature_audits/conditional_dual_diffusion_optical_sar_clustering_abstract_20261029_v1/report.md).
- [ ] Keep CDD clustering evidence separate from optical-guided SAR-output JDSR and fallback claims; reopen the protocol review only if the full paper or a verified author artifact becomes accessible.

## SDIF optical–SAR diffusion-fusion literature review — 2026-10-29

- [x] Review the full publisher article and record SDIF's fused-image output, SAR-BM3D preprocessing, diffusion objective, semantic term, author-reported dataset/training details, and unresolved inference-time label availability.
- [x] Add SDIF to the matrix and update the literature, research, and risk records. Matrix status: 134 rows, 78 historical-membership confirmed, 56 unresolved. See D-2026-10-29-015 and the [audit](../../experiments/literature_audits/sdif_remote_sensing_image_fusion_fulltext_20261029_v1/report.md).
- [ ] Keep SDIF's fusion/classification results separate from optical-guided SAR-output JDSR evidence. Revisit inference-time label availability only if an accessible implementation or author clarification supplies a direct answer; no project experiment or gate depends on this paper.

## SAR split-Gibbs posterior-imaging source review — 2026-10-29

- [x] Read the full arXiv v2 author manuscript and record the complex Fourier/additive-Gaussian likelihood, split-Gibbs likelihood/prior updates, data protocol, and reporting limits. Add it as separate SAR measurement-domain prior art; keep the related IEEE ICASSP record separate until version identity is verified. Matrix status: 133 rows, 78 historical-membership confirmed, 55 unresolved.
- [ ] Keep the source's reported gains out of local evidence and do not use it to justify a Gamma likelihood over normalized PNGs or optical-guided raster JDSR. Revisit version lineage/code only if an author-controlled source provides a clear link. See D-2026-10-29-014 and the [audit](../../experiments/literature_audits/sar_split_gibbs_posterior_imaging_fulltext_20261029_v1/report.md).

## Flow posterior-transport literature review — 2026-10-29

- [x] Review the full FlowDPS author manuscript and the full 2026 posterior-transport preprint; add the theory record to the matrix and research notes. Matrix status is now 132 rows: 78 confirmed historical members and 54 entries with unresolved historical membership.
- [ ] If a posterior-sampling route is later authorized and reaches an eligible phase, first establish a data-supported likelihood and distinguish posterior-sample fidelity from point-estimate quality; predeclare distributional, observation-consistency, diversity, and uncertainty checks. The current evidence does not authorize code, data staging, or a run. See D-2026-10-29-013 and the [audit](../../experiments/literature_audits/flow_posterior_transport_source_review_20261029_v1/report.md).

## SEN1-2 C1 scene-split provenance review — 2026-10-29

- [x] Verify against the official SEN1-2 paper that the C1 group key follows the parent ROI/scene unit and that whole-scene separation is the documented split unit.
- [x] Record the exact limit: retained metadata has no ROI footprint or product/acquisition IDs, so geographic-footprint and acquisition independence for this assignment remain unverified. See D-2026-10-29-012 and the [audit](../../experiments/data_audits/sen1_2_scene_split_metadata_review_20261029_v1/report.md).
- [ ] If authoritative ROI-coordinate and source-product metadata becomes available, check cross-root spatial footprints and acquisition overlap. The official mediaTUM route returned HTTP 429 on 2026-10-29 and yielded no crosswalk; do not retry without a material access change. Otherwise retain the source-scene-folder-disjoint description and current exploratory-only eligibility. Do not access confirmation payload or alter the B1/B2/B3 gates.

## July manuscript status review — 2026-10-29

- [x] Confirm the root July paper directory is an internal engineering pilot, not a current submission manuscript. Record the draft's test-associated text exposure without reproducing or using its values. See D-2026-10-29-011 and the [audit](../../experiments/literature_audits/manuscript_draft_status_review_20261029_v1/report.md).
- [ ] Keep the 100-pair mini-pilot cohort and its reported outcomes out of model selection, gate changes, and formal manuscript claims. Any future formal paper must use eligible prospective evidence under the approved protocol.

## SAR-to-EO flow-matching prior-art review — 2026-10-29

- [x] Review the primary arXiv manuscripts for ContraFM-S2O and ReFlowSET, record both as reverse-task SAR-to-EO flow-matching prior art, and update the matrix to 131 rows (78 historical-membership confirmed; 53 unresolved).
- [x] Check the existing OGSFlow repository title/paper status; no public paper link was found, so retain its repository-level evidence boundary.
- [ ] Keep the charter candidate inactive pending explicit approval. Do not change H1/B1/C2 status or run experiments from these reverse-task results. See D-2026-10-29-010.

## γ-Bridge prior-art follow-up — 2026-10-29

- [x] Add the primary-source review to the matrix and research records; keep all reported performance attributed to the authors.
- [ ] If the charter is approved and C3/C4 become eligible, compare a suitable SAR-only Gamma-structured generative control under the authorized data and observation protocol; verify source/code availability first.
- [ ] Keep the C2 runner, data acceptance, and existing gate dependencies in order. This literature update authorizes no staging, training, or model selection.

## Charter-review exposure boundary — 2026-10-29

- [x] Record the broad status-search exposure conservatively as `test_set_accessed=true`, `test_set_used=false`; no metric value surfaced or was used. Keep test identities and measurements closed. The C2 protocol's own false/false flags are unchanged. See D-2026-10-29-005.
- [ ] Continue only with explicitly scoped primary-source and non-test data/protocol review. Do not use any test-associated text or measurements for model choice, threshold setting, or claims.

## Research-charter revision proposal — 2026-10-29

- [x] Prepare the full inactive charter candidate, sectionwise change log, and supporting literature/data-protocol audit. No experiment or gate changed; see D-2026-10-29-004.
- [x] Reread the full source brief and verify that the candidate covers its numbered requirements; compare the frozen C2 settings with the v2 JSON. Clarified official SEN1-2 priority in C1; no omission remains. See D-2026-10-29-009.
- [ ] Review and explicitly approve or revise the candidate before adopting it or editing protected `AGENTS.md`/`README.md` research-direction text.
- [x] Reconcile the C2 protocol-v2 frozen-artifact status with `CURRENT_STATE.md`, `MASTER_RESEARCH_PLAN.md`, `PROJECT_FACTS.md`, the decision log, and this action list. D-2026-10-29-004 records protocol freeze separately from execution readiness; the end-to-end runner/acceptance work remains open.
- [ ] Before any C1 staging or training, satisfy the v2 acceptance list, including an end-to-end CUDA runner and its targeted tests, fresh CUDA preflight, and verified train/development source inputs. Preserve confirmation/test boundaries and B1/B2/B3 gates. The scene-split paper supports the grouping unit but does not resolve footprint or acquisition independence for this assignment.

## C2 matched controls — 2026-10-29

- [x] Implement the matched ×2/×4 SAR-only/HR-optical candidate, train/development reader and CUDA-only fixed generator; synthetic GPU contracts passed. No trained comparison or quality result exists. See D-2026-10-29-001/002/003.
- [x] Freeze the prospective screen design in protocol v2, including training recipe/budget, seeds, matched/permuted/missing mapping, common metrics, group-aware uncertainty and decision/stop rules. The design treats the noisy pseudo-target as a representation-domain reference. The v2 artifact is frozen; execution acceptance remains pending.
- [x] Reconcile protocol-v2 status with current records: v2 freezes the design; execution acceptance is still incomplete. See D-2026-10-29-004.
- [x] Complete a read-only execution-acceptance inventory: helper/model/reader/generator components exist, but the C2-specific staging entrypoint and end-to-end train/evaluation runner plus runner test are absent from the scoped inventory. See D-2026-10-29-007.
- [x] Audit the frozen protocol binding path: the current generator/reader do not enforce the pinned C1 sample-manifest hash, and the available stage helper targets an older train/val cohort. Record the exact missing runner boundaries without touching data or code. See D-2026-10-29-029.
- [ ] Implement and review a C1-v2 train/development-only staging boundary and CUDA-only end-to-end runner. Enforce the exact C2 protocol ID/hash and pinned C1 manifest hash; audit shared target/optical array parity across the four cells; implement all eight trainings and development group-balanced PSNR checkpoint selection with the frozen tie-break; run all four paired contrasts with complete identity/group checks and group-aware outputs; bind metric/evaluator settings and fail closed on undefined values; add targeted tests for these contracts and CUDA/resource/output preflight. Preserve the frozen JSON/hash.
- [ ] After runner checks pass, obtain a fresh approved GPU/resource preflight and verify train/development inputs before generating fixed arrays in a unique directory. Then run all eight frozen trainings, evaluate the optical controls, and produce group-aware paired results. Keep confirmation/test payloads and metrics untouched.
- [ ] Analyze outcomes against declared limits, then choose Continue/Pivot/Stop before architecture search or diffusion. Preserve B1 `not_promoted / stop_at_b1` and the B2/B3 gate.

## Phase C1 v2 payload validation before model selection — 2026-10-28

- [x] Assess residual source and overlap risks: accept C1 v2 for prospective source-folder-disjoint exploratory C2 development use under the explicit limits in D-2026-10-28-075. No geographic/acquisition independence or untouched confirmation overlap has been established.
- [ ] Freeze and implement a fair C2 ×2/×4 fixed-input protocol and matched SAR-only/optical paths. Existing model construction, training entrypoint and paired crop are hardcoded ×2; adapt them in a versioned path, verify shapes/degradation/evaluator and use a fresh GPU preflight before model runs.

- [x] Poll narrow-overlap audit session `77692` to completion: no candidate translations in SAR or optical; 115 optical windows hit an excluded repeated signature. Temporary images were removed. See D-2026-10-28-074.

- [x] Screen frozen train/development samples for exact integer-translated narrow overlaps in both modalities. No candidates were retrieved; 115 optical windows were excluded as repeated signatures, and the report states coverage and limitations. Confirmation imagery remains untouched.
- [x] Decide prospective development eligibility with optical exclusions, unverified geographic/acquisition relationships and untouched confirmation explicitly documented; historical B1 gate preserved. See D-2026-10-28-075.

- [x] Poll session `26461` to completion; the train/development payload audit passed file-hash and PNG schema checks, reported zero duplicates in its fixed-grid screen, and removed temporary payloads. See D-2026-10-28-071.

- [x] Complete the initial authenticated 1,800-pair cohort inventory and train/development-only representation, target, degradation, and evaluator audits. The old 30 groups were already consumed by B1 v2; the representation and synthetic checks remain domain-limited. See the [C1 v1 inventory](../../protocols/mmcanet_phase_c1/protocol_report.md) and [audit index](DATA_REPRESENTATION_AUDIT.md).
- [x] Retain the official per-group source inventory, file sizes, and SHA-512 values; verify 131 groups, 142,506 matched patch indices, zero cross-group exact-file duplicates, and zero exact hash overlap with historical B1 v2 inputs. See D-2026-10-28-069 and the [C1 v2 report](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/report.md).
- [x] Freeze metadata-level roles before patch sampling: 92/19/20 source groups and 1,600/200/200 train/development/confirmation pairs under seed 20261028. This is not yet a full C1 pass.
- [x] Stage train/development payloads temporarily; verify file sizes, SHA-512 and PNG schema, record representation statistics and remove staged payloads. Confirmation payload remains unaccessed.
- [x] Run cross-role exact decoded-pixel and fixed-grid 128×128 window checks. Both found zero matches within their stated scope; geographic/acquisition independence, arbitrary-offset overlap and fine registration remain open.
- [x] Assess C1 v2 for limited prospective, source-folder-disjoint exploratory C2 development use under D-2026-10-28-075. Do not claim geographic/acquisition independence, untouched confirmation spatial independence, or fine registration. Preserve B1 `not_promoted / stop_at_b1`.

## Local source-lineage connector route — 2026-10-28

## Local source-lineage connector route — 2026-10-28

- [x] Recheck the currently catalogued Drive metadata search and list-drives actions after the tools reappeared. Three document searches and one list call all failed with backend `Unknown tool`; no metadata was returned. See D-2026-10-28-066 and exposure group 515.
- [ ] Resume metadata search only after the Drive backend actually returns results, or when a new authoritative subset-generation/file-to-source-group artifact is available. Keep the local numeric-name 2k cohort unmapped and model selection paused; do not repeat the same unavailable route.
- [ ] Preserve B1 `not_promoted` / `stop_at_b1`; do not enter B2/B3.

## CloudLULC cloud-degraded fusion prior art — 2026-10-28

- [x] Review the arXiv abstract and author-controlled project README; add CloudLULC as adjacent SAR–optical reliability-guided semantic-mapping prior art. Its output is LULC classification, so it does not test H1 or establish SAR restoration, SR, geometric mismatch tolerance, or restoration fallback. Matrix: 121 rows (78 confirmed historical; 43 unresolved); at least four historical entries remain unaccounted for against the 125-row lower bound. See D-2026-10-28-065 and the [audit](../../experiments/literature_audits/cloudlulc_cloud_degraded_semantic_fusion_20261028_v1/report.md).
- [ ] Reopen technical review only when the full manuscript or versioned author code is directly accessible. Check split/source-group independence, cloud protocol, geometric mismatch experiments, and paper-to-run lineage; do not use author-reported segmentation scores as restoration evidence.
- [ ] Keep the local source-group mapping/split prerequisite open and model selection paused. Preserve B1 `not_promoted` / `stop_at_b1`; do not enter B2/B3.

## Direction-aware SAR–optical matching metadata lead — 2026-10-28

- [x] Register the publisher-indexed title/authors/issue/date and DOI as metadata-only optical–SAR registration prior art. Direct article text was inaccessible; method, split, reference points, results, and code remain unverified. No local experiment or gate changed. See D-2026-10-28-064 and the [audit](../../experiments/literature_audits/direction_aware_sar_optical_matching_metadata_20261028_v1/report.md).
- [ ] Reopen technical review only if publisher full text or an author manuscript/versioned artifact becomes accessible. Check method, data/source groups, reference-point independence, controlled mismatch, code, and paper-to-run lineage. Do not infer a local registration threshold or use this matching paper as restoration evidence.

## Source-lineage metadata route — 2026-10-28

- [x] Recheck the advertised Drive search with three focused metadata-only queries. All calls failed before returning metadata with a backend transport error; the CUA inventory had no browser and the in-app browser was unavailable. This is an access failure, not a negative source search. See D-2026-10-28-055 and exposure group 507.
- [x] Re-probe current connector availability through one metadata-only Drive search and `list_drives`; both returned backend `Unknown tool`, and CUA exposed no apps or browsers. This confirms the source route remains unavailable without implying that no source record exists. See D-2026-10-28-060 and exposure group 512.
- [ ] Resume the source-record search only after the Drive backend or a browser surface is operational. Keep the local numeric-name 2k cohort unmapped and model selection paused; do not repeat the same calls while the backend is unavailable.

## Related-work matrix reconciliation — 2026-10-28

- [x] Restore Ma et al.'s 2026 flood-mapping preprint row using the contemporaneous D-2026-10-18-09 inclusion record. At that decision point the matrix had 118 rows: 78 historically confirmed and 40 with unresolved original membership. The shift-invariant objective is for segmentation labels; it does not establish optical-guided SAR image restoration. See D-2026-10-28-061 and the [row-restoration audit](../../experiments/literature_audits/flood_mapping_shift_invariant_row_restore_20261028_v1/report.md).
- [ ] Recover remaining original matrix titles only from a user-provided copy or a new authoritative inventory. Keep historical membership unresolved where direct inclusion evidence is absent; do not infer it from current citations.

## MTDN adjacent SAR multitask prior art — 2026-10-28

- [x] Verify MTDN against the publisher-indexed article text, Wuhan University author profile, and author-group repository README/code. Add it as adjacent SAR-only temporal despeckling/change-detection prior art; its change mask does not establish optical reliability, spatial SR, H1, or a matched local baseline. The matrix now has 119 rows: 78 historically confirmed and 41 with unresolved original membership. See D-2026-10-28-063 and the [audit](../../experiments/literature_audits/mtdn_adjacent_multitask_despeckling_20261028_v1/report.md).
- [ ] Keep local optical utility unresolved and B1 at not_promoted / stop_at_b1; do not enter B2/B3. Model-forward work remains CUDA-only after a fresh preflight in the approved GPU-visible profile.

## PLISA registration prior-art screen — 2026-10-28

- [x] Verify the publisher-indexed citation, method premise, and reported SEN1-2 split counts. PLISA's pseudo-label stage assumes already co-registered pairs; the accessible article text does not establish source-group separation or independent geodetic checkpoints. Add it as registration-only prior art, not as local residual evidence. See D-2026-10-28-058 and the [audit](../../experiments/literature_audits/plisa_optical_sar_registration_publisher_indexed_20261028_v1/report.md).
- [ ] Keep local fine-registration uncertainty open. Resume only with an authoritative source/product crosswalk or independently referenced local checkpoints; do not infer residuals from patch geometry or use them to alter B1/B2 gates.

## Artifact cleanup follow-up — 2026-10-28

- [x] Recheck output and experiment artifact paths, sizes, interrupted-run folders, and checkpoint-to-run manifest statuses. No `smoke`-named files or clearly disposable run payloads were identified in the reviewed roots; preserve the audit/config/command records and completed-run checkpoints. See D-2026-10-28-054 and the [cleanup audit](../../experiments/data_audits/artifact_cleanup_gpu_followup_20261028_v1/report.md).
- [x] Verify the newly surfaced Sentinel-1 joint SR/despeckling paper was already indexed and audited; do not add a duplicate. Record the path exposure in group 506.

## PolSAR/SinSAR diffusion SR source lead — 2026-10-28

- [x] Follow the newly surfaced DOI/Xplore pointer and Crossref-sourced ORCID search-index pointer. The IEEE and ORCID direct pages remained JavaScript-gated, the DOI route was inaccessible, and only secondary technical text surfaced; exclude it. See D-2026-10-28-053 and the [audit addendum](../../experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/report.md#addendum--source-route-recheck-2026-10-28).
- [ ] Reopen only if an official publisher/DOI record or author-controlled manuscript becomes directly readable. Keep the title, DOI, method, and result claims outside the formal matrix until then; do not retry the same routes without changed access.

## ICSP 2024 SAR diffusion SR metadata — 2026-10-28

- [x] Verify title, author list, and proceedings location from the IEEE-copyrighted ICSP proceedings table of contents, and bind the abstract to the matching IEEE Xplore DOI record. Keep method details at abstract level; the full paper is not available through the checked route. See D-2026-10-28-052 and the [audit](../../experiments/literature_audits/icsp2024_sar_diffusion_sr_metadata_20261028_v1/report.md).
- [ ] Reopen method/protocol review only if the publisher or author full text becomes accessible. Check degradation, source-group split, target and metric protocol, code, and paper-to-run lineage; do not infer efficacy from the abstract.

## B1F CUDA-only model entrypoints — 2026-10-28

- [x] Require explicit `--device cuda` and fail closed before data/output staging in B1F training, evaluation, and optical-pairing diagnostic entrypoints. Record the actual device in manifests; `py_compile` passed. Historical CPU runs and their manifests remain unchanged. See D-2026-10-28-050.
- [x] Audit all executable `MMCANet/src` training/inference entrypoints: they call the shared CUDA requirement or restrict `--device` to `cuda`; the shared contract refuses CPU fallback. This was a source-only check, with no launch or model forward. See D-2026-10-28-056.
- [ ] Before any separately authorized model-forward work, use the approved CUDA-visible profile, rerun the CUDA preflight, and pass `--device cuda`. Do not restart completed diagnostics or enter B2/B3 while the B1 gate is `not_promoted` / `stop_at_b1`.

## v4 output-head and target-range contract — 2026-10-28

- [x] Trace the v4 training and fixed-protocol target paths: canonical preprocessing clips targets to `[0,1]`, matching the default sigmoid head. Static code establishes range compatibility only; it does not establish bright-target preservation or quantify sigmoid saturation. See D-2026-10-28-057.
- [ ] Keep output-head selection unchanged unless an eligible prospective development split and CUDA-run evidence show a specific saturation problem. Do not use the contaminated development partitions or test data for that decision; B1 remains `not_promoted` / `stop_at_b1`.

## OSR-Net author and evidence-level correction — 2026-10-28

- [x] Correct the matrix's author field to the full nine-author roster exposed by the official MDPI index. Resolve the article's cited arXiv DOI to the separate MultiResSAR review, not an OSR-Net preprint; update the citation boundary. See D-2026-10-28-059 and the [audit](../../experiments/literature_audits/osr_net_publisher_indexed_text_update_20261028_v1/report.md).
- [x] Recheck the official search-index and direct publisher HTML/PDF routes on 2026-10-28. The index still surfaced excerpts from the same record, the HTML page returned HTTP 429, and the direct `/pdf` route was inaccessible through the reader; no new official full-text route appeared. See the access-route recheck in the [audit](../../experiments/literature_audits/osr_net_publisher_indexed_text_update_20261028_v1/report.md#access-route-recheck-2026-10-28).
- [ ] Reopen full-text protocol review only if the correct OSR-Net article/PDF or a versioned author artifact becomes directly accessible. Verify dataset subset, split, reference-point independence, and run/code lineage; do not treat the linked MultiResSAR review as an OSR-Net manuscript or interpret registration RMSE as a SAR-restoration mismatch threshold.

## RGSNet abstract-level adjacent prior art — 2026-10-28

- [x] Review the SSRN-indexed abstract and linked public GitHub repository. Record RGSNet as SAR–optical reliability-guided semantic segmentation prior art; the repository contains only a README and says code will be released upon acceptance. The accessible evidence does not establish restoration, SR/despeckling, geometric-mismatch robustness, or fallback. See D-2026-10-28-045 and the [audit](../../experiments/literature_audits/rgsnet_reliability_guided_sar_optical_segmentation_20261028_v1/report.md).
- [ ] Reopen only if the full paper or versioned implementation becomes publicly accessible. Keep the row at abstract-only evidence level and do not change B1 `not_promoted` / `stop_at_b1` or enter B2/B3.


## MambaR-ESRGAN official implementation route — 2026-10-28

- [x] Check the official University of Waterloo IVC project portal, the lab publication listing, and the accepted author PDF for a versioned implementation link. No author-controlled MambaR-ESRGAN code route surfaced; the portal's old `/code` endpoint was inaccessible, so code availability remains unverified rather than proven absent. See D-2026-10-28-044 and the [audit addendum](../../experiments/literature_audits/mambar_esrgan_fulltext_audit_20261025_v1/report.md#addendum--official-implementation-route-recheck-2026-10-28).
- [ ] Reopen only if a new official paper-page link, author-controlled repository, or versioned artifact appears. Do not use a third-party repository as the paper implementation or reproduction; this prior-art review does not change B1 `not_promoted` / `stop_at_b1` or authorize B2/B3.

## Bibliography-to-matrix crosswalk — 2026-10-28

- [x] Crosswalk both manuscript bibliographies against the 12-column matrix; restore the two rows already documented in D-2026-10-25-107 but missing from the current CSV, and add Hughes & Schmitt (2019) at abstract-only evidence level. The matrix is now 114 rows: 77 confirmed historical members and 37 unresolved additions. See D-2026-10-28-043 and the [audit](../../experiments/literature_audits/bibliography_matrix_crosswalk_20261028_v1/report.md).
- [ ] Recover the exact pre-corruption title/DOI list only if a user-provided copy or materially new authoritative source becomes available. Keep the matrix marked incomplete; do not infer historical membership from current manuscript citations.
- [x] Search user attachments, current project file names, and reachable Git history for a complete pre-corruption matrix copy. None was found; a similarly named matrix in the separate RadarDistill project was excluded. Continue to require a user-provided or authoritative source and do not infer absent rows. See exposure group 500.

## DPS1-SAR data-source screen — 2026-10-28

- [x] Review the official DataPort metadata and primary data-descriptor; add DPS1-SAR as a conditional SAR-only despeckling candidate. Its files are subscription-gated, and its source groups, target-stack membership, representation, and split remain unresolved. See D-2026-10-28-042 and the [audit](../../experiments/literature_audits/dps1_sar_source_screen_20261028_v1/report.md).
- [ ] Reopen only if authorized file-level metadata or source code becomes available. Verify source-group IDs, whether each input appears in its references, representation, registration, reuse terms, and prospective split suitability before considering any use. This does not authorize model selection or change the local source-provenance prerequisite.

## Literature-review/matrix crosswalk — 2026-10-28

- [x] Cross-check the reviewed literature against the 12-column matrix and add the four source-backed omissions: Capella SAR SR, 3MOS, SOMA-1M, and conditional SAR400. Historical membership remains unresolved; see D-2026-10-28-041 and the [audit](../../experiments/literature_audits/related_work_matrix_literature_review_crosswalk_20261028_v1/report.md).
- [x] Keep these adjacent records out of task-matched baseline and optical-utility claims; leave B1 `not_promoted` and B2/B3 gated.
- [x] Fill the three blank `code_data_link` cells for optical-guided SAR despeckling records: link the 2019 entry to the official GNLM repository already recorded in D-2026-10-18-07, and state the source-bounded no-link status for the 2015 and abstract-level Gaya 2024 entries. See D-2026-10-26-046.
- [x] Recheck the official GNLM GitHub search-index route. It identifies a public repository matching the paper and lists MATLAB entry points; its indexed README summarizes informational/nonprofit use. Direct repository/license retrieval and the paper-to-commit binding remain unverified. See the [recheck audit](../../experiments/literature_audits/gnlm_author_repository_status_recheck_20261028_v1/report.md).
- [ ] Inspect the exact MATLAB source and full license, and determine whether a repository version is bound to the reported paper run, only if direct official file access becomes available. Do not describe repository existence as an exact paper reproduction.

## Sim2Real SAR joint-restoration prior art — 2026-10-28

- [x] Verify the author arXiv full text and add Sim2Real as adjacent SAR-only joint despeckling/sidelobe-reduction prior art. Its effective reflectivity target and qualitative-only real-image sidelobe assessment limit the physical claim; it has no SR or optical input. Historical matrix membership remains unresolved. See D-2026-10-28-040 and the [audit](../../experiments/literature_audits/sim2real_joint_despeckling_sidelobes_20261028_v1/report.md).
- [ ] Keep it out of task-matched SAR-only JDSR baseline claims; do not use it to alter the B1 gate.

## ADAM prior-art boundary — 2026-10-28

- [x] Review the DLR-hosted published text and add ADAM to the related-work matrix as adjacent SAR joint despeckling/compression prior art. Its MERLIN→JPEG2000 comparison does not establish a joint-SR or optical-guidance result; see D-2026-10-28-036 and the [full-text audit](../../experiments/literature_audits/adam_joint_compression_despeckling_20261028_v1/report.md).
- [ ] Keep ADAM out of task-matched local JDSR baseline claims. Continue model selection only after the authoritative local source-group mapping/split prerequisite is met; B1 remains `not_promoted` and B2/B3 remain gated.

## Source-lineage connector status (2026-10-28)

- [x] Recheck the currently listed Google Drive metadata-only search with three focused queries for source groups, numeric-name mapping, and subset-generation records. The backend returned `Unknown tool` and no metadata; do not retry until it executes successfully. This is an access failure, not evidence that the source record is absent. See D-2026-10-28-037 (and the earlier single-query failure in D-2026-10-28-033).
- [x] Recheck after the search function appeared in the tool catalog: three metadata-only document queries again returned `Unknown tool`; the current browser inventory has no browser session. No Drive metadata or file content was returned. Keep the route paused until the backend executes successfully; this does not establish that the source record is absent. See exposure group 500.
- [ ] Resume local-subset source mapping only when an authoritative file-to-source-group record or a functioning connector route becomes available. Keep model selection paused in the meantime.

## Muzeau paper-to-code run lineage — 2026-10-28

- [x] Inspect all 14 reachable commits in the author-linked repository: seven config-bearing commits set 10 epochs, three earlier configs omit the epoch setting, and no versioned 100-epoch configuration was found. See D-2026-10-28-038 and the [history audit](../../experiments/literature_audits/muzeau_author_repository_config_history_20261028_v1/report.md).
- [ ] Reopen reproduction only if a run-bound config/checkpoint or a versioned author artifact links the paper's reported run to a specific source revision. Keep the paper-to-run mapping unresolved until then.

## HSFMamba reproducibility route (2026-10-28)

- [x] Recheck the paper-linked author GitHub index: its OGSSR entry exposes a data-share link but no toolkit/results or Algorithm implementation entry; the author catalog page was inaccessible. No data/code was fetched. Treat code availability as unverified outside the checked route; see D-2026-10-28-034 and the [full-text audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md).
- [ ] Reopen implementation search only if the authors add a versioned code link or a new author-controlled repository becomes directly discoverable. Do not use an unverified third-party repository as a reproducibility source.

## DSRKD protocol evidence (2026-10-28)

- [x] Recheck the matrix-linked ResearchGate page. It reports no full text and offers a request directly to the authors; no request was sent. This does not resolve the abstract-versus-related-work dataset description. See D-2026-10-28-035.
- [ ] Keep DSRKD's dataset, target construction, degradation, split, and teacher/student protocol unresolved. Reopen only if a new accessible primary manuscript or official implementation appears; do not contact authors without explicit user authorization.

## Canonical degradation operator verification (2026-10-28)

- [x] Revalidate normalization, FFT filtering, anisotropic blur, Gamma statistics, pre-speckle replay, and MMCANet/Cascade canonical-function bindings using synthetic-only unit checks: 14 operator tests plus one binding test and one replay test passed. No model forward pass, project image, test payload, metric file, or checkpoint was used. See D-2026-10-28-027.
- [ ] Keep representation-domain speckle labeled as controlled synthetic noise. Mathematical sampler tests do not establish calibrated SAR-domain statistics or validate the local physical observation model; obtain product/target evidence before making physical ENL or sensor-fidelity claims.

## CUDA-only reusable model interfaces (2026-10-28)

- [x] Require CUDA in the reusable Cascade/SFT interfaces and in `train_joint_jdsr.py`, `infer_joint_jdsr.py`, `train_swinir_joint.py`, `infer_swinir_joint.py`, `train_ogsrn_srun.py`, `train_ogsrn_sortn.py`, and `infer_ogsrn.py`. Device preflight now precedes run/output directory creation. Python compilation and bounded CPU-fallback scans passed; no training, inference, data, or gate was accessed. See D-2026-10-28-023/024.
- [x] Make reusable Cascade/SFT shell launchers and the SFT seed-matrix command pass `--device cuda` explicitly so saved commands record the required accelerator. Shell syntax and Python compilation passed; launchers were not executed. See D-2026-10-28-026.
- [x] Add and run a CUDA-only regression case for missing/zero optical with cross-attention enabled; the fallback matched the SAR backbone exactly on synthetic inputs. See D-2026-10-28-039.
- [x] Close the remaining model-forward CPU/auto routes in `scripts/` training, inference, validation, calibration, checkpoint-selection, benchmark, and matrix entrypoints. They now require `--device cuda` (or run the same preflight in the no-flag representation audit), check CUDA before data/output work, and fail closed. Python compilation passed; no run or smoke was started. See D-2026-10-28-062.
- [ ] For any future authorized model-forward task, use the approved CUDA-accessible profile, perform a fresh preflight, and pass `--device cuda`; do not rely on `auto` or CPU fallback. Preserve B1 `not_promoted` / `stop_at_b1`; B2/B3 remain gated.

## MHFNet publisher-indexed text recheck (2026-10-28)

- [x] Clarify that MHFNet's US-Loss handles detector box-label shifts at the training-example level; FAFM aligns features and DMP-Fusion reconciles detections. The publisher index does not expose exact data partitions or a controlled offset sweep, and direct open remains HTTP 403. Keep it as downstream detection prior art only; see D-2026-10-28-021 and the [audit](../../experiments/literature_audits/mhfnet_publisher_indexed_text_recheck_20261028_v1/report.md).

## Manuscript eligibility and historical pilot text exposure (2026-10-28)

- [x] Confirm the formal 2k paper directory is an archive and the July manuscript is an internal engineering pilot; neither is a current submission draft. The train/validation overlap and failed B1 promotion gate prevent using them as independent evidence. See D-2026-10-28-031.
- [x] Disclose group 473: historical test-metric prose surfaced in the pilot source during this review; no value was used and no image, payload, or metric file was opened. Current flags are `test_set_accessed=true`, `test_set_used=false`; cumulative flags remain `true/false`; clipped research-planning output in group 474 was excluded.
- [ ] Rebuild submission claims only after an eligible source-group split and a promotion-eligible result.
- [ ] Continue excluding the 100 historical mini-pilot identities from future model selection or tuning.

## OMA-SSR train/test protocol evidence (2026-10-28)

- [x] Recheck publisher Sections 2.4 and 3.1. The written protocol uses the whole test set in the training loss while separately listing train/test splits; record this as test contamination as written, without claiming implementation behavior. The 2026-10-28 exact DOI/correction recheck surfaced no separate correction item; actual implementation behavior remains unknown. See D-2026-10-28-017/051 and the [audit](../../experiments/literature_audits/oma_ssr_test_loss_protocol_recheck_20261028_v1/report.md).
- [ ] Keep OMA-SSR reported test metrics out of independent-evaluation evidence unless a primary correction, author-provided clarification, or versioned implementation resolves the data flow. Do not contact authors without explicit user authorization.

## Nullable metric consumers and CUDA execution (2026-10-28)

- [x] Handle undefined ratio-correlation values and coverage in Phase-A2, joint-vs-cascade, and formal aggregation consumers. Five synthetic regression tests and Python compilation passed; no project data or model run was used. See D-2026-10-28-001.
- [x] Close CPU/`auto` fallback in reusable JointJDSR/OGSRN/SwinIR train/inference CLIs; see D-2026-10-28-024. Protocol-specific evaluation/benchmark/matrix launchers remain outside that repair.
- [ ] Run future model-forward work only in a CUDA-accessible approved profile after a fresh preflight, passing `--device cuda` explicitly. Do not use CPU/`auto` defaults. B1 remains `not_promoted` and no B2/B3 work is authorized. See D-2026-10-28-022/024.
- [ ] Preserve B1 `not_promoted` / `stop_at_b1`; do not start B2/B3 without an authorized gate change.

## GPU profile preflight and artifact cleanup (2026-10-28)

- [x] Record group 456 for the current-state review and MediaTUM metadata-route audit. Historical result-bearing text was excluded; no local test payload or metric was accessed. See D-2026-10-28-006.
- [x] Verify the approved elevated AISAR profile: CUDA tensor execution succeeds on the RTX 3050 Laptop GPU and the joint-vs-cascade CLI offers only CUDA. The ordinary restricted shell remains unable to reach the driver. No model process was active, so no new experiment was started; the earlier CPU v2 diagnostic and CUDA v3 repeat are already recorded. See D-2026-10-28-003.
- [x] Remove six generated Python cache directories (six files, 92,505 bytes); the metadata-only scan found no additional disposable array/raster/temp/smoke payload. Keep manifests/configs/audits and cited high-value diagnostics. See the cleanup audit at ../../experiments/data_audits/artifact_cleanup_gpu_followup_20261028_v1/report.md.
- [x] Record group 455 for the bounded state/action and cleanup review. Historical research text was excluded from new claims; no local test image, metric file, or payload was opened. Current task flags are false/false; cumulative flags remain true/false. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [ ] For the next authorized model-forward task, use the approved CUDA profile, run a fresh preflight, and set --device cuda; keep B1 not_promoted and do not enter B2/B3.

## SEN1-2 source-group route recheck (2026-10-28)
- [x] Recheck the official TUM open-dataset listing and ISPRS publisher record. They support the aggregate SEN1-2 description but provide no file-level mapping for the local numeric subset; the direct MediaTUM page could not be fetched through this reader. Keep the existing requirement for an authorized file-to-source-group record; see exposure group 493.

- [x] Check the official GovData catalog alternate route and its resource/metadata links. The record is discoverable, but its linked metadata is unavailable through the current reader; the direct TUM route remains rate-limited. No local data or project test material was opened. See D-2026-10-28-002.
- [x] Retrieve the official MediaTUM record through OAI-PMH. It confirms collection-level SEN1-2 metadata but provides no per-sample source-group mapping. The separate item JSON export confirms CC BY 4.0; the OAI `restrictedAccess` tag's specific semantics remain unresolved. No data archive was downloaded. See D-2026-10-28-005/010 and the [metadata audit](../../experiments/data_audits/sen1_2_mediatum_oai_metadata_recheck_20261028_v1/report.md).
- [x] Try the unqueried OAI-PMH `ListSets` endpoint through the available web reader. The reader returned an inaccessible-URL error before retrieving XML; this is not evidence that the repository has no sets. Do not repeat through the same reader without a new access route; see D-2026-10-28-020.
- [x] Search bounded public Schmitt/TUM-related GitHub routes for a SEN1-2 source-file crosswalk. No author/TUM-controlled mapping surfaced; the distinct SEN12MS toolbox and a third-party task derivative do not map local numeric filenames and were not accepted as provenance. See D-2026-10-28-008.
- [x] Read the official MediaTUM item JSON export and use the published rsync route in `--list-only` mode. The item record reports CC BY 4.0 and exposes a checksum-index file; guest child/descendant listings are empty. The retained upstream hash audit reports zero byte-identical matches across 3,600 fixed train/validation inputs. No archive or file payload was downloaded. See D-2026-10-28-010.
- [x] Search project scripts, model code, and the `data/` tree for an explicit local numeric-filename-to-upstream path map or sidecar; none surfaced in this bounded text/code check. No image contents were opened. See D-2026-10-28-013.
- [ ] Obtain an authorized subset-generation record or explicit mapping from local numeric filenames to version-pinned upstream file paths/source groups before creating a new selection split. The author-maintained QXS-SAROPT README and primary paper expose no such crosswalk; do not repeat that route without a new artifact or endpoint. Keep the local numeric 2k subset unidentified and B1 `not_promoted` until then; byte-level matching alone has already failed for the audited train/validation inputs. See D-2026-10-28-025.
- [x] Recheck the newly advertised Google Drive metadata-search tool with three focused queries; each returned backend `Unknown tool` and no file metadata. Treat this as a connector failure, not a negative source search. Do not retry until the backend is executable. See D-2026-10-28-028.

## Research-text exposure disclosure (2026-10-28)

- [x] Record group 493 for the completed-run/device reconciliation, 222-link README check, and official SEN1-2 metadata-route review. Historical result-bearing text was excluded; no test payload or local value was used.
- [x] Record group 494 for the objective/state reread and RGSNet source audit. Historical test/result prose surfaced in continuity excerpts and was excluded; no local test payload, metric file, or result value was used. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [x] Record group 495 for the complete objective reread, required entry checks, bounded current-state/action/registry/decision review, matrix row-count check, and fresh process/CUDA check. Historical test/result prose surfaced and was excluded; no test payload, metric file, checkpoint, or result value was accessed or used. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [x] Record group 496 for the OSR-Net author/source recheck, the official publisher-page search-index review, local matrix crosswalk, and two failed official full-text retrieval routes. External publisher-reported registration performance was treated only as the article's own highlight. No local test payload, metric file, model, or result value was accessed or used. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [x] Record group 500 for the required objective reread, current repository/state review, metadata-only Drive availability recheck, matrix-copy search, and literature-matrix code-link cleanup. No local test payload, metric file, or result value was accessed or used. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [x] Record group 501 for the GNLM author-group repository search-index review and direct repository/license fetch failures. No code or data was downloaded, and no local test payload, metric file, or result value was accessed or used. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).

- [x] Record group 457 for the bounded current state/action/decision and experiment-registry review. Historical development/result text was excluded; no local test metric or payload was accessed. See D-2026-10-28-007.
- [x] Record group 459 for official MediaTUM item/rsync metadata checks, the bounded local mapping-script search, and the primary-source diffusion-route recheck. Historical local and secondary paper result text was excluded; no local test image/metric, archive, or checksum file was accessed or downloaded. See D-2026-10-28-012.
- [x] Record group 492 for the full research-objective reread, bounded state/action review, and clipped README/current-status inspection. Surfaced historical local development-result text was excluded; no test payload or metric file was opened and no result value was used. The README now includes the bounded MambaR-ESRGAN official-code-route finding. See the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).

- [x] Record groups 453–454 for the clipped nullable-metric/CUDA follow-up and official source-route review. Historical result-bearing prose was excluded; no local test image, identity, metric file, payload, or dataset archive was opened. Those continuations' flags were `test_set_accessed=true`, `test_set_used=false`. See D-2026-10-28-001, D-2026-10-28-002, and the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).

## Research-text exposure disclosure (2026-10-27)

- [x] Record group 452 for the clipped current-document and archived-paper review. Historical development/result prose and fixed-test workflow metadata are excluded; no local test metric value, image, metric file, or payload was opened or used. See D-2026-10-27-053 and the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).

## Related-work source map (2026-10-27)

- [x] Add an evidence-bounded trace for current registration, tie-point benchmark, SAR-only self-supervised despeckling, and reverse-task alignment/reliability sources already reviewed in the matrix. No paper or result claim was added. See D-2026-10-27-054 and the [claim–evidence map](claim_evidence_map.md#supplemental-2026-prior-art-trace-2026-10-27).
- [x] Add Lian et al. (2026) as publisher-indexed structure-confidence SAR–optical registration prior art. The direct article/PDF/XML reader routes were unavailable; do not promote the entry to full-text status or transfer its registration results to SAR restoration. See D-2026-10-28-030 and the [audit](../../experiments/literature_audits/registration_structure_confidence_phase_congruency_20261028_v1/report.md).
- [x] Recheck a focused author-copy route: the earlier multi-query output was clipped and excluded, OpenAlex was inaccessible, and an exact-title arXiv search returned no results. No usable author copy was obtained; keep the entry at publisher-indexed level. See D-2026-10-28-032.
- [ ] Reopen the article's full text only if a new accessible official PDF/XML or author-controlled copy becomes available. Check its source-group split, tie-point reference independence, controlled displacement/warp protocol, and code/run lineage; do not repeat the blocked routes without changed access state.
- [ ] Keep the broader local contribution-to-eligible-run map incomplete until admissible prospective evidence exists. Do not update historical performance/robustness claims or reuse the archived paper as a submission draft.

## Ratio-residual edge-correlation numerical validity (2026-10-27)

- [x] Mark Pearson correlation unavailable when float32 residual/gradient variation is below input precision; report per-metric undefined count and coverage. The generic evaluator's 26 synthetic known-case tests pass. This does not alter the separate B1 metric policy; see D-2026-10-27-051 and the [audit](../../experiments/data_audits/ratio_residual_edge_correlation_20261027_v1/report.md).
- [ ] Keep historical generic-evaluator correlations under their original evaluator lineage. If a future admissible comparison needs the corrected metric, generate a new version-bound result from authorized non-test inputs; do not rewrite old CSVs or access test outputs.

## README literature-inventory count reconciliation (2026-10-27)

- [x] Replace stale 100/76/24 and 102/76/26 snapshots with the current 103/77/26 matrix counts and corrected missing-inventory bounds. No matrix rows changed; see [D-2026-10-27-022](DECISION_LOG.md#d-2026-10-27-022--align-readme-literature-inventory-counts-with-current-matrix).

## Remaining literature-inventory snapshot reconciliation (2026-10-27)

- [x] Update three older action-list notes from the superseded 99/76/23 snapshot to the current 103/77/26 counts and corrected missing-inventory bounds; preserve their existing checkbox states. See [D-2026-10-27-023](DECISION_LOG.md#d-2026-10-27-023--reconcile-remaining-matrix-counts-in-next-actions).

## GPU-only model-forward execution (2026-10-27)

- [x] Make the reusable MMCANet and MMCANetV2 training/inference CLIs CUDA-only with an early availability check; Python syntax compilation passed.
- [x] Close the CPU fallback in eight standalone legacy MMCANet training/inference entrypoints through one shared CUDA device contract. Ten synthetic contract tests and `py_compile` passed; no model run occurred. See D-2026-10-27-052.
- [x] Compare the completed direct-fusion CPU/CUDA diagnostic outputs over 1,200 paired scalar rows with matching frozen-input and checkpoint hashes; differences are recorded descriptively without a tolerance or gate change. See D-2026-10-27-047 and the [comparison audit](../../experiments/diagnostics/b1_v2_cpu_cuda_output_reproducibility_20261027_v1/report.md).
- [ ] Run future model-forward diagnostics/training only after a fresh CUDA preflight in a GPU-accessible execution profile and pass `--device cuda` explicitly. The restricted follow-up shell has no NVIDIA device nodes or PyTorch; defer model work there rather than using a CPU or `auto` path. See D-2026-10-28-022.

## Failed patch-oracle result pruning (2026-10-27)

- [x] Delete the generated per-image and summary CSVs from failed v2/v3 patch-oracle attempts; keep failure manifests, commands, configs, preflight records, and the v3 report. See [cleanup audit](../../experiments/data_audits/artifact_cleanup_failed_t_patch_oracle_20261027_v1/report.md).

## Standalone draft render cleanup (2026-10-27)

- [x] Remove the rebuildable, noncanonical `IEEEtran/mmcanet_jdsr_draft.pdf` (223,671 bytes). Retain the distinct `.tex`, `.bib`, and IEEE class for provenance; no current-paper references or LaTeX build sidecars were found. See D-2026-10-27-007 and the [manuscript-lineage audit](manuscript_lineage_audit.md).

## Related-work matrix original-inventory recovery (2026-10-27)

- [x] Check the current Git index, reachable path history, and every unreachable blob for the exact 12-column CSV schema. The matrix is untracked, has no reachable path revision, and none of four blobs matched; see [D-2026-10-27-005](DECISION_LOG.md#d-2026-10-27-005--rule-out-current-git-objects-as-a-matrix-recovery-source) and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).
- [x] Reconcile pre-corruption inclusion decisions for GCOANet, Gaya et al.’s IGARSS 2024 paper, and the program-level IGARSS 2025 SAR-diffusion record. The 2025 row is limited to title/authors/session facts from the official program; the preserved 46-row prefix remains unchanged. See D-2026-10-27-016 and the recovery audit addendum.
- [x] Correct the inventory lower bound: the 119-row read predates six rows added by D-2026-10-25-46, establishing at least 125 pre-corruption rows. Exact final count remains unknown; the current 103-row matrix has 77 confirmed and 26 unresolved rows, leaving at least 22 original records absent from the current inventory.
- [ ] Recover the exact original title/DOI inventory from a user-provided copy or a materially new authoritative source. The Google Drive search tool resurfaced in the catalog, but three metadata-only calls returned backend `Unknown tool` and no files (D-2026-10-27-017). Do not infer membership from identifier candidates or retry Drive/Git routes unless their backend/source state changes. Keep the current matrix marked incomplete.



- [x] Add Zhang and Wang (2026) as a publisher-backed SAR–optical registration prior-art row; retain unresolved historical membership and do not transfer its results to local B2. See D-2026-10-27-008 and the existing audit. The current matrix is now 103 rows (77 confirmed; 26 unresolved); at least 22 original records are absent based on the 125-row lower bound.
- [x] Add two publisher-backed rows for SAG-CR and CGD-Net to the related-work matrix as reverse-task current evidence. Keep their optical-output boundary explicit; exact implementation lineage and detailed stress settings remain open. Historical matrix membership is unknown. See the [matrix completion audit](../../experiments/literature_audits/related_work_matrix_sagcr_cgdnet_completion_20261027_v1/report.md). See [D-2026-10-27-009](DECISION_LOG.md#d-2026-10-27-009--index-sag-cr-and-cgd-net-as-reverse-task-prior-art).

## Sentinel-1 IW SLC candidate route (2026-10-27)

- [x] Complete the asset-free rights/product/target review after the metadata screen. The review found no defensible product-bound self-supervision target yet; raw IW I/Q is not treated as independent MERLIN targets. See D-2026-10-26-147 and the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).
- [ ] Reopen product access or split work only when authorized sidecar credentials and a frozen deramping/calibration/geometry/registration/target protocol are available. Until then, do not access assets or pixels, create a split, or train/infer. B1 remains `not_promoted`; B2/B3 remain gated.

## Direct optical fusion-input bypass and GPU preference (2026-10-27)

- [x] Run the predeclared zero-training fusion-stem intervention in [v2](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v2/runs/attempt_001/report.md). Direct optical-HR/context channels were zeroed while both attention updates stayed active; prediction sensitivity was about 0.0243% relative RMS and nearly identical under matched/permuted optical. This remains interpretation-only and cannot support selection or promotion.
- [x] Preserve the v1 fixed-path preflight failure separately; it failed before model inference and its temporary data, duplicate copies, bytecode, and logs were removed. See the [failure manifest](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v1/runs/attempt_001/manifest.json) and [cleanup record](../../experiments/diagnostics/b1_v2_optical_direct_fusion_bypass_20261027_v1/cleanup.json).
- [ ] Keep optical-use causality open. These component bypasses are out-of-training-distribution, and the reused development cohort cannot select or promote models. B1 remains `not_promoted`; do not enter B2/B3.
- [x] Apply the GPU-only rule to the frozen v3 direct-fusion repeat: an approved elevated preflight identified `cuda:0`, and the run completed with test flags false/false. The earlier v2 CPU run remains historical and was not rerun. For future model-forward diagnostics and training, require a fresh GPU preflight and explicit `--device cuda`; if unavailable, defer compute without CPU/`auto` fallback.

## Approved run-interface and cross-attention diagnostic (2026-10-27)

- [x] Confirm the approved residual/high-pass training and validation-inference interfaces remain implemented; no duplicate source edit or test rerun was needed. See D-2026-10-27-001 and the earlier focused verification records.
- [x] Preserve the interrupted v1 launch as an incomplete record and remove its task-owned temporary staging/logs.
- [x] Repair the diagnostic output-directory interface in a new frozen v2 audit, run the predeclared CPU intervention, verify 1,200 finite scalar rows, 200 identities, three seeds, 15 source folders, hashes, and false/false test flags, then remove recreated temporary inputs and low-value logs. See D-2026-10-27-002 and the [report](../../experiments/diagnostics/b1_v2_optical_cross_attention_ablation_20261027_v1/runs/attempt_001/report.md).
- [ ] Keep causal optical-use attribution open. The result is interpretation-only and cannot be used for selection/promotion on the overlapping development validation; B1 remains `not_promoted`, and B2/B3 remain gated.

## Historical pilot test-text exposure (2026-10-26)

- [x] Disclose the clipped paper-source search that surfaced historical test-subset metric text; exclude all surfaced/unseen result text and record task flags as accessed/used = true/false. See D-2026-10-26-138 and the [access audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).
- [x] Stop any future selection or tuning on the pilot's 100 test identities. Keep the recorded B1 v2 gate unchanged; make no source-overlap claim for the pilot subset.

## B1 v2 optical-pairing sensitivity (2026-10-26)

- [x] Run the zero-training matched/permuted/missing optical diagnostic against the SAR-only checkpoint on fixed validation. It found no stable metric-level advantage for correctly paired optical; B1 remains `not_promoted`. See D-2026-10-26-130 and the [report](../../experiments/diagnostics/b1_v2_optical_pairing_sensitivity_20261026_v1/report.md).
- [ ] Continue source/product provenance and precise local registration evidence; the current cohort remains one fall ROI-season and fine co-registration is unverified.
- [x] Compute streaming output-difference norms for matched/permuted/missing conditions and the SAR-only checkpoint. Matched/permuted mean relative RMS was 0.0105%; matched/missing and matched/SAR-only were about 0.129%. Keep this interpretation-only and do not use it to promote B1 or open B2/B3. See [D-2026-10-26-134](DECISION_LOG.md#d-2026-10-26-134--record-prediction-difference-norms-and-task-artifact-cleanup) and the [report](../../experiments/diagnostics/b1_v2_prediction_difference_norms_20261026_v1/report.md).
- [x] Inspect the three EMA C2 checkpoints' residual and reliability output projections. The final residual projection remains small after zero initialization; this is compatible with the weak paired-output sensitivity but does not establish why it occurs. No activation maps or new inference were run. See [D-2026-10-26-142](DECISION_LOG.md#d-2026-10-26-142--audit-optical-candidate-residual-head-parameters).
- [x] Run a predeclared interpretation-only activation audit for matched versus cross-group-permuted optical. Gate mean was about 0.467 in both conditions, while gated contribution/base RMS was about 0.129%; this argues against a globally near-zero gate as the sole explanation for low output sensitivity, but does not identify cause or prove no optical utility. No quality metrics or target values were used; B1/B2/B3 status is unchanged. See [D-2026-10-26-154](DECISION_LOG.md#d-2026-10-26-154--record-interpretation-only-optical-gate-activation-audit) and the [report](../../experiments/diagnostics/b1_v2_optical_gate_activation_20261026_v1/report.md).
- [ ] Keep causal optical-use attribution open: the cross-attention bypass audit found a tiny, nearly pairing-invariant prediction change while direct optical fusion remained active. It does not isolate residual-head magnitude, high-pass filtering, or pseudo-target effects, and is not a quality or robustness test. Do not repeat the activation audit or use the overlapping validation cohort for selection; any further mechanism analysis needs a distinct, predeclared question. Keep B1 `not_promoted` and B2/B3 gated.
- [x] Remove the task-owned validation staging copy and generated Python caches after recording the summary. No registered run, checkpoint, configuration, source data, or result summary was deleted; see the [cleanup record](../../experiments/artifact_cleanup_after_norms_20261026_v1.json).
- [ ] Do not inspect test images or read test metrics.

## Validation inference split interface (2026-10-26)

- [x] Add explicit `--split validation` support to the MMCANetV2 inference runner while preserving the default `test` behavior. Validation outputs are marked `debug_only`, and the manifest records split-specific test access/use flags. No run or data access occurred; see [D-2026-10-26-128](DECISION_LOG.md#d-2026-10-26-128--add-explicit-validation-split-to-the-mmcanetv2-inference-runner).

## Artifact cleanup and interface approval follow-up (2026-10-26)

- [x] Recheck outside source-data and fixed-test trees for disposable caches, bytecode, temporary files, replayable arrays, and smoke/scratch outputs. None were found; keep the cited cleanup audit, required configs, partial matrix-recovery snapshot, and registered scientific records. The user-approved validation-split and residual-filter run interfaces were already present under D-2026-10-26-128 and D-2026-10-23-46, so no duplicate code edit or run was needed. See D-2026-10-26-151.

## SAR object-image SR publication metadata screen (2026-10-26)

- [x] Verify the official author-lab citation and DOI route; add Cho et al. as a metadata-level current-evidence matrix row. The accessible publisher page is blocked, so method, input modalities, data, degradation, targets, split, metrics, and code remain unverified. Historical matrix membership remains unknown. See D-2026-10-26-153 and the [audit](../../experiments/literature_audits/sar_object_sr_multires_metadata_20261026_v1/report.md).
- [ ] Reopen the technical review only if an accessible author manuscript or publisher full text becomes available; do not use secondary summaries to fill protocol fields.

## PNG container metadata lineage audit (2026-10-26)

- [x] Scan `data/sar` and `data/opt` PNG container metadata without reading pixel payloads. The files contain no embedded text or ancillary source metadata; numeric basename overlap does not identify source groups. See the [audit report](../../experiments/data_audits/png_header_lineage_20261026_v1/report.md).
- [ ] Continue source/product provenance and precise co-registration work using direct release, acquisition, or sidecar evidence. Do not infer a dataset identity from filenames or patch overlap, and keep B2/B3 gated.

## Published dataset lineage crosswalk (2026-10-26)

- [x] Compare the canonical SEN1-2 and QXS-SAROPT tiling descriptions with local overlap evidence. The overlap pattern is a compatibility clue only; dataset identity remains unresolved. See the [crosswalk report](../../experiments/data_audits/dataset_lineage_source_paper_crosswalk_20261026_v1/report.md).
- [x] Review the author-hosted SEN1-2 full manuscript for source-product and preprocessing details; record these as collection-level facts only, without attributing them to local PNGs. See D-2026-10-26-136.
- [x] Inspect the pinned SEN12MS geospatial overview utility and its unresolved location-plot issue. It supplies no cross-release or local numeric-filename mapping, and its generated map is not verified ground truth; see the [source audit](../../experiments/literature_audits/sen12ms_geospatial_utility_route_20261026_v1/report.md).
- [x] Review the primary SEN1-2 paper's public scene-generation description and retry its TUM landing page once; the article gives no `s1_i`-keyed coordinate/product/date crosswalk, and TUM returned HTTP 429. This does not establish what metadata may be present inside the archive; keep the mapping unresolved.
- [ ] Seek a direct local-file to source archive/scene/acquisition mapping and precise registration evidence. Do not use patch geometry or numeric names as source identity.

## SMAF-Net optical–SAR registration prior-art screen (2026-10-26)

- [x] Add the publisher-abstract record and keep it distinct from SAR restoration; historical matrix membership remains unknown. See [D-2026-10-26-059](DECISION_LOG.md#d-2026-10-26-059--add-smaf-net-as-abstract-level-optical-sar-registration-prior-art).
- [x] Review the corresponding-author-uploaded full text for architecture, training, dataset families, metric definitions, ablations, and geometric stress tests. Record the unclear QXS split/correct-match reference and the absence of local-warp/restoration evaluation; do not use its results as local evidence. See [D-2026-10-26-071](DECISION_LOG.md#d-2026-10-26-071--review-smaf-net-full-text-as-registration-only-prior-art) and the [audit](../../experiments/literature_audits/smafnet_optical_sar_registration_abstract_screen_20261026_v1/report.md).
- [ ] Reopen only if a public author-controlled code route or a file-level dataset split/reference-point protocol becomes available. Keep SMAF-Net as registration-only context; do not infer local cohort lineage or use its reported pixel error as a B2 tolerance.


## Source-aligned B1 input-record continuity (2026-10-26)

- [x] Rebind the retained source/protocol manifests to the completed run's recorded hashes and confirm the train/validation candidate paths remain available. This verifies path availability only; no image payload or test path was opened, and no input tensor was rebuilt. See [D-2026-10-26-060](DECISION_LOG.md#d-2026-10-26-060--bind-the-retained-b1-source-manifest-to-its-run-record).
- [x] Recheck the raw verifier-output digests: expected hashes differ from both retained train/validation `verification.json` files, while each split's `config.json`, `files.csv`, `manifest.json`, and `split.txt` hashes match. The recorded original verifier outputs were temporary and removed; see the [digest recheck](../../experiments/data_audits/b1_v2_verification_record_hash_recheck_20261026_v1/report.md).
- [ ] Keep the raw verifier-record binding unresolved. The 2026-10-27 trace confirms the bound raw `/tmp/.../verify_val.json` was pruned; the retained validation `verification.json` has a different whole-file digest even though its summary and all 1,400 payload hashes agree with the bound `files.csv`. A final-newline-only reconstruction does not match. Do not rewrite historical records or regenerate the removed output; revisit only if the original bytes or an exact retained source become available. Future runs must retain the exact verifier output whose digest is bound. See the [lineage note](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_revalidation_inputs_20261021_v1/README.md#verifier-output-digest-lineage-follow-up--2026-10-27).
- [ ] Keep source-release/version mapping, parent-scene/acquisition independence, and fine pixel registration unresolved until authoritative metadata is available. Do not rerun B1 or enter B2/B3.
- [x] Recheck the official QXS-SAROPT repository, download survey, and author issues #3, #4, and #13. No file-level map or alternate public mirror is exposed; this does not attribute the local cache to QXS. The author states the raw GF-3 SAR is third-party copyrighted and only authorized processed/cropped samples could be released. No form was submitted. See [D-2026-10-26-126](DECISION_LOG.md#d-2026-10-26-126--recheck-official-qxs-saropt-author-source-routes) and the [focused audit](../../experiments/data_audits/qxs_author_issue_lineage_recheck_20261026_v1/report.md).

## Related-work matrix recovery (priority; 2026-10-26)

- [x] Run a strict DOI locator crosswalk: 71 of 103 unique review DOI mentions match current matrix identifiers/primary links; 32 are unmatched locator leads, not a count of missing original rows. No row was added from the count. See D-2026-10-26-088 and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).
- [x] Add SARGANet as current SAR-only ×2 SR prior art from the full Springer article. Record its simulated target and unspecified group split, and exclude its test-informed loss-weight results from local evidence. Historical membership remains unknown. See D-2026-10-26-097 and the [source audit](../../experiments/literature_audits/sarganet_publisher_fulltext_screen_20261026_v1/report.md).
- [x] Add Hu et al.’s DB-CR as a source-backed adjacent SAR-assisted optical cloud-removal diffusion row; preserve its reverse-task boundary and unknown historical membership. See D-2026-10-26-092 and the [matrix recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).
- [x] Screen unmatched DOI `10.1109/LGRS.2026.3742531` against the official IEEE abstract; classify it as SAR-to-optical reverse translation, not direct SAR restoration, and do not count it as a reconstructed matrix row. See D-2026-10-26-091.

- [x] Check unreachable objects in the current Git database for a recoverable original matrix; no unreachable commit or readable matrix blob was present. This closes only the current checkout route. See [D-2026-10-26-078](DECISION_LOG.md#d-2026-10-26-078--close-the-unreachable-git-object-route-without-changing-the-matrix).
- [x] Check the available local Zotero API, connected Drive search, official SEN1-2 TUM record/PDF, and bounded local directory names for the original row inventory or a local data-source crosswalk. No inventory or authoritative crosswalk was recovered; do not repeat these routes without a new source. See [D-2026-10-26-135](DECISION_LOG.md#d-2026-10-26-135--record-unavailable-original-matrix-and-data-lineage-routes) and the [route audit](../../experiments/data_audits/original_matrix_and_source_routes_20261026_v1/report.md).
- [x] Check current matrix structure and resolve its two blank identifier cells with clearly labeled source locators/statuses. The matrix is 97×12 with unique titles and populated IDs; this does not restore the original inventory or confirm historical membership. See [D-2026-10-26-144](DECISION_LOG.md#d-2026-10-26-144--record-two-source-locators-without-inventing-dois) and the [identifier audit](../../experiments/literature_audits/related_work_matrix_identifier_completion_20261026_v1/report.md).
- [x] Supersede this duplicate recovery item with the current 2026-10-27 action at the top of this file. The exact row-level inventory remains unresolved; current evidence is 103 rows (77 confirmed historical members, 26 unresolved additions), against a pre-corruption lower bound of 125, leaving at least 22 entries unaccounted for; if the final count was 125, the gap is 22–48. Do not infer historical membership from identifiers or current-evidence additions.
- [x] Add Mamba-SCR as an adjacent SAR-assisted optical cloud-removal row from the existing publisher-text audit. Keep historical matrix membership unresolved and do not use its performance text as local evidence. See D-2026-10-26-107.
- [x] Add Muzeau et al. (EUSAR 2024) from the author-hosted manuscript as current SAR-only integrated SR/despeckling-supervision prior art. Keep historical membership unresolved; the VDE proceedings route is currently inaccessible. See D-2026-10-26-110.
- [x] Complete the R-DDPM full-text degradation/evaluation screen. Its row already belongs to the preserved 46-row prefix; keep that row byte-for-byte unchanged and record unresolved target/split/code details in a companion audit. See D-2026-10-26-112.
- [x] Add Ye et al. (2021) as current measured Sentinel-1/2 registration evidence; keep the held-out point checks distinct from surveyed ground truth and do not transfer residuals to local B2 thresholds. Original matrix membership remains unknown. See D-2026-10-26-087 and the [primary-source audit](../../experiments/literature_audits/sentinel12_measured_registration_checkpoints_20261025_v1/report.md).
- [x] Add OSdataset as a conditional optical–SAR registration/dataset source from the IEEE indexed abstract and author repository; keep its display-domain, source-group, date, registration, and rights limits explicit. Do not identify it with the local 2k cohort. See D-2026-10-26-068.
- [x] Add MuChaPro as abstract-level InSAR phase/coherence restoration prior art. Keep its phase/coherence task distinct from optical-guided intensity-SAR joint restoration; do not transfer its results to the local cohort. See D-2026-10-26-066.
- [x] Record Dong et al. as abstract-level complex-valued SAR SR prior art with its institutional-index/citation-chain source limitations; do not transfer its reported metrics to the local cohort. See D-2026-10-26-065.
- [x] Add Attention OffsetNet as abstract-level optical–SAR registration prior art; retain its protocol/code and historical-membership boundaries and do not treat its abstract residual values as local alignment evidence. See D-2026-10-26-064.
- [x] Add SDS-SAR as current abstract-level SAR-only self-supervised despeckling prior art from the existing publisher/author source audit; do not count it as recovered historical membership or a task-matched joint-restoration baseline. See D-2026-10-26-062.
- [x] Preserve the 46 rows present before reconstruction byte-for-byte as the prefix of the current matrix. The last confirmed pre-corruption inventory had at least 73 more rows; it was not complete.
- [x] Reconstruct the seventeen rows with pre-corruption membership evidence (six foundational additions, SFT, MambaR-ESRGAN, two MLOG records, Lan et al., RQ-Net, and five diffusion/consistency records). See the [reconstruction rows](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/confirmed_reconstructed_rows.csv).
- [x] Append those 17 confirmed rows to the current matrix after checking the 12-column schema, normalized title/DOI disjointness, and exact preservation of the existing 46-row byte prefix; see [D-2026-10-26-041](DECISION_LOG.md#d-2026-10-26-041--append-confirmed-reconstructions-while-keeping-the-matrix-incomplete).
- [x] Add MD-Glow as a current source-backed SAR-only multi-task prior-art row; original matrix membership remains unknown and the DOI remains unverified. The official CVF record resolves its pages to 536–543; do not count it toward reconstruction. See D-2026-10-26-042/090 and the [audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md).
- [x] Resolve MD-Glow pagination as 536–543 from the official CVF proceedings record; keep the DOI unverified because the official BibTeX omits it and the secondary listing conflicts on pages. See D-2026-10-26-090.
- [ ] Revisit only the MD-Glow DOI if an official IEEE or DOI-registry record directly binds it to this article; do not promote the secondary DOI listing. The 2026-10-27 direct DOI-resolver and Crossref single-work opens were inaccessible, and the exact IEEE-domain query returned no record; do not repeat these routes without a new official source.
- [x] Add WMoE-Net as abstract-level alignment/reliability fusion prior art for oriented ship detection; keep it distinct from SAR restoration and do not count it toward original-matrix recovery. See [D-2026-10-26-043](DECISION_LOG.md#d-2026-10-26-043--add-wmoe-net-as-abstract-level-adjacent-prior-art) and the [source screen](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md).
- [x] Screen the title-matching WMoE-Net GitHub repository at README/metadata level. Keep author/publisher linkage unverified, do not treat it as an official implementation, and do not adopt its linked data package. See [D-2026-10-26-044](DECISION_LOG.md#d-2026-10-26-044--keep-the-wmoe-net-repository-at-candidate-code-status).
- [x] Add GCOANet as abstract-level optical-assisted SAR despeckling prior art; keep it distinct from joint SR, leave full protocol/mismatch robustness unresolved, and do not count it toward original-matrix recovery. See [D-2026-10-26-045](DECISION_LOG.md#d-2026-10-26-045--add-gcoanet-as-abstract-level-optical-assisted-despeckling-prior-art) and the [source audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md).
- [x] Keep the 2015 Optical-Driven Nonlocal filter and 2019 Guided patchwise nonlocal SAR despeckling as historically confirmed rows based on D-2026-10-18-07; Gaya et al. (2024) remains a current evidence addition with unresolved original membership. See D-2026-10-26-046 and D-2026-10-26-051.
- [x] Add Meraoumia et al. (2023) as SAR-only multitemporal despeckling prior art; keep it distinct from optical-guided joint SR and do not bind its TerraSAR-X stacks to SEN12TS. See [D-2026-10-26-047](DECISION_LOG.md#d-2026-10-26-047--record-multitemporal-sar-despeckling-without-binding-it-to-sen12ts) and the [source screen](../../experiments/literature_audits/multitemporal_merlin_tgrs2023_abstract_screen_20261026_v1/report.md).
- [x] Revalidate the captured 46-row prefix byte-for-byte before further recovery. One HSFMamba row differed in two description fields and was restored from the preserved partial snapshot; its later source-version caveat remains in the literature review/integration audit. See [D-2026-10-26-050](DECISION_LOG.md#d-2026-10-26-050--restore-the-captured-46-row-matrix-prefix-byte-for-byte).
- [x] Recover four historical rows explicitly named in D-2026-10-18-02 and the contemporaneous diffusion-update action: Kuang et al. RSDM/LSIR, SMSDNet, GUSO/FHReg, and Onat quality-aware SAR despeckling fusion. Their sources and membership evidence are recorded in the [confirmed-row list](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/confirmed_reconstructed_rows.csv) and D-2026-10-26-049; do not use their published scores as local evidence.
- [x] Recover seven more historical entries explicitly directed to the matrix in D-2026-10-18-07, D-2026-10-19-03/08/27, D-2026-10-22-06, and D-2026-10-23-24. The MONTI row in D-2026-10-19-32 was already present in the 46-row prefix. The confirmed ledger now has 28 rows; see D-2026-10-26-051.
- [x] Add Ebel et al. (2021) as a current SEN12MS-CR dataset/source row. The TUM author page and publication record verify the dataset paper and published metadata, while the 169/175 ROI discrepancy, release-specific split, local numeric-2k provenance, and historical matrix membership remain unresolved. Keep the dataset conditional and distinct from SAR restoration. See [D-2026-10-26-048](DECISION_LOG.md#d-2026-10-26-048--verify-sen12ms-cr-paper-identity-and-keep-the-source-conditional) and the [source identity follow-up](../../experiments/literature_audits/sen12mscr_author_archive_route_20261026_v1/report.md#source-identity-and-dataset-statistics-follow-up-2026-10-26).
- [x] Cross-check dated matrix-inclusion decisions through D-2026-10-23-24 and run a bounded phrase sweep through D-2026-10-25-45. No additional absent historical row was found in that match set; this does not prove completeness. Seven current evidence additions remain unresolved. See D-2026-10-26-051/052.
- [x] Preserve the search boundary: the sweep used explicit matrix-reference phrases only; it cannot establish the missing rows outside those matches. Keep the original 119-row list marked unavailable and retain the 39–45-row gap. See D-2026-10-26-052.
- [x] Search tracked Git history and project filenames for the original 119-row matrix; no tracked version was found, and the accessible `/tmp` partial snapshot contains only the 46-row prefix. Continue reconstruction from source-specific evidence and leave the unresolved inventory gap visible.
- [x] Extend the filename-only original-matrix lookup across `/home/shy/AIR` and Codex attachments; no additional JDSR original/backup was found. Do not reuse other projects’ matrices. See D-2026-10-26-075 and the [recovery audit](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/report.md).
- [ ] Compare recovered title/DOI coverage with the pre-corruption row-level list if it becomes available. The current 104-row matrix has 77 historically confirmed rows and 27 unresolved additions; the known pre-corruption inventory lower bound is 125, leaving at least 21 entries unaccounted for; if the final count was 125, the gap is 21–48, and the exact count remains unknown. The [candidate identifier index](../../experiments/data_audits/related_work_matrix_recovery_20261026_v1/recovery_candidate_identifiers.csv) remains a locator only.

- [x] Add Liao et al. (2025) as current adjacent registration-validation evidence after checking the publisher DOI record; keep historical membership unverified and do not treat the paper as local alignment evidence. See D-2026-10-26-053.
- [x] Add MambaDIP as current SAR-only despeckling/contrast-enhancement prior art. The accepted author manuscript treats spatial SR as future work and includes no optical input; do not count it as historical matrix recovery or a task-matched joint baseline. See [D-2026-10-26-055](DECISION_LOG.md#d-2026-10-26-055--add-mambadip-as-sar-only-despeckling-prior-art) and the [source screen](../../experiments/literature_audits/mambadip_sar_only_prior_art_screen_20261026_v1/report.md).
- [x] Fill the missing SwinIR and Transformer arXiv identifiers from arXiv and official proceedings records; no other DOI/ID cells remain blank. See [D-2026-10-26-056](DECISION_LOG.md#d-2026-10-26-056--fill-two-missing-arxiv-identifiers-in-the-matrix) and the [identifier audit](../../experiments/literature_audits/related_work_matrix_identifier_completion_20261026_v1/report.md).
- [x] Check the available decision log before 2026-10-16 and classify the recovery candidate index: no older decision sections exist, and its sole historical-membership candidate (SMSDNet) is already restored. Keep the remaining original-list gap visible; see [D-2026-10-26-057](DECISION_LOG.md#d-2026-10-26-057--keep-historical-matrix-recovery-bounded-by-direct-membership-evidence) and the [boundary audit](../../experiments/data_audits/related_work_historical_membership_boundary_20261026_v1/report.md).



## New direct joint-SAR diffusion literature lead (2026-10-26)

- [x] Record the newly surfaced “Frequency-enhanced diffusion prior network for joint SAR image despeckling and super-resolution” as an unverified lead only. The visible listing is secondary; IOP/DOI routes and author-manuscript searches yielded no primary full text. No method or numerical claim was used and no matrix row was added. See D-2026-10-26-077 and the [source screen](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).
- [x] Recheck the frequency-enhanced diffusion paper's bibliographic record on 2026-10-28. Crossref API HTTP 200 confirms the DOI/title/journal/date metadata only; IOP full text remains inaccessible and secondary technical claims remain excluded. See D-2026-10-28-011/014 and the [source screen](../../experiments/literature_audits/frequency_enhanced_joint_sar_diffusion_lead_20261026_v1/report.md).
- [x] Follow Crossref's official IOP HTML/PDF links with metadata-only HEAD requests and search ORCID/Xijing/arXiv/GitHub for an author copy. IOP returned bot-manager challenge HTML; no manuscript or author-controlled code surfaced. No challenge was bypassed and no technical text was retrieved. See D-2026-10-28-015.
- [x] Check OpenAlex for a legal author-copy route. Its metadata lists only the IOP PDF and no repository full text, with a CC BY-NC-ND label that is not verified against the article page. Crossref's license URI redirects to IOP's general copyright/permissions page; keep reuse terms unresolved. See D-2026-10-28-016.
- [ ] Reopen only when the full publisher article, an author manuscript, or versioned author implementation is directly accessible; verify task, degradation, split, targets, metrics, and code linkage before classifying technical prior art.

## IGARSS 2025 physics-inspired SAR diffusion source route (2026-10-27)

- [x] Recheck the official paper/session pages and primary DOI routes. The program metadata is confirmed; the displayed manuscript control is not actionable in the current reader, and the DOI candidate remains secondary/unverified. Do not update the matrix or bibliography from the secondary lead. See D-2026-10-27-021 and the [route audit](../../experiments/literature_audits/igarss25_physinspired_sar_diffusion_route_recheck_20261027_v1/report.md).
- [ ] Reopen only when an accessible IEEE publisher record or manuscript becomes available; review the full method and protocol before using it as technical prior art.

## Compute preflight (2026-10-26)

- [x] Check all installed Conda environments. `nvidia-smi` cannot reach the driver; CUDA is unavailable in `AISAR` (PyTorch 2.8.0+cu128) and `pytorch-img2img` (2.4.0+cu121); `ldm` torch import fails on `libtorch_cpu.so`; base Python has no torch. No experiment or data access occurred. See D-2026-10-26-073.
- [ ] Recheck the GPU/driver before any future training. Do not repeat B1 or enter B2/B3 while their existing promotion/source-provenance gates remain closed.

## OGSRN official PDF route status (2026-10-26)

- [x] Recheck the SciOpen publisher page and official ScienceDirect page/PDF routes. SciOpen exposes Open Access status, the CC BY-NC-ND 4.0 license, metadata, and abstract but no extracted PDF link/body; direct ScienceDirect returned 403 and the guessed `/pdfft` route returned cache miss. See [D-2026-10-26-023](DECISION_LOG.md#d-2026-10-26-023--keep-ogsrn-at-publisher-abstract-evidence-after-pdf-route-recheck).
- [ ] Keep the article protocol unresolved: SciOpen text says a PDF-viewer button opens the paper, but the available reader exposes no actionable URL/body and no interactive browser surface was available. Reopen when an actual publisher PDF/body or author-controlled full manuscript can be retrieved; do not infer protocol from the abstract or related patent. See [D-2026-10-26-029](DECISION_LOG.md#d-2026-10-26-029--keep-ogsrn-at-publisher-abstract-evidence).

## OGSRN author-group patent route (2026-10-26)

- [x] Inspect Shenzhen University patent CN113421190B and compare its inventors and technical description with the CJA OGSRN publisher record. It describes bicubic-upsampled/channel-expanded LR SAR at the test-time optical branch, but exact patent-paper implementation equivalence remains unverified. See [D-2026-10-26-022](DECISION_LOG.md#d-2026-10-26-022--treat-cn113421190b-as-related-ogsrn-patent-evidence) and the [audit addendum](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md#addendum--2026-10-26-author-group-patent-route).
- [ ] Continue seeking an author/publisher-linked full article or explicit patent-to-article version mapping. Do not use the patent as a matched baseline or as a substitute for the article's dataset, split, degradation, metrics, or code.


## SMSDNet full-text route recheck (2026-10-26)

- [x] Recheck the official IEEE indexed abstract and metadata and try the direct article/PDF routes. The article page returned HTTP 418 and the PDF route was inaccessible; the targeted author/institution search produced no verified manuscript. Keep this at abstract-level; do not treat its performance claims as evidence for optical-guided joint restoration. See D-2026-10-26-021 and the [route audit](../../experiments/literature_audits/smsdnet_publisher_fulltext_route_20261026_v1/report.md).
- [ ] Review its full method and protocol only if a new accessible IEEE or author-controlled full-text route appears.

## SEN12MS-CR author archive route (2026-10-26)

- [x] Search the author GitHub namespace for a current download route. The author-code script links season-level S1/S2 archives on TUM `dataserv`; no train-only ROI manifest or version crosswalk was found. See [D-2026-10-26-010](DECISION_LOG.md#d-2026-10-26-010--keep-sen12ms-cr-conditional-after-finding-season-level-author-links) and the [audit](../../experiments/literature_audits/sen12mscr_author_archive_route_20261026_v1/report.md).
- [ ] Do not fetch season-level bundles. Reopen only when an authoritative, version-pinned train-only ROI/source manifest or a selectively accessible train-only archive becomes available; preserve the test boundary and local B1 gate.


## SEN12TS temporal source-group route (2026-10-26)

- [x] Screen repeated public observations at the frozen 2,000 centers using metadata only. The listing has 2,716 rows and 350 centers with repeated S1 dates under the same track/lookpass proxy; the old representative split is not valid for the expanded cohort.
- [x] Apply the predeclared 30-component rule to expanded temporal metadata. Only four proxy components were found, so no split was created.
- [x] Record that the bounded header-tag screen was denied by the execution environment before receiving bytes; it provides no tag evidence and was not retried.
- [x] Use the official tile-center naming contract to freeze a separate geographic tile-group split: 1,600 train and 400 development groups, all 2,716 observations assigned, no tile crossover. See [D-2026-10-26-001](DECISION_LOG.md#d-2026-10-26-001--accept-the-documented-sen12ts-tile-group-for-exploratory-splitting) and the [audit](../../experiments/data_audits/sen12ts_official_tile_group_split_20261025_v1/report.md).
- [x] Derive the four S1 stack-date proxies per row from the official 12-day spacing and quantify within-tile window overlap. The metadata screen finds repeated/overlapping date proxies and 209 acquisition proxies crossing the tile split; no pseudo-reference independence claim is supported. See [D-2026-10-26-002](DECISION_LOG.md#d-2026-10-26-002--treat-sen12ts-stack-windows-as-overlapping-acquisition-proxies) and the [audit](../../experiments/data_audits/sen12ts_stack_window_overlap_20261026_v1/report.md).
- [x] Reconcile the metadata-only split/window audits with the completed 43-tile pilot and 38-tile supplement. The paired train/development screen already concluded with a stop on further sampling; the new metadata findings do not reopen that question or support a clean reference. See [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit) and [D-2026-10-26-002](DECISION_LOG.md#d-2026-10-26-002--treat-sen12ts-stack-windows-as-overlapping-acquisition-proxies).

## Research-record text-output disclosure (continued, 2026-10-25)

- [x] Update the exposure audit to twenty-five grouped outputs, including three additional truncated reads; the attached plan was fully reread in bounded slices, while unseen spans from two broad searches, one capped state excerpt, focused mediaTUM and HSFMamba/DSRKD status searches, and a capped risk-register inventory remain excluded. See [D-2026-10-25-187](DECISION_LOG.md#d-2026-10-25-187--extend-truncated-output-disclosure-after-reading-the-research-plan) and the [access audit](../../experiments/data_audits/research_goal_text_read_exposure_20261025_v2/report.md).

## Fixed dev2k source-group metadata (2026-10-25)

- [x] Inspect only fixed train/dev manifests and split-manifest schema. They contain per-file provenance paths and pair hashes, but no parent scene/acquisition/source-group IDs; do not treat `group_leakage_limit` as a source grouping. See [D-2026-10-25-186](DECISION_LOG.md#d-2026-10-25-186--fixed-dev2k-manifests-do-not-provide-source-groups) and the [audit](../../experiments/data_audits/fixed_dev2k_source_group_schema_20261025_v1/report.md).
- [x] Scan all PNG chunk headers for the project-local train/development SAR and optical files while skipping `IDAT`; no ancillary or provenance chunks were present in 3,600 files. This closes only the embedded-header route. See the [source-lineage addendum](../../experiments/data_audits/local_2k_source_lineage_candidate_20261025_v1/report.md#addendum--2026-10-26-jdsr-traindev-png-container-metadata).
- [ ] Obtain an authorized upstream archive/version manifest with parent-scene, acquisition, or documented source-group IDs before constructing a prospective group-disjoint split or selecting models. Keep B1 `not_promoted`; B2/B3 remain gated.

## OMA-SSR author-code route (2026-10-25)

- [x] Search exact-title/DOI author-code routes and inspect the official Wiley HTML. The data are available on reasonable request, but no public author/publisher-linked code or supplementary route was verified. See [D-2026-10-25-185](DECISION_LOG.md#d-2026-10-25-185--record-oma-ssr-code-route-as-unverified) and the [audit](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/report.md).
- [x] Recheck the exact DOI/title GitHub route and publisher correction/code-link surface on 2026-10-28; no author/publisher-linked implementation or separate correction item surfaced. The Crossmark route remains inaccessible, so absence is not established. See D-2026-10-28-051 and the [audit addendum](../../experiments/literature_audits/oma_ssr_author_code_route_20261025_v1/report.md#addendum--2026-10-28-exact-doicodecorrection-recheck).
- [ ] Revisit only if an author/publisher-linked versioned implementation or an authorized dataset/protocol package becomes available; do not treat the article as a reproducible baseline while its source and data package are unverified.

## OGSRN publisher-abstract boundary (2026-10-25)

- [x] Recheck the official SciOpen and ScienceDirect routes. SciOpen exposes the abstract and a PDF-viewer prompt without extractable PDF content/link; ScienceDirect returns HTTP 403. Classify OGSRN as training-time optical-space supervision for SAR SR with SAR-only inference. See [D-2026-10-25-183](DECISION_LOG.md#d-2026-10-25-183--keep-ogsrn-at-the-publisher-abstract-boundary) and the [audit](../../experiments/literature_audits/ogsrn_publisher_abstract_boundary_20261025_v1/report.md).
- [x] Search exact-title/DOI GitHub and author routes. A community repository claims an OGSRN implementation, but its owner is not a listed paper author and no publisher/author link was found. Do not adopt it as official code or a reproduction source; see [D-2026-10-25-184](DECISION_LOG.md#d-2026-10-25-184--classify-the-ogsrn-github-repository-as-unverified-community-code).
- [ ] Reopen full-protocol review only if an official PDF or author manuscript becomes accessible. Do not use abstract performance language as evidence or treat OGSRN as a matched baseline.

## Research-record text-output disclosure (2026-10-25, follow-up)

- [x] Exclude unseen spans and all surfaced historical result text from fifteen newly truncated output groups; cumulative test flags remain true/false. See [D-2026-10-25-182](DECISION_LOG.md#d-2026-10-25-182--disclose-truncated-research-record-outputs-follow-up) and the [access audit](../../experiments/data_audits/research_goal_text_read_exposure_20261025_v2/report.md).

## OGRN publisher-abstract boundary (2026-10-25)

- [x] Recheck the publisher-indexed OGRN abstract and bounded author-code routes. The abstract identifies optical-derived training guidance and SAR-only inference, but full-text routes return HTTP 403 and no author-controlled repository was located. See [D-2026-10-25-181](DECISION_LOG.md#d-2026-10-25-181--classify-ogrn-as-training-time-optical-guidance) and the [audit](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/report.md).
- [x] Recheck the exact DOI against targeted ResearchGate and Shenzhen University routes; no author manuscript/code appeared, and the publisher result remained abstract-only. See D-2026-10-26-076 and the [route audit](../../experiments/literature_audits/ogrn_publisher_abstract_boundary_20261025_v1/report.md).
- [ ] Reopen full-protocol review only if an author manuscript, official accessible PDF, or versioned implementation becomes available. Do not infer split, degradation, target, or metrics from the abstract; keep OGRN out of matched-baseline claims.

## Research-record text-output disclosure (2026-10-25)

- [x] Exclude four truncated local-document outputs from evidence and retain conservative cumulative test-access flags; see [D-2026-10-25-180](DECISION_LOG.md#d-2026-10-25-180--disclose-truncated-research-record-outputs) and the [access audit](../../experiments/data_audits/research_goal_text_read_exposure_20261025_v1/report.md).

## SEN12MS-CR conditional ROI-grouped source route (2026-10-25)

- [x] Screen Mamba-SCR's publisher-level SEN12MS-CR subset and Data Availability statements: 49 summer ROIs split 40/4/5, with a Zenodo pointer for sample indices and scripts. The paper-specific split has no verified crosswalk to the original TGRS holdout or current 175-ROI catalogue, so it does not resolve the source route. See [D-2026-10-25-174](DECISION_LOG.md#d-2026-10-25-174--treat-mamba-scr-split-metadata-as-non-authoritative) and the [audit](../../experiments/literature_audits/mamba_scr_sen12ms_cr_split_route_20261025_v1/report.md).
- [x] Attempt metadata-only access to the cited Zenodo DOI page and official API. The current web route reported both direct opens inaccessible and exact-DOI search returned no primary record metadata; a secondary ResearchGate result was excluded. This does not establish global record unavailability. See [D-2026-10-25-175](DECISION_LOG.md#d-2026-10-25-175--keep-the-mamba-scr-zenodo-record-unverified-after-inaccessible-metadata-routes).
- [x] Review the newly accessible publisher-full-text copy for method and geometry scope. It describes wavelet SAR-feature enhancement and optical–SAR feature-level fusion for cloud-free optical output; the reviewed robustness analyses vary cloud cover and land-cover category, and no explicit shift/registration stress or SAR-only fallback was located. Keep the paper adjacent to, not evidence for, SAR-output mismatch robustness; see [D-2026-10-25-176](DECISION_LOG.md#d-2026-10-25-176--classify-mamba-scr-as-optical-output-fusion-without-explicit-geometric-stress-evidence).
- [x] Add Mamba-SCR to the source-only claim evidence map as optical-output prior art with its geometry and source-split boundaries. The broader contribution-specific code-to-experiment map remains open.
- [x] Recheck the official Zenodo metadata route after the record page became reachable on 2026-10-26. Its Rights/License field is blank, the record’s split/archive description conflicts with the paper, and the scene metadata contents were not opened; a direct JSON API request was inaccessible via the current web tool. See [D-2026-10-26-054](DECISION_LOG.md#d-2026-10-26-054--keep-local-lineage-unresolved-and-sen12tp-unadopted) and the [route audit](../../experiments/data_audits/source_lineage_routes_recheck_20261026_v1/report.md). Keep all-scenes metadata and test identities unopened.
- [x] Compare the primary 2020 JPRS and TGRS dataset descriptions with the current author page and TUM catalogue. Keep the 157,521-patch JPRS description (149/10/10 ROI split), the 122,218-patch TGRS train-plus-holdout total (169 ROIs), and the current 175-ROI catalogue entry separate; no source explicitly reconciles all counts. See [D-2026-10-25-173](DECISION_LOG.md#d-2026-10-25-173--record-sen12ms-cr-publication-version-statistics-without-merging-them) and the [lineage audit](../../experiments/literature_audits/sen12ms_cr_release_lineage_followup_20261025_v1/report.md).
- [ ] Obtain an authorized, version-pinned training-only source manifest; reconcile the 169/175 ROI count and the archive/version relationship; verify ROI/source groups, spatial separation, SAR domain for the selected release, paired acquisition dates, and S1–S2 registration before freezing a prospective development split.
- [ ] Keep the release unadopted until those checks pass. Do not inspect or use its holdout/test identities, imagery, or measurements; do not change the local B1 gate.

## Optional anisotropic-blur characterization (2026-10-25)

- [x] Add synthetic characterization for the optional blur's normalized center impulse response, axis rotation, and zero-padding edge attenuation; the focused degradation tests pass 13/13. See [D-2026-10-25-169](DECISION_LOG.md#d-2026-10-25-169--characterize-optional-blur-with-synthetic-tests) and the [audit](../../experiments/data_audits/anisotropic_blur_characterization_20261025_v1/report.md).
- [ ] Do not enable the blur in a physical or formal acquisition model until the product-axis mapping, point-spread assumptions, and border policy are predeclared and validated. Current tests only characterize synthetic code behavior.

## Canonical controlled-degradation semantics (2026-10-25)

- [x] Audit the shared HR→LR implementation, its representation/intensity/amplitude noise modes, operation order, and optional blur boundary/axis semantics. See [D-2026-10-25-168](DECISION_LOG.md#d-2026-10-25-168--bound-the-canonical-degradation-to-representation-domain-evidence) and the [audit](../../experiments/data_audits/canonical_degradation_semantics_20261025_v1/report.md).
- [x] Run synthetic canonical-binding, normalization, FFT, and Gamma-moment tests in `AISAR`: 11 passed. They do not validate physical SAR statistics or calibrated ENL.
- [x] Add a synthetic FFT-cutoff regression: the legacy 64-bin request retains bins ±31, while the prospective 129-bin request retains the 2× support through ±64. The focused degradation suite now passes 17 tests; see [D-2026-10-25-170](DECISION_LOG.md#d-2026-10-25-170--lock-fft-cutoff-support-with-a-synthetic-regression) and the [audit](../../experiments/data_audits/frequency_cutoff_regression_20261025_v1/report.md).
- [x] Characterize the full 129-bin mask plus bilinear 256→128 resampling: synthetic bin 63 matches the analytic per-axis gain and the first excluded bin 65 is suppressed before aliasing. The focused suite now passes 18 tests; see [D-2026-10-25-171](DECISION_LOG.md#d-2026-10-25-171--characterize-bilinear-response-at-the-two-x-cutoff) and the [audit](../../experiments/data_audits/frequency_resampler_response_20261025_v1/report.md).
- [ ] Keep the local protocol described as controlled representation-domain degradation on noisy pseudo-targets. Do not claim physical Gamma speckle/ENL without calibrated source quantities and a matched measurement model. If optional blur is enabled later, predeclare orientation and border handling.

## Paper archive labels and cleanup disposition (2026-10-25)

- [x] Correct the root README to identify `paper/MMCANet_formal/` and `paper/MMCANet/` as historical archives, not current submission manuscripts. Keep their old fixed-set/held-out results out of current selection, promotion, and submission claims; see [D-2026-10-25-167](DECISION_LOG.md#d-2026-10-25-167--classify-both-paper-directories-as-historical-archives) and the [audit](../../experiments/data_audits/paper_directory_scope_20261025_v1/report.md).
- [x] Synchronize the manuscript/archive classification into the A–E [project fact sheet](PROJECT_FACTS.md).
- [x] Recheck the registered artifact inventory. No additional clearly disposable payloads or named smoke/failed/temporary outputs were identified, so retain the remaining traceable records and protected source data; see the [cleanup inventory](../../experiments/artifact_cleanup_current_inventory_20261025_v1.json).
- [ ] Draft a current manuscript only after admissible prospective evidence exists; do not reuse archived fixed-set or held-out result text. Keep B1 not promoted and B2/B3 gated while source/group and registration prerequisites remain unresolved.

## Historical pilot manuscript scope (2026-10-25)

- [x] Audit only the main LaTeX source. It is explicitly an internal July 2026 pilot with a historical held-out mini set, no validation set, and test-result prose; it is not the current submission manuscript. The generated result fragments and PDF were not opened. See [D-2026-10-25-166](DECISION_LOG.md#d-2026-10-25-166--retain-the-july-pilot-draft-as-historical-only) and the [scope audit](../../experiments/data_audits/paper_legacy_pilot_scope_20261025_v1/report.md).
- [ ] Keep the draft and its linked result fragments as historical evidence. Do not use them for current selection or submission claims. Create a separate current manuscript revision only from admissible prospective evidence; do not rebuild the old full draft routinely.

## SOTANet helper/source-chain check (2026-10-25)

- [x] Check the sibling repository's tracked-tree names and history, the local preparation helper's Git status/history, and the exact configured source mount. History contains only a `sar2opt` visualization example path, not a QXS/SEN1-2 source crosswalk; the ignored helper hard-codes a currently absent SEN1-2 source root and does not bind or write to the QXS-named cache. See [D-2026-10-25-163](DECISION_LOG.md#d-2026-10-25-163--do-not-treat-the-ignored-sotanet-helper-as-source-lineage) and the [audit](../../experiments/data_audits/sotanetwork_helper_lineage_check_20261025_v1/report.md).
- [x] Recheck the local checkout identity and bounded public GitHub root excerpt. Both identify CycleGAN/pix2pix rather than a versioned SAR–optical data source; the remote snapshot is not pinned to a current SHA. Close this repository route as non-authoritative for provenance; see [D-2026-10-25-165](DECISION_LOG.md#d-2026-10-25-165--close-the-sotanetwork-named-checkout-as-a-data-provenance-route) and the [identity audit](../../experiments/data_audits/sotanetwork_repo_identity_recheck_20261025_v1/report.md).
- [ ] Keep the local numeric 2k source, cache version, source groups, and rights unresolved. Do not infer lineage from this helper, the QXS cache name, numeric basenames, or overlap components. Reopen only with a versioned archive/manifest or an authoritative source-group crosswalk; B1 remains `not_promoted`, B2/B3 remain gated.
- [x] Disclose the truncated cross-document route search; exclude its unseen span and retain cumulative flags `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-162](DECISION_LOG.md#d-2026-10-25-162--disclose-truncated-research-state-route-search) and the [access audit](../../experiments/data_audits/research_state_route_search_truncation_20261025_v1/report.md).

## DSRKD Tsinghua institutional source route (2026-10-25)

- [x] Check the corresponding author's official Tsinghua Department of Electronic Engineering profile, achievements page, academic-papers route, and linked homepage. The profile confirms affiliation but provides no DSRKD manuscript/code; its displayed achievements end in 2015, the academic-papers page timed out, and the linked personal homepage was inaccessible. See [D-2026-10-25-160](DECISION_LOG.md#d-2026-10-25-160--keep-dsrkd-protocol-unresolved-after-the-tsinghua-institutional-route) and the [route audit](../../experiments/literature_audits/dsrkd_tsinghua_profile_route_20261025_v1/report.md).
- [ ] Keep DSRKD's dataset, target construction, degradation, split, and teacher/student protocol unresolved. Do not repeat this institutional route unless its pages or links change; resume only if a new accessible primary manuscript or author-controlled code route appears. No method claim or phase gate changes.
- [x] Disclose truncated local research-context reads and exclude their unseen spans from evidence; cumulative flags remain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-161](DECISION_LOG.md#d-2026-10-25-161--disclose-truncated-research-context-reads) and the [access audit](../../experiments/data_audits/research_context_read_truncation_20261025_v1/report.md).

## Lan et al. optical–SAR registration full-text audit (2026-10-25)

- [x] Review the publisher-formatted article and separate its HongTu-1/Google Earth registration evidence, manual-control-point RMSE, and synthetic affine stress from natural local residual registration. Add the source and transfer limits to the literature matrix, review, and pilot Related Work; see [D-2026-10-25-158](DECISION_LOG.md#d-2026-10-25-158--classify-lan-et-al-as-registration-only-prior-art) and the [audit](../../experiments/literature_audits/lan_multilevel_registration_fulltext_20261025_v1/report.md).
- [ ] Keep local B2/B3 closed. Resume local geometry work only with an exact product/baseline crosswalk and spatially separate checkpoints against a documented common reference; do not set a B2 perturbation from this paper's 1.33-pixel statistic or synthetic affine ranges.
- [x] Disclose the truncated multi-document research-text search; exclude its output from evidence and retain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-159](DECISION_LOG.md#d-2026-10-25-159--disclose-truncated-research-document-search-output) and the [access audit](../../experiments/data_audits/research_context_search_truncation_20261025_v1/report.md).

## SAR-DDPM primary-text and code audit (2026-10-25)

- [x] Review the full author manuscript and the author-linked repository's README, training script, data loader, model, and training loop. Record synthetic training/results separately from the limited real-scene ENL evidence; see [D-2026-10-25-156](DECISION_LOG.md#d-2026-10-25-156--record-sar-ddpm-scope-and-keep-local-diffusion-deferred) and the [audit](../../experiments/literature_audits/sar_ddpm_fulltext_code_audit_20261025_v1/report.md).
- [x] Add SAR-DDPM to the literature review, diffusion feasibility note, research-gap boundary, and related-work matrix as SAR-input-conditioned despeckling prior art.
- [ ] Keep local diffusion deferred. Do not use the mutable public code as a reproduction baseline until a pinned source/checkpoint and runnable conditioning path are bound to the paper's results.
- [ ] Continue prospective source/product/target lineage and independent-registration work before any new optical-utility model selection; B1 remains not_promoted, and B2/B3 remain gated.
## Registration-error sensitivity full-text audit (2026-10-25)

- [x] Review the full author-uploaded Wu et al. (2022) article and update the existing literature row. Its translation sweep and feature-level-fusion ranking concern land-cover classification; the reported GCP RMSE is fit error, not an independent checkpoint result. See [D-2026-10-25-157](DECISION_LOG.md#d-2026-10-25-157--bound-registration-error-classification-evidence-to-its-task).
- [ ] Keep local fine alignment unresolved. Recover exact source-product/baseline mapping and spatially separate checkpoints against a documented common reference before any B2 work; do not transfer the paper's shift magnitudes or enter B2/B3 while B1 is not promoted.

## So2Sat LCZ42 author metadata route (2026-10-25)

- [x] Screen the pinned author-associated training geolocation metadata. Same-EPSG checks found no cross-city overlap; a buffered common-CRS envelope screen found one candidate among 835 different-EPSG city pairs. The exact patch-level attempt failed closed because transformed boundaries were non-convex; its provisional output was removed and excluded from evidence. See [D-2026-10-25-154](DECISION_LOG.md#d-2026-10-25-154--keep-so2sat-city-labels-out-of-disjoint-splits) and the [audit](../../experiments/data_audits/so2sat_lcz42_cross_epsg_envelope_screen_20261025_v1/report.md).
- [ ] Keep So2Sat unadopted and do not use city labels as a source-disjoint split. Revisit exact cross-EPSG patch geometry only if a valid restoration target, product/acquisition lineage, applicable reuse terms, and measured S1/S2 timing/registration first make this classification dataset task-relevant. Do not inspect validation/test files.

## Noisy pseudo-target risk identity (2026-10-25)

- [x] Review Noisier2Noise's single-realization assumptions against its author manuscript and the local fixed-input source path. Its correction requires a known, sampleable source-noise law, which the normalized PNG pipeline does not establish; see [D-2026-10-25-179](DECISION_LOG.md#d-2026-10-25-179--bound-noisier2noise-to-known-source-noise-models) and the [audit](../../experiments/literature_audits/noisier2noise_known_noise_boundary_20261025_v1/report.md).
- [x] Clarify Noise2Noise's indexed-target Eq. 6 against the population risk condition E[Z|X]=E[S|X]; the local clean-target relation remains unestablished. See [D-2026-10-25-178](DECISION_LOG.md#d-2026-10-25-178--clarify-noise2noise-conditional-means-in-the-population-formulation) and the [derivation](../../experiments/literature_audits/noise2noise_population_mean_clarification_20261025_v1/report.md).

- [x] Add the distribution-free Bayes-risk decomposition showing why optical gain against `T=X+E` need not order optical gain against latent `X`; record it as theory only. It does not authorize a run or change B1/B2/B3. See [D-2026-10-25-152](DECISION_LOG.md#d-2026-10-25-152--record-the-exact-noisy-pseudo-target-bayes-risk-decomposition).
- [ ] Keep source/target lineage and residual-registration verification ahead of any new optical-utility experiment; the identity supplies no missing local residual assumption or clean reference.

## OGSFlow repository versioned review (2026-10-25)

- [x] Recheck the public repository landing page and README. A root snapshot (crawled yesterday) lists `assets/paper` and four commits; a separate README snapshot (crawled five days ago) says the public paper link and final citation will be added. No commit SHA binds the snapshots, and no paper filename was obtained. Exact folder/history/API routes were inaccessible, and shell network access was denied. See [D-2026-10-25-151](DECISION_LOG.md#d-2026-10-25-151--keep-the-new-ogsflow-repository-signal-unverified) and the [audit](../../experiments/literature_audits/ogsflow_author_repository_versioned_review_20261025_v1/report.md).
- [ ] Keep OGSFlow as mutable README-level prior-art only. Reopen only if a readable paper/citation or pinned accessible commit and file listing becomes available; do not repeat the current cache-miss routes. No code, data, split, experiment, result, or gate changed.

## PolSAR/SinSAR diffusion SR bibliographic lead (2026-10-25)

- [x] Make one bounded direct GET to the exact IEEE record after the indexed-reader route failed. HTTP 202 returned no structured citation or abstract fields; no PDF or link was opened. See [D-2026-10-25-149](DECISION_LOG.md#d-2026-10-25-149--retain-the-polsarsinsar-diffusion-sr-item-as-an-unverified-lead) and the [direct-record audit](../../experiments/literature_audits/polsar_sinsar_diffusion_ieee_direct_check_20261025_v1/report.md).
- [ ] Keep this as an unverified lead. Reopen only if a readable official publisher record or author-controlled manuscript becomes available; do not repeat the same IEEE request or infer technical claims from secondary listings.

## Artifact cleanup and run-interface follow-up (2026-10-25)

- [x] Follow-up cleanup removed the unselected R1 seed-2 `last.pth` checkpoint marked `no_feasible_checkpoint`; its manifest, config, log, and validation record remain. The approved residual/high-pass run interface was already implemented and verified under D-2026-10-23-46. The metadata-only cleanup follow-up found no further unambiguous low-value artifacts to remove. See [D-2026-10-26-020](DECISION_LOG.md#d-2026-10-26-020--prune-the-unselected-r1-seed2-checkpoint) and the [cleanup record](../../experiments/artifact_cleanup_followup_20261026_v1.json).
- [x] Recheck remaining artifacts by path, size, and run-manifest status. No smoke/debug/scratch checkpoints, generated array payloads, or additional explicitly unselected checkpoint remained; retain completed runs, audit figures, and scientific error analyses. See D-2026-10-26-082 and the updated cleanup audit.


- [x] Recheck cache and low-value artifact names with bounded filesystem metadata scans; no further deletions were warranted. Keep the remaining interruption/config/source/recovery records and high-value error analyses. See the [cleanup audit](../../experiments/data_audits/artifact_cleanup_run_interface_20261025_v1/report.md).
- [x] Reconfirm the approved MMCANetV2 residual-filter interface repair remains implemented under [D-2026-10-23-46](DECISION_LOG.md#d-2026-10-23-46--bind-mmcanetv2-residual-filters-to-one-run-configuration); its prior `py_compile` and 23-test verification remain the evidence, so no duplicate source edit or test run was needed. This follow-up's output disclosures are in [D-2026-10-26-012](DECISION_LOG.md#d-2026-10-26-012--disclose-truncated-aggregate-and-interface-confirmation-outputs) and [D-2026-10-26-013](DECISION_LOG.md#d-2026-10-26-013--extend-truncated-output-disclosure-to-42-groups).
- [ ] Avoid repository-wide text searches over experiment CSV/JSON/Markdown. If cleanup resumes, use path-limited filename/metadata checks and keep test flags conservative; the latest truncated search is excluded from evidence.

## SpaceNet 6 expanded dataset candidate screen (2026-10-25)

- [x] Review the author manuscript, official page, official-linked release article, and author processing Gist. The page and September 2020 article agree on 202 released strips (the article says 202 unique); the April 2020 arXiv v1 manuscript reports 204. The two-strip difference remains unexplained because no versioned inventory maps the counts. No S3 listing or data access was performed; see D-2026-10-25-138.
- [ ] Keep it conditional as an auxiliary external-data candidate. Reopen only after a versioned strip manifest reconciles the count, a train/development-only strip-footprint and spatial-overlap audit is completed, cross-modal checkpoints are documented, and a noisy-target protocol is predeclared. Do not access the challenge test or final-scoring areas; do not use this screen to promote B1 or enter B2/B3. See [D-2026-10-25-134](DECISION_LOG.md#d-2026-10-25-134--keep-spacenet-6-as-a-conditional-auxiliary-candidate) and the [audit](../../experiments/literature_audits/spacenet6_expanded_candidate_screen_20261025_v1/report.md).

## Sentinel-1/2 measured registration precedent (2026-10-25)

- [x] Review Ye et al.'s primary Sentinel-1/2 co-registration study. Six 10 m IW GRD/L1C pairs show terrain-dependent residual checkpoint errors after geometric correction; the 48 held-out checkpoints are correspondences from those pairs, not surveyed external truth. See [D-2026-10-25-133](DECISION_LOG.md#d-2026-10-25-133--use-sentinel12-registration-checkpoints-as-a-method-template-only) and the [audit](../../experiments/literature_audits/sentinel12_measured_registration_checkpoints_20261025_v1/report.md).
- [ ] Keep local fine alignment unmeasured. Resume only with an exact local source-product/baseline crosswalk and spatially separate checkpoints tied to a documented common reference. Do not transfer the paper's shifts/RMSE to SEN1-2 or use them to set a B2 shift; B1 remains `not_promoted`, B2/B3 remain gated.

## GLCNet prior-art and repository review completed (2026-10-25)

- [x] Add the publisher-indexed primary article and author-repository review to the literature audit, matrix, research gaps, and manuscript Related Work. Classify GLCNet as SAR-only self-supervised despeckling; it does not test optical-guided JDSR or SR.
- [x] Record the paper/repository iteration-count mismatch and ambiguous 20-image validation path without inferring test leakage. Do not reproduce it until a pinned paper-matched configuration and verified data/reference lineage are available. See [D-2026-10-25-132](DECISION_LOG.md#d-2026-10-25-132--classify-glcnet-as-sar-only-self-supervised-despeckling-prior-art).
- [ ] Keep local model selection paused. Continue only with a materially new, verifiable source/group or registration-evidence route; B1 remains `not_promoted`, B2/B3 remain gated.

## Manuscript-text access disclosure (2026-10-25)

- [x] Disclose that a broad manuscript search surfaced historical local pilot test-result text; no value was used and no raw test metrics or payload were opened. Preserve the historical cohort's exclusion from model selection and cumulative flags `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-131](DECISION_LOG.md#d-2026-10-25-131--disclose-historical-pilot-test-result-text-surfaced-during-literature-search).

## Sentinel-1 IW absolute-geolocation follow-up (2026-10-25)

- [x] Review Schubert et al.'s 2017 primary S1A/B calibration article. It quantifies IW SLC and GRDH absolute ALE on surveyed calibration sites, and identifies subswath-dependent azimuth offsets; this corrects the earlier 2015-only picture that lacked quantitative IW results. See [D-2026-10-25-130](DECISION_LOG.md#d-2026-10-25-130--record-quantified-iw-absolute-geolocation-without-transferring-it-to-local-s1s2-registration) and the [audit](../../experiments/literature_audits/sentinel1_iw_geolocation_followup_20261025_v1/report.md).
- [ ] Keep local S1–S2 residual unmeasured. Resume only with exact source-product/baseline mapping and independent common-reference checkpoints; do not derive local offsets from absolute ALE statistics.

## Sentinel-1A absolute-geolocation transfer audit (2026-10-25)

- [x] Review Schubert et al.'s primary commissioning study. It uses DGPS-surveyed corner reflectors and reports corrected 2014–2015 StripMap ALE consistency; its IW results are preliminary and unquantified. Absolute S1 accuracy does not provide a local S1–S2 residual. See [D-2026-10-25-129](DECISION_LOG.md#d-2026-10-25-129--keep-sentinel-1a-absolute-geolocation-calibration-separate-from-local-s1s2-residuals) and the [audit](../../experiments/literature_audits/sentinel1_geolocation_accuracy_primary_20261025_v1/report.md).
- [ ] Keep local fine alignment unverified. Reopen only with an authorized exact-product/baseline crosswalk and spatially separate, accuracy-documented common-reference checkpoints. Do not infer a B2 shift or open test data.


## SEN12TS temporal-reference screen completed (2026-10-25)

- [x] Verify the official S1/S2 band map and run the frozen 43-tile train/development pilot; schema and `<1%` mask-coverage checks passed.
- [x] Select the second hash-ranked tile in each eligible linkage component and run the 38-tile supplement (26 train, 12 development); five components had only one candidate.
- [x] Pair pilot and supplement by component and record the descriptive result. Local-mean changes were similar in cohort medians, while paired differences varied and residual correlations stayed positive. See [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit) and the [supplement report](../../experiments/data_audits/sen12ts_temporal_reference_supplement_20261025_v1/report.md).
- [x] Stop further external sampling for this temporal-reference question. Do not call the reference independent/clean or use this cohort for model training/selection. Reopen only with materially new source/product/group provenance or an independently supported reference protocol; keep B1/B2/B3 unchanged.

## QXS source-route refresh and exposure disclosure (2026-10-25)

- [x] Recheck official QXS-SAROPT issue #13. It remains open, with no maintainer reply or alternate mirror; this does not resolve local source provenance or groups. See [D-2026-10-25-93](DECISION_LOG.md#d-2026-10-25-93--keep-local-source-unresolved-after-qxs-issue-status-refresh).
- [x] Disclose the truncated SOTANet page response and exclude its unreconstructable text from claims; preserve cumulative session flags `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-94](DECISION_LOG.md#d-2026-10-25-94--disclose-truncated-sibling-repository-page-read).
- [ ] Keep model selection paused. Next source work must use a new, verifiable access route that can establish source/version, rights, and group metadata; do not repeat the same QXS issue/page checks without new external activity.

## Gamma sampler moments and physical-domain boundary (2026-10-25)

- [x] Verify the canonical Gamma sampler at `L=1,4,8` across intensity, representation, and amplitude modes using a predeclared in-memory CPU audit. All nine conditions passed; no local image or model data were used. See [D-2026-10-25-85](DECISION_LOG.md#d-2026-10-25-85--verify-gamma-sampler-moments-without-claiming-physical-speckle) and the [audit](../../experiments/diagnostics/gamma_sampler_moments_20261025_v1/report.md).
- [ ] Keep representation-domain noise labeled controlled synthetic noise. Before claiming physical ENL or deploying an intensity/amplitude-domain model, establish calibrated input quantities and a target/noise protocol supported by the source product.
- [ ] Do not add another sampler implementation or expand the look matrix until a valid SAR source and an experiment question require it; the current blocker is physical/source validity, not the Gamma draw's moments.

## CDSE metadata candidate route screens (2026-10-25)

- [x] Screen official CDSE STAC/OData documentation and terms; the earlier web reader could not retrieve item searches.
- [x] Query the official STAC v1 endpoint directly for a fixed point and November–December 2025. Both complete result pages were returned without a `next` link: 41 S1 GRD and 72 S2 L2A items.
- [x] Identify metadata-only candidate groups and a clear same-day pair. Ten S1 records share ascending relative orbit 44/IW/VV+VH; nine S2 item records have product cloud cover ≤20% across five datatakes. Three distinct S1/S2 datatake pairs fall within 24 h. The selected pair has a 6.602 h gap; both product polygons cover a candidate approximately 2.56 km square probe.
- [x] Freeze and screen four separated point anchors for the same two-month interval. All eight pages were complete; each site yielded low-cloud, close-date candidates. Shared S1 datatakes `513146` (Hamburg/Frankfurt) and `41531` (Berlin/Hamburg), and shared S2 datatake `GS2A_20251226T103501_054903_N05.11` (Hamburg/Frankfurt) show that sites are not independent groups.
- [x] Fetch only the 29 exact candidate Item metadata records and build a projected footprint/acquisition graph. The 19 paired-candidate edges plus two same-datatake edges form nine structural components; 141 positive-area footprint-overlap edges connect all 29 items and all four anchors into one component, unchanged at 1 m², 100 m², and 1 km² thresholds. See [D-2026-10-25-102](DECISION_LOG.md#d-2026-10-25-102--reject-the-four-anchor-cdse-pool-as-a-group-disjoint-cohort) and the [graph audit](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/report.md).
- [x] Reject this four-anchor product pool as a prospective group-disjoint train/development cohort under the frozen product-footprint rule. Do not create a split or use the one connected component for model selection.
- [x] Freeze and run the 8-AOI, May–July 2026 global metadata screen. Three AOIs form three independent components; one AOI is incomplete, so the maximum possible count is four, below the threshold of six. Stop this route without following pagination or requesting assets; see [D-2026-10-25-147](DECISION_LOG.md#d-2026-10-25-147--stop-the-global-cdse-route-at-metadata-screening).
- [ ] Keep CDSE closed at metadata screening. Reopen only if materially new source/version/group or target-protocol evidence appears; do not repeat the Berlin, Germany four-anchor, or global AOI screens. Before any future asset access, resolve calibration, AOI-specific cloud masks, reference validity, and product terms. Do not change B1/B2/B3.
- Evidence: [global AOI screen](../../experiments/data_audits/cdse_global_aoi_metadata_screen_20261025_v1/report.md), [focused live audit](../../experiments/data_audits/cdse_stac_pairing_live_20261025_v1/report.md), [four-anchor audit](../../experiments/data_audits/cdse_stac_multi_aoi_metadata_20261025_v1/report.md), [graph audit](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/report.md), [graph JSON](../../experiments/data_audits/cdse_acquisition_footprint_graph_20261025_v1/graph.json), [earlier documentation route audit](../../experiments/literature_audits/copernicus_data_space_pairing_route_20261025_v1/report.md), [CDSE STAC docs](https://documentation.dataspace.copernicus.eu/APIs/STAC.html), and [Sentinel data terms](https://dataspace.copernicus.eu/terms-and-conditions).

## Continuity exposure and assistant temporary-file cleanup (2026-10-25)

- [x] Record the truncated combined continuity read as `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no result value was used. See [D-2026-10-25-83](DECISION_LOG.md#d-2026-10-25-83--disclose-truncated-continuity-read-and-remove-rebuildable-status-dumps) and the [access audit](../../experiments/data_audits/continuity_doc_truncation_text_exposure_20261025_v1/access_manifest.json).
- [x] Remove the two assistant-generated `/tmp/jdsr_goal_status_*.txt` status dumps (625,554 bytes total) without reading contents; their exact paths and sizes are recorded in the [cleanup audit](../../experiments/artifact_cleanup_continuation_20261025_v2.json).
- [x] Resolve the header route through the public Azure Blob cloud URI and complete the bounded audit for all frozen candidates; do not repeat the earlier failed proxy/S3 requests. See D-2026-10-25-95.

## SEN12TS 2k metadata candidate and header route (2026-10-25)

- [x] Freeze a candidate-construction and temporal-reference protocol before TIFF-header access; use one exact-name pair per tile center, a fixed deterministic rank, a >4 km center guard, and prior-date VV power averaging only as a noisy pseudo-reference.
- [x] Generate a metadata-only 2,000-pair candidate manifest. The greedy screen retained 3,082 centers; the chosen candidate set has a 4.087 km minimum center spacing. It is not a train/dev split.
- [x] Resolve a verified, bounded byte-range route through the public Azure Blob mirror and inspect CRS/affine, band count/dtype, and paired SAR/optical grids for all 2,000 candidates without reading pixel blocks.
- [x] Apply the predeclared conservative 400 m footprint guard. All 2,000 tile groups remain; the minimum center-minus-radius lower bound is 1,225.281 m. This is spatial tile separation, not proof of parent-scene independence.
- [x] Create a metadata-only train/development assignment by connected components of shared SAR date/track/pass and optical date. It yields 1,101/899 pairs across 28/15 components, with no test split.
- [x] Verify the 19-band SAR and 14-band optical channel mapping against versioned source documentation; run the train/development-only pilot and second-tile supplement under frozen protocols. This establishes schema feasibility only; temporal independence remains unproven. See D-2026-10-25-98 and the two audit reports.

## SEN12TS public-hosting and tile-group route (2026-10-25)

- [x] Confirm the author repo describes a public 1.69 TB collection with repeated same-tile SAR/optical observations; Source Cooperative exposes AOI and modality folders and identifies the dataset under CC BY-NC 4.0.
- [x] Confirm the official documentation's filename schema carries tile-center latitude/longitude, acquisition dates, and SAR track/look/pass. Treat coordinate plus spatial audit as a candidate group key, not yet a validated split.
- [x] Obtain the complete Sumatra `s1/` and `s2/` metadata listings: 8,212 objects per modality, 8,212 exact basename pairs, 5,827 tile-center coordinates, and at most seven observations per tile. The compressed manifest contains names and object metadata only; no TIFF was fetched.
- [x] Run a coordinate-only nearest-neighbor pre-screen. It found 2,873 center pairs below the nominal 3.62 km tile diagonal and 5,301 below 4 km; this flags potential overlap only.
- [x] For the 2,000-pair candidate, verify exact GeoTIFF CRS/affine and paired grids for every pair, then apply the 400 m footprint bound; all 2,000 groups remain with a 1,225.281 m minimum clearance lower bound. This does not identify parent scenes. See D-2026-10-25-95 and the [header audit](../../experiments/data_audits/sen12ts_geotiff_header_audit_20261025_v1/report.md).
- [ ] Resolve the global tile-count discrepancy by enumerating or reconciling all AOIs only if the Sumatra spatial and temporal protocol proves viable.
- [x] Audit actual candidate tile transforms/footprints and create a train/development manifest from available acquisition proxies. Treat its 43 components as proxies only, not verified source groups; see D-2026-10-25-96 and the [split audit](../../experiments/data_audits/sen12ts_sumatra_acquisition_group_split_20261025_v1/report.md).
- [ ] Predeclare and validate a leave-input-date-out pseudo-reference only if independent acquisitions are available; temporal reflectivity change and correlated speckle remain competing risks. This dataset is not clean SAR ground truth.
- Evidence: [route audit](../../experiments/data_audits/sen12ts_public_listing_route_20261025_v1/report.md), [author repository](https://github.com/SEL-Columbia/sen12ts), [Source Cooperative listing](https://source.coop/sen12ts/sen12ts/paired_imagery), [dataset README and terms](https://source.coop/sen12ts/sen12ts/README.md), and [documentation PDF](https://data.source.coop/sen12ts/sen12ts/Documentation.pdf).

## QXS-SAROPT public distribution route recheck (2026-10-25)

- [x] Recheck the official author repository, releases, and the public mirror-request issue. The repository exposes only its README and preview image; its download link goes to a Wenjuan form that this web route cannot inspect; Releases is empty; issue #13 remains open without a maintainer response. Tags could not be fetched, so no conclusion is made about tags.
- [x] Preserve the source boundary: this does not prove that local files are QXS-SAROPT and does not provide a source-group map or canonical split. No form was submitted and no data were downloaded.
- [ ] Keep the local 2k dataset lineage unresolved. Restore an authoritative archive/version and patch-to-source mapping, or identify a new accessible source with verifiable source groups and an auditable prospective split before resuming model selection.
- [x] In parallel, screen one manageable, georeferenced paired SAR–optical source with authoritative scene/tile/date identifiers and terms. The frozen OPERA RTC-S1/Sentinel-2 screen returned no qualifying four-date paired stack; do not adopt or widen its window post hoc. See [D-2026-10-25-121](DECISION_LOG.md#d-2026-10-25-121--stop-the-frozen-operasentinel-2-aoi-screen).
- Evidence: [route audit](../../experiments/data_audits/qxs_author_data_route_recheck_20261025_v2/report.md), [author repository](https://github.com/yaoxu008/QXS-SAROPT), [releases](https://github.com/yaoxu008/QXS-SAROPT/releases), and [issue #13](https://github.com/yaoxu008/QXS-SAROPT/issues/13).

## Research-state index output disclosure (2026-10-25)

- [x] Disclose the truncated `NEXT_ACTIONS.md` structure-extraction response as `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no result value was used. See [access audit](../../experiments/data_audits/research_state_action_index_exposure_20261025_v1/report.md).

Updated: 2026-10-25. Do not read test imagery or measurements.

## Continuity excerpt exposure (2026-10-25)

- [x] Record the truncated combined read of `CURRENT_STATE.md` lines 1–32 and `NEXT_ACTIONS.md` lines 1–80 as `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no value was used. Preserve prior cohort exclusions. See [D-2026-10-25-76](DECISION_LOG.md#d-2026-10-25-76--disclose-truncated-continuity-excerpt) and the [access audit](../../experiments/data_audits/session_context_excerpt_exposure_20261025_v2/access_manifest.json).

## Session context excerpt disclosure (2026-10-25)

- [x] Record the truncated continuity reads of the first 120 lines of `CURRENT_STATE.md` and first 75 lines of `NEXT_ACTIONS.md` as `test_set_accessed=true`, `test_set_used=false`; no payload or metric file was opened and no value was used. Preserve existing cohort exclusions. See [D-2026-10-25-74](DECISION_LOG.md#d-2026-10-25-74--disclose-truncated-session-context-excerpt) and the [access audit](../../experiments/data_audits/session_context_excerpt_exposure_20261025_v1/report.md).

## SEN12TS conditional source/task-fit screen (2026-10-25)

- [x] Review the dataset documentation, current product listing, and author repository overview. Tile-center/date names and georeferenced TIFFs may support grouping all dates by tile; current records specify CC BY-NC 4.0. The dataset is 1.69 TB, has no general restoration split, and provides observed RTC SAR rather than independent clean SAR. The documentation's printed total tile count conflicts with its AOI row sum. See [D-2026-10-25-73](DECISION_LOG.md#d-2026-10-25-73--keep-sen12ts-as-a-conditional-tile-grouped-source-candidate) and the [source audit](../../experiments/literature_audits/sen12ts_source_task_fit_screen_20261025_v1/report.md).
- [ ] Keep SEN12TS conditional. Reconsider only if an authorized manageable subset, versioned tile metadata, spatial-overlap evidence, and an independent-reference/temporal-stationarity protocol are available. Do not use its land-cover example split as a restoration split or change B1/B2/B3 gates.

## SEN12TS temporal-reference feasibility (2026-10-25)

- [x] Screen the primary dataset documentation, SAR2SAR, and multi-temporal MERLIN. The four same-track acquisitions span 36 days; short-baseline scene stability competes with temporal speckle dependence, and the documented coherence layers are too coarse to certify 10 m pixelwise independence. Keep any future average labeled as a pseudo-target. See [D-2026-10-25-75](DECISION_LOG.md#d-2026-10-25-75--keep-sen12ts-temporal-references-conditional) and the [audit](../../experiments/literature_audits/sen12ts_temporal_reference_feasibility_20261025_v1/report.md).
- [x] Before pixel access, freeze a manageable train/development sample, versioned tile/acquisition map, component-level selection, and a leave-input-date-out reference in linear γ⁰ power. Measure temporal dependence, radiometric drift, masks, and coarse registration in the pilot and second-tile supplement. Results do not validate an independent target.
- [x] Record that the bounded audit does not reopen B1, B2, or B3.

## Research-record search output disclosure (2026-10-25)

- [x] Record the truncated multi-document `rg` output caused by unescaped backticks in a shell double-quoted pattern as `test_set_accessed=true`, `test_set_used=false`; no test payload was opened and no value was used. See [D-2026-10-25-71](DECISION_LOG.md#d-2026-10-25-71--disclose-truncated-research-record-search-output) and the [access audit](../../experiments/data_audits/sen12tp_doc_search_exposure_20261025_v1/report.md).
- [ ] Do not inspect or repeat the truncated output. Quote future literal patterns with single quotes and search only the specific source/record needed.

## HSFMamba author-linked implementation route (2026-10-25)

- [x] Inspect the linked public repository's OGSSR section. It lists an external dataset share but its Algorithm table has no implementation entry; do not count HSFMamba as a reproduced baseline. See D-2026-10-25-69 and the [audit](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md).
- [x] Leave the external dataset share unopened because it does not resolve code availability or the current local protocol question.
- [ ] Revisit implementation lineage only if a direct, versioned author-code route appears. Keep paper-level claims and local B1/B2/B3 gates unchanged.

## README project-entry audit (2026-10-25)

- [x] Structural check: the planned research, status, data, model, protocol, reproduction, roadmap, literature, paper, limitations, repository, and next-step headings are present; all inspected local file-link targets exist.
- [x] Spot-check selected non-quantitative status, data-domain, test-boundary, and roadmap statements against the current project rules and `CURRENT_STATE.md`; no contradiction was found in this subset.
- [x] Verify local quantitative Key Findings against their cited train/validation source records and non-test manifests. Counts and directions match the train-only metadata screens, source-aligned B1 v2 validation audit, C2 diagnostic, bootstrap rederivation, SIFT training-list probe, B1F development diagnostics, and ReFlowSET train-list audit. The source-aligned run binding reuses byte-identical frozen input metadata. The C2 checkpoint-selection cohort and the unverified bootstrap folder identities are now explicit in README. This audit did not use the historical text-exposed mini cohort and accessed no test imagery or measurement payloads.
- [x] Verify the documented historical diagnostic's config, seed-0 fusion checkpoint, two configured parent checkpoint paths, validation metadata, and output-path behavior using path metadata and source only. The fixed validation directory is read-only and has no `.npy` payloads, the diagnostic hard-binds it, and the preparation helper rejects nonempty output directories; no command was run and no payload was read.
- [x] Apply source-supported README corrections only: link the current source-aligned run manifest and input binding, state that all three C2 checkpoints were selected on the reused validation cohort, and identify the 15 bootstrap folders as unverified parent-scene/acquisition units. No metric values, gate, or promotion state changed.

## Artifact cleanup review (2026-10-25)

- [x] Review existing cleanup records and current output/cache metadata. Prior passes already removed prediction arrays and temporary caches; no additional deletion was justified without risking necessary run, audit, or historical result records. See the [retention audit](../../experiments/data_audits/artifact_retention_review_20261025_v1/report.md) and [D-2026-10-25-20](DECISION_LOG.md#d-2026-10-25-20--retain-only-evidenced-artifacts-after-cleanup-review).
- [x] Perform the requested follow-up using exact path/type/size metadata. No in-repository arrays, caches, or named smoke/failed artifacts remain under the experiment/output/checkpoint roots. Keep the three `error`-named CSVs because the cascade matrix cites them as error-analysis evidence. No further in-repository deletion was justified; see the [cleanup record](../../experiments/artifact_cleanup_residual_20261025_v3.json) and [D-2026-10-25-23](DECISION_LOG.md#d-2026-10-25-23--retain-cited-error-analysis-and-close-residual-cleanup).
- [x] Disclose the legacy held-out aggregate result text surfaced in `CURRENT_STATE.md` during state review. Do not use that historical cohort for any future selection or tuning; preserve its original run flags. See the [access audit](../../experiments/data_audits/cleanup_search_text_exposure_20261025_v1/report.md) and D-2026-10-25-24.

## Standard SEN12MS conditional source route (2026-10-25)

- [x] Check whether the numeric scene/data token is globally unique across the author training-list season branches. It is not: 78 of 119 numeric tokens recur in multiple branches, so preserve branch/season in filename keys. This does not indicate shared or disjoint geography; see the [namespace audit](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/report.md).
- [x] Check the public author repository and official catalog for a train-only geospatial crosswalk or supplied validation list. The reviewed repo tree/README exposes neither; it mentions only an overview notebook for all ROIs. The notebook itself and all map outputs remain unopened to avoid whole-dataset/holdout coordinates. GovData exposes no usable resource or license field, and TUM returned HTTP 429; see the [access audit](../../experiments/literature_audits/sen12ms_train_id_namespace_20261025_v1/access_manifest.json).

- [x] Screen the standard 2019 release separately from SEN12MS-CR-TS using the primary paper, author README/license, catalog metadata, and author training-list metadata only. The training list contains 162,555 unique rows grouped into 226 provisional ROI/scene filename keys across four season branches. This is a candidate-capacity count, not evidence of geographic independence; 256-pixel patches at stride 128 overlap by 50%. See the [audit](../../experiments/literature_audits/sen12ms_standard_train_metadata_20261025_v1/report.md).
- [x] Screen the newly surfaced FDL GCS mirror as a possible metadata route. Its indexed README exposes a bulk `SEN12MS` sync command, not a train-only bounds/crosswalk file; the available browser could not fetch a bucket listing, and this says nothing about underlying permissions. Do not download the mirror or treat its route as resolving source geometry. See the [route audit](../../experiments/literature_audits/sen12ms_gcs_mirror_metadata_route_20261025_v1/report.md) and [D-2026-10-25-21](DECISION_LOG.md#d-2026-10-25-21--do-not-use-the-fdl-bulk-mirror-as-a-metadata-shortcut).
- [x] Recheck only the FDL raw README URL through a bounded direct reader; it returned HTTP 404 and no body was read. Do not guess alternate raw paths or list the bucket from this response. See [D-2026-10-25-150](DECISION_LOG.md#d-2026-10-25-150--leave-the-fdl-sen12ms-mirror-route-unresolved-after-404) and the [direct-route audit](../../experiments/literature_audits/sen12ms_fdl_readme_direct_route_20261025_v1/report.md).
- [x] Screen the current TorchGeo SEN12MS loader and author-linked TUM page. TorchGeo documents bulk four-season/three-modality archives plus separate train/test lists, while the TUM share page exposes only a JavaScript shell to the available reader. No train-only geospatial crosswalk was found; no archive, list, label, or raster was opened. See the [route audit](../../experiments/literature_audits/sen12ms_torchgeo_tum_route_20261025_v1/report.md) and [D-2026-10-25-25](DECISION_LOG.md#d-2026-10-25-25--close-the-sen12ms-torchgeo-and-tum-route-without-bulk-download).
- [x] Screen the official mediaTUM node-export API as a metadata-only route. Its docs support node/children export, but the candidate SEN12MS record endpoint was inaccessible through the available reader and its share token was not verified as a node ID. No train-only geometry, record terms, or useful fields were obtained; see the [route audit](../../experiments/literature_audits/sen12ms_mediatum_metadata_api_route_20261025_v1/report.md) and [D-2026-10-25-40](DECISION_LOG.md#d-2026-10-25-40--keep-mediatum-api-as-an-unverified-metadata-route).
- [x] Recheck the candidate `1474000` page through an authorized direct reader after the web reader returned HTTP 429. The direct page returned HTTP 200 but had a generic title; unscoped text hits for SEN12MS and `m1474000` do not bind the numeric ID to a structured record. Do not treat the node as confirmed; no JSON or child request was made. See [D-2026-10-25-148](DECISION_LOG.md#d-2026-10-25-148--keep-mediatum-node-identity-unverified-after-direct-page-check).
- [ ] Reopen mediaTUM only when new authoritative TUM search or structured record metadata binds the exact numeric node to standard SEN12MS; the frozen exact-domain search returned no results. Then inspect record-only metadata; request direct-child metadata only if it can be filtered to train-only fields without exposing holdout/test identities. Do not repeat the current landing-page/search request, request recursive children, archive files, split-list contents, or imagery. Keep the route unverified until a train-only spatial/source crosswalk and applicable access terms are directly evidenced.
- [ ] Determine whether authorized train-only geospatial/source metadata can be obtained without enumerating holdout/test identities or downloading the full archive. Continue only through a newly identified selective train GeoTIFF metadata endpoint or authoritative crosswalk. Keep the paper's CC-BY data statement and the author's research/education-only, non-commercial data license as separate source facts; do not assume redistribution rights beyond the reviewed terms.
- [ ] Check candidate ROI/scene groups for cross-group geographic overlap, then define a representation-domain SAR noisy-pseudo-target task and freeze a group-disjoint train/validation split before any model selection. Keep the candidate unadopted until these checks pass; the local numeric 2k lineage is unchanged, B1 remains not_promoted, and B2/B3 remain gated.

## Frequency-specific optical utility under registration uncertainty (2026-10-25)

- [x] Derive the frequency-wise incremental LMMSE gain for a linearly related optical coefficient under an independent per-example random translation; distinguish unknown shifts from a known, exactly correctable global shift. Record assumptions and limits in the [analysis](optical_information_analysis.md#frequency-resolved-linear-utility-under-random-residual-translation-2026-10-25) and [H4 addendum](research_hypotheses.md#h4-theoretical-addendum--frequency-specific-effect-of-unknown-residual-shift-2026-10-25).
- [x] Derive how the same one-frequency model's absolute LMMSE gain varies with additive SAR-noise power when optical quality and shift coherence are held fixed. Record why this does not transfer directly to Gamma looks or SR scales in the [analysis](optical_information_analysis.md#sar-noise-severity-under-fixed-optical-reliability-2026-10-25) and [H1 addendum](research_hypotheses.md#h1-theoretical-addendum--additive-sar-noise-sensitivity-2026-10-25).
- [ ] Only after authorized source/registration metadata and a prospective group-disjoint split exist, decide whether a predeclared severity-by-optical and frequency-stratified utility test is feasible. Freeze each LR condition and use a matching SAR-only control; treat additive-noise monotonicity as a restricted reference, and do not use current contaminated development partitions or test measurements.

## SAMDFuse author-source follow-up (2026-10-25)

- [x] Inspect the public author README and exposed SFEM/SGFM/fusion modules, and compare its repository citation status with the publisher-indexed journal record. Record semantic/cloud-degradation fusion as adjacent prior art; do not infer geometric-shift robustness, SAR-output restoration, or fallback from these sources. See the [audit](../../experiments/literature_audits/samdfuse_author_source_followup_20261025_v1/report.md).
- [ ] If the authors later publish a version-pinned full pipeline or the publisher full text becomes directly readable, reconcile the final paper-to-code version and check whether it contains explicit geometric registration or a SAR-only output path. Do not reproduce it as a SAR restoration baseline from the current partial release.

## MD-Glow SAR multi-task prior art (2026-10-25)

- [x] Screen the official CVF indexed paper text beyond the abstract: model/loss, synthetic training setup, ablations, comparison tables, and qualitative SR path. Record MD-Glow as SAR-only multi-task prior art; its quantitative tables evaluate despeckling, not joint SR. See the [audit](../../experiments/literature_audits/md_glow_indexed_fulltext_screen_20261025_v1/report.md) and [D-2026-10-25-08](DECISION_LOG.md#d-2026-10-25-08--record-md-glow-as-sar-multi-task-despeckling-sr-prior-art).
- [ ] Revisit unexposed protocol details only if the full PDF or an official supplement becomes directly readable; keep source-group separation, exact random downsampling, and numeric task-loss weights unresolved. Also resolve the CVF/DBLP page-range conflict only from authoritative publisher metadata; cite without pages meanwhile.

## MuChaPro InSAR spatial-regularization prior art (2026-10-25)

- [x] Record the IEEE abstract and HAL metadata as adjacent InSAR phase/coherence prior art; keep the full method/protocol unknown because the IEEE and HAL article routes were not readable. See the [audit](../../experiments/literature_audits/muchapro_insar_regularized_inversion_abstract_20261025_v1/report.md).
- [ ] Review the full article only if its official IEEE/HAL text becomes directly accessible. Do not infer optical guidance, intensity-SAR SR, mismatch robustness, or local physical-model validity from this phase/coherence study.

- [x] Rechecked the current `AISAR` runtime on 2026-10-25: PyTorch 2.8.0+cu128 is installed, CUDA is unavailable with zero visible devices, and `nvidia-smi -L` exits 9 because the driver cannot be reached. No model run was started. Continue source-independent work; GPU restoration alone does not reopen B2/B3.

- [x] Screen Gaya et al.'s 2024 IGARSS optical-assisted SAR despeckling paper using the institutional abstract and official program entry. Record direct optical/self-supervised despeckling prior art, with full-text protocol and results unknown; no joint-SR, mismatch, or fallback claim follows. See the [audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md) and [D-2026-10-25-04](DECISION_LOG.md#d-2026-10-25-04--record-gaya-et-al-as-abstract-level-optical-assisted-sar-despeckling-prior-art).
- [x] Recheck the official IEEE Xplore document and direct stamp/PDF routes on 2026-10-28; neither was accessible through the reader. An author-associated HAL CV index points to the already-known `hal-04676452v1` entry, but its record/document routes remain blocked and supplied no full text. See [D-2026-10-28-029](DECISION_LOG.md#d-2026-10-28-029--keep-gaya-igarss-2024-at-abstract-level-after-ieee-route-recheck).
- [ ] Revisit the paper's methods only if an official or author-hosted full-text route becomes accessible; keep the method abstract-level meanwhile.

- [x] Screen the author sources for MultiResSAR's task fit and release lineage. The paper describes a registration benchmark and an 850-pair Umbra test subset; the current repository exposes only 1,100 pairs without a visible subset/source-group manifest or dataset license. Do not download or adopt for SAR restoration. See the [audit](../../experiments/literature_audits/multiresar_dataset_provenance_screen_20261025_v1/report.md) and [D-2026-10-25-03](DECISION_LOG.md#d-2026-10-25-03--keep-multiresar-unadopted-for-sar-restoration).
- [x] Remove the residual project-named `/tmp` status/resume dump; retain source data and registered run/audit evidence. No cache or smoke/scratch-named workspace directories remained. See the [cleanup record](../../experiments/artifact_cleanup_residual_20261025_v1.json).
- [x] Prune the rebuildable epoch 10/20/30/40 checkpoint snapshots from the legacy MMCANetV2-v2 and SFT folders, retaining endpoint/default weights and all run records. The workspace has no generated arrays or cache/smoke/scratch artifacts; see the [cleanup audit](../../experiments/artifact_cleanup_checkpoint_pruning_20261025_v2.json).
- [x] Screen SARptical from its primary paper, author repository, and current DLR data-use page. The 2018 paper describes high-resolution Berlin TerraSAR-X/UltraCAM pairs with 3D center-point correspondence and only first-approximation patch alignment; exact per-pair residuals and patch-to-source groups are not exposed. DLR's current page makes TerraSAR-X terms use-dependent, and the reviewed author README gives no dataset license/version manifest. Keep it unadopted for restoration; see the [audit](../../experiments/literature_audits/sarptical_dataset_provenance_screen_20261025_v1/report.md) and [D-2026-10-25-01](DECISION_LOG.md#d-2026-10-25-01--keep-sarptical-unadopted-pending-rights-and-source-lineage).
- [ ] Reconsider SARptical only if explicit SAR and optical reuse terms and a versioned patch-to-acquisition/view manifest become available. Any later use would need source-grouped spatial splits and a task-specific SAR restoration target/degradation; do not treat center-point matching as dense pixel registration or as evidence for the Sentinel-1/2 B1 condition.

- [x] Screen EarthMiss as a possible independent SAR–optical source. Its official project page states a city-wise split but also limits use to academia and prohibits secondary development; its Zenodo v1 record exposes no license value, and the restoration target/task is not defined. Do not download, derive, or train on it unless use terms are clarified and group/target metadata are permitted. See the [candidate audit](../../experiments/data_audits/earthmiss_research_candidate_screen_20261024_v1/report.md) and [D-2026-10-24-45](DECISION_LOG.md#d-2026-10-24-45--keep-earthmiss-unadopted-pending-restoration-use-permission-and-target-fit).
- [x] Screen the newly surfaced Figshare v2 QXS-derived semantic-mask record using its public metadata only. The page identifies QXS-SAROPT as upstream and warns that the displayed CC BY 4.0 does not remove upstream image restrictions; it exposes no local filename/source-group crosswalk. Do not download or adopt the archive without resolved upstream terms and an authorized source map. See the [route audit](../../experiments/data_audits/figshare_qxs_semantic_masks_route_20261024_v1/report.md) and [D-2026-10-24-44](DECISION_LOG.md#d-2026-10-24-44--classify-the-figshare-qxs-derived-deposit-as-a-non-resolving-provenance-route).
- [x] Make the historical MMCANetV2 validation evaluator fail closed unless checkpoint test-access flags are explicitly false and its validation split/manifest hashes match the fixed inputs. Eight synthetic tests and `py_compile` passed. The protocol module's remaining failure is the preserved historical B1 source-hash mismatch; no historical artifact or gate changed. Generated bytecode, pytest cache, and this turn's synthetic pytest sessions were removed; see the [cleanup audit](../../experiments/artifact_cleanup_validation_contract_20261024_v1.json) and [D-2026-10-24-43](DECISION_LOG.md#d-2026-10-24-43--bind-historical-mmcanetv2-validation-to-checkpoint-input-metadata).
- [x] Remove the remaining three synthetic-interface pytest scratch sessions and two empty Python cache directories; keep fixed-split metadata and registered research artifacts. See the [cleanup record](../../experiments/artifact_cleanup_residual_20261024_v1.json). No training, inference, metric calculation, selection, or gate change occurred.
- [x] Disclose that the cleanup-total text search unintentionally surfaced stored `fixed_test_dev2k` metric rows from the broad `experiments` search root. The numeric output was not analyzed or used; no test image/array was opened. Treat those measurements as accessed and do not use them for future tuning/selection. This later event does not alter the B1 v2 run manifests' run-time flags; see the [disclosure audit](../../experiments/data_audits/legacy_test_metric_text_exposure_20261024_v1/report.md) and [manifest](../../experiments/data_audits/legacy_test_metric_text_exposure_20261024_v1/access_manifest.json).
- [x] Repair the legacy MMCANet-v4 inference contract: source-free `zero`/`missing` modes no longer require optical files, inference loads fixed LR without reading HR target payloads, and shape checks use configured upscale. Seven synthetic interface tests, including end-to-end CPU inference for both modes, and `py_compile` passed; generated caches were removed. No real data or test payload was used; B1/B2/B3 remain unchanged. See [D-2026-10-24-29](DECISION_LOG.md#d-2026-10-24-29--make-mmcanet-v4-inference-source-free-modes-independent-of-optical-and-target-payloads).

## Completed B1 v2 execution — current-source full matrix (2026-10-23)

- [x] Recheck authorized CUDA/AISAR availability and confirm no prior training process was active. RTX 3050 Laptop GPU: 4,096 MiB total, 3,387 MiB free at preflight.
- [x] Reverify the frozen fixed train/validation inputs from source and pixel hashes: 1,600/200 rows, metadata byte-identical to the retained reference, with test flags false.
- [x] Preserve the four-source-hash mismatch audit for the previous v2 root; create a unique current-source root and capture 26 source hashes, environment, exact commands, and input bindings. Do not combine old sealed runs with this matrix.
- [x] Complete/seal all six current-source runs, verify their 25/10 epoch logs, checkpoint and canonical validation hashes, and run the frozen aggregator. The gate is `not_promoted`: mean ΔPSNR `+0.001617 dB` fails `+0.05 dB`; mean ΔSSIM `−0.000892` fails zero. All seed ΔPSNR values are positive and other guardrails pass. Stop at B1; do not run B2/B3.
- [x] Confirm that all 26 source hashes matched at sealing, verify the five aggregate output hashes, and record the two expected post-run D-2026-10-23-46 source changes (`MMCANetV2/model.py`, `scripts/train_mmcanet_v2.py`) without altering the sealed hashes. Test flags remain false; see the [source/objective audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).
- [x] Recompute source-group/seed mean ΔPSNR and ΔSSIM from the canonical paired rows and compare with the retained group summary. ΔSSIM is negative in all 45 group/seed cells; positive group-mean ΔPSNR counts vary by seed (7/15, 9/15, 12/15). Record only as a read-only proxy-target diagnostic; do not change the gate or select a subgroup.
- [x] Audit the sealed C2 loss weights and training curves against the strict validation-score implementation. The only nonzero terms are reconstruction and residual; all seeds select epoch 1, while epoch-10 validation loss is lower but SSIM/selection score are lower and residual contribution is larger. Record a possible objective/metric tension, not a proven cause; do not retune on this cohort.
- [x] Trace the active C2 loss and current model path, bind the observations to the run's recorded source hashes, and document the post-run source change. The run-bound loss file is unchanged; model/trainer differences are the documented D46 interface maintenance. Do not infer causality or tune the failed cohort. See the [audit](../../experiments/diagnostics/b1_v2_c2_objective_code_audit_20261023_v1/report.md).
- [x] Summarize per-image direction consistency from the hash-bound 600-row B1 v2 paired validation CSV. PSNR image means were positive for 129/200 identities, while SSIM was worse for 192/200 and edge-strength ratio error was worse for all 200 in every seed. Preserve as descriptive failure analysis only; do not select images/subgroups, retune, or change the frozen gate. See the [audit](../../experiments/diagnostics/b1_v2_per_image_heterogeneity_20261023_v1/report.md).
- [x] Synthesize the canonical gate, per-image, and objective/code evidence into a bounded C2 failure analysis. The failed gate is established, but no causal root is isolated; objective mismatch, timing/geometry, proxy-target/degradation, and frozen-backbone capacity remain competing hypotheses. Do not retune on this cohort or run B2/B3. See the [synthesis](../../experiments/diagnostics/b1_v2_c2_root_cause_synthesis_20261023_v1/report.md).
- [x] Remove per-epoch checkpoint/validation outputs and the exact temporary fixed-input workspace after sealing; retain selected best/last checkpoints, canonical validation results, complete epoch summaries, configs, commands, manifests, aggregate, environment/source/input audits, and [pruning audit](../../experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/artifact_pruning_audit.json).
- [x] Repair MMCANetV2 run configuration wiring: the residual kernel flags now configure both the model high-pass branch and matching loss filters, while inference restores the saved model settings. Default behavior remains 5 / 1.0; `py_compile` passed and `tests/test_mmcanet_v2.py` passed 23/23 in the existing `AISAR` CPU environment. No experiment was run.
- [x] Require an explicit `--validation-dir` in both B1 trainer CLIs whenever `--protocol-id` is set, while retaining the legacy default for non-B1 calls. Four targeted parser cases passed and both trainer files passed `py_compile`; the full suite's single failure remains the preserved historical Phase-B1 source-hash mismatch. No experiment was run; see D-2026-10-24-19.
- [x] Screen the newly surfaced Figshare QXS-derived package using page/catalog metadata only. Its v2 record names QXS-SAROPT but has no reviewed itemized file inventory or source-group crosswalk, warns that filename matching does not prove co-registration, and leaves upstream imagery terms controlling; no archive was downloaded. This route does not resolve local 2k lineage; see the [audit](../../experiments/data_audits/figshare_qxs_semantic_repackage_source_screen_20261024_v1/report.md).
- [x] Check for the record's author-controlled processing notebook/code route. Targeted Kaggle/GitHub searches found no matching notebook/repository, and the linked Figshare author profile returned 403. Keep this route closed unless a directly accessible metadata/code path appears; do not download the dataset archive. See the [route audit](../../experiments/data_audits/figshare_qxs_semantic_repackage_source_screen_20261024_v1/report.md).


## P0 — Restore development split integrity before model work

- [x] Compare the numeric local train/validation names against the official ReFlowSET QXS-SAROPT train list only. Exact basename membership was 1,310/1,600 local train and 157/200 local validation (1,467/1,800 total); 333 names absent from that list remain unclassified. This does not establish dataset identity, source groups, shared pixels, or leakage; no other ReFlowSET split was accessed. The `main` snapshot is unpinned. See the [audit](../../experiments/data_audits/reflowset_numeric2k_train_membership_20261024_v1/report.md) and D-2026-10-24-16.

- [x] Prune rebuildable numeric outputs and low-value generated artifacts. Removed registered `.npy`/`.npz` experiment arrays, fixed-input arrays, named smoke/failed artifacts, caches/build outputs, and legacy pilot-snapshot payloads while retaining configs, manifests, metrics/report records, logs, audit figures, source PNGs, standalone paper PDFs, and current experiment checkpoints. Four checkpoints from the run explicitly invalidated for train/test overlap were removed; its run records remain. The pilot archive's original manifest is retained as a pre-prune inventory and no longer matches the reduced snapshot. Pixel payload contents were not read in the numeric-array or 2026-10-22 pruning passes. The 2026-10-22 follow-up also removed the ignored BasicSR setuptools/Cython cache (334 files, 10,473,691 bytes) and 19 exact duplicate diagnostic figures from candidate-summary v2–v4, retaining canonical v5 figures and all versioned CSV/JSON/manifests. See the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).
- [x] Remove 200 replayable cascade validation preview PNGs and three unreferenced attention-demo figures (203 files, 17,515,272 bytes), and remove the 11 corresponding empty preview directories while retaining the manifests, CSVs, command/config records, checkpoints, and selected audit figures. The broad text-search scope may have included historical fixed-test output text; treat that set as accessed and do not use it for selection. The prospective candidate test directory was not enumerated. See the [cleanup/access audit](ARTIFACT_CLEANUP_20261021.md) and [disclosure record](manuscript_text_search_access_disclosure_20261022.json).
- [x] Finish the named SOTANet `sar2opt` cache crosswalk, excluding the 100 legacy mini test identities before source reads. All 1,700 permitted local pairs had zero exact modality or paired matches among 14,400 candidate train/val composites. This closes the exact named-cache route for local group recovery; see the [audit report](../../experiments/data_audits/sen1_2_sotanet_named_cache_crosswalk_20261019_v1/report.md).
- [x] Verify one deterministic candidate train pair against official TUM SEN1-2 files. Both panels match exactly. Metadata found the same 30 filename-derived groups in both candidate train and val, so do not reuse those partitions for independent validation; see the [probe](../../experiments/data_audits/sen1_2_named_cache_official_pair_probe_20261021_v1/report.md).
- [x] Freeze and validate two prospective source-folder cohorts before model metrics. The 24/6 v1 cohort remains a valid audit; v2 is preferred for model work and uses 1,600/200 samples across 15 train and 15 disjoint fall validation source folders. All selected pairs match official SEN1-2 pixels and both declared exact/±128-grid audits pass. The candidate test directory was not enumerated. See the [v1 audit](../../experiments/data_audits/sen12_scene_dev1800_auth_20261021_v2/report.md), [v2 protocol](../../protocols/sen12_scene_dev1800_v2/protocol.md), and [v2 audit](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md).
- [x] Address the six-cluster limitation before training. Metadata-only capacity analysis showed a 15/15 allocation was feasible; v2 now provides 15 fall-only validation source folders and has passed official source and overlap checks. Keep its parent-ROI/season limits explicit; use source-folder-aware uncertainty.
- [x] Freeze the new cohort's representation-domain 256→128 degradation, fixed-input metadata, target/mask/metric definitions, and source-group-aware paired uncertainty plan. Keep the registered B1 gate and guardrails unchanged; do not reuse contaminated-split checkpoints. The protocol and original metadata are in [`protocols/sen12_scene_dev1800_v2/b1_revalidation_v1.md`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v1.md) and [`sen12_scene_dev1800_v2_b1_revalidation_inputs_20261021_v1`](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_revalidation_inputs_20261021_v1/). A CPU rebuild recreated and verified all train/validation inputs against their frozen hashes, then removed the payloads again; see the [rebuild audit](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_input_rebuild_20261021_v1/).
- [x] Estimate the three-seed training budget using the authorized RTX 3050 Laptop 4-GiB profile and a synthetic batch-2 optimizer run. The compute-only projection is about 2.1 GPU hours for the declared steps; validation, I/O, metrics, and checkpoint overhead are excluded. This is an estimate, not a full training feasibility run.
- [x] Freeze the prospective B1 registration interpretation and bind it through the run interface. Xu et al.'s full 2023 SOPatch article reports limited registration accuracy for original SEN1-2 geographic pairing; its reprocessed subset uses spring/winter only and does not cover v2's fall-only validation groups. No fall residual-error distribution or traceable fine-registration reference is available. [`registration_condition_addendum_v1.json`](../../protocols/sen12_scene_dev1800_v2/registration_condition_addendum_v1.json) therefore limits the experiment to source-matched nominal georegistration and explicitly leaves fine pixel alignment unverified. The sealer writes its ID/path/hash to run and validation manifests; the aggregator checks and reports the same binding. This does not change the frozen B1 configuration or numeric gate. No B2/B3 until the prospective B1 screen passes.
- [x] Implement and validate a source-group-aware paired B1 result aggregator. It enforces exact validation identity/group membership, all three paired seeds, fixed `files.csv` hashes, consistent regenerated manifests, and the historical B1 gate while reporting 15-group bootstrap uncertainty; 12 synthetic-fixture tests and `py_compile` pass. No model run has occurred. See [`b1_aggregation_contract_v1.md`](../../protocols/sen12_scene_dev1800_v2/b1_aggregation_contract_v1.md) and `scripts/aggregate_source_group_b1.py`.
- [x] Rebuild train/validation fixed inputs on CPU and verify every fixed LR, target, optical, and mask artifact, plus the frozen `files.csv`, split, and source-protocol hashes. The validation summary records 1,600/200 complete formal inputs, false test flags, and no model selection; generated payloads were removed after verification. See the [rebuild audit](../../experiments/data_audits/sen12_scene_dev1800_v2_b1_input_rebuild_20261021_v1/).
- [x] Exhaustively audit exact 51/52-pixel edge overlaps in fixed train/validation SAR and RGB optical sources, verify the shared target strips, and check B1F fit/inner-dev against the train-only overlap graph. The scan found 580 exact train–validation geometries and 440 B1F fit–inner-dev geometries; see the [audit report](sen1_2_train_val_spatial_overlap_audit.md) and its machine-readable run artifacts. No test data was accessed.
- [x] Retrieve the TUM SEN1-2 supporting document and upstream SHA-512 index through the read-only rsync endpoint; compare only fixed train/validation source files. The 564,768-entry index yielded zero exact matches among 3,600 local PNGs and recovered no source IDs. This excludes byte-identical copies only; see the [audit summary](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/summary.json) and [source note](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/upstream_sources.json). No imagery or test data was accessed.
- [x] Locate and pixel-crosswalk the neighboring `SOTANet/datasets/sar2opt-QXS` cache. It contains 20,000 composite PNG entries in local train/val/test folders (16,000/2,000/2,000). Of 1,800 current fixed train/validation identities, 1,626 same-name files in the cache's train/val folders match both local SAR and optical panels exactly; another 174 names appear only in the cache's test folder by filename. The test payloads were not opened. See the [reproducible crosswalk audit](../../experiments/data_audits/sar2opt_qxs_cache_crosswalk_20261019_v1/report.md).
- [x] Review the primary QXS-SAROPT paper and author repository for source grouping and split metadata. The paper describes three large SAR source images, manually selected subregions/tie points, and patch filtering but publishes no patch-to-source crosswalk or canonical full-dataset split; its 70%/test statement is for a BNN example. The public repository contains a README and project image, with a download-request link but no source-group or split manifest. This does not resolve the local source chain; see [CURRENT_STATE.md](CURRENT_STATE.md) and the [literature review](literature_review.md).
- [x] Inspect the official QXS-SAROPT maintainer issue history for source-group, raw-scene, and alternate-download routes. Issue #3 says city-to-patch labels were not made; issue #4 says release of original SAR scenes requires source-center consent; issue #13 requests a Google Drive/Hugging Face copy because the README points to Baidu. The 2026-10-24 refresh confirms the repository still exposes only `README.md` and the project image, the Releases page has no releases, and issue #13 remains open without a response. No patch map or accessible alternate mirror was found; no archive was downloaded and no contact was made. See [the current-state refresh](CURRENT_STATE.md#qxs-saropt-official-route-recheck-2026-10-24), [issue #3](https://github.com/yaoxu008/QXS-SAROPT/issues/3), [issue #4](https://github.com/yaoxu008/QXS-SAROPT/issues/4), [issue #13](https://github.com/yaoxu008/QXS-SAROPT/issues/13), and [D-2026-10-19-36](DECISION_LOG.md#d-2026-10-19-36--keep-qxs-cache-lineage-and-source-groups-unresolved).
- [x] Recheck local SOTANet Git/cache-root metadata and exact-name public routes on 2026-10-25. The cache remains ignored and unversioned; the current official repository/issues still expose no cache-specific source map or alternate archive. A bounded public exact-name search found no path-specific generator. This closes only the current route screen; see the [follow-up audit](../../experiments/literature_audits/qxs_sotanet_cache_route_followup_20261025_v1/report.md) and D-2026-10-25-22.
- [x] Recheck the primary QXS paper’s stated CC BY license and the author README’s data-request route. The official paper declares CC BY for QXS-SAROPT, but the linked form was not accessible through the available read-only route and no archive was fetched; because the local cache’s exact upstream version remains unverified, local-cache license and source-group status remain open.
- [x] Bound the downstream ReFlowSET QXS split-list lead using only the official train configuration/list and current local train/validation names. The official config points to `splits/qxs-saropt/train_eo_list.txt` and uses a numeric EO filename with a derived SAR path; the published train-list count is 16,001. Current B1 v2 local splits contain 1,600 train and 200 validation names, all in the `ROIs...` namespace, and the retained source manifest points to the sibling `SOTANet/datasets/sar2opt` candidate tree. This name-schema mismatch prevents a direct basename crosswalk and does not establish dataset lineage or image overlap. No test list, test identity, payload, or metric was accessed. This no-match conclusion is scoped to the ROI-named B1 v2 cohort; the numeric subset was checked separately below; see [CURRENT_STATE.md](CURRENT_STATE.md) and [D-2026-10-23-29](DECISION_LOG.md#d-2026-10-23-29--keep-reflowset-qxs-list-out-of-local-source-crosswalk).
- [x] Scan only the neighboring QXS-named cache's top-level train/validation PNG containers for embedded source metadata. All 18,000 files contain only `IHDR`/`IDAT`/`IEND` chunks; IDAT payload bytes were skipped without pixel decoding. No `test/` directory or JDSR fixed-test data was accessed. This closes the embedded-container-metadata route for those files, not the upstream lineage or scene-group question. See the [audit report](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/report.md).
- [x] Check filename schema and SOTANet repository history using directory entries and Git metadata only. All train/validation basenames are numeric PNG IDs with no sidecars or shared names; no QXS preparation reference appears in reachable tracked history/path names. These IDs do not encode verified source groups; see the supplemental section in the [audit report](../../experiments/data_audits/sar2opt_qxs_png_metadata_20261019_v1/report.md).
- [ ] Verify the cache's upstream archive/version, dataset license, local split semantics, and source-scene metadata. The cache is Git-ignored in the neighboring user SOTANet clone and has no dataset-specific manifest or license file. Exact matches establish pixel equivalence to this local QXS-named cache, not official QXS-SAROPT provenance. The official QXS paper declares CC BY for its release, but the local cache is not version-linked to that release. The maintainer says the release did not include city-to-patch labels and the original SAR scenes cannot be shared without source-center consent; the 2026-10-23 repository refresh found no authorized release package/group map or official mirror. The new WMoE-Net-linked Zenodo `QXS-SAROPT-SHIP` record is not linked from the original QXS repository and has no visible license/source manifest; it does not establish local lineage. The 174 train/validation names in its `test` folder are a possible source-partition risk, not verified official-test leakage. Until an authorized archive/version and verifiable source groups are recovered, keep the project data label provisional, preserve all current splits/results, and do not use current validation or B1F inner-dev for model selection or promotion.
- [x] Screen the WMoE-Net-linked QXS-SAROPT-SHIP Zenodo candidate and search the exact article title on the publisher route. The official record/abstract now verifies WMoE-Net as oriented ship-detection prior art, but full text and online-publication date remain unavailable; the separate dataset record still has no visible license/source groups and is not tied to the original QXS release or local cache. Do not download/adopt it without lineage and rights. See the [updated screen report](../../experiments/literature_audits/wmoenet_qxs_ship_source_screen_20261024_v1/report.md).
- [ ] After authoritative source identity and source-group metadata are recovered, map permitted non-test train/validation IDs to source paths and scene groups appropriate to that family. The 1,626 exact cache matches recover same-name files and the sibling cache's local partition labels, but not parent scenes/acquisition groups; the 174 cache-test names remain unverified at pixel level. If SEN1-2 is established, recover ROI/season and `s1_i`/`s2_i` IDs using the [TUM supporting document](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt); if another dataset is established, use its authoritative identifiers. The TUM byte-hash crosswalk found no exact matches. A targeted local-history check found `MMCANet/src/data_prepare.py` was first tracked empty, while `scripts/prepare_dev2k_formal_split.py` only hashes and partitions numeric filenames. The third-party SIFT pair-list lead also has no crosswalk to these filenames. Only after mapping should a prospective split be assigned by verified source groups and checked for spatial overlap; current numeric IDs and overlap-connected components are not source groups.
- [x] Add and verify the fail-closed native-run sealing step that binds completed trainer outputs to the frozen B1 protocol and writes the selected-best-epoch validation artifacts required by the source-group aggregator. Candidate sealing also verifies the per-seed calibration record and binds it to the train-only manifest, first train identities, configured weight, and same-seed SAR-only checkpoint. Synthetic tests cover both roles and a six-run handoff through the aggregator (23 focused sealing/aggregation tests passed on 2026-10-22). The latest full suite reported 234 passed and one failure: the historical source-hash check differs from current untracked source files; see [CURRENT_STATE.md](CURRENT_STATE.md).
- [x] Freeze and verify the versioned v2 evaluator contract for the frozen v1 undefined-median failure: require mean TBR on all 200 identities, represent ancillary median TBR availability and coverage explicitly, and preserve the original strict score, required metrics, thresholds, and mean-based gate. The v2 protocol, trainers, sealer, aggregator, and six-run synthetic handoff now enforce this policy; v1 remains unchanged. See [D-2026-10-22-16](DECISION_LOG.md#d-2026-10-22-16--freeze-the-b1-v2-optional-median-policy) and the [v2 protocol](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.md). B2/B3 remain gated pending the complete v2 screen.

## P0 — Close the validation diagnostic analysis

- [x] Inspect paired per-image metric deltas for matched vs SAR-only and matched vs permuted using only saved CSV rows. Report distributions, quantiles, sign counts, and fixed-mask metric directions; state these are conditional on one failed-screen checkpoint.
- [x] Descriptively stratify by SAR-only baseline PSNR quartile without using the result to select a model, threshold, subset, or paper figure.
- [x] Cross-check metric directions against `Early-FusionCascadeJDSR/eval_metrics.py`; retain the noisy pseudo-target limitation.
- [x] Record the result in `optical_information_analysis.md` and `EXPERIMENT_REGISTRY.md`. No gate changes.
- [x] Compare the saved all-zero-optical output directly with the same checkpoint's SAR-only decoder base, then audit the matcher/fusion/loss path. All eight paired metric contrasts regress under zero optical; code has no explicit reject state or inference bypass. This is conditional evidence from the existing one-seed diagnostic, not a causal finding or a new experiment.
- [x] Test the B1F nine-offset matcher-entropy confidence as a target-free correlate of saved matched-optical utility on the same 200 validation identities. The fixed all-eight-positive rule failed; do not carry this cue forward as a global utility/rejection signal. No metric recomputation, training, selection, or test access; see the [run artifact](../../experiments/dev2k_formal/phase_b1f_matcher_entropy_utility_20261018_v1/) and D-2026-10-18-36.

## P0 — B1F failure analysis

- [x] Summarize the saved 10-epoch inner-dev trajectory without changing selection: all epochs failed the strict SSIM and TBR gates, including the epoch with peak PSNR. Record that this rules out checkpoint choice as the sole sufficient explanation but does not identify a causal root cause.
- [x] Check saved attention-diagnostic ranges as descriptive evidence only; do not interpret softmax entropy/offsets as reliability or a causal mechanism.
- [x] Preserve `stop_at_b1f`; do not select a different epoch, rerun B1F, or change its gate.
- [x] Inventory the local `phase_b1f_minimal_review_2026-10-16.tar.gz` by member names and read its review manifest. It is a dated B1/B1R/B1F evidence snapshot, not a SEN1-2 source archive; its manifest excludes raw data/checkpoints/test artifacts and predates the 2026-10-19 spatial-overlap audit. The saved metric CSVs were not reopened; see [CURRENT_STATE.md](CURRENT_STATE.md).

## P0 — Evaluation metric audit

- [x] Trace exact-set/lineage, range, dtype, shape, mask, and metric code paths in `eval_metrics.py`.
- [x] Fix silent non-finite metric omission in aggregate/bootstrap summaries; exact-match `+∞` PSNR now fails aggregation explicitly.
- [x] Correct high-frequency radius help/manifest wording without changing the calculation or historical metric values.
- [x] Add regression coverage for non-finite aggregation and run evaluator, pairing-diagnostic tests, and Python compilation.
- [x] Add synthetic known-case checks for empty/degenerate masks, zero reference denominators, and target-over-prediction ratio convention.
- [x] Recompute all four fixed masks for the 200 validation targets, compare stored arrays and hashes, quantify percentile-tie expansion, and check mask intersections.
- [x] Audit the six B1 seed/version validation output sets for row/prediction counts, exact target matches, non-finite PSNR, missing targets, shape, and available manifest lineage.
- [x] Distinguish ratio-image `target/prediction` from B1's prediction-TBR/target-TBR ratio; confirm the B1 evaluator does not report ratio-image statistics.
- [x] Audit Phase-1 R1 and S-R1b three-seed manifests, fixed validation/split hashes, crop/mask protocol, source hashes, retained aggregate PSNR, and available test-use metadata without opening test data.
- [x] Search the worktree and available local archives for Phase-1 R1/S-R1b per-image outputs and their matching evaluator source. All six per-image directories are empty; the pilot archive has no matching outputs and its evaluator copy hash differs. Exact-match counts remain unverified from aggregate-only artifacts.
- [ ] Search another authorized archive only if one becomes available; retain historical R/S outputs and gates and do not access test arrays or measurements.

- [x] Audit the Phase-A2 fixed validation masks and saved per-image/summary/paired arithmetic. All 18 groups contain the exact 200 validation names; 652 summary metric cells and 486 paired mean/bootstrap fields reproduce from the saved rows. This checks recorded arithmetic and mask artifacts only: 16 of 18 prediction sets are absent, and the manifests omit evaluator hashes, so an independent historical metric re-evaluation is not possible.
- [x] Audit Phase-B1's three-seed evaluation/evaluation_v2 manifests and evaluator source lineage. All six manifests use the same fixed validation-manifest hash and record 200 validation outputs at 256×256 with test flags false. The earlier `evaluation` commands omit the high-frequency radius; `evaluation_v2` records 0.25 explicitly. The current entrypoint defaults to 0.25 and sets evaluation crop to zero, while the training config's 64-pixel crop is for training. However, the entrypoint is untracked in the current checkout, `eval_metrics.py` is modified, and the run records a dirty tree without a source snapshot or evaluator hash. Thus the historical evaluator/crop/mask implementation cannot be independently attested from current source; this does not establish a metric defect or change B1's saved gate result.
- [x] Source-review paired multi-method aggregation. It pairs sorted sample identities and rejects duplicate rows/non-finite metrics, but only reports image-iid bootstrap intervals and trusts the matrix producer for expected count, split/input/mask lineage, and test-access status; see [evaluation audit](evaluation_audit.md#paired-multi-method-aggregation-source-review-2026-10-19).
- [x] Implement the source-valid B1 aggregator: it checks exact expected validation identities/order and split/list hashes, binds the frozen `files.csv` digest (whose rows contain SAR/optical, LR, target, and mask payload hashes), pairs all three seeds, uses the authoritative 15-group map, and reports per-image deltas plus source-group bootstrap intervals. Synthetic fixtures cover these checks; no complete native run has been sealed or aggregated. The seed-0 attempt stopped before sealing on an evaluator-contract error. See the [aggregation contract](../../protocols/sen12_scene_dev1800_v2/b1_aggregation_contract_v1.md).
- [x] Extend the source-group aggregator with per-seed per-image median and sample SD, overall median and sample SD across the three seed means, and Markdown tables for all paired metrics and per-seed image distributions. The synthetic regression test passed; the mean-based gate and source-group bootstrap calculations are unchanged. No model metrics were generated.
- [ ] Continue evaluator-version/crop/mask audits for other selected validation-only reports if an authorized source snapshot becomes available. The Phase-1 R/S recorded evaluator source remains unavailable in the current worktree/Git history; keep Phase-A2 labeled arithmetic-checked but evaluator-version-unverified unless its missing prediction artifacts are recovered from an already-authorized source.
- [x] Specify a prospective input-conditioned high-return proxy region for future fixed-input comparisons. Construction does not read HR target values, but the LR input is synthetically derived from the target, so this is not a statistically target-independent ROI or a physical sparse point-scatterer ROI. Historical masks, metrics, and gates remain unchanged. See [protocol](bright_return_roi_protocol.md).
- [ ] Establish a physical point-target evaluation only if independently identified target locations and an appropriate SAR product become available; current normalized PNGs do not establish these inputs.
- [x] Replace the split-metadata test's `FixedDev2kDataset("fixed_test_dev2k", ...)` construction with metadata-only checks and an `np.load` guard after discovering that three full-suite runs loaded the first test sample arrays. Record the access incident without changing historical experiment flags.
- [x] Scan remaining test code for fixed-test array readers and rerun the suite after confirming the metadata-only boundary. The other dataset test reads train/validation only, test-path evaluator tests reject before dataset construction, and the guarded suite passed (200 passed, 2 warnings) without fixed-test array loads.
- [x] Disclose that a later broad repository text search traversed `fixed_test_dev2k/files.csv` and historical report HTML. Its truncated output included a report excerpt, so exact fixed-test text exposure is unknown; no arrays were loaded and no value was used in a decision. Exclude these artifacts from future source searches.
- [x] Disclose that a 2026-10-19 repository inventory search included the legacy pilot manuscript and surfaced aggregate held-out result statements in its source text. No test images, arrays, per-image metric files, aggregate metric files, or fixed-test CSVs were opened in that command; the surfaced statements were not used in a research decision. Future source inventories should exclude result-bearing manuscript text when the no-test-measurement boundary applies.
- [x] Disclose that a 2026-10-19 status-log tail also surfaced a historical held-out aggregate-summary paragraph. No underlying test images, arrays, per-image CSV, or aggregate metrics file was opened, and the surfaced values were excluded from this turn's source-provenance assessment. Future state-log reads should target only the current section needed and avoid historical result summaries when the no-test-measurement boundary applies.

## P0 — Historical B1 source snapshot integrity

- [x] Reproduce the historical B1 source-hash test failure and identify the four mismatching current source paths without changing the recorded hash manifest or artifacts. Check the public GitHub repository's branches and per-path commit history read-only: only `main` at the recorded commit exists, and none of the four paths has a remote commit history. The current checkout cannot attest to the original run source; recovery remains contingent on an authorized exact archive.

## P1 — Complete first-pass literature and data context
- [x] Scan only the frozen dev2k train/validation SAR and optical PNG containers for embedded provenance metadata. All 3,600 files contain only `IHDR/IDAT/IEND`; no ancillary metadata was found and no pixel/test payload was read. This closes the embedded-container route, not upstream source, acquisition, calibration, or target provenance. See the [audit](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/report.md).
- [x] Review Bertrand et al.'s 2025 MEOW SAR–optical registration study and author repositories. Record its multiseason dataset, registration-only task, controlled blur/speckle protocol, split-direction inconsistency, and limits for SAR restoration and local cohort alignment. The pinned wrapper implementation was also inspected: Gamma(shape=L, rate=L) is transformed with `20 log10(sample)` before SAR clipping and normalization; this does not establish physical fidelity or paper-table lineage. Do not adopt its data or treat its matching-rate gains as restoration evidence; see [literature review](literature_review.md#optical-sar-registration-literature-and-scope-boundary).

- [x] Re-parse the related-work matrix after the 2026-10-22 Mendeley page recheck. Fix the pre-existing unquoted comma in the SORF evidence cell; all 85 data rows now parse against the 12-column schema.

- [x] Review Hänsch et al.'s 2026 SpaceNet 9 full paper and official challenge page. Record the tie-point-based, spatially varying optical–SAR registration benchmark and its limits for this SAR-restoration study; no dataset payloads were downloaded and no local residual was inferred. See [literature review](literature_review.md#optical-sar-registration-literature-and-scope-boundary).
- [x] Resolve SpaceNet 9's dataset-level rights statement and accessible split: the official challenge page states CC BY-SA 4.0; the official README lists three labeled development triples and keeps public-test tie-point labels/private-AOI files hidden. Do not use the three training scenes as held-out validation or start a dataset run now. See [D-2026-10-25-135](DECISION_LOG.md#d-2026-10-25-135--keep-spacenet-9-as-registration-prior-art-not-held-out-validation) and the [audit](../../experiments/literature_audits/spacenet9_rights_split_screen_20261025_v1/report.md).

- [x] Screen the author-published OSdataset as an alternative data candidate. Its repository describes 20 GF-3/Google Earth parent scene pairs and 512/256-pixel registered crops, but stored SAR is normalized and `imadjust`-enhanced 8-bit amplitude; no machine-readable crop-to-scene map, acquisition dates, or grouped split is exposed, and the Google Earth source terms are unresolved. No files were downloaded or opened. Keep it unadopted; see the [dataset review](literature_review.md#osdataset-alternative-sar-optical-source-candidate-2020).
- [ ] Before downloading or adopting OSdataset, establish an authoritative right to redistribute/use its Google Earth-derived patches; the author README's CC-BY-NC label does not resolve Google's additional redistribution/copying restrictions. Then verify archive/version, recover a crop-to-parent-scene map, and audit overlap, acquisition timing, registration residuals, stored-pixel semantics, and split lists using authorized non-test material only. Treat it at most as a representation-domain controlled-degradation candidate unless calibrated source data and an independent SAR reference are established.
- [x] Recheck Ma et al.'s full 2026 flood-mapping preprint against its newly identified official repository and data links. The paper describes a 104-cluster spatial/temporal S1 split and a ±2-pixel loss on S1 flood predictions versus S2-derived labels; it separately trains a SAR-only CVAE. It also says S2 patch/stride was selected using a held-out test benchmark. The repository's documented Zenodo download covers S2+PRISM only; S1/CVAE data terms remain unverified. Nested source directories and the Zenodo record were inaccessible; no archive, split identities, or local test data were opened.
- [x] Review the full SARGANet 2026 publisher article as adjacent SAR-SR prior art. Record its simulated 10 m→5 m target ambiguity, unspecified scene-group split, test-subset loss-weight selection, dual-polarization/metric reporting conflicts, and lack of an article code link. No dataset archive, split identities, or local test artifacts were accessed; see the [review](literature_review.md#sar-super-resolution-and-joint-restoration) and [decision D-2026-10-19-24](DECISION_LOG.md#d-2026-10-19-24--keep-sarganet-as-adjacent-sar-sr-prior-art).
- [x] Review Ahn et al.'s full 2025 Capella SAR-SR paper. Record the real SAR source but bicubic-synthetic LR, patch-after-extraction split, missing source-group/polarization/product details, and lack of optical or joint-despeckling evaluation. The publisher PDF is available; no dataset imagery, code, split identities, or local test artifacts were accessed. See the [review](literature_review.md#sar-super-resolution-and-joint-restoration) and [decision D-2026-10-19-25](DECISION_LOG.md#d-2026-10-19-25--classify-capella-sr-as-real-source-synthetic-degradation).
- [ ] Inspect Ma et al.'s implementation or S1/CVAE data only if direct source files and a verifiable dataset record become accessible. Do not treat the MIT code license as dataset permission or use this downstream flood-mapping resource as a restoration benchmark without verifying target, radiometric domain, registration, and data terms.



- [x] Review Zhu et al.'s 2025 Ka-SAR integrated speckle-suppression/SR author manuscript. Record the ×4 Ka-band SAR-only inference path, optical-example screening and terrain-dependent feature similarity, and the lack of pixelwise HR evidence for real strip-map results; this does not change local experiment gates.
- [x] Complete a standalone HSFMamba full-text audit. The manuscript confirms direct inference-time optical-guided joint SAR SR/denoising, BM3D-derived SAR references, bicubic plus Gaussian/Gamma noise, CFSM/FSAM/RSSL architecture and author-reported ablations. It leaves the Gamma equation/domain, independently auditable scene split, quantitative residual registration, mismatch fallback, implementation and dataset terms unresolved. The haze example is qualitative and adaptive optical-quality perception is future work. Preserve the distinction between author-reported OGSSR results and local evidence; see the [report](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/hsfmamba_fulltext_audit_20261024_v1/access_manifest.json). No local experiment is implied.
- [x] Recheck DSRKD through the author research page, IEEE record, and ResearchGate record. IEEE article extraction is blocked and the ResearchGate page requires a full-text request; the dataset-source discrepancy remains unresolved.
- [x] Run a fresh exact-title/DOI search and recheck the current author publication page for a public manuscript or code link; none was located. Do not infer the missing protocol from secondary descriptions.
- [ ] Obtain an authorized copy of the DSRKD full text before making dataset/training-protocol claims. A further 2026-10-22 exact-title/DOI and author-domain search found no primary manuscript/code route; the official author homepage still links the paper to IEEE Xplore only, while ResearchGate offers a full-text request. The conflict between the abstract's synthetic/SAR wording and HSFMamba's natural-image description remains unresolved. Do not repeat these routes without a new access path.
- [x] Screen the newly surfaced OGSFlow repository (2026-10-24): its README provides a code tree, method summary, OGSRL archive route, and a CC BY-NC 4.0 claim. The paper citation is still forthcoming; linked dataset/evaluation/training files were not retrievable, the archive was not opened, and OGSSR terms remain unresolved. Keep this as a README-level prior-art lead. See the [audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md).
- [x] Verify bibliographic metadata and source links for OGSRN, OMA-SSR, HSFMamba, and the OS_SR preprint. HSFMamba’s author-linked dataset/resource route is audited above; model code remains unverified. Code access for the other methods remains unverified.
- [ ] Verify CN122694669A's publication number, applicant, legal status, full claims, and publication date against the official CNIPA record before relying on it in the manuscript. The 2026-10-19 search found only a secondary patent-text index; the official CNIPA query portal did not expose the specific record, and exact-number searches across CNIPA/WIPO and Google Patents returned no item-level record. A bounded WIPO web-index query on 2026-10-22 surfaced unrelated records only. On 2026-10-25, an exact-number search restricted to the CNIPA domain returned only general lookup guidance, while direct access to `epub.cnipa.gov.cn` timed out in the available web tool; no item record or claims were retrieved. Treat this as a novelty lead, not efficacy evidence, and stop repeating these routes until an official portal/API or item-level source is reachable.
- [x] Retrieve and review the full MLOG v1 preprint through a public full-text copy matching the Research Square title/authors/DOI. Record its SEN1-2 subset, BM3D-prefiltered bicubic SR protocol, metrics, 10-image test set, and SAR-derived pseudo-optical inference input; it does not establish joint despeckling/SR or controlled misregistration robustness.
- [x] Recheck the 2025 SPIE/*Journal of Applied Remote Sensing* DOI and publisher full-text routes (DOI `10.1117/1.JRS.19.026506`); both were inaccessible through the current web interface. J-GLOBAL displays a JST machine-translated abstract naming MLOG and OWG/MLN/RMB, but this is secondary evidence and does not establish version lineage or protocol.
- [ ] Obtain the primary 2025 article or an author/publisher full-text copy only if a new public route appears. A bounded 2026-10-19 check of the Shenzhen University author profile and ResearchGate article record surfaced no manuscript; ResearchGate states no full text is available, and the publisher/DOI and Crossref routes could not be fetched. Do not submit an access request or repeat these routes without new evidence. If a copy becomes reachable, verify the method, data, degradation, splits, metrics, inference inputs, registration-error tests, and exact relationship (if any) to Research Square MLOG v1; do not transfer preprint results before then.
- [x] Complete a bounded primary-source update of SAR diffusion/despeckling and SAR SR diffusion through 2026-10-18; record method/task boundaries, fidelity risks, acceleration claims, and code/full-text limits in `literature_review.md`, `diffusion_feasibility.md`, and `related_work_matrix.csv`. This is a scoped first pass, not an exhaustive review.
- [x] Recheck SharpSAR at the official journal page and its issue-page HTML link. Record Tables 1–4 and the metric tradeoffs; the full HTML route returned a cache miss and the article body/protocol remain unreviewed. The abstract's 8.12% aggregate-improvement basis is unspecified. No local data, test measurements, or model experiments were accessed; this does not change B1/B2.
- [x] Recheck the newly readable official article record on 2026-10-26. It exposes the abstract and indexed tables, but the `RichHTML` and `PDF (PC)` controls returned the same page in the available reader; no interactive browser was available. Keep the method and protocol unresolved; see [D-2026-10-26-039](DECISION_LOG.md#d-2026-10-26-039--keep-sharpsar-at-abstract-and-indexed-table-evidence) and the [route audit](../../experiments/literature_audits/sharpsar_fulltext_route_20261026_v1/report.md).
- [x] Recheck the official `RichHTML`/`PDF (PC)` controls and embedded frame through the web reader on 2026-10-27. Both controls returned the same article-record page; the embedded request returned `Cache miss`, and no manuscript body or usable PDF was exposed. The full protocol remains unresolved; see the [route audit addendum](../../experiments/literature_audits/sharpsar_fulltext_route_20261026_v1/report.md#addendum--2026-10-27-web-reader-recheck).
- [x] Check SciOpen's article page and bounded exact-title/DOI, Nankai, author, and GitHub routes. The publisher-platform page shows a `PDF (2.6 MB)` control but the text reader exposes no actionable PDF link/body; the linked journal HTML route returns `Cache miss`, and no author-controlled manuscript surfaced. Keep all protocol details unresolved; see the [route audit addendum](../../experiments/literature_audits/sharpsar_fulltext_route_20261026_v1/report.md#addendum--2026-10-27-sciopen-and-author-route-check).
- [ ] Review SharpSAR's full method and protocol only if a new public publisher/author route becomes accessible. The official issue-page HTML link returned a cache miss; do not infer datasets, splits, or degradation details from the abstract/tables alone.
- [x] Complete a bounded primary-source pass on registration-error sensitivity, misalignment-aware SAR-optical detection, matcher deployment sensitivity, and uncertainty-based SAR-optical correspondence confidence. Record that these do not test restoration-image fidelity.
- [x] Extend that bounded source pass to the 2026 SMSDNet abstract, title/metadata for a SAR-only joint diffusion method, GUSO/FHReg registration study, and publisher abstract for quality-aware SAR despeckling fusion; keep each task boundary explicit. The GUSO author README was rechecked on 2026-10-23: downloads require a form, dataset terms are academic-only/noncommercial and require prior written consent for derivative development, while GitHub's MIT label is for code. No form was submitted or data downloaded; keep GUSO as registration-only prior art, with paper protocol, source-group splits, archive contents, and provider terms unverified.
- [x] Review the MambaDIP full author manuscript and IEEE record. Record its SAR-only per-image despeckling/Retinex-enhancement task, 8-bit HRSID and synthetic aerial evidence, Gamma-MAP/ENL-M method, and explicit placement of SR in future work; no code link was identified in the manuscript. It is not a matched joint-SR baseline and does not change B1/B2.
- [x] Review the primary 2026 ReFlowSET and ContraFM-S2O arXiv manuscripts for the flow-matching route. Both are SAR-to-EO generation, not optical-guided SAR restoration; record their task, architecture, QXS split descriptions, and limits in the [literature review](literature_review.md#sar-to-eo-flow-matching-adjacent-reverse-direction-work) and [diffusion feasibility note](diffusion_feasibility.md#flow-matching-work-in-the-reverse-sar-to-eo-direction). No local data or model experiment was run.
- [x] Recheck Onat's publisher-linked PLFSID route after the landing page became retrievable. The publisher record dates acceptance to 2026-06-09 and online publication to 2026-06-14. The accessible README says `main.m` is demo-only, algorithm/fusion scripts are withheld, and an academic-use license is planned; the listed file/raw routes returned cache misses and the publisher full text returned 403. A targeted search surfaced the official issue TOC's Full Text/PDF listing, but direct TOC and DOI-PDF routes were inaccessible; exact-title/author/institution/ORCID/arXiv searches found no alternate full copy. No code or raw data was inspected. This closes the currently available route set only; see the [route audit](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/report.md) and [access manifest](../../experiments/literature_audits/onat_quality_fusion_route_update_20261024_v1/access_manifest.json).
- [ ] Revisit Onat's full method only if the full article, actual algorithm source, or effective version-specific reuse terms become accessible. Then verify quality-map and pseudo-reference construction, inference inputs, datasets/splits, mismatch protocol, and code/data-to-table lineage. Do not repeat the same inaccessible routes, infer these details from the abstract, submit an access request, or download the repository's raw images under the current provisional terms.
- [x] Recheck the primary 2025 SAG-CR article on supervised alignment and injected SAR offsets for SAR-assisted optical cloud removal. Publisher-indexed text confirms 1/3/5/7-pixel shifts; direct article retrieval returns HTTP 403 and the linked GitHub repository currently reports empty, so implementation/table lineage is unavailable. Its task boundary is recorded in the literature review.
- [x] Review and follow up the 2026 CGD-Net publisher abstract and author repository. The architecture note clarifies that the reliability module combines cloud-mask/prior reliability for optical-reflectance output; it does not implement SAR-output fallback. The README split counts are 10,826/1,443/2,165, but identities were not inspected. Exact stress magnitudes/results and individual code/config/test files remain unavailable. No code/data was executed or accessed; see the [source follow-up](../../experiments/literature_audits/cgdnet_architecture_source_followup_20261024_v1/report.md).
- [x] Recheck DSRKD through the current author homepage, direct IEEE Xplore, author GitHub/title searches, and DBLP open-access metadata pointers. The author page provides only an architecture summary, IEEE is JavaScript-gated, and no official code/full text surfaced; the dataset/training conflict remains unresolved. See the [route audit](../../experiments/literature_audits/dsrkd_access_route_recheck_20261023_v1/report.md). Do not repeat these routes without a new source.
- [x] Review the full author manuscript for the 2026 SAR-only joint diffusion method; architecture, datasets, ×4 setup, metrics, adaptation budget, and limitations are recorded. Code and exact degradation/group-split protocol remain unverified.
- [x] Identify the 2026 Cao et al. SAR-only joint-diffusion article from an abstract-level source. The abstract describes a Transformer denoiser followed by diffusion SR on ×4 FAIR-CSAR-V1.0 and calls the cascade jointly optimized; no performance claim or protocol inference was carried forward.
- [ ] Review Cao et al.'s primary full text if it becomes publicly retrievable. Establish whether the cascade is jointly trained or sequential, identify the exact FAIR-CSAR representation, target/degradation, scene split, metrics, code, and runtime; do not use abstract-only results as a baseline or claim SAR-physics validity. The publisher route was blocked and the accessible record offers no full text; a 2026-10-19 exact-title/DOI/arXiv/GitHub search found only the existing ResearchGate abstract, which still states no full text is available. Do not request access from the authors or repeat these routes without a new access clue.
- [x] Review the full EUSAR 2024 SAR SR/despeckling article and linked repository/config. Record its subaperture degradation, noisy-to-denoised supervision, one-scene patch validation, and task boundary; the paper's 100-epoch SR description differs from the repository's current 10-epoch base config. Do not attribute the reported result to that config or attempt reproduction until the code/run version is reconciled.
- [x] Check Lee and Son's 2026 MONTI paper as direct evidence on SAR-SR restoration sequencing. The manuscript describes structure restoration followed by latent-diffusion speckle synthesis and a comparison with single-stage baselines using separate FWD structure/speckle metrics; no reversed-order or matched joint-restoration comparison appears in the targeted text reviewed here. The official workshop site confirms accepted, non-archival status. The full split/training protocol and code remain unverified; no local data or experiments were accessed. See [literature review](literature_review.md#diffusion-beyond-denoising), [research gaps](research_gaps.md), and [H8](research_hypotheses.md).
- [x] Refine H8's comparison design by separating task order from training coupling: stage-wise and end-to-end cascades in both `D→SR` and `SR→D` orders, plus a direct-joint control. Added an operator-level derivation and matched-protocol requirements; this is conceptual work only, with no experiment or B1/B2 gate change. See [restoration-order analysis](restoration_order_analysis.md) and [H8](research_hypotheses.md).
- [x] Complete a second targeted primary-source review for direct restoration behavior under controlled residual shifts and optical/SAR negative transfer. The 2026-10-18 pass reviewed Ma et al.’s shift-invariant S1 flood-mapping preprint but found no optical-conditioned SAR-output restoration under controlled post-registration shifts; the search remains bounded, not exhaustive.
- [x] Add the OMA-SSR full-text protocol and its unresolved test-set-loss wording to the review and risk register.
- [x] Check upstream SEN1-2 patch-overlap and registration context against a primary source; local parent-patch mapping remains unresolved.
- [x] Audit the SOMA AAAI record, full arXiv manuscript, current public `main` loader, train/test transform use, and all six reachable Git commits. The only code-bearing commit (`1003432`, 2025-11-17) adds the loader and train/test scripts with ±50 px/±20° and fixed scale 1.0; later commits change only the README. No earlier matching loader/config exists in the reachable history.
- [ ] Establish the exact run revision, transform config, and RNG/validation protocol used for each SOMA comparison and ablation table. The paper does not provide run-specific manifests, and the tracked repository has no transform config or checkpoint-to-table mapping. Until author-side provenance is available, do not use the public code as a reproducible implementation of the main-comparison setting.
- [x] Check two 2026 adjacent primary sources: SAMDFuse for degradation-aware registered SAR–optical image fusion and SORF for a joint registration/fusion preprint; record their task boundary and code/full-text availability limits. Rechecked SORF on 2026-10-21: the indexed SSRN author abstract remains accessible while direct page retrieval returns 403; the linked repository still has only a README and two commits, with code/model release pending.
- [x] Review the 2026 SASRF publisher article for real-world cross-sensor SR under residual geometry; record that it uses optical PMS/WFV pairs and manual test-landmark alignment metrics, not SAR output or an injected shift sweep.
- [x] Review the primary VGSRF-Net article for Retinex-visible-prior SAR feature reconstruction and synthetic speckle robustness; record that it evaluates a fused output on well-registered inputs, not SAR restoration fidelity under spatial mismatch.
- [x] Review the 2026 WEMFusion publisher article for discrepancy-aware SAR–optical feature fusion; record its co-registration assumption and distinguish its fused-image output from SAR restoration.
- [x] Review the full author-uploaded 2015 Optical-Driven Nonlocal SAR Despeckling article alongside the 2019 GNLM manuscript/code. Record the 2015 multiscale-ENL sigmoid blend of optical-guided bilateral predictor weighting and SAR-BM3D, its one-scene real-data evaluation, and the absence of controlled spatial-shift or joint-SR evidence; retain GNLM's SAR-domain predictor rejection and qualitative stale-guide example.
- [x] Add Amieva et al.'s 2026 same-sensor Sentinel-1 IW-to-SM joint SR/despeckling study as SAR-only prior art. Record indexed full-text protocol/results only with attribution, its three-image test set, and its task distinction from optical guidance; the repository PDF could not be directly retrieved through the current web tool.
- [x] Recheck Amieva et al.'s methods and result tables through the newly surfaced official institutional search-index text. The indexed published-version excerpts cover sections 3.3–3.6 and 5 and Tables 2–8, including the three-image test split, the pre-split full-image-set percentiles, ESRT/training/loss comparisons, and MERLIN+bicubic metrics. Direct PDF opening still times out, so this is not a visual/full-byte PDF review; exact input/reference statistic pooling, independent geographic source-group crosswalk, registration residuals, and code lineage remain unresolved. See the [audit](../../experiments/literature_audits/amieva_indexed_fulltext_update_20261024_v1/report.md) and [D-2026-10-24-10](DECISION_LOG.md#d-2026-10-24-10--update-amieva-evidence-to-indexed-method-and-results-text).
- [x] Identify the 2025 GCOANet publisher abstract as optical-assisted SAR despeckling with conditional-diffusion reference generation; the official indexed abstract names non-local filtering, interval downsampling, and an all-directional gradient loss, and reports real/simulated experiments plus data availability on request. It says references are generated during training and test but does not specify test-time generator inputs, so optical use at inference remains unresolved.
- [x] Recheck GCOANet full-text availability. On 2026-10-28 the official publisher article route still returned HTTP 403; the ResearchGate author-copy page offered a request rather than a full text, and a Wuhan University domain query surfaced no author/institution manuscript. This adds no method detail; no data/code were retrieved. Do not repeat these routes unless a new full-text link appears.
- [ ] Review GCOANet's datasets, target/noise domain, registration protocol, metrics, ablations, runtime, and code if an authorized full-text copy becomes retrievable. The 2026-10-23 official index search added no protocol detail beyond the abstract. Bounded publisher/GitHub/Mendeley/Zenodo searches also found no matching primary dataset or code release for a separate non-author community-list claim of 30,000 GF-3/Google Earth/S1/S2 patches; do not treat that lead as verified. Keep the protocol fields unresolved and do not repeat blocked routes without a newly available author or institutional copy.
- [x] Refresh the bounded primary-source search for SAR-output restoration under residual optical mismatch. The 2026-10-18 search returned SAR-to-optical translation, registration work, OGSRN training-time optical feedback, and a metadata-only TGRS cloud-removal lead (Xu et al.; full text/protocol unavailable). No accessible primary source established quantitative SAR-output fidelity under controlled/measured residual shifts with fallback. Search remains non-exhaustive and does not establish a novelty gap.
- [x] Review the full arXiv manuscript for QDFNet (arXiv:2512.22447). Record learned-token magnitude/direction quality scoring and orthogonalized reliability-weighted fusion for optical–SAR object detection under whole-modality dropout; its reviewed experiments do not test controlled spatial offsets or SAR-output fidelity. Treat generic quality-aware feature fusion as adjacent prior art, not evidence of a restoration fallback mechanism.
- [x] Continue the bounded targeted search for quantitative SAR-output restoration under controlled/measured residual registration error, especially joint SR, and reliability/fallback in that task. The 2026-10-22 continuation refined SAG-CR and added CGD-Net; both reconstruct optical imagery. No direct task-matched study with residual shifts, SAR-fidelity metrics, and SAR-only fallback was located in this bounded pass; it does not establish a novelty gap.
- [x] Review primary joint-registration/fusion papers for misaligned IR/VIS (IMF, IEEE TCSVT 2024; HMMF, AAAI 2026). Both provide adjacent progressive/joint registration-fusion designs and synthetic deformation stress protocols, but evaluate fused IR/VIS outputs rather than SAR restoration fidelity or SAR-only fallback. Keep modules as deferred literature candidates; no experiment gate changes.
- [x] Review the 2026 GFS-Net publisher abstract and publisher-linked repository as adjacent optical–SAR geometry/fusion prior art. The abstract reports deformable alignment, amplitude–phase fusion, and gated recalibration for land-cover classification; the full article was inaccessible and the code/model link returned 404, so no restoration, controlled-shift, or fallback claim is inferred.
- [x] Inspect only the fixed train/validation manifests and their generation code for provenance fields. They record source-file hashes, shapes/dtypes/modes, normalization/degradation settings, and split hashes; all 3,600 train/validation source PNG hashes still match. The builder schema contains no parent scene, tile, original-patch, acquisition-time, product-level, or geospatial identifiers.
- [x] Read the accessible 2018 SEN1-2 author manuscript for source grouping, acquisition/product, and export preprocessing details. It describes 258 scenes/ROIs, four ROI-season groups, 256×256 patches at stride 128, Sentinel-1 VV IW GRD σ⁰ in dB without speckle filtering, and Sentinel-2 L1C RGB bands 4/3/2 with per-band clipping/scaling/contrast stretch. It says scene/season splits are possible but gives no exact pair timestamps or local-ID mapping. The linked TUM record returned HTTP 429 on 2026-10-18; keep local timing, radiometry, and grouping unresolved. A 2026-10-19 recheck made the record page accessible and added corpus-level archive/license metadata, but did not recover local IDs or per-pair provenance; see the updated source audit below.
- [x] Run a bounded exact-title/GitHub search for SEN1-2 source mapping. It surfaced a third-party 64×64 SIFT keypoint-matching derivative with original ROI/subfolder/patch path strings, but no mapping to this repository's numeric PNG IDs; treat it as a search lead, not an upstream crosswalk ([repository README](https://github.com/YuChuang1205/SEN1-2-patch-dataset)). No archive, imagery, or project test artifacts were accessed.
- [x] Inspect only PNG container chunk metadata for the 3,600 SAR/optical source PNGs referenced by fixed train/validation manifests. No ancillary chunks or embedded text keys were present; IDAT image payloads were skipped and no fixed-test path was opened, so this local metadata route does not recover source IDs.
- [x] Trace the canonical fixed-degradation chain using code and train/validation manifests/CSV metadata only. The 256→128 protocol applies a 63-bin symmetric FFT passband after requesting 64 bins, then bilinear resize, negative clamp, looks-2 Gamma multiplication in the representation domain, and no output clamp. A synthetic response check confirms that bins 32–64 are suppressed; fixed CSV metadata show every train/validation LR sample exceeds 1 after the unclipped multiplier (median above-one pixel fractions 18.13%/18.20%). The existing data-free degradation/model suite passed (35 tests). This is a controlled representation-domain composite-noise protocol, not calibrated Gamma speckle; see [D-2026-10-18-27](DECISION_LOG.md).
- [x] Declare a prospective cutoff/range policy in [D-2026-10-18-35](DECISION_LOG.md): retain historical 63-bin tensors unchanged; use a 129-bin symmetric `±64` support for a future standard ×2 matrix; apply the current representation-domain Gamma multiplier without post-noise clipping and allow shared LR values above 1. The original cutoff intent remains undocumented; the policy does not authorize an experiment.
- [x] Cross-check the frozen B1 v2 degradation config against the exact run-bound train/validation input metadata. Both configs use 129 bins and match `input_binding.json`; all six child runs bind to the same fixed manifest hashes. The 2026-10-21 archive is a distinct preparation record despite matching config and file-list hashes. Keep the CLI default 64 for historical replay and explicitly pass 129 for future v2 inputs; no code or gate change. See the [audit](../../experiments/data_audits/sen12_v2_degradation_contract_crosscheck_20261024_v1/report.md).
- [ ] Before any new degradation matrix or model experiment, resolve source/target provenance and decide whether the study remains an explicitly representation-domain stress condition or has enough evidence for a distinct physical-domain model. Keep the historical tensors and gates fixed.
- [x] Review Díaz-Paz et al.'s 2026 Sentinel-1 multitemporal reference dataset from the primary article and Mendeley record. It is a SAR-only despeckling candidate with 512×512 VV/VH inputs and 5–25-acquisition temporal averages; it does not supply optical guidance or an SR target. The paper specifies a five-character scene/target-area identifier, chronological acquisition indices, and a metadata CSV with coordinates/GEE IDs, which could support area-grouped splitting and acquisition tracing if confirmed in the released files. Record the unresolved per-target stack membership, rescaling/registration details, stationarity assumption, and ENL-table anomaly in [literature_review.md](literature_review.md).
- [x] Recheck the official article and Mendeley record for processing sequence and downloadable inventory. The article says images are rescaled to 0–255, registered to a designated reference, and then fused; the exact range mapping, averaging representation, registration method/error, and per-`GT_x` acquisition membership remain unspecified. The official [OAI-PMH `GetRecord`](https://data.mendeley.com/oai?verb=GetRecord&identifier=oai%3Adata.mendeley.com%2F6nzd6x25vw.1&metadataPrefix=datacite) returns public DataCite record metadata but no file inventory; the page still has no `Files` entries, and the PMC JATS export has no supplementary-material element. The separate [file-list API](https://data.mendeley.com/api/docs/) requires OAuth2. No release-file inventory or imagery was inspected.
- [x] Compare the stated GEE source with Google's official [`COPERNICUS/S1_GRD` catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD). That collection provides calibrated, orthocorrected backscatter converted to dB, but the dataset paper does not name its GEE collection ID and reports a 0–255 rescaling before registration/fusion. Treat the GEE catalog domain as conditional source context; it does not establish the released TIFF domain or target-stack membership.
- [x] Inspect the 2024 Toronto precursor's primary article and linked author code. The code averages the same 24 August acquisition used as the paired noisy input into its ten-date reference, and the article splits non-overlapping patches from one scene between train and validation. This is confirmed for the 2024 release only; do not transfer it as a fact about the 2026 extension.
- [x] Derive the conditional input/reference noise-residual covariance when the noisy acquisition is included in an idealized independent-speckle temporal mean. For an `N`-image mean the conditional correlation is `1/√N`; use this only to explain why stack membership matters, not as a measured property of the 2026 release. See [literature review](literature_review.md#external-multitemporal-sar-reference-candidate-2026).
- [x] Review the full accessible 2024 IEEE GRSL multitemporal-reference/assessment article cited by the 2026 dataset paper. It uses two 20×20 homogeneous ROIs for ENL and describes the 5 September acquisition as reference; the linked repository currently uses 24 August, so retain an article/code lineage discrepancy. The ROI context means the 2026 ENL outlier is an audit flag, not proof of a metric error.
- [x] Review Guan et al.'s near-real SAR despeckling paper and linked author repository as a possible SAR-only reference resource. Record the article's 250-train/20-test split versus the repository's 250-train/20-validation wording, and its later 936-pair 256×256 release with no verified mapping back to the paper split. Do not treat the GLRT/AML references as independent clean ground truth or as optical/SR evidence.
- [ ] Before considering this release for any SAR-only work, inspect an authorized archive copy and its metadata for the 936-pair lineage, source-scene groups, target-generation stack membership, registration, representation/radiometric domain, and dataset-specific license. Keep it a candidate until those checks are possible; no training or test-data access is implied.
- [x] Review De Paepe et al.'s full MOCEM paper on metadata-conditioned SAR-only joint despeckling/sidelobe reduction. Record its effective reflectivity target, CAD/background realism boundary, split, and limited real-data evidence; it does not provide optical guidance or SR evidence.
- [ ] If considering MOCEM as a real-SAR baseline, first locate a public or authorized code/data release. A bounded 2026-10-19 check of the primary [arXiv HTML](https://arxiv.org/html/2601.01541) and targeted repository searches found no code/data-availability statement or public repository; the [CatalyzeX listing](https://www.catalyzex.com/author/Jean-Christophe%20Louvigne) offers only a “Request Code” route. Do not design a reproduction until an authorized release exists. If one appears, verify the real Umbra/Capella evaluation protocol and simulator/data-use terms.
- [x] Audit the SARLO-80 paper and official Hugging Face card, reconstruction guide, and dataset metadata without downloading shards or imagery. Record the current train-only visible release, historical card count/file-schema conflict, approximate affine pairing, on-demand Google Satellite TMS dependency, and distinct SAR/optical licensing boundaries.
- [ ] Before using SARLO-80 for a geometry diagnostic, obtain a pinned, authoritative split/group manifest and reconcile the paper's train/validation/test claim with the Hub-visible train-only release; verify optical-source dates/terms and quantify registration residuals on permitted samples. This is a candidate audit only, not a training or project-test-data action.
- [ ] Before adopting the 2026 external candidate, obtain its metadata CSV and file list through a public route or authorized access, then check the scene/target-area identifiers and chronological acquisition indices, whether each noisy input is included in its `GT_x` stack, the mapping into 0–255 and representation actually averaged, registration quality, and whether the groups support independent evaluation. The public OAI-PMH `GetRecord` route is available for record metadata only; it returns no file list. The official page still has no entries under `Files`, the separate API file-list endpoint requires OAuth2, and the article JATS export has no supplementary-material element. No authenticated file access is available here, and no archive or release files were retrieved. The article has no linked generation-code repository. The 2024 precursor makes stack-membership verification a priority but does not establish the 2026 construction. Do not start model training or use the reported ENL table as validated evidence before this audit.
- [x] Recheck the OGSRN publisher routes for full text. SciOpen labels the article open access and presents a PDF-viewer prompt, but its PDF was not retrievable through the available interface (no browser surface); the official ScienceDirect article fetch returned HTTP 403. The abstract verifies training-time optical-space feedback and SAR-only inference; full protocol and official code remain open.
- [x] Check the primary SSRN record for Li et al.'s 2022 Optical-Guided Residual Learning Network. Its abstract describes HR-optical-derived feature weighting for supervised SAR SR, but does not specify inference-time optical use; opening the official page timed out, so the PDF remains unreviewed. Keep as abstract-level prior art with protocol/version lineage unresolved.
- [x] Recheck official SSRN and publisher/DOI full-text routes for the 2022 OGRN preprint/journal article. No full text was retrievable. The same author list and near-identical titles support a likely preprint/journal pairing, but exact version relationship and revisions remain unverified; keep data, degradation, inference inputs, metrics, results, and code unknown.
- [x] Open the lead author’s official Shenzhen University profile; it directly lists the separate JARS 16(3), 036503 article dated 2022-07-20. This verifies journal citation metadata only; the DOI, article body, and exact preprint/version relationship remain unresolved. See D-2026-10-26-095 and the [bibliographic audit](../../experiments/literature_audits/ogrn_2022_author_profile_recheck_20261026_v1/report.md).
- [x] Run a bounded exact-title/GitHub search for the 2022 OGRN implementation; no official code repository surfaced. This does not establish that no private or unindexed code exists.
- [ ] Compare the 2022 OGRN full texts and resolve exact version/revision differences only if an authorized SSRN PDF or publisher/author copy becomes publicly retrievable.
- [x] Recheck the 2025 OS_SR SSRN record and full-text route. The 20-page preprint's abstract describes optical high-frequency feature mapping and multiscale SAR reconstruction; the landing page is accessible, but direct PDF access returned HTTP 403 and exact-title/DOI searches found no alternate full text. Compare its abstract with the same authors' OMA-SSR journal article: the disclosed module descriptions differ, but no source establishes whether OS_SR is an earlier version. Keep the two records separate and lineage unresolved; see [D-2026-10-18-25](DECISION_LOG.md).
- [ ] Revisit OS_SR's full protocol and publication lineage only if the SSRN PDF or an author/publisher copy becomes directly readable; do not transfer OMA-SSR results or protocol to OS_SR. The 2026-10-22 Delivery.cfm recheck returned 403 and the Open PDF link redirected to the abstract page, so do not repeat those same routes without an access change.
- [x] Compare exact 128-pixel shared edge strips and diagonal corner blocks between the listed train/validation source SAR and optical PNGs at horizontal, vertical, and diagonal 50%-stride offsets. No exact matches were found at those offsets. The later exhaustive 51/52-pixel scan found 580 exact train–validation overlap geometries at 204/205-pixel translations; see the [audit report](sen1_2_train_val_spatial_overlap_audit.md).
- [ ] Resolve optical/SAR acquisition timing, preprocessing lineage, physical units, and co-registration only from source manifests or upstream metadata; do not infer from filenames or normalized PNG appearance. The local PNGs have neither sidecars nor ancillary container metadata. The official record is reachable and exposes corpus-level metadata, archive size/file count, preparation note, and license, but no local filename crosswalk or pair-level acquisition fields; the share link timed out and export endpoints remain inaccessible. See the [PNG metadata audit](../../experiments/data_audits/dev2k_png_metadata_20261023_v1/report.md).
- [ ] Obtain upstream scene/tile/patch identifiers for a group-level split and source-overlap audit. Local basenames are numeric and the split manifest records no parent/source IDs; do not infer groups from those names. Until then, retain the leakage/alignment limitation; exact file hashes do not detect distinct overlapping crops.
- [x] Recheck whether an authoritative SEN1-2 source record exposes a local train/validation filename crosswalk, without opening test arrays. On 2026-10-19 the official mediaTUM record became accessible and confirmed the corpus-level record, 43.7 GB / 564,770-file archive listing, GEE/MATLAB preparation note, and CC BY 4.0 right. The share link timed out and JSON/CSV exports remained inaccessible; no local filename crosswalk or file inventory was retrieved.
- [x] Recheck the documented mediaTUM metadata export route for SEN1-2 record 1436631. The record page now opens, but direct JSON/CSV exports remain inaccessible through the available web interface and the linked file server timed out; no export, file inventory, archive, or imagery was retrieved. This updates the 2026-10-18 access note in [CURRENT_STATE.md](CURRENT_STATE.md).

- [x] Review the author preprint/card and retrieve the public `D_lite` version record, manifest, README, checksum list, loader, and examples package. DOI differences are concept/version pairs (`22646276`/`22646277` for D_lite and `22699046`/`22699047` for the full corpus); the NLM's 740,847 count versus the paper/card's 740,746 remains unexplained. Metadata-file MD5s match the Zenodo API record; class archive SHA-256 entries were read but the large archives were not downloaded or checked.
- [x] Audit all D_lite manifest rows and probe three example pairs. The manifest has no split and repeats source-scene/tile IDs, so pair-randomized splitting is unsafe. Example rasters share nominal geotransforms and pass sidecar byte hashes, but cross-modal matches/correlation are too weak to bound residual alignment; sample optical dtype/nodata metadata is inconsistent and exact scaling remains unresolved. See [the audit](strata_s12_dlite_audit.md).
- [ ] If STRATA-S12 becomes relevant to a later geometry study, first obtain broader authorized samples and independent cross-modal tie points, resolve raster/sidecar value and nodata semantics, and construct source-grouped partitions with spatial-overlap review. Do not use it as a restoration benchmark or reopen local optical-fusion training: it provides no clean SAR reference or paired restoration target, and the local B1/B2 gates are unchanged.


- [x] Review the full C-DiffSET arXiv v4 and official project/repository code. Record its reverse SAR-to-EO task, training-only confidence weighting, per-file QXS split without source grouping, and public checkpoint selection through the loader named test; table-to-code lineage remains unresolved. No split identities, image payloads, local test data, or model runs were accessed. This literature/code audit does not change the B1/B2 gates.

- [x] Formalize the optical-information analysis into geometry, reflectance/texture, and semantic/context channels; derive the direct noisy-observation and general scalar conditional-Gaussian cases, and distinguish their Bayes-risk relations from local pseudo-target metrics. This is theory documentation only and does not authorize a new screen or alter B1/B2.
- [x] Review Zhang et al.'s 2026 CRDRS publisher abstract and official repository/code. Record its SAR-to-optical direction, optical-residual diffusion target, public `TestDataset` evaluation behavior, and unresolved split/code-to-table semantics. The split mapping, sample identities, image payloads, local data, and model were not accessed; do not infer published leakage or transfer the target-variance claim to SAR restoration. See [literature](literature_review.md#crdrs-residual-diffusion-for-sar-to-optical-translation), [diffusion feasibility](diffusion_feasibility.md#crdrs-residual-diffusion-in-reverse-sar-to-optical-translation), and [decision D-2026-10-19-22](DECISION_LOG.md#d-2026-10-19-22--keep-crdrs-as-reverse-direction-residual-diffusion-prior-art).

- [x] Extend the bounded diffusion literature taxonomy to SAR despeckling posterior sampling, raw-measurement SAR imaging consistency, and consistency models. Record Wang et al. (2025), Wang et al. (2024), Gao et al. (2026), CM-CR, and CM4IR with their task boundaries, evidence levels, and access limits in the literature review, feasibility note, and matrix; no local experiment or gate change follows.
- [x] Check the official IGARSS 2025 program for the title and authors of *Physics-Inspired Diffusion Model for SAR Image Despeckling*. The program confirms an oral presentation but provides no accessible manuscript/abstract; record existence only, with no method/result inference.
- [x] Review the 2025 3MOS version-of-record paper and author repository, plus the 2026 SOMA-1M preprint and author repository, as potential alignment-data routes. Record 3MOS's random post-crop 6:2:2 split, 50% patch overlap, dataset-specific CC BY-NC-ND label and third-party source terms; record that SOMA-1M's public repo currently exposes a README/test links while other subsets remain in progressive release. Neither is a SAR-restoration replacement or a local-source crosswalk. No archives, images, split identities, or local test data were accessed; see [literature review](literature_review.md#recent-multi-source-saroptical-alignment-datasets) and [D-2026-10-19-37](DECISION_LOG.md#d-2026-10-19-37--keep-3mos-and-soma-1m-as-alignment-prior-art).
- [x] Recheck the complete GNLM author manuscript at the official arXiv HTML route and verify its journal DOI. The method's SAR-domain predictor rejection is calibrated under an equal-signal Gamma-speckle model; its stale-guide example is a single qualitative building-change case after manual registration, not a controlled spatial-shift test. This strengthens the prior-art boundary without changing B1/B2 or authorizing a new experiment; see the [literature review](literature_review.md#optical-guided-sar-despeckling-and-predictor-reliability) and [D-2026-10-22-09](DECISION_LOG.md#d-2026-10-22-09--retain-the-narrow-gnlm-mismatch-boundary).
- [x] Verify citation identity from Crossref metadata: title, five authors, IEEE journal, volume, pages, and 2026 year. The official IEEE article page/text remains inaccessible, so this supports citation metadata only. See [D-2026-10-25-155](DECISION_LOG.md#d-2026-10-25-155--verify-da-physdiff-citation-metadata-only) and the [audit](../../experiments/literature_audits/da_physdiff_crossref_metadata_20261025_v1/report.md).
- [ ] Keep DA-PhysDiff outside the technical evidence matrix and make no method/result claim until the official IEEE article or an author-controlled primary text is accessible. Do not repeat the current blocked IEEE/DOI-reader routes absent a new access path.
- [ ] Review SAR-CDL's complete journal article or an author manuscript only if a new accessible full-text route appears. Direct ScienceDirect retrieval and the SSRN PDF route returned HTTP 403; the indexed publisher excerpts and SSRN abstract are not a full-text review. Verify the unrolling details, split recipe, real-SAR sources/reference metrics, code, and 2023-preprint-to-2025-journal lineage; do not repeat the blocked routes absent a change in access.

## P2 — Decide whether a new method experiment is justified

- [x] State whether a target-free, inference-available optical cue is supported for local joint restoration: none is currently supported. The B1F nine-offset matcher-entropy candidate failed its fixed all-eight-positive utility direction rule on 200 validation images. GNLM demonstrates predictor rejection for despeckling and GCOANet describes diffusion-based reference generation in an optical-assisted despeckling method, but the abstract does not clarify test-time optical input and neither provides local evidence for joint-SR utility or controlled-mismatch fallback.
- [x] Decide whether another Level-0 optical-fusion screen is justified on the current 2k subset. The single matcher-entropy cue diagnostic is complete and failed its prospective interpretation rule; no further screen is justified without materially new source/target evidence or a distinct, preregistered cue question. Do not repeat the pairing diagnostic or change historical gates; see [D-2026-10-18-17](DECISION_LOG.md) and [D-2026-10-18-36](DECISION_LOG.md).
- [x] Record the ordinary-sandbox `AISAR` probe (CUDA unavailable/device count 0; `nvidia-smi` driver failure) as a historical visibility result. A later authorized execution profile found an RTX 3050 Laptop GPU with 4 GiB and completed a bounded synthetic forward audit; the default Python still lacks PyTorch. See [resource audit](model_resource_audit.md).
- [x] Profile C2 and B1F parameter counts, trainable counts, batch-1 synthetic inference latency, and PyTorch tensor allocation peaks with strict checkpoint loading and test-access flags verified. This does not measure training peak memory or establish model quality; see [resource audit](model_resource_audit.md).
- [x] Produce a synthetic batch-2 optimizer profile and a compute-only estimate after freezing the source-valid split and protocol. The three-seed projection is about 2.1 GPU hours; full-run duration, validation/I/O overhead, and actual training peak memory remain unmeasured. Do not present the estimate as training feasibility.
- [x] Pause optical-fusion development on this dataset because no inference-available hypothesis is currently supported. Keep B1 failed and B2 gated; see [D-2026-10-18-17](DECISION_LOG.md).
- [x] Source-audit the active MMCANetV2 C2/v2_b forward path and run synthetic CPU checks for missing, masked, all-zero and variable-size inputs. The missing/zero modes reproduce the SAR base exactly in the source-level smoke setup; this does not establish trained-model or misregistration robustness, and it does not change B1/B2.
- [x] Close the SwinIR-Joint→MMCANetV2 base-checkpoint compatibility gap: require a verifiable run-config hash, exact train/validation fixed-manifest hashes, full architecture/behavior config, and parameter count before strict loading. Updated the baseline writer and all loader callers; synthetic mismatch and acceptance tests pass. Historical checkpoints without the new metadata must be regenerated before reuse. No data or research gate was changed.
- [x] Audit the SAR-only input/target contract from train/validation manifests and source code only. The fixed task maps 128×128 SAR representation inputs to 256×256 PNG-derived noisy pseudo-targets using synthetic ×2 degradation; it supports representation-domain target-consistency analysis only, not clean despeckling or calibrated SAR claims. No arrays or test artifacts were read; see [D-2026-10-18-18](DECISION_LOG.md).
- [x] Close the current public-code route for the third-party SIFT labels. The dataset README's code link resolves to RRL-Net training/testing code that uses prepared archives; the dataset repository exposes precomputed lists, not a label-generation script. The 434 matched rows and 219 zero-offset positive points therefore remain annotation-schema observations, not independent registration evidence. Do not repeat absent a new method/source artifact; see the [audit](../../experiments/data_audits/sen12_sift_tiepoint_probe_20261023_v1/report.md) and D-2026-10-23-40.
- [x] Check whether the original-data path hard-coded by the neighboring SOTANet helper is locally available. It is absent in this execution namespace, and the bounded path/mount scan found no alternate local source directory; this does not rule out unmounted or external copies. No source payload or test data was accessed. See the [path-availability audit](../../experiments/data_audits/dev2k_configured_source_root_probe_20261023_v1/report.md).
- [ ] Before proposing another model experiment, resolve the original numeric 2k subset's upstream/parent-scene, acquisition, product-level, and target provenance, or complete the audit needed to adopt a better-specified SAR dataset/reference. The separate official 1,800-pair cohort has verified source pixels and source-folder separation, but acquisition/product metadata, fine fall registration, and an independent reference remain unresolved. The 2026 multitemporal dataset is only a candidate until its input/target independence and radiometric/registration protocol are verified. Do not infer missing information from PNG appearance or numeric basenames; do not access test imagery or measurements.

- [x] Review the full 2018 SEN1-2 author manuscript and retained TUM supporting document for acquisition/registration semantics. The paper documents same-season S1/S2 filtering and per-ROI mosaicking, but no pairwise timestamps, product IDs, time-gap rule, or measured residual alignment; the TUM record returned HTTP 429 on the 2026-10-23 recheck. Pixel mixing across collection members is possible but unmeasured. See the [source audit](../../experiments/literature_audits/sen12_pair_timing_registration_20261023_v1/report.md) and D-2026-10-23-38.
- [x] Search for an author-maintained SEN1-2 GEE generator or product-date crosswalk using the exact paper title and source identifiers. No public author code route surfaced in the bounded search; this does not establish that no such code exists. Do not repeat without a new link or access route.
- [ ] For the official 1,800-pair cohort, obtain an authorized source manifest/crosswalk binding exact `s1_i`/`s2_i` folders to product IDs and acquisition timestamps; quantify fall residual alignment from a traceable independent reference if one exists. The paper and current TUM supporting document are insufficient. Do not use the present B1 validation metrics to estimate temporal or geometric error.

## P3 — Manuscript only after method evidence

- [x] Identify and distinguish the current formal 2k submission preview, the earlier engineering pilot, and the active MMCANetV2 research track. Record paper/PDF lineage and the formal comparison's `adaptive_development_test` role in `manuscript_lineage_audit.md`; preserve both historical PDFs.
- [x] Complete the field-level numeric-prose reconciliation for the formal draft’s legacy `TwoStreamFrequencyGuidedUNet` version against generated summaries, keeping historical adaptive-test evidence distinct from active MMCANetV2 and out of current method selection; see `manuscript_lineage_audit.md`.
- [x] Map the formal main-body table/figure groups to their generated artifacts, source aggregates, evaluation manifests, and freeze/selection records. Confirm that all 10 learned-model inference manifests carry checkpoint hashes matching the freeze manifest; the expanded comparison is explicitly `adaptive_development_test`. Correct Figure 3's direct-concatenation caption to distinguish the frozen seed-0 chart from the expanded three-seed table, preserving historical `main.pdf` and the corrected rebuild separately.
- [ ] Restore the exact frozen learned checkpoints and training-manifest bytes from an authorized archive if one becomes available. Current audit found 0/10 selected learned checkpoint paths present, 0/10 current training-manifest hashes matching the frozen hashes, but 10/10 selection-manifest hashes and 10/10 formal inference-manifest checkpoint hashes matching. For the separate historical B1 source snapshot, the manifest's recorded commit is current HEAD, but four source paths are absent from that tree and have no path-specific commits in reachable refs; do not rewrite hashes or scan unreachable objects that may contain experiment payloads. Do not substitute current `best.pth` files or regenerate historical test outputs.
- [x] Reconcile manually stated numeric prose claims against the generated source fields while keeping all historical adaptive-test values out of current method selection; see `manuscript_lineage_audit.md`.
- [x] Repair the optional `optimized_optical_stress.tex` lineage path for future rebuilds: the aggregator binds the table and aggregate source in the paper manifest, and the submission auditor validates them when included. Preserve the existing historical PDF; its old manifest remains unbound and its prior audit pass does not cover this table.
- [x] Map each planned claim to implementation, protocol, manifest, checkpoint, per-image CSV, aggregate, and figure status in [claim_evidence_map.md](claim_evidence_map.md). The B1 run artifacts are validation-only by manifest, but evaluator/source lineage and a final manuscript-linked comparison figure remain unresolved; the map records these as limitations rather than verified support.
- [x] Resolve active-manuscript ambiguity: no MMCANetV2-specific manuscript exists. `paper/MMCANet/mmcanet_jdsr_draft.tex` is the archived engineering pilot, the formal source is a historical legacy-model study, and a distinct unclassified draft remains under `IEEEtran/`; only its rebuildable PDF was removed. Preserve the source/bibliography and do not transfer its claims to V2. See D-2026-10-27-007.
- [x] Mark `paper/MMCANet_formal/` as a historical preview after the split-overlap and provenance audits; preserve its old source/results without using them for current model selection or promotion. Record the 2026-10-22 test-labelled text-search exposure and access flags in `CURRENT_STATE.md` and the dedicated disclosure manifest.
- [ ] Do not report the active task as submission-ready until formal, multi-seed, fair, reproducible evidence and visual PDF QA exist.

## Prospective B1 v2 evaluator and run interface (2026-10-22)

Historical continuation snapshot: the pending seed-2 and aggregation lines below were superseded by the completed current-source matrix recorded at the top of this file. They are not active tasks; the current B1 gate failed, so B2/B3 must not run.

- [x] Freeze `sen12_scene_dev1800_b1_v2` with mean TBR required on all validation identities and ancillary median TBR represented by explicit defined/undefined status, unavailable values, and per-role/seed coverage. Keep the v1 protocol, strict score, and promotion gate unchanged; see [v2 protocol](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.md) and [D-2026-10-22-16](DECISION_LOG.md#d-2026-10-22-16--freeze-the-b1-v2-optional-median-policy).
- [x] Wire protocol selection through both trainers, the run sealer, and the source-group aggregator. The B1 trainer path now emits all eight gate metrics, and v2 sealing/aggregation verify median status and coverage manifests.
- [x] Make `--validation-dir` mandatory in the B1 residual-weight calibration CLI and function API after the seed-1 invocation fell back to the legacy `fixed_val_dev2k` manifest. Add CPU regression checks that omission fails before file access in both interfaces; see [D-2026-10-23-23](DECISION_LOG.md#d-2026-10-23-23--require-an-explicit-validation-manifest-for-b1-calibration).
- [x] Rebuild fixed train/validation inputs from retained metadata, confirm byte-identical frozen file-list hashes, and start the new unique v2 run root after CUDA preflight. Do not resume the failed v1 model run; preserve false run-level test flags. SAR-only seed 0 is complete and sealed; see the current state and registry.


- [x] Finish the v2 seed-0 optical candidate: residual weight was calibrated from its deterministic first train batch only, then C2 was trained and sealed against the same-seed SAR-only checkpoint. No gate was evaluated.
- [x] Superseded by the complete current-source matrix below: seed-2 SAR-only and optical-candidate runs were completed and sealed in the unique 2026-10-23 root. Preserve the earlier interruption as historical; do not resume or rerun it. See `experiments/sen12_scene_dev1800_b1_revalidation_20261023_source_aligned_v1/`.
- [x] Superseded by the current-source six-run aggregation: the unchanged gate returned `not_promoted` (`stop_at_b1`). Do not enter B2/B3. See the aggregate report in the current-source root.
- [x] Remove the temporary B1 v2 train/validation arrays and staged PNG panels after their metadata hashes matched the retained references. The later current-source matrix rebuilt and verified its inputs, completed all six runs, and pruned replayable payloads after sealing; no seed-2 resumption remains active.

## Artifact cleanup — 2026-10-21

- [x] Remove the verified byte-identical mini copies, the explicitly named smoke checkpoint, the local pytest cache, and generated pixel payloads from the legacy July 1900/100 outputs. A later 2026-10-22 pass pruned generated pixels/build caches from the pilot snapshot and removed four checkpoints from its train/test-overlap-invalidated run; the original manifest and audit records remain, but the manifest describes the pre-prune tree. Keep original source images; see the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).
- [x] Disclose that duplicate verification read the 100-pair legacy mini test subset and its matching source PNGs. Do not use those identities for future selection; a follow-up metadata-only check found all 100 in fixed train and none in validation, without reading the fixed-test manifest.
- [x] Remove the previously inventoried 8,800 rebuildable `.npy`/`.npz` payloads in registered formal runs. The sensitivity, experiment, and later pilot-snapshot cleanup passes removed generated array payloads while retaining configs, manifests, metrics, logs, figures, and current checkpoints. The old pilot manifest now describes its pre-prune snapshot; see the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).
- [x] Remove two redundant root-level legacy `last.pth` copies after byte-for-byte comparison with the preserved frozen-archive copies (46,061,958 bytes). Run records and archive copies remain; see the [cleanup audit](ARTIFACT_CLEANUP_20261021.md).
- [x] Repair `test_fixed_protocol_train_validation_contract` so it uses generated `tmp_path` fixtures rather than the pruned fixed-input arrays; syntax compilation and the targeted test passed (`1 passed`). The full suite was not rerun.


## Historical B1 v2 continuation checkpoint (2026-10-22; superseded by the current-source matrix)

- [x] Prepare explicit seed-2 baseline, calibration, candidate, and sealing commands from the successful seed-1 commands and current parsers. At this historical checkpoint the commands awaited CUDA preflight and input reconstruction; the later current-source matrix completed and sealed that seed pair, so no execution remains pending here.
- [x] Superseded by the later completed current-source matrix; the archived interrupted seed-2 attempt remains historical and was not resumed. The current shell's GPU preflight is unavailable, so do not restart training from this historical checklist.
- [x] Seed-2 SAR-only, train-only calibration, and paired C2 candidate were completed and sealed in the unique 2026-10-23 current-source root; the earlier partial attempt remains an archived failure record.
- [x] The current-source six-run aggregate evaluated the unchanged B1 v2 gate and returned `not_promoted` / `stop_at_b1`. Keep B2/B3 gated.
- [x] Remove the current B1 v2 train/validation arrays and staged panels from `/tmp` after the seed-2 interruption while CUDA is unavailable; the retained manifests, hashes, run records, and commands support reconstruction after a fresh GPU preflight.
- [x] Add strict epoch-boundary resume to the SwinIR-Joint baseline and MMCANetV2 candidate trainers. Resume is bound to the same incomplete run, unchanged CLI settings, split/input manifests, and `last.pth`; checkpoints restore all training/RNG/DataLoader state and carry the epoch records needed to rebuild run outputs. Focused synthetic tests pass. The archived seed-2 checkpoint was pruned, so its retry remains from scratch after GPU restoration.

## MONTI two-stage SAR-SR prior-art update (2026-10-23)

- [x] Review the official workshop record and indexed OpenReview manuscript sections on method, data, results, and limitations. The paper reports structure restoration followed by diffusion speckle synthesis, three SR-backbone comparisons, and separate real-input distributional and synthetic paired metrics. See the [audit report](../../experiments/literature_audits/monti_two_stage_sar_sr_20261023_v1/report.md) and D-2026-10-23-31.
- [x] Search the exact title, authors, OpenReview ID, GitHub, and SI Analytics domain for a public code repository. No official repository surfaced; code availability remains unresolved because direct PDF access is blocked and indexed search can omit links. See the [audit report](../../experiments/literature_audits/monti_two_stage_sar_sr_20261023_v1/report.md).
- [ ] Revisit the full PDF only if OpenReview access or an author copy becomes directly available. Do not repeat the current browser-verification route without an access change.
- [ ] Keep H8's matched order/joint comparison prospective and gated on valid source/target provenance, group-disjoint development data, frozen inputs, and compute.

## Local QXS-named cache generator/config audit (2026-10-23)

- [x] Inspect local SOTANet split helpers and QXS-named training options without reading test payloads. The tracked helper is an unseeded per-file splitter; the ignored SEN1-2 preparation script writes a different output path; ignored run options bind only to the local QXS-named path. None is linked to cache generation or carries an upstream version/source-group manifest. See the [local generator/config audit](../../experiments/data_audits/sar2opt_qxs_local_generator_audit_20261023_v1/report.md) and D-2026-10-23-30. The existing P0 lineage and mapping actions remain open pending an authorized archive/version or authoritative source-group crosswalk.

## SEN12MS-CR-TS conditional candidate audit (2026-10-23)

- [x] Review the primary paper, author dataset page, public README/loader, and TUM record response. The paper reports 53 ROI groups (40 train / 13 holdout), 30 paired S1/S2 dates per ROI in 2018, and a maximum 14-day acquisition gap. The loader parses calendar dates from filenames. The dataset is for optical cloud removal, has no clean SAR restoration target, is approximately 2 TB, and its dataset-specific license remains unresolved; the TUM record returned HTTP 429.
- [x] Recheck the author-group download helper and official article copy for a train-only metadata route and rights statement. The helper exposes per-region/per-modality archives with test archives separately named; the reviewed sections contain no standalone metadata crosswalk. The article's CC BY 4.0 notice applies to the paper and its data section points to the author website, but it gives no separate dataset license. The author repository listing has no root-level LICENSE. No payload or holdout identity was accessed. See the [route audit](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/report.md).
- [ ] Keep it unadopted until dataset-specific rights are established and a SAR task defines the input/target stack, input exclusion, temporal-change assumptions, representation domain, and registration checks. If authorized, inspect only train-side metadata and form a prospective ROI-grouped split from the 40 train ROIs. Keep the 13-ROI holdout unopened; do not run training or alter the local B1 gate. Revisit the same public routes only if an official rights statement or separate train-only metadata release appears. See the [candidate audit](../../experiments/literature_audits/sen12ms_crts_candidate_20261023_v1/report.md), [route audit](../../experiments/literature_audits/sen12ms_crts_download_route_20261023_v1/report.md), and [access manifest](../../experiments/literature_audits/sen12ms_crts_candidate_20261023_v1/access_manifest.json).

## SEN12TP conditional candidate audit (2026-10-23)

- [x] Review the version-of-record paper, Zenodo record, author dataloader README, dataset enumeration behavior, official terms for listed source components, and a rights-metadata recheck. The Zenodo record's Rights/License field is blank; OpenAIRE's merged entry displays a version-unspecified `CC BY` label, while Zenodo's generic developer and Help pages give different default-license descriptions. None establishes the selected license for this record or clears the compiled archive/curated metadata. The Zenodo public file list shows a 1.3 MB metadata JSON, five train archives, and one test archive, while its prose says four train archives. No metadata contents or archive data were accessed. Exact rights, mapping between the released ~90/10 train/test archives and the paper's 80/10/10 split, and train-side metadata remain unresolved. The dataset has no clean SAR restoration target; its 10 m grid also exceeds the stated ~20 × 22 m SAR effective resolution. One article-level test-metric text surfaced in search output and was not used; see the [access manifest](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/access_manifest.json) and [audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).
- [x] Inspect the author dataset source for a train-only spatial audit route. `load_patch` returns bbox/CRS/affine metadata; the default 256-pixel windows use stride 249. This route could audit permitted training TIFFs without opening the all-scenes metadata JSON, but the mutable source is unpinned and does not enforce parent-ROI grouping. No code was executed and no dataset or test content was accessed; see D-2026-10-25-70 and the [audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).
- [x] Cross-check the publisher's data-availability identifier and rights notice against the current Zenodo record. The article links `7342059`, while the current v1 page is `7342060`; the former returned HTTP 429, and the article CC BY 4.0 notice does not resolve the dataset record's blank license. See D-2026-10-25-72 and the [audit](../../experiments/literature_audits/sen12tp_candidate_20261023_v1/report.md).
- [ ] Resolve the `7342059`/`7342060` relationship and record-specific dataset license only if an authoritative metadata route becomes accessible. Keep the all-scenes metadata JSON and every test asset unopened; do not download the compiled dataset or start a model experiment while rights and archive/split mapping remain unresolved.
- [ ] Do not download the compiled dataset or run a model experiment until package/curation rights, archive/split mapping, and a task-appropriate noisy-pseudo-target protocol are established. An alternative assembled from original provider feeds would need exact product IDs, processing versions, and source credits in a separate manifest. If rights resolve, pin the loader revision, use only authorized train-side TIFFs for a spatial audit, group every dated scene by parent `roi_id` before patch extraction, and keep the all-scenes metadata JSON plus released test payload and measurements unopened. B1 remains `not_promoted`; B2/B3 remain gated.

## Manuscript result-text exposure disclosure (2026-10-23)

- [x] Record the targeted TeX/Bib search exposure to aggregate held-out result text from the legacy pilot. No PDF, test image, tensor, prediction, metric file, test manifest, or test directory was opened; no values were used for selection, tuning, checkpoint choice, paper updates, or gate decisions. Historical run flags remain unchanged. See the [disclosure report](../../experiments/data_audits/paper_tex_result_text_exposure_20261023_v1/report.md) and [access manifest](../../experiments/data_audits/paper_tex_result_text_exposure_20261023_v1/access_manifest.json).
- [ ] Exclude that legacy pilot sample set from further selection or tuning and do not reread its result-bearing manuscript sections. Continue only independent provenance, protocol, or code work that does not depend on those held-out results.

## B1 v2 bootstrap interval verification (2026-10-23)

- [x] Independently rederive the eight source-group bootstrap summaries from the hash-bound validation aggregate. All means, 95% intervals, and positive-draw probabilities reproduce within `2.71e-20`; see the [audit](../../experiments/diagnostics/b1_v2_source_group_bootstrap_rederivation_20261023_v1/report.md).
- [ ] Keep interpretation conditional on the three fixed seeds and the single fall ROI-season. The gate remains `not_promoted`; B2/B3 stay gated pending valid provenance and prospective development data.

## README and runtime follow-up (2026-10-23; updated 2026-10-24)

- [x] Add the independent bootstrap audit to README Key findings and verify all 101 local link destinations resolve.
- [x] Recheck the execution environment on 2026-10-24: system Python 3.13.5 lacks PyTorch; `AISAR` is Python 3.13.7 with PyTorch 2.8.0+cu128/CUDA runtime 12.8 but CUDA is unavailable, device count is 0, and `nvidia-smi` exits 9. No model run was launched; the result matches the prior availability check.
- [x] Reflect the ProSR full-text audit in README Key findings, keeping its discrete SAR ×4 prior art separate from the optical-guided joint-restoration question. See the [audit](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/report.md).
- [ ] Continue only authorized provenance or primary-source work with a new access route. Do not treat GPU availability alone as permission to enter B2/B3; the unchanged B1 gate failed.

## SAR400 conditional denoising candidate (2026-10-23)

- [x] Screen the publisher abstract, author dataset page, and university metadata record. SAR400 is described as a SAR-noise suppression test set with four 100-item noise groups; its public metadata does not establish `gt` provenance, physical domain, generator parameters, sample grouping/split, or package rights. See the [audit](../../experiments/literature_audits/sar400_candidate_20261023_v1/report.md).
- [ ] Keep the archive unadopted. Revisit only if a direct full-text or author metadata route resolves source/target semantics, noise construction, and dataset-specific terms. If cleared, predeclare a separate SAR-only denoising stress-test protocol; it cannot substitute for optical/JDSR evidence or change the B1/B2/B3 gate.

## SOMA registration prior-art screen (2026-10-23)

- [x] Review the full arXiv v1, AAAI record, and linked public training/test/loader code. Record SOMA as registration-only prior art and identify an unresolved paper/code transform-range mismatch plus an unexplained Table 1/Table 3 CMR discrepancy. See the [audit](../../experiments/literature_audits/soma_sar_optical_registration_20261023_v1/report.md).
- [ ] Do not adopt SOMA or its certainty map as a restoration reliability mechanism. Reconsider only after the local B1/provenance gates permit a defined downstream SAR-output question; any use must separately audit source split/rights and reproduce a pinned implementation against fixed registration perturbations.

## ISSMSAR prior-art audit (2026-10-23)

- [x] Review official publisher-indexed primary text for the multi-temporal SAR-only joint-restoration method, its GaoFen-1 optical-derived pseudo-SAR training targets, reported real-source examples, and metric tables. Direct publisher HTML returned HTTP 429; XML/PDF were inaccessible through the web reader. See the [audit](../../experiments/literature_audits/issmsar_fulltext_audit_20261023_v1/report.md) and D-2026-10-23-57.
- [ ] Revisit exact split/group, noise/downsampling, real-scene reference, and code details only if a directly readable publisher or author-controlled source becomes available. Do not repeat the same inaccessible routes without an access change. No local experiment is authorized by this literature review.

## Project facts and evidence index (2026-10-23)

- [x] Add a repository-wide A–E index for verified facts, bounded observations, hypotheses, open questions, and unsupported claims; link it from README and the claim–evidence map. This is a navigation aid and does not replace CURRENT_STATE.
- [ ] Continue the P0 provenance route: obtain an authorized upstream archive/version and verifiable parent-scene/source-group mapping for the numeric 2k subset. The ReFlowSET basename intersection is only downstream split membership; the 333 local train/validation names absent from its train list remain unclassified. Until provenance and source groups are verified, do not use the contaminated development partitions for model selection or promotion; preserve the failed B1 gate and keep B2/B3 gated.


## SDS-SAR prior-art screen (2026-10-23)

- [x] Review the publisher-indexed article record and inspect the official author repository's visible test/model files. Classify SDS-SAR as SAR-only self-supervised despeckling prior art; the current public test path is a median-filter placeholder and does not reproduce the paper. See the [audit](../../experiments/literature_audits/sds_sar_primary_source_20261023_v1/report.md) and D-2026-10-23-61.
- [ ] Revisit implementation only if the authors publish an actual training/inference path and the method/data source and reuse terms can be verified. Do not use the placeholder as a baseline or alter B1/B2/B3.


## SAR-only temporal-average adaptation prior art (2026-10-23)

- [x] Review primary indexed sections for Lattari et al. (2019) on speckle modeling, synthetic pretraining, temporal-average SAR fine-tuning, and real-domain metrics. Record the reference as pseudo-clean and the result as denoising-only prior art; no task-matched JDSR baseline or verified code/data route was found. See the [audit](../../experiments/literature_audits/lattari_sar_despeckling_2019_primary_20261023_v1/report.md) and D-2026-10-23-62.
- [x] Add Lattari et al. (2019) to the current literature matrix as SAR-only despeckling prior art; keep historical matrix membership unresolved and do not treat it as a task-matched JDSR baseline. See D-2026-10-26-080.
- [ ] Do not adopt the temporal-average stacks or reported metrics locally unless exact source acquisitions, stack/patch membership, rights, and code/data access can be verified; keep B1/B2/B3 unchanged.

## SAR2SAR full-text and implementation audit (2026-10-23)

- [x] Review arXiv v3's speckle model, three-stage training protocol, synthetic/real evaluation boundary, and ablation; inspect the author-linked repository route and the related author-group inference package without downloading data or weights. Record it as SAR-only denoising prior art, not a matched JDSR baseline. See the [audit](../../experiments/literature_audits/sar2sar_primary_fulltext_20261023_v1/report.md) and D-2026-10-23-63.
- [ ] Do not use the related package locally unless a task-matched SAR input protocol, prospective valid split, model/checkpoint lineage, and data/weight terms are established. Do not repeat the currently inaccessible GitLab route without an access change; keep B1 `not_promoted` and B2/B3 gated.


## Completed artifact cleanup continuation (2026-10-24)

- [x] Prune the remaining 12 per-epoch CSV/log files from recorded failed/interrupted runs, remove 178 empty legacy output directories, and remove three stale rebuildable LaTeX build logs while preserving configs, commands, manifests, failure/interruption audits, registered results, and selected checkpoints. No test/metric payload or gate was accessed. See [cleanup report](ARTIFACT_CLEANUP_20261024.md).

## Corley et al. registration benchmark audit (2026-10-24)

- [x] Read full arXiv v4 and inspect the official CVF record and current official repository sweep entry. Record the three-scene descriptive scope, exact §5.2 protocol grid, and unresolved difference between the paper and mutable-main reproduction defaults. See the [audit](../../experiments/literature_audits/corley_pretrained_matcher_fulltext_20261024_v1/report.md).
- [ ] Keep the paper as registration-protocol context only. Do not enter B2/B3 or infer restoration utility from matcher/reprojection scores; preserve the B1 `not_promoted` stop and prioritize authorized source-provenance/group mapping.

## GCOANet patent-route follow-up (2026-10-24)

- [x] Check the new CN120450996A lead through secondary patent indexes and attempt the official CNIPA item route. The indexes render an SAR-plus-optical prediction module and QXSLAB_SAROPT split details, but the official record and paper-to-patent binding remain unverified. See the [route audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md).
- [x] Check the surfaced same-acronym GitHub `GCOANet` candidate. Its README describes gene/CpG/miRNA cross-omics, so it is not evidence of the SAR article's code; this does not establish whether another official code release exists. See [D-2026-10-25-142](DECISION_LOG.md#d-2026-10-25-142--exclude-the-same-acronym-cross-omics-repository-as-gcoanet-paper-code).
- [x] Recheck the official CNIPA guidance and exact publication/application-number routes on 2026-10-28. The official portals returned timeout/HTTP 412 or no item-level result; secondary patent text was excluded. See D-2026-10-28-019 and the [route audit](../../experiments/literature_audits/gcoanet_related_patent_route_20261024_v1/report.md).
- [ ] Keep the journal article abstract-level and the local QXS source lineage unresolved. Revisit only if the official patent document, authorized article full text, or an author-controlled code repository explicitly tied to the article becomes accessible; do not use secondary-index text to choose a method, infer local data origin, or alter B1/B2/B3.

## OGSFlow repository follow-up (2026-10-24)

- [x] Re-read the currently accessible official repository README. It describes validation calibration of the Stage-2 source-noise scale, frames it as a perception–distortion control, and names HR-trained semantic downstream evaluations. The README now visibly maps S1/S2/RGF to source paths and advertises a CPU synthetic-data smoke command, but the linked source pages remain inaccessible and the commands were not run. Validation-split separation from later selection/reporting is unresolved, not a leakage finding. No numeric results or blind-split contents were accessed. See [updated audit](../../experiments/literature_audits/ogsflow_public_repo_screen_20261024_v1/report.md) and D-2026-10-24-17/18/42.
- [ ] If the official citation or source/document routes become directly accessible, inspect the calibration estimator and determine whether its `val` identities overlap S2 model-selection/reporting data; inspect protocol definitions without opening blind split identities or payloads.
- [ ] Revisit method and data details only if the repository publishes the paper citation or its linked source/evaluation/dataset documents become directly readable. Keep OGSRL separate from OGSSR, inspect only permitted non-test metadata first, and leave all blind test payloads unopened. Do not use the repository lead to reopen B2/B3 or the local B1 gate.


## C2 residual diagnostic (2026-10-24)

- [x] Complete the zero-training, three-seed diagnostic on the exact 200-row B1 v2 validation split; verify all nine per-sample input hash fields against the retained input metadata. Keep per-image/aggregate CSVs, config, command, manifests, and figures; remove regenerated fixed payloads and intermediate arrays. See the [diagnostic record](../../experiments/diagnostics/b1_v2_c2_actual_residual_diagnostic_20261024_v1/report.md).
- [x] Record that overall residual evidence is mixed: actual PSNR gain is small, SSIM remains negative, the tested global alpha does not improve strict score, and the target-assisted oracle is not deployable. Leave B1 `not_promoted` and B2/B3 gated.
- [ ] Prioritize an authorized numeric-2k source/version and parent-scene crosswalk, or a prospectively defined group-disjoint split, before any new model selection. Do not retune C2 or reopen the current B1 gate from these diagnostics.

## C2 gate-to-utility alignment diagnostic (2026-10-24)

- [x] Predeclare and run a zero-training analysis of input-derived mean gate activation against per-image actual-minus-base strict-score change on the same 200 validation names across seeds 0/1/2. Mean seed-wise Spearman rho was +0.8549, with source-folder cluster-bootstrap 95% interval [+0.7259,+0.9107]; all seeds were positive. This is bounded support for composite-score ranking on the reused cohort only.
- [x] Record the limit: secondary rho values were weak/inconsistent for ΔPSNR and ΔSSIM; the composite also contains edge/TBR terms. Do not infer calibrated reliability, identify a causal metric component, fit a threshold, or change B1/B2/B3.
- [ ] Resume with authorized source/version and parent-scene/target/registration provenance, or define and verify a prospective independent cohort before any new model selection. Do not repeat gate-cue thresholding on these same validation identities.

## Local source-volume availability (2026-10-24)

- [x] Recheck the configured SEN1-2 source root, current mount table, NTFS partition metadata, and sandbox device access. Two candidate data volumes are present but unmounted; the sandbox cannot mount them read-only, so no partition contents were inspected. See the [availability audit](../../experiments/data_audits/dev2k_unmounted_ntfs_partition_probe_20261024_v1/report.md).
- [ ] When an authorized read-only mount or verified source archive/crosswalk is available, inspect only the candidate source directory names and permitted train/validation metadata needed to establish local numeric-2k lineage and source groups. Do not enumerate test identities or open test payloads. Until then, continue independent research and keep model selection/B2/B3 stopped.

## MLOG-related patent prior art (2026-10-24)

- [x] Review the CN114359043B patent text for MLOG architecture, train/test modality paths, degradation, sample counts, metrics, and ablations. Keep the patent separate from the 2022 preprint and 2025 JARS record because version lineage is unverified. See the [audit](../../experiments/literature_audits/mlog_related_patent_cn114359043b_20261024_v1/report.md).
- [ ] Resume the 2025 JARS method/protocol review only if a directly readable publisher or author-controlled full text appears; do not repeat the blocked DOI route absent an access change. Do not transfer patent details to the journal article or change B1/B2/B3.

## ECDM SAR despeckling prior-art review (2026-10-24)

- [x] Review publisher-indexed method, synthetic and real-data target construction, metrics, and runtime; record ECDM as SAR-only diffusion despeckling prior art. Direct HTML returned HTTP 429 and PDF/XML routes were inaccessible; code availability, split grouping, input-date membership in the SAR average, and product-transfer details remain unresolved. See the [audit](../../experiments/literature_audits/ecdm_sar_despeckling_fulltext_audit_20261024_v1/report.md) and D-2026-10-24-22.
- [ ] Revisit only if a directly accessible publisher/author full text or paper-linked code route appears. Do not repeat the blocked routes or implement local diffusion without a valid target, task-matched baseline, and authorized protocol. Keep B1 `not_promoted`; B2/B3 remain gated.


## DualCDM reverse-direction diffusion prior-art review (2026-10-24)

- [x] Review the publisher-indexed method, SEN1-2/S1S2 split descriptions, reported optical-domain metrics, large-image holdout, and paper-linked code/data roots. Record the reverse SAR-to-optical task boundary and unresolved source-group/release lineage. See the [audit](../../experiments/literature_audits/dualcdm_sar_to_optical_audit_20261024_v1/report.md) and D-2026-10-24-23.
- [ ] Revisit source files or dataset metadata only if a directly accessible, versioned author route with explicit reuse terms appears. Do not repeat current cache-miss/API routes, download the S1S2 archive, or use this work to claim optical-guided SAR-output efficacy. Keep B1 `not_promoted`; B2/B3 remain gated.

## Attention OffsetNet registration prior-art screen (2026-10-24)

- [x] Record the institution-hosted bibliographic entry and author abstract as registration-only prior art, and carry its scope into `research_gaps.md`. Keep patch RMSE distinct from full-scene RANSAC residual; see D-2026-10-24-24 and the [audit](../../experiments/literature_audits/attention_offsetnet_author_page_screen_20261024_v1/report.md).
- [ ] Revisit the full protocol only if an accessible full paper or author-controlled code route appears. Check scene-level split, offset labels, aggregation, residual statistics, and run lineage. Do not transfer its residual numbers to the local cohort or change B1/B2/B3.

## Manuscript test-result text exposure (2026-10-24)

- [x] Disclose that a broad manuscript-source search surfaced historical pilot test-result numbers. They were not analyzed or used; no test image/array was opened. Treat the measurements as accessed and exclude them from tuning, selection, and promotion. See the [disclosure audit](../../experiments/data_audits/manuscript_test_result_text_exposure_20261024_v1/report.md) and [manifest](../../experiments/data_audits/manuscript_test_result_text_exposure_20261024_v1/access_manifest.json).
- [ ] Keep the result-bearing manuscript claim audit deferred until it can be reviewed without exposing or using those historical test measurements. Continue source-independent literature and protocol work; B1 remains `not_promoted`, B2/B3 remain gated.

## MMCANetV2 inference interface (2026-10-24)

- [x] Require explicit `--fixed-dir`, report model tensors by optical mode, and load only LR plus the optical payload when the selected mode needs it. Synthetic tests confirmed missing optical and target directories are accepted for `zero`/`missing`, while normal mode loads optical without target. See D-2026-10-24-26/27.
- [ ] Keep the test-only runner unused until the frozen research gate authorizes final evaluation; the source-level loader contract has not been exercised on real data.

## Focused inference-interface smoke cleanup (2026-10-24)

- [x] Remove the three residual pytest temporary sessions and `pytest-current` symlink after recording their aggregate counts and sizes; remove the workspace pytest cache and temporary compile cache. No experiment outputs or source data were touched. See the [cleanup record](ARTIFACT_CLEANUP_20261024.md#focused-inference-interface-test-cleanup).

## Formal-test promotion interface repair (2026-10-24)

- [x] Require one hash-bound promoted B1 decision before preparing formal test inputs, freezing models, running formal test/stress/degradation/efficiency entrypoints, selecting display identities, or rendering qualitative test figures. Record the run interface change and synthetic verification under D-2026-10-24-30.
- [x] Replace target/intensity-driven identity and region selection with filename-hash identity ranking and filename-hash crop/profile coordinates; bind renderer input to the pre-test selection manifest.
- [ ] Keep formal test, B2, and B3 closed while the current B1 gate is `not_promoted`. Reconsider only after a valid prospective group-disjoint development protocol supports a new B1 decision; the interface repair itself is not promotion evidence.
## Lower-level test interface repair (2026-10-24)

- [x] Bind direct inference/evaluation entrypoints to the verified formal freeze and selected model; pass this context from the main, optical-stress, and degradation-sensitivity runners. See D-2026-10-24-31.
- [x] Add synthetic tests proving direct inference/evaluation rejects test-marked fixed inputs before split or payload access; run them with the focused evaluator and MMCANet-v4 interface suites (49 passed).
- [ ] Keep all formal test runs closed while B1 is `not_promoted`. After a valid prospective group-disjoint development protocol supports a new B1 decision, recreate the source-hash-bound freeze and use only the predeclared formal runners.

## SAR-conditioned flow matching for SAR-to-optical translation (2026-10-24)

- [x] Add a bounded primary-source screen for the IEEE early-access abstract and indexed ICLR workshop author manuscript. Record the reverse task, SAR2Opt setup, and workshop Table 1's PSNR/prose inconsistency; keep the IEEE/workshop version relationship unverified. No local experiment or gate changed. See [audit](../../experiments/literature_audits/sar_conditioned_flow_matching_s2o_20261024_v1/report.md).
- [ ] Review the IEEE version only if its full text becomes directly accessible; verify whether it changes the workshop protocol or metric table. Do not reuse workshop numbers as journal results or as SAR-restoration evidence.

## ProSR discrete SAR-SR prior-art audit (2026-10-24)

- [x] Review the full manuscript, author repository, dataset-card metadata, and Umbra source-data license pages. Record the discrete-token ×4 task, SLC spectral-crop protocol, amplitude representation, metric trade-off, source-disjoint train/validation split, code availability, and separate dataset/source license statements. See the [audit](../../experiments/literature_audits/prosr_fulltext_audit_20261024_v1/report.md) and D-2026-10-24-36.
- [ ] Reconsider dataset or method reproduction only if an authorized route provides the model code or enough implementation detail and the curated-dataset/source-image reuse terms are reconciled. Do not use ProSR as a matched optical-guided joint-restoration baseline or change B1/B2/B3.


## Legacy artifact pruning (2026-10-24)

- [x] Remove reproducible legacy panels, detailed pilot metrics, raw test-run logs, and superseded formal-test result detail while preserving configs, commands, manifests, split records, audit summaries, checkpoints, source data, and current B1 evidence. See the [cleanup record](ARTIFACT_CLEANUP_20261024.md#legacy-result-pruning-2026-10-24).
- [x] Record the path-level exposure of legacy metric-audit sample IDs/ranking labels. Exclude that historical cohort from future selection; do not alter prior run flags. See the [disclosure audit](../../experiments/data_audits/legacy_output_path_metadata_exposure_20261024_v1/report.md).
- [ ] Keep B1 `not_promoted`; do not reopen B2/B3 or use the exposed legacy cohort for tuning or promotion.

## Redundant completed-run checkpoint pruning (2026-10-25)

- [x] Remove `last.pth` only from completed runs that retain the matching `best.pth` and full run records; retain all best checkpoints, metrics, manifests, configs, and logs. See the [cleanup manifest](../../experiments/artifact_cleanup_continuation_20261025_v1.json).
- [x] Confirm no repository cache or separately named smoke artifact remained in the metadata inventory. Canonical SAR/optical inputs and fixed splits remain untouched.

## Pilot manuscript text-exposure disclosure (2026-10-25)

- [x] Record that a targeted SFT citation/source query surfaced held-out text from the legacy pilot draft; no surfaced values were used. Exclude the named historical cohort from future selection, tuning, promotion, and quantitative manuscript claims; preserve the historical record. See the [access disclosure](../../experiments/data_audits/manuscript_citation_search_text_exposure_20261025_v1/report.md).

## H8 task-order literature boundary (2026-10-24)

- [x] Review primary-source camera-pipeline studies that compare task order and sequential versus joint restoration. Record Qian et al. (ICCP 2022) as direct camera-order prior art and Xing and Egiazarian (CVPR 2021) as fixed-order joint/cascade prior art; broad task-order novelty is unsupported. See the [audit](../../experiments/literature_audits/restoration_order_camera_pipeline_20261024_v1/report.md).
- [ ] Keep any H8 SAR comparison prospective and blocked on authoritative source/target provenance, a verified group-disjoint development split, frozen inputs/protocol, and available compute. Do not reopen B2/B3 while B1 is not_promoted.

## TUM SEN1-2 record metadata route (2026-10-24)

- [x] Check TUM's current dataset catalogue and publication route for a public SEN1-2 sample/scene manifest. The catalogue confirms only the official published count and links the paper; the expected mediaTUM record returned HTTP 429. No local lineage was established. See the [route audit](../../experiments/data_audits/tum_sen1_2_record_metadata_route_20261024_v1/report.md).
- [ ] Recover an authorized exact upstream archive/version and a verifiable patch-to-parent-scene/source-group map for the local numeric 2k subset. Do not infer lineage from count or numeric filenames, and do not select/promote models using the contaminated local development partitions.

## FAF-CD prior-art boundary (2026-10-24)

- [x] Review the full arXiv manuscript, official MONTI workshop page, and author repository README. Record its feature-rectification/frequency-gated change-detection scope and distinguish modality-side appearance/radiometric stress from geometric mismatch testing. See the [audit](../../experiments/literature_audits/faf_cd_imperfect_multimodal_stress_20261024_v1/report.md) and D-2026-10-24-40.
- [ ] Keep any restoration claim limited to SAR-output evidence. Revisit FAF-CD only if a new version reports controlled geometric shifts or SAR restoration/fallback evaluations; do not transfer its change-detection metrics to the local B1 gate.

## SEORation/OpenSEP pair-curation evidence (2026-10-24)

- [x] Review indexed primary-manuscript excerpts and the official MONTI page. Record that QXS-SAROPT is external to OpenSEP construction and used for bidirectional retrieval evaluation; RSLIP scene compatibility is not pixel-registration evidence. See the [audit](../../experiments/literature_audits/seoration_opensep_pair_curation_20261024_v1/report.md) and D-2026-10-24-41.
- [ ] Keep local QXS/SEN1-2 source identity and parent-scene groups unresolved until an authorized versioned source archive or authoritative crosswalk is available. Do not infer local lineage from the shared QXS name or use the paper's retrieval scores as restoration evidence.

## Gaya thesis route follow-up (2026-10-25)

- [x] Check official Télécom Paris group pages for a direct Gaya thesis route. The pages list the 2025 thesis but provide no document link; no thesis text was obtained.
- [x] Search theses.fr and HAL using the exact thesis title/author route. No candidate thesis record surfaced in the available web search; do not treat this as proof of unavailability or repeat the same queries without a new direct link.
- [x] Check the official IGARSS paper detail and proceedings links for the 2024 full text. The detail page lists no paper resources and the proceedings gateway is closed/JavaScript-only in the available reader; no PDF was obtained. Do not repeat this route unless access changes; see the updated [2024 audit](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).
- [x] Follow the newly surfaced official GRETSI archive link to the full 2023 Fuse-MERLIN paper; record its method, one-scene qualitative evidence, explicit lack of quantitative validation, and preliminary two-pixel registration statement in the [audit](../../experiments/literature_audits/gaya_fuse_merlin_gretsi2023_fulltext_20261025_v1/report.md).
- [x] Sync the 2023/2024 distinction into the project-facts index and README; remove the duplicate 2024 Gaya entry from the facts index.
- [ ] Keep the 2023/2024 version relationship unresolved unless a publisher or author source explicitly links the papers. Do not use either paper as a local baseline or as evidence of joint SR, systematic mismatch robustness, or quantitative fallback. See [D-2026-10-25-66](DECISION_LOG.md#d-2026-10-25-66--classify-fuse-merlin-as-single-scene-qualitative-despeckling-prior-art).

## Run-interface approval search disclosure (2026-10-25)

- [x] Disclose the truncated historical decision-log text search conservatively as `test_set_accessed=true`, `test_set_used=false`; no test payload or metric file was opened and no value was used. Preserve historical run flags and cohort exclusions. See [D-2026-10-25-67](DECISION_LOG.md#d-2026-10-25-67--disclose-truncated-run-interface-decision-log-search) and the [access audit](../../experiments/data_audits/run_interface_approval_decision_log_text_exposure_20261025_v1/report.md).
- [ ] Keep future run-interface searches scoped to the source files and the single relevant decision entry; do not repeat broad historical decision/state searches.

## SpeckleDiff-2D SAR despeckling diffusion prior art (2026-10-25)

- [x] Record the IEEE abstract and official IGARSS program as abstract-level prior art: SAR-despeckling diffusion trained on optical-dataset-derived synthetic speckle. The abstract does not expose inference inputs or establish joint SR, optical-at-inference use, mismatch robustness, or fallback. See the [audit](../../experiments/literature_audits/specklediff2d_abstract_screen_20261025_v1/report.md).
- [x] Add SpeckleDiff-2D to the current literature matrix as abstract-level SAR diffusion despeckling prior art; keep historical matrix membership and all unreported protocol details unresolved. See D-2026-10-26-080.
- [ ] Review the full paper or official code only if a direct accessible publisher/author route appears. Keep local diffusion deferred; this citation does not reopen B1/B2/B3 or authorize training.

## SAR degradation-domain boundary (2026-10-25)

- [x] Derive the conditional moments for the current representation-domain Gamma multiplier and the conditions under which an unclipped inverse log transform has finite moments. Preserve the current condition as a representation-domain stress test; see [audit](representation_domain_gamma_composition_20261025.md) and [D-2026-10-25-12](DECISION_LOG.md#d-2026-10-25-12--bound-the-inverse-moments-of-representation-domain-gamma-noise).
- [x] Audit `configs/dev2k_formal/degradation_sensitivity.json` by metadata only. It is predeclared for post-freeze evaluation on `fixed_test_dev2k`, uses representation-domain looks 1/2/4/8, and sets `test_set_used_for_selection=false`; it is not a development run. See D-2026-10-25-47.
- [x] Inspect the sensitivity runner and freeze verifier as source only. They require a completed freeze, promoted B1 gate, and matching frozen config hash before the runner reads the test split identities. The current B1 gate is `not_promoted`; no test payload was opened and no code change was needed.
- [ ] Keep this configuration sealed until a method is frozen and final-test evaluation is authorized. Do not use it to select degradation settings, models, or claims.
- [ ] Before proposing a physical-speckle experiment, recover authoritative calibrated-domain input metadata and define/validate the noise model in that domain. This is not an active experiment or a prerequisite for source-independent research under the current gate.

## Prospective parent-ROI development cohort (2026-10-25)

- [x] Aggregate the authenticated v2 train/validation group assignment by parent ROI-season. The v2 validation is all fall; train has 1,494 fall pairs and only 106 summer pairs from one folder. Source-folder disjointness does not supply parent-ROI independence. See the [feasibility audit](../../experiments/data_audits/sen12_scene_dev1800_parent_roi_feasibility_20261025_v1/report.md) and [D-2026-10-25-13](DECISION_LOG.md#d-2026-10-25-13--do-not-derive-a-parent-roi-holdout-from-the-v2-cohort).
- [x] Screen the author SEN12MS-to-SEN1-2 scene-index route against both primary dataset papers and a pinned author README. Shared ROI roots are documented, but no stable per-scene `s1_i` crosswalk is; no archive or split-list content was accessed. See [D-2026-10-26-018](DECISION_LOG.md#d-2026-10-26-018--keep-sen12ms-scene-index-crosswalk-unverified) and the [audit](../../experiments/literature_audits/sen12ms_sen1_2_scene_crosswalk_20261026_v1/report.md).
- [ ] Obtain authorized non-test metadata mapping source-scene folders to parent ROI locations/acquisition products, with enough unused scenes across multiple ROI-sampling-batch/season roots for a prospective train/validation split. Do not treat folder names alone, matching seed roots, or equal `s1_i` strings across releases as geographic/acquisition IDs. If justified, freeze a new B1 screen with the registered gate unchanged before training or selection; do not recycle the already-used v2 validation identities as a holdout. Keep B2/B3 gated unless a prospective B1 screen passes. The 2026-10-25 runtime preflight confirmed no CUDA device in `AISAR`; defer runs until a fresh preflight shows driver/device availability and the split prerequisite is met.



- [x] Quantify group-equal uncertainty for the existing three-seed source-aligned B1 diagnostic using the predeclared 15-folder bootstrap; the very small PSNR mean is positive while SSIM is negative, and no gate changed. Keep the open parent-ROI/source-metadata prerequisite above as the next model-selection blocker. See [D-2026-10-26-015](DECISION_LOG.md#d-2026-10-26-015--record-conditional-source-folder-bootstrap-uncertainty) and the [diagnostic](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_bootstrap_20261026_v1/report.md).

- [x] Clarify the SEN1-2 hierarchy: `ROIs<seed>_<season>` is an ROI-sampling-batch/season root, and `s1_i`/`s2_i` are source-scene folders. Keep scene-folder disjointness distinct from unverified geographic/acquisition independence; see [D-2026-10-26-016](DECISION_LOG.md#d-2026-10-26-016--clarify-sen1-2-root-and-scene-folder-units) and the [audit addendum](../../experiments/data_audits/sen12_scene_dev1800_parent_roi_feasibility_20261025_v1/report.md#addendum--official-folder-hierarchy-clarification-2026-10-26).

## Amieva et al. SM subaperture prior-art audit (2026-10-25)

- [x] Review the arXiv v1 full text and official CVF proceedings metadata for the distinct SAR-only Stripmap subaperture-enhancement paper. Record the noisy full-aperture target, same-grid task, whole-dataset percentile preprocessing caveat, two-scene test, reported smoothing/detail trade-off, and unresolved code/data lineage. Keep it separate from the authors' IW→SM journal paper. No local experiment or gate follows. See [D-2026-10-25-17](DECISION_LOG.md#d-2026-10-25-17--classify-amieva-sm-subaperture-enhancement-as-sar-only-prior-art).
- [x] Add this distinct Amieva et al. (2026) study to the current matrix as SAR-only same-grid enhancement; retain the noisy-reference, normalization, and task-boundary caveats, and leave historical membership unknown. See D-2026-10-26-083.


## Dong et al. complex-valued SAR SR prior-art screen (2026-10-25)

- [x] Record the available abstract and primary-paper citation as bounded prior art for subaperture-based complex SAR SR; keep the reported percentages author-attributed and the unavailable protocol explicit. See [D-2026-10-25-18](DECISION_LOG.md#d-2026-10-25-18--classify-dong-complex-sar-sr-as-abstract-level-prior-art).
- [ ] Revisit exact task, scale, data/split, metric/comparator, despeckling objective, and code lineage only if the publisher full text or a direct author-hosted version becomes accessible. Do not infer these details from secondary summaries or use this screen to authorize training/model selection.


## MMCANetV2 source architecture boundary and search disclosure (2026-10-25)

- [x] Trace the model-level optical-absence path, SAR base, optional alignment, local cross-attention, residual filter, and structural gate target. Record code-level fallback separately from mismatch robustness. See [D-2026-10-25-19](DECISION_LOG.md#d-2026-10-25-19--distinguish-absent-optical-fallback-from-mismatch-robustness).
- [x] Disclose that a broad source-location search had a truncated response and may have surfaced test-associated result text; no values were used and test_set_accessed is conservatively true. See the [access report](../../experiments/data_audits/architecture_search_text_exposure_20261025_v1/report.md).
- [ ] Do not run B2 shifts, wrong-texture rejection, stale-content, or fallback promotion tests unless the registered B1 gate first passes on a valid prospective group-disjoint development cohort. The source audit did not measure model performance, complexity, or runtime.
- [ ] If a valid prospective model run becomes authorized, record trainable/total parameters, reliable MACs/FLOPs, latency, peak memory, and clamp/high-brightness behavior alongside zero/missing/shift/wrong-texture diagnostics.

## NEXT_ACTIONS search text-exposure disclosure (2026-10-25)

- [x] Conservatively record the truncated unchecked-action search as test-associated text access, with no measurement used and no payload or metric file opened. Preserve historical run flags and do not use any potentially surfaced cohort for future model selection, tuning, or manuscript claims. See the [access report](../../experiments/data_audits/next_actions_search_text_exposure_20261025_v1/report.md) and [manifest](../../experiments/data_audits/next_actions_search_text_exposure_20261025_v1/access_manifest.json).

## FIL-Net SIFT annotation provenance follow-up (2026-10-25)

- [x] Check the lead-author homepage, IEEE article route, and surfaced OpenReview route for a primary annotation method. No full text, supplement, or author-controlled generation script was accessible; keep the third-party point list out of registration-ground-truth claims. See the [route audit](../../experiments/literature_audits/filnet_annotation_method_route_20261025_v1/report.md) and [manifest](../../experiments/literature_audits/filnet_annotation_method_route_20261025_v1/access_manifest.json).
- [ ] Revisit only if the full primary paper/supplement or an author-controlled annotation script becomes directly accessible; do not repeat the same blocked routes.

## README reproduction and roadmap alignment (2026-10-25)

- [x] Label the historical CPU pairing command as audit-only and currently non-runnable because its generated validation payloads were pruned; align the roadmap with the source-group split prerequisite and sequential B1/B2/B3 gates. No experiment or gate changed.


## Current-state search text-exposure disclosure (2026-10-25)

- [x] Record the historical held-out aggregate summary surfaced by a targeted `CURRENT_STATE.md` search as `test_set_accessed=true`, `test_set_used=false`; exclude that cohort from all future selection/tuning and preserve historical records. See [the access audit](../../experiments/data_audits/current_state_search_text_exposure_20261025_v1/report.md) and D-2026-10-25-31.


## Multiplicative-speckle theory boundary (2026-10-25)

- [x] Derive the scalar intensity-domain LMMSE optical-utility trend for independent unit-mean Gamma speckle with fixed optical reliability. The result is monotone only under those assumptions and does not authorize a local screen.
- [ ] Keep any severity-by-optical experiment gated on authorized signal-domain/source metadata and a prospective source-group-disjoint split; predeclare a test for nonmonotonicity if those prerequisites are met.


## SEN1-2 patch-matching train-list route check (2026-10-25)

- [x] Check the author repository tree and training-list page without opening list contents. The 60.6 MB train list is not rendered by the current reader and has no visible commit pin; this route did not produce local source-group counts or a numeric-ID crosswalk. See the [audit](../../experiments/literature_audits/sen1_2_patch_train_list_metadata_route_20261025_v1/report.md) and D-2026-10-25-33.
- [x] Close this route for local source grouping: the repository is a separate patch-matching annotation set and no exact mapping to local numeric restoration IDs exists. Do not fetch `finally_train.txt` or inspect `finally_test.txt` without a new, verified crosswalk. See [D-2026-10-26-024](DECISION_LOG.md#d-2026-10-26-024--close-the-patch-matching-train-list-route-for-local-source-grouping).

## Noisy target and degraded-input dependence (2026-10-25)

- [x] Trace the hash-bound B1 v2 train/validation preprocessing metadata and code. The HR pseudo-target and synthetic LR share the same preprocessed SAR tensor; the LR then receives the configured representation-domain multiplier. Record the bounded one-mode risk derivation in `optical_information_analysis.md` and D-2026-10-25-34. No data or metric payload was opened.
- [x] State in the pilot Method and Limitations that LR is generated from the same SAR observation as the noisy HR pseudo-target, so full-reference metrics measure agreement with a correlated proxy and cannot establish clean-SAR recovery. The code trace does not identify target/input noise covariance or prove target-noise independence; see the [claim–evidence map](claim_evidence_map.md#same-observation-pseudo-target-boundary-2026-10-25) and D-2026-10-25-55.
- [ ] Before any future model screen, audit whether any reference is independent of the input SAR observation and what evidence supports target-noise independence. If no independent clean reference exists, retain proxy-target language and do not claim clean-SAR utility. Keep current B1/B2/B3 gates unchanged.

## Multivariate pseudo-target risk ordering (2026-10-25)

- [x] Extend the one-mode derivation to a vector linear-Gaussian degradation and construct a 2-D counterexample where optical risk reduction against the pseudo-target exceeds that against the latent signal. Together with the scalar attenuation case, this rules out a universal bias direction even under independent additive residual assumptions. See [D-2026-10-25-35](DECISION_LOG.md#d-2026-10-25-35--bound-multivariate-pseudo-target-risk-ordering) and the [analysis](optical_information_analysis.md#a-multivariate-counterexample-to-one-mode-target-risk-ordering-2026-10-25).
- [ ] Keep the target/input and residual-independence audit as a prerequisite for any new source-valid model comparison. The derivation does not authorize a local correction, experiment, or gate change.

## Draft bibliography and coverage audit (2026-10-25)

- [x] Check all seven citation keys originally used by the draft against publisher, author, or institutional metadata; no metadata mismatch was found. See the [audit](../../experiments/literature_audits/draft_bibliography_integrity_20261025_v1/report.md).
- [x] Add OMA-SSR, HSFMamba, and the original SFT citation to Related Work/BibTeX, `related_work_matrix.csv`, and the scoped `claim_evidence_map.md` trace; link SFT to the local SFTLayer implementation without claiming exact paper reproduction. Distinguish the prior task contracts and keep reported results out of the local claim. See the [integration audit](../../experiments/literature_audits/related_work_integration_20261025_v1/report.md).
- [x] Disclose the result-bearing manuscript text surfaced during the SFT matrix sync. Do not use the historical cohort for any future selection, tuning, promotion, or quantitative claims; preserve the original run flags. See the [access audit](../../experiments/data_audits/sft_matrix_sync_manuscript_text_exposure_20261025_v1/report.md) and D-2026-10-25-42.
- [ ] Complete the broader claim-to-source map and remaining Related Work coverage before reusing the draft. Do not update local performance claims until the prospective data/split and experiment gates support them.

## Pilot manuscript section text-exposure disclosure (2026-10-25)

- [x] Record the targeted Introduction/Related Work read that surfaced qualitative held-out result prose, with `test_set_accessed=true`, `test_set_used=false`; no numeric performance metric, table, image, array, or metric payload was opened, and no content was used. Preserve the historical cohort exclusion. See the [access report](../../experiments/data_audits/manuscript_section_text_exposure_20261025_v1/report.md) and [manifest](../../experiments/data_audits/manuscript_section_text_exposure_20261025_v1/access_manifest.json).
- [x] Complete the source-only citation/code/protocol/audit inventory for candidate claims C1–C9 in `claim_evidence_map.md`; distinguish candidate implementations from eligible results and list missing run/figure evidence. Do not query further result-bearing manuscript text for local quantitative claims. See D-2026-10-25-48.
- [ ] Bind local effectiveness/robustness claims to eligible prospective runs, checkpoints, per-image records, aggregates, and figures only after the source-group split and registered gates permit them. Keep unsupported claims unchanged meanwhile.

## Foundational bibliography trace (2026-10-25)

- [x] Map the ten current bibliography entries to primary/institutional metadata or the existing primary-source audit, and link local SwinIR, attention, SSIM, and SAR2SAR-topology adaptations with explicit reproduction/evidence boundaries. Six missing foundation rows were added to `related_work_matrix.csv`; the scoped trace is in `claim_evidence_map.md`. See D-2026-10-25-46.
- [x] Map the C3–C9 candidate code and retained protocol/audit records, with absent eligible runs/figures explicit; see the new C3–C9 trace in `claim_evidence_map.md` and D-2026-10-25-48. Keep the manuscript body closed for local result claims while the text-exposed pilot cohort remains excluded.

## Phase-A2 C8 artifact-path inventory (2026-10-25)

- [x] Verify the presence of the retained fixed-protocol record, baseline inventory, cascade/paired/joint summary tables, and per-image table by path only. Phase-A2 contains screen/confirmation manifests; no metrics, manifest bodies, checkpoints, predictions, or figure content were opened. Link the paths in the C8 row of `claim_evidence_map.md`; see D-2026-10-25-49.
- [x] Open Phase-A2 manifest JSON files adjacent to `command.txt` and extract only run status and test-access flags: all 27 checked runs record `test_set_used=false` and `test_set_accessed=false`. Other manifest fields were not examined or used. This metadata check does not establish metric validity or independent generalization; see D-2026-10-25-50.
- [ ] Keep the Phase-A2 comparison as a protocol-specific historical reference. Do not infer independent generalization or broad joint-over-cascade superiority from the contaminated shared development validation; no new run is authorized by this inventory.

## MMCANetV2 absent-optical interface check (2026-10-25)

- [x] Run `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_mmcanet_v2.py::test_v2_output_shape_and_zero_optical_fallback`; 1 passed on synthetic CPU inputs. The test verifies exact SAR-backbone fallback for missing/explicitly absent optical input and a finite, shape-correct normal-input forward. No result file was saved.
- [ ] Keep geometric-shift and wrong-texture robustness claims unsupported. The unit test is not a B2 evaluation; do not run that gate while B1 remains `not_promoted` and the source-group split prerequisite is unresolved.

## MambaR-ESRGAN prior-art update (2026-10-25)

- [x] Review the IEEE-accepted author manuscript and add its SAR-only joint SR/despeckling scope plus protocol limits to `related_work_matrix.csv`, `literature_review.md`, and `research_gaps.md`; see D-2026-10-25-52 and the [audit](../../experiments/literature_audits/mambar_esrgan_fulltext_audit_20261025_v1/report.md).
- [ ] Verify the final publisher metadata and any official code/data route only if a readable, authoritative source becomes available. Do not treat the reported setup as a task-matched local baseline or update manuscript performance claims from it.

## Pilot SAR-only joint-restoration citation integration (2026-10-25)

- [x] Add DSRKD, ISSMSAR, MD-Glow, and MambaR-ESRGAN to the draft Related Work/BibTeX and C9 prior-art map, with protocol and evidence boundaries; the four new keys parse in BibTeX. See D-2026-10-25-53.
- [ ] Continue the broader literature/source-coverage audit before reusing the pilot draft. Do not use the historical text-exposed cohort for selection, tuning, promotion, or quantitative claims; do not treat these sources as task-matched local baselines.



## Pilot optical-guided despeckling prior-art integration (2026-10-25)

- [x] Add the 2015 Optical-Driven Nonlocal filter, 2019 GNLM, and GCOANet to draft Related Work/BibTeX and the C9 map. Preserve the single-scene, qualitative stale-guide, and abstract-only boundaries; the three new BibTeX keys parse. See D-2026-10-25-54.
- [x] Resolve the GCOANet author-name conflict from the official ScienceDirect indexed record: use “Zeming Geng”; keep the conflicting secondary spelling out of the verified record. The direct publisher page remains blocked, and full-text protocol review stays open.
- [x] Integrate Gaya et al.'s abstract-level self-supervised optical-assisted SAR-despeckling prior art into the draft/BibTeX and C9 source map. Do not infer its unexposed target, inference, registration, or result protocol; see D-2026-10-25-65 and the [source screen](../../experiments/literature_audits/gaya_multimodal_sar_despeckling_2024_screen_20261025_v1/report.md).
- [ ] Keep the broader claim-to-source audit open; the accessible Gaya evidence does not support joint SR, controlled mismatch robustness, or a quantitative SAR-only fallback.

## Same-observation pseudo-target clarification (2026-10-25)

- [x] Bind the method statement to the preparation code: the HR target tensor is passed directly into LR degradation and also saved as the reference. Clarify in the pilot draft that metrics are correlated pseudo-target agreement, not independent clean-SAR recovery. No run, metric, or gate changed; see D-2026-10-25-55.
- [x] Disclose that the bounded manuscript read surfaced qualitative held-out-result prose and a pooled LR input-range summary with unspecified split scope. Neither was used; treat the summary as potentially test-associated and record `test_set_accessed=true`, `test_set_used=false`. No image, array, checkpoint, or metric payload was opened; preserve the historical cohort exclusion. See the [access audit](../../experiments/data_audits/manuscript_method_conclusion_text_exposure_20261025_v1/report.md).
- [ ] Keep local effectiveness and robustness claims gated on a valid prospective source-group split and eligible evidence; B1 remains `not_promoted`, and B2/B3 remain gated.

## Gamma multiplier notation (2026-10-25)

- [x] Match the pilot equation to the sampler source: shape=`L`, rate=`L`, unit mean and variance `1/L`. This resolves notation only; keep the representation-domain and non-calibrated-physics limits explicit. See D-2026-10-25-56.
- [x] Run synthetic moment regressions for intensity, representation (`L=2`), and amplitude-square multipliers; all three pass. The representation case directly covers the pilot equation's configured mode/look count. This validates sampler moments only, not local data or physical speckle assumptions.

## Reproduction-path search text-exposure disclosure (2026-10-25)

- [x] Record that the bounded README/config/script/research-doc search returned a capped, truncated result set. No test payload was opened and no visible result was used; because remaining matches cannot be inspected, set `test_set_accessed=true`, `test_set_used=false` conservatively. Preserve the historical cohort exclusion; see D-2026-10-25-57 and the [access audit](../../experiments/data_audits/reproduction_path_search_text_exposure_20261025_v1/report.md).

## MHFNet author repository route (2026-10-25)

- [x] Inspect the public source ZIP statically after the reader-only route could not preview it. Record the RGBT filename pairing, four-channel formatting, the apparent loader/YAML channel-order question, and the two-branch DCNv4/CGA detection architecture. Treat the channel-order observation as unresolved because the archive is mutable and the exact article run/data binding is not available. See D-2026-10-25-62 and the [audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).
- [ ] Revisit only if a version-pinned article source/run mapping or a maintainer response becomes available. Then check dataset versions/groups, mismatch construction, label handling, and the observed channel-order question. Do not treat the detector as a SAR restoration baseline.

## Run-interface search exposure (2026-10-25)

- [x] Disclose the broad truncated historical-text search as `test_set_accessed=true`, `test_set_used=false`; no payload was opened or content used. See D-2026-10-25-59 and the [audit](../../experiments/data_audits/run_interface_decision_log_text_exposure_20261025_v1/report.md).

## MHFNet prior-art integration (2026-10-25)

- [x] Add MHFNet to Related Work as downstream misalignment-aware ship-detection prior art and state the restoration boundary. Add its DOI-backed reference; do not transfer detection results to restoration. See D-2026-10-25-60 and the [source/integration audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).
- [ ] Continue the broader Related Work claim-to-source audit; do not make novelty-by-absence claims from the current non-exhaustive search.

## MHFNet repository issue follow-up (2026-10-25)

- [x] Review public issues #1 and #2. Treat their unanswered questions/allegations as unverified reports; do not infer code defects or protocol behavior from them. See D-2026-10-25-61 and the [audit](../../experiments/literature_audits/mhfnet_author_repository_20261025_v1/report.md).
- [x] Inspect the currently linked mutable archive to resolve part of the dual-branch loader question without executing code or downloading data. The exact article-run mapping and issue #2 allegation remain unverified.
- [ ] Revisit only if a maintainer response or version-pinned source/run mapping provides new evidence.

## RQ-Net optical–SAR registration prior art (2026-10-25)

- [x] Screen the official publisher-indexed text for RQ-Net's task and architecture; classify it as registration/matching prior art, not SAR-output restoration evidence. Direct full-text opening failed; do not claim detailed split, perturbation, or code verification. See D-2026-10-25-63 and the [audit](../../experiments/literature_audits/rqnet_registration_indexed_primary_source_20261025_v1/report.md).
- [x] Add the bounded claim to the pilot Related Work, bibliography, literature review, related-work matrix, and claim map.
- [x] Compile and visually check a temporary Related Work-only excerpt; the citation resolves. The full manuscript was not rebuilt, and preview files were removed.
- [ ] Revisit exact datasets/splits, controlled shift/warp evaluation, and code lineage only if a directly accessible publisher/author full text or versioned code route becomes available. Do not repeat the blocked page-open/search routes.

## Pilot abstract text-exposure disclosure (2026-10-25)

- [x] Record the historical abstract text exposure as `test_set_accessed=true`, `test_set_used=false`; no value was used and no test payload or result-bearing file was opened. Preserve historical run flags and exclude this cohort from future selection/tuning. See D-2026-10-25-64 and the [audit](../../experiments/data_audits/draft_abstract_text_exposure_20261025_v1/report.md).


## Workspace cleanup and approved run-interface repair (2026-10-25)

- [x] Inventory artifact roots by path, type, and size only. No exact cache, smoke, scratch, temporary, or generated array candidates were found; no files were deleted. See [cleanup record](../../experiments/artifact_cleanup_continuation_20261025_v4.json) and [D-2026-10-25-86](DECISION_LOG.md#d-2026-10-25-86--disclose-cleanup-inventory-text-exposure-and-record-no-deletion).
- [x] Confirmed the approved MMCANetV2 residual-filter run-interface repair is already implemented: training applies the configured kernel/sigma to both model and loss, checkpoint metadata records the model filter, and inference restores it. See [D-2026-10-23-46](DECISION_LOG.md#d-2026-10-23-46--bind-mmcanetv2-residual-filters-to-one-run-configuration); no duplicate source edit was needed.
- [x] Record truncated metadata/document text exposure as `test_set_accessed=true`, `test_set_used=false`; no surfaced value was used and no test payload was opened. See the [access audit](../../experiments/data_audits/workspace_cleanup_access_exposure_20261025_v1/report.md).


## Fixed SAR representation contract (2026-10-25)

- [x] Audit the fixed-development target/degradation contract from code and config only. It is an 8-bit-PNG-derived log representation with representation-domain Gamma degradation; source radiometry and an independent clean reference remain unresolved. See [D-2026-10-25-87](DECISION_LOG.md#d-2026-10-25-87--bound-the-fixed-dev-sar-target-to-its-representation-domain) and the [audit report](../../experiments/data_audits/fixed_val_sar_representation_contract_20261025_v1/report.md).
- [ ] Resolve original PNG lineage and source-group metadata before using physical SAR terminology or prospective model selection. Preserve B1/B1R/B1F history and keep B2/B3 gated.


## Local 2k source lineage (2026-10-25)

- [x] Recheck the official QXS-SAROPT repository and sibling SOTANet cache metadata. No authoritative file-to-scene/group crosswalk is available through those routes; the numeric 2k lineage remains unresolved. See [D-2026-10-26-054](DECISION_LOG.md#d-2026-10-26-054--keep-local-lineage-unresolved-and-sen12tp-unadopted) and the [route audit](../../experiments/data_audits/source_lineage_routes_recheck_20261026_v1/report.md).

- [x] Compare the local train/validation overlap geometry with the original QXS-SAROPT and SEN1-2 tiling descriptions using primary sources. QXS is geometrically compatible under one reading, but lineage remains unresolved. See [D-2026-10-25-88](DECISION_LOG.md#d-2026-10-25-88--keep-local-2k-source-unresolved-despite-qxs-compatible-overlap) and the [audit](../../experiments/data_audits/local_2k_source_lineage_candidate_20261025_v1/report.md).
- [ ] Recover a versioned source archive/manifest or authoritative source-group crosswalk for local basenames 8000–9999. Do not infer source from geometry or filenames. If unavailable, use a new prospective split only after independent source groups and spatial buffers are verified. Keep model selection paused on contaminated development partitions.


## SEN1-2 byte-crosswalk follow-up (2026-10-25)

- [x] Reconcile the stored train/validation SEN1-2 upstream hash crosswalk with its result CSV and current split/source-map metadata hashes. The 3,600 rows have zero exact byte matches; current payload bytes were not re-read. See [D-2026-10-25-89](DECISION_LOG.md#d-2026-10-25-89--record-zero-exact-sen1-2-source-matches-with-current-metadata-binding).
- [ ] Recover an authoritative source manifest/crosswalk or verified upstream archive before assigning identity; exact-byte non-matches do not rule out re-encoding or re-tiling. Keep model selection paused until source groups and a prospective group-disjoint split are verified.


## Future dataset labels (2026-10-25)

- [x] Correct new split manifests and report generators to preserve the project-labeled 2k subset name while marking upstream source/version unverified; leave historical outputs untouched. See [D-2026-10-25-91](DECISION_LOG.md#d-2026-10-25-91--label-source-as-unverified-in-future-generated-artifacts).
- [ ] Keep source attribution provisional until an authorized versioned archive or authoritative source-group crosswalk is recovered.


## QXS-named local cache crosswalk (2026-10-25)

- [x] Rebind the stored pixel crosswalk to current train/validation source-map metadata and sibling repository revision. Exact pixel matches establish a local cache relationship, not official QXS-SAROPT provenance. See [D-2026-10-25-92](DECISION_LOG.md#d-2026-10-25-92--treat-qxs-named-cache-matches-as-local-lineage-only).
- [ ] Recover the sibling cache's authorized upstream archive/version, license, source groups, and original split semantics; do not use its `test/` name-only overlap as evidence of pixel identity. Keep source-specific claims and model selection paused.

## 2026-10-25 — next gate for the AOI route

1. [x] 审查通用 Sentinel 数据权利与 attribution 条款；80 个冻结候选的 STAC Item 元数据全部成功，并确认 S1 校准/噪声 XML、S2 反射率/SCL/产品元数据资源均有 catalog 条目。没有保存 asset href 或访问产品文件。见 [D-2026-10-25-104](DECISION_LOG.md#d-2026-10-25-104--bound-cdse-asset-metadata-and-repeat-stack-evidence) 与 [asset inventory](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/report.md)。
2. [x] 从冻结的原始 S1 元数据确定 12 个 AOI 均有同平台/轨道方向/相对轨道的 3+ acquisition 组，最长组跨度 47–59 天；这只是多时相伪参考可行性，不能证明稳定性或独立干净 target。见 [repeat summary](../../experiments/data_audits/cdse_candidate_asset_inventory_20261025_v1/s1_repeat_summary.json)。
3. [blocked: registered S3 access required] The selected CDSE assets use `s3://eodata`; official S3/OData product access requires registered-account credentials, and none are configured in this environment. Do not repeat unauthenticated requests. See [D-2026-10-25-105](DECISION_LOG.md#d-2026-10-25-105--pause-cdse-sidecar-fetch-until-authenticated-access-is-configured) and the [access-path audit](../../experiments/data_audits/cdse_sidecar_schema_review_20261025_v1/report.md). Resume the already frozen five-item/11-sidecar sample once a secure credential source is available.
4. [ ] While CDSE sidecars are blocked, continue independent, source-backed literature/protocol work; do not claim calibration or reference validity from catalog metadata alone.
5. [ ] After sidecar review, resolve authorized source/acquisition groups, spatial buffers, target construction, and cloud/registration protocol before pre-registering a prospective train/development split. Do not create a split, train, or evaluate before that audit passes.
6. Maintain B1 `not_promoted`, keep B2/B3 gated, and do not read test imagery or metrics.

证据：[屏查报告](../../experiments/data_audits/cdse_global_aoi_group_feasibility_20261025_v1/report.md)。

## STRATA-S12 current availability and rights recheck (2026-10-25)

- [x] Recheck the current official Hugging Face card and European Commission Sentinel Data Legal Notice. The card reports the full corpus upload complete and integrity-verified on the Hub; the legal notice states attribution requirements for modified Sentinel data. No archive or image was fetched, so the card's integrity statement remains author-reported. See [D-2026-10-25-144](DECISION_LOG.md#d-2026-10-25-144--update-strata-s12-full-corpus-availability-and-sentinel-attribution-record) and the [D_lite audit follow-up](strata_s12_dlite_audit.md#current-full-corpus-availability-and-upstream-terms-2026-10-25).
- [x] Review the author-linked evaluation harness README: it covers frozen-representation transfer to reBEN classification, not SAR restoration or alignment. The current Zenodo API reader route was inaccessible; do not repeat it without an access change. See [D-2026-10-25-145](DECISION_LOG.md#d-2026-10-25-145--classify-the-strata-s12-evaluation-harness-as-representation-transfer-evidence).

## Research-record text-output disclosure (2026-10-25)

- [x] Record the two truncated text-only inspection outputs, exclude their surfaced spans from evidence, and retain cumulative flags `test_set_accessed=true`, `test_set_used=false`. No test payload was opened and no surfaced value was used. See [D-2026-10-25-106](DECISION_LOG.md#d-2026-10-25-106--disclose-truncated-research-document-text-outputs) and the [access audit](../../experiments/data_audits/research_record_search_truncation_20261025_v1/report.md).

## Misalignment-aware restoration prior-art audit (2026-10-25)

- [x] Rebuild the updated Related Work draft in a temporary output directory; both new citations resolve and the affected page was visually checked. Remove the temporary PDF/render afterward.
- [x] Add Shibata et al.'s cross-modal translated-guidance filter and Bungert/Ehrhardt's joint reconstruction-registration method to the source map and Related Work; narrow the novelty boundary to the unresolved SAR-specific question. No local experiment or gate changed. See [D-2026-10-25-107](DECISION_LOG.md#d-2026-10-25-107--bound-general-misalignment-aware-restoration-prior-art) and the [source audit](../../experiments/literature_audits/misalignment_robust_reconstruction_prior_art_20261025_v1/report.md).

## Approved run-interface request (2026-10-25)

- [x] Confirm the approved MMCANetV2 residual-filter interface repair is already present under D-2026-10-23-46; avoid a duplicate source edit or redundant test run.
- [x] Disclose the truncated documentation output conservatively as cumulative `test_set_accessed=true`, `test_set_used=false`; no payload or result was used. See [D-2026-10-25-109](DECISION_LOG.md#d-2026-10-25-109--confirm-approved-residual-filter-interface-and-disclose-truncated-context-read) and the [access audit](../../experiments/data_audits/run_interface_approval_context_text_exposure_20261025_v1/report.md).

## Artifact cleanup inventory (2026-10-25)

- [x] Recheck current output, experiment, checkpoint, and data roots by path/type/size. No further disposable cache, array, temporary, or explicitly named smoke/failed artifact remained; keep referenced records and checkpoints. See [D-2026-10-25-110](DECISION_LOG.md#d-2026-10-25-110--retain-remaining-artifacts-after-bounded-cleanup-inventory) and the [inventory](../../experiments/artifact_cleanup_current_inventory_20261025_v1.json).

## RABASAR temporal-superimage conditions (2026-10-25)

- [x] Review the full RABASAR TGRS paper and cross-check SAR2SAR/MERLIN independence and change assumptions. Record temporal stability and speckle decorrelation as separate conditions; see [D-2026-10-25-113](DECISION_LOG.md#d-2026-10-25-113--keep-temporal-sar-means-as-conditional-pseudo-references) and the [audit](../../experiments/literature_audits/rabasar_temporal_superimage_assumptions_20261025_v1/report.md).
- [ ] Keep the SEN12TS temporal-reference question stopped unless materially new source/product evidence or an independent reference becomes available. The 2026-10-26 tile split and stack-window metadata screen do not meet that reopening condition. Do not transfer physical-domain assumptions to normalized PNGs or treat the temporal mean as clean SAR truth. See [D-2026-10-25-98](DECISION_LOG.md#d-2026-10-25-98--stop-sen12ts-temporal-reference-sampling-after-the-paired-audit) and [D-2026-10-26-002](DECISION_LOG.md#d-2026-10-26-002--treat-sen12ts-stack-windows-as-overlapping-acquisition-proxies).

## Active-goal document read (2026-10-25)

- [x] Finish reading the user-provided research-goal document from line 1501 through 2075 in five bounded sections. The earlier truncated 1501–1800 response remains excluded; the follow-up reads completed without truncation. See [D-2026-10-25-111](DECISION_LOG.md#d-2026-10-25-111--disclose-truncated-research-goal-document-output) and the [access audit](../../experiments/data_audits/goal_document_read_truncation_20261025_v1/report.md).
- [x] Select the next source-only literature task from the current research state: audit how multi-temporal SAR despeckling papers define pseudo-reference independence and temporal stability. No local result was used; see [D-2026-10-25-112](DECISION_LOG.md#d-2026-10-25-112--disclose-current-research-state-text-exposure).


## Temporal geometric-mean source review (2026-10-25)

- [x] Exclude the truncated HAL `hal-03095153v1` search output from evidence and record the public-source text disclosure. No local test payload or result was accessed or used; see [D-2026-10-25-114](DECISION_LOG.md#d-2026-10-25-114--disclose-truncated-temporal-geometric-mean-source-search) and the [audit](../../experiments/data_audits/geometric_mean_source_search_truncation_20261025_v1/report.md).
- [x] Verify citation metadata and the associated institutional code-repository route; record that neither paper body nor code files were accessible, without inferring method assumptions. See [D-2026-10-25-115](DECISION_LOG.md#d-2026-10-25-115--keep-temporal-geometric-mean-review-at-metadata-only-status) and the [source-route audit](../../experiments/literature_audits/temporal_geometric_mean_primary_route_20261025_v1/report.md).
- [ ] Resume technical review only if the full paper or code files become available from an authorized primary source. Do not repeat the same inaccessible HAL/DOI/IEEE/GitLab routes; continue independent source/protocol work meanwhile.


## Project-context search disclosure (2026-10-25)

- [x] Exclude the truncated combined research-record search from evidence. No surfaced historical value was used and no test payload was opened; see [D-2026-10-25-116](DECISION_LOG.md#d-2026-10-25-116--exclude-truncated-project-context-search-output) and the [audit](../../experiments/data_audits/project_context_search_truncation_20261025_v1/report.md).
- [ ] Choose the next source/protocol task from bounded, single-file task-status lines only; do not use the truncated search response.


## OSdataset rights route (2026-10-25)

- [x] Review the current official Google Earth Additional Terms and Geo Guidelines. They do not establish blanket rights to copy or redistribute a dataset-scale crop archive; the 2025 terms also do not determine the historical 2020 permissions. See [D-2026-10-25-118](DECISION_LOG.md#d-2026-10-25-118--keep-osdataset-rights-unresolved-under-current-google-earth-terms) and the [audit](../../experiments/literature_audits/osdataset_google_earth_terms_20261025_v1/report.md).
- [ ] Keep OSdataset unadopted. Resume only with source/provider identification plus version-specific terms or authoritative permission covering the intended research use and redistribution; do not download the archive on the CC-BY-NC label alone.


## Manuscript-build text exposure (2026-10-25)

- [x] Disclose that full-draft LaTeX QA processed three generated historical result fragments; no result page was viewed and no value used. Treat the unmapped historical cohort as accessed and exclude it from future selection/tuning. Preserve historical flags and gates. See [D-2026-10-25-108](DECISION_LOG.md#d-2026-10-25-108--disclose-historical-pilot-result-text-processed-by-manuscript-build) and the [audit](../../experiments/data_audits/paper_build_result_text_exposure_20261025_v1/report.md).


## Completed artifact cleanup follow-up (2026-10-25)

- [x] Remove four transient CDSE execution-error JSON outputs (4,352 bytes) while retaining audit summaries/manifests and the scientific error-analysis tables; see [D-2026-10-25-119](DECISION_LOG.md#d-2026-10-25-119--remove-transient-execution-error-outputs) and the [cleanup record](../../experiments/artifact_cleanup_error_outputs_20261025_v1.json).
- [x] Confirm the previously approved MMCANetV2 residual-filter run-interface repair is already implemented under D-2026-10-23-46; user reconfirmed approval on 2026-10-26, and no duplicate source edit or redundant validation run was needed. See D-2026-10-26-058.


## OPERA/Sentinel-2 route screen (2026-10-25)

- [x] Complete the frozen metadata-only 12-AOI screen and preserve the sanitized results and group summary; see [D-2026-10-25-121](DECISION_LOG.md#d-2026-10-25-121--stop-the-frozen-operasentinel-2-aoi-screen).
- [x] Stop this fixed date window after zero AOIs met the four-date optical-pair rule; do not widen dates or relax thresholds using this result.
- [ ] Continue with a separate evidence-led source/target route or literature task from the current research backlog. Do not repeat CDSE requests without authorized credentials or adopt QXS/OSdataset without resolved lineage/rights.


## Completed low-value artifact cleanup (2026-10-25)

- [x] Remove three transient failed metadata-attempt records from the SEN12TS temporal-reference pilot while retaining its final audit and successful summaries; see [D-2026-10-25-122](DECISION_LOG.md#d-2026-10-25-122--remove-transient-failed-metadata-attempt-outputs) and the [cleanup record](../../experiments/artifact_cleanup_error_outputs_20261025_v2.json).
- [x] Confirmed no repository `__pycache__`, pytest cache, generated `.npy`/`.npz` result arrays, or explicitly named smoke/scratch output remains in the experiment/output/checkpoint roots. `BasicSR/basicsr/metrics/niqe_pris_params.npz` is a required metric resource and was retained.


## B1 v2 source-group status and failure analysis (2026-10-25)

- [x] Recheck the separate `sen12_scene_dev1800_v2` cohort against the current group-split requirement: official source-scene folder mapping, 15/15 group separation, exact source-pixel verification, declared stride-grid checks, and current cohort/config/group hashes all match. Its scope remains one fall ROI-season; it does not prove acquisition or geographic independence.
- [x] Confirm the existing frozen three-seed source-aligned B1 aggregate is bound to that cohort and protocol (30/30 source/run/output hashes match), uses 200 validation identities per run, and retains false test-access/use flags. The result is `not_promoted`; no B2/B3 transition.
- [x] Analyze source-folder sign consistency without opening per-image rows: 10/15 across-seed group-mean ΔPSNR values were positive, while all 15 group-mean ΔSSIM values were negative. See [D-2026-10-25-123](DECISION_LOG.md#d-2026-10-25-123--keep-the-source-group-b1-screen-at-stop-at-b1) and the [diagnostic](../../experiments/diagnostics/sen12_scene_dev1800_b1_group_heterogeneity_20261025_v1/report.md).
- [x] Review noisy-target self-supervision assumptions against primary Noise2Noise, SAR2SAR, and multi-temporal MERLIN sources. The local LR/target pair shares one preprocessed SAR observation, and its conditional-mean relation to latent reflectivity is unverified; see [D-2026-10-25-177](DECISION_LOG.md#d-2026-10-25-177--bound-noisy-pseudo-target-interpretation-by-self-supervision-assumptions) and the [audit](../../experiments/literature_audits/noisy_pseudo_target_n2n_sar2sar_boundary_20261025_v1/report.md).
- [ ] Continue fine co-registration investigation only through materially new exact source-product mapping and spatially independent checkpoints against a documented reference. Do not transfer literature offsets, repeat blocked catalog routes, retrain the same candidate, use contaminated numeric-2k/B1F partitions for selection, or enter B2/B3.

## TUM source-product timestamp route outcome (2026-10-25)

- [x] Try the frozen single-record mediaTUM OAI-PMH route for record `1436631`. `Identify` returned HTTP 429 and `GetRecord` was inaccessible through the available reader; no metadata body or exact folder-to-product/timestamp mapping was obtained. Close this route as inconclusive rather than infer that the metadata is absent. See [D-2026-10-25-124](DECISION_LOG.md#d-2026-10-25-124--close-the-blocked-tum-producttimestamp-metadata-route) and the [audit](../../experiments/literature_audits/sen12_scene_product_timestamp_route_20261025_v1/report.md).
- [x] Keep source-product dates and fine co-registration unresolved after the blocked TUM route and official SEN1-2 paper review. Reopen only with an authorized direct metadata reader or materially new authoritative crosswalk; do not retry the blocked endpoint or estimate registration from B1 metrics. See [D-2026-10-25-124](DECISION_LOG.md#d-2026-10-25-124--close-the-blocked-tum-producttimestamp-metadata-route) and [D-2026-10-25-128](DECISION_LOG.md#d-2026-10-25-128--keep-sen1-2-paper-evidence-at-collection-level).
- [x] Disclose the truncated combined research-record tail read; no historical result was used and no test payload was accessed. See [D-2026-10-25-125](DECISION_LOG.md#d-2026-10-25-125--disclose-truncated-research-record-tail-output).
- [x] Audit official Sentinel-1/2 product geolocation specifications. They do not establish local S1-to-S2 mosaic residuals for the SEN1-2 fall cohort; retain the frozen nominal-georegistration wording and do not derive a shift level. See [D-2026-10-25-126](DECISION_LOG.md#d-2026-10-25-126--keep-product-geolocation-specifications-separate-from-local-s1s2-registration) and the [audit](../../experiments/literature_audits/sentinel12_official_geolocation_specs_20261025_v1/report.md).
- [x] Review an independent-checkpoint optical–SAR registration precedent. Liao et al. report spatially distributed manual checkpoints separate from adjustment controls, but use GF-03/GF-07/ZY-3 and do not quantify SEN1-2 residuals. Keep it as a protocol template only; see [D-2026-10-25-127](DECISION_LOG.md#d-2026-10-25-127--treat-independent-saroptical-checkpoints-as-a-validation-template-only) and the [audit](../../experiments/literature_audits/sar_optical_independent_tiepoint_validation_20261025_v1/report.md).
- [x] Screen the official SEN1-2 paper and current authoritative source records for exact fall product IDs and independent point-residual evidence. They establish collection-level processing but no product crosswalk or separate check points; this is a current evidence gap, not proof that metadata does not exist. See [D-2026-10-25-128](DECISION_LOG.md#d-2026-10-25-128--keep-sen1-2-paper-evidence-at-collection-level) and the [audit](../../experiments/literature_audits/sen1_2_independent_reference_availability_20261025_v1/report.md).
- [x] Check the TUM SIPEO open-datasets index and official GovData catalog metadata route. The pages add no product manifest; the Turtle export and package API could not be read. Do not repeat these routes unless the metadata endpoint becomes accessible. See [D-2026-10-25-141](DECISION_LOG.md#d-2026-10-25-141--keep-the-official-sen1-2-catalog-route-at-metadata-only-status).
- [ ] Reopen only if an authorized route supplies exact products, a reference with documented positional accuracy, and spatially separate check points. Until then, retain the registration limitation and do not derive a shift or enter B2/B3.


## DP-SAR abstract-level prior art (2026-10-25)

- [x] Record the IEEE indexed abstract as adjacent complex-PolSAR scattering-fidelity SR prior art; keep its author-reported metrics abstract-level and separate from the local PNG cohort. See [D-2026-10-25-136](DECISION_LOG.md#d-2026-10-25-136--keep-dp-sar-at-abstract-level-as-adjacent-polsar-sr-prior-art) and the [audit](../../experiments/literature_audits/dp_sar_pol_sar_sr_abstract_screen_20261025_v1/report.md).
- [ ] Verify author metadata and full text only if a new authoritative IEEE or author-controlled route appears. Until then, do not add it to the formal bibliography, use it as a baseline, or transfer its metrics to local experiments.
- [ ] Continue source-group and independent-registration evidence work under the existing gates. Do not repeat B1, enter B2/B3, or open test assets.


## PolSAR/SinSAR diffusion SR source lead (2026-10-25)

- [x] Check the newly surfaced lead against the IEEE document page, DOI, PDF route, and exact-title publisher search. The publisher page is a JavaScript/robot shell and no readable primary abstract or author manuscript was found; secondary listings remain excluded. See the [source-route audit](../../experiments/literature_audits/polsar_hr_sinsar_diffusion_source_route_20261025_v1/report.md).
- [ ] Reopen only if an accessible IEEE record, author-controlled page, or manuscript becomes available. Until then, do not put the item in the formal bibliography/matrix or use its secondary-listed methods/results.


## Evaluator metric implementation audit (2026-10-25)

- [x] Correct the even-size FFT radius center in the auxiliary high-frequency diagnostic; the regression failed before the fix and passed afterward, with 29 focused metric tests recorded under [D-2026-10-25-139](DECISION_LOG.md#d-2026-10-25-139--center-even-sized-fft-diagnostics-on-the-shifted-zero-bin).
- [x] Keep all pre-fix historical outputs unchanged and mark their high-frequency values as non-comparable to future values. No gate or model-selection decision changes.
- [ ] Full-suite verification remains pending: the base environment has `rasterio` but no `torch`; AISAR, ldm, and pytorch-img2img have `torch` but no `rasterio`. Do not install or alter dependencies; retry only if an existing environment changes. The 2026-10-26 environment recheck is recorded in D-2026-10-26-148.


## Dual-channel SAR–optical registration prior art (2026-10-25)

- [x] Screen the 2026 AIP Advances article and record its method, random pair-level Data-R split, synthetic geometry tests, and 200-pair SEN1-2 transfer evaluation. Do not treat it as local alignment ground truth or SAR restoration evidence; see [D-2026-10-25-140](DECISION_LOG.md#d-2026-10-25-140--treat-the-dual-channel-registration-article-as-prior-art-only).
- [ ] Reopen only if the publisher PDF, released dataset/code, or traceable source-group mapping becomes available. Continue local registration work only with exact product mapping and independently referenced, spatially separate check points; keep B2/B3 closed.

## Session text-read disclosure (2026-10-25)

- [x] Exclude three truncated current-state/research-record text outputs; no surfaced value was used and no test payload or raw metric was opened. Cumulative flags remain `test_set_accessed=true`, `test_set_used=false`. See [D-2026-10-25-143](DECISION_LOG.md#d-2026-10-25-143--disclose-truncated-current-state-text-reads).

## Artifact cleanup recheck (2026-10-26)

- [x] Recheck current artifact metadata against the cleanup request. No remaining experiment pixel arrays or clearly named low-value smoke/failed binary outputs were identified; prior cleanup records cover removed prediction payloads, caches, and incomplete-run detail.
- [x] Retain the runtime metric resource, completed A2 comparison checkpoints, source/configuration snapshot, formal result records, and compact negative-result audits. No additional deletion was warranted. See the [cleanup recheck](../../experiments/data_audits/artifact_cleanup_recheck_20261026_v1/report.md) and D-2026-10-26-004.

## Research-plan continuity reread (2026-10-26)

- [x] Reread the complete attached long-term research plan in nine bounded ranges (2,065 lines). The goal remains active; continue from current evidence and the highest-priority executable action. See [D-2026-10-26-005](DECISION_LOG.md#d-2026-10-26-005--record-full-research-plan-reread-and-thirty-two-output-groups).

## Research-record output disclosure (2026-10-26, follow-up)

- [x] Exclude the unseen remainder of the bounded action-index/path-search output and update the cumulative count to 33 groups. No result or test measurement was used. See [D-2026-10-26-006](DECISION_LOG.md#d-2026-10-26-006--extend-truncated-output-disclosure-to-thirty-three-groups).

## Research-record output disclosure (2026-10-26, audit-path inventory)

- [x] Exclude the unseen suffix of the bounded SEN12MS-CR audit-path listing and update the cumulative disclosure to 34 groups. No result or test payload was used. See [D-2026-10-26-007](DECISION_LOG.md#d-2026-10-26-007--record-truncated-sen12ms-cr-audit-path-inventory).

## Research-record output disclosure (2026-10-26, research-gap/diffusion excerpt)

- [x] Exclude the unseen suffix of the combined research-gap/diffusion excerpt and update the cumulative disclosure to 35 groups. No measurement or result was used. See [D-2026-10-26-008](DECISION_LOG.md#d-2026-10-26-008--record-truncated-research-gap-and-diffusion-excerpt).

## Research-record output disclosure (2026-10-26, dataset-route excerpt)

- [x] Exclude the unseen suffix of the bounded dataset-route decision excerpt and update the cumulative disclosure to 36 groups. No measurement or result was used. See [D-2026-10-26-009](DECISION_LOG.md#d-2026-10-26-009--record-truncated-multitemporal-dataset-decision-excerpt).

## Research-record disclosure (2026-10-26, source-cohort excerpt)

- [x] Exclude the unseen combined source-cohort excerpt and update the cumulative disclosure to 37 groups. No model measurement was used. See [D-2026-10-26-011](DECISION_LOG.md#d-2026-10-26-011--record-truncated-source-cohort-state-excerpt).


## Research-record disclosure (2026-10-26, matrix post-write/interface excerpts)

- [x] Exclude the unseen output from groups 216–217 and update the cumulative disclosure through 217. The SARGANet matrix addition is retained with its original prefix intact. The approved residual-filter run interface remains complete under D-2026-10-23-46, so no duplicate source edit or test rerun was needed. See [D-2026-10-26-099](DECISION_LOG.md#d-2026-10-26-099--disclose-clipped-post-write-and-interface-context-outputs).


## Current related-work status synchronization (2026-10-26)

- [x] Synchronize current matrix counts in README, master plan, experiment registry, and research risks to 94 rows (74 confirmed historical memberships, 20 unresolved additions, 25–45 original entries unaccounted). Preserve dated older snapshots. The planned HSFMamba full-text audit already exists; the OGRN publisher full-text route remains unavailable, so no method details were added. See D-2026-10-26-100.


## 2022 OGRN DOI route boundary (2026-10-26)

- [x] Preserve the candidate DOI only as unverified audit metadata: the official author profile omits it and direct SPIE/DOI/Crossref routes were inaccessible. Do not add a bibliography entry or separate matrix row unless a primary record or authorized manuscript confirms the DOI and version relationship. See D-2026-10-26-101.


## Claim–evidence map coverage update (2026-10-26)

- [x] Map the publisher-audited SARGANet paper to a narrow SAR-only SR prior-art claim (`RW-SR-SAR`). Keep the broad related-work audit open; do not use its reported metrics as an untouched or task-matched comparison. See D-2026-10-26-102.


## C1/C2 B1 v2 evidence-map update (2026-10-26)

- [x] Add the prospective source-aligned B1 v2 run to C1/C2 claim traces. Preserve `not_promoted`, its +0.001617 dB mean ΔPSNR and −0.000892 mean ΔSSIM, and the one-root/15-folder scope. Keep B2/B3 gated. See D-2026-10-26-103.


## OGRN training-time optical prior-art mapping (2026-10-26)

- [x] Map OGRN 2024 as publisher-abstract-level prior art for optical-derived training features with SAR-only SR inference. Keep it distinct from real-optical inference, joint denoising, mismatch rejection, and mismatch-triggered fallback; leave the broader claim-to-source audit open. See D-2026-10-26-104.


## Optical-guided SAR-SR claim-map coverage (2026-10-26)

- [x] Add OS_SR as abstract-level optical-guided SAR-SR prior art under `RW-SR-OPT`, keeping it separate from OMA-SSR and preserving the unresolved version relationship. See D-2026-10-26-105. Existing action 628 retains the condition for any future full-protocol review.

## Muzeau et al. source route (2026-10-26)

- [x] Reopen the author-hosted EUSAR 2024 manuscript and check the linked VDE proceedings route. Add the source-backed row and claim boundary; keep current evidence separate from historical-matrix recovery. See D-2026-10-26-110.
- [ ] Reopen paper-to-code lineage only if a pinned author release or a readable official proceedings route becomes available.

## Gaya et al. GRETSI 2023 matrix coverage (2026-10-26)

- [x] Add the full-text-audited Fuse-MERLIN paper as a current-evidence row with its one-scene, qualitative-only, despeckling scope; preserve unknown historical membership and separate it from the 2024 IGARSS paper. See D-2026-10-26-116.
- [ ] Recover the exact pre-corruption title/DOI inventory from direct records before assigning historical membership to unresolved additions. The current matrix has 104 rows (77 confirmed; 27 unresolved); the known original-count lower bound is 125, so at least 21 records remain unaccounted for; if the original count was 125, the gap is 21–48.

## Requested artifact cleanup (2026-10-26)

- [x] Recheck workspace cache, generated-payload, smoke-name, and error/failure-name records using metadata only; remove stale pytest scratch and the temporary status snapshot found in `/tmp`. Zero workspace cache/generated-payload matches remain; evidence-bearing error records and all cleanup audits are retained. See D-2026-10-26-118.
- [x] Inspect the selected `outputs/` test100/archive/run directories by path, file type, and size. They contain run/audit records only; no prediction or metric payload was found in those inspected subtrees. Preserve those records. See D-2026-10-26-120.

## Authoritative status reconciliation (2026-10-26)

- [x] Put the current B1 stop decision, data/evaluation limits, 97-row matrix count, source-registration risk, and resource status at the top of `CURRENT_STATE.md`; correct the two stale README matrix counts. Historical entries remain intact. See D-2026-10-26-122.

## B1 v2 multi-metric failure diagnosis (2026-10-26)

- [x] Extend the existing PSNR/SSIM group diagnosis to EPI, edge-strength, TBR, and bright-target errors using the frozen 15-group validation summary. Group identifiers are not emitted or ranked; the original B1 v2 gate is unchanged. See D-2026-10-26-123.
- [ ] Keep B2/B3 gated. Continue only source/product lineage and independent fine-registration evidence work until the existing B1 decision and split requirements permit another phase.

## Final task-owned /tmp cleanup (2026-10-26)

- [x] Remove the remaining reproducible JDSR status/diff snapshots and generated bytecode from /tmp (52 files; 4,069,167 bytes). The partial matrix-recovery snapshot remains as cited evidence; workspace cache and generated-payload/smoke/scratch scans are clear. See D-2026-10-26-139 and the [cleanup report](../../experiments/data_audits/artifact_cleanup_tmp_continuation_20261026_v1/report.md).
- [x] Record clipped local text-search output as exposure group 316; exclude the clipped output and do not use any surfaced result value. Continue the existing rule against test-set selection.
- [x] Remove the remaining task-specific Python cache-prefix tree (43 files; 963,511 bytes) and one manifest-verified duplicate Phase-A2 checkpoint copy (22,090,399 bytes). Keep the matrix-recovery snapshot, canonical checkpoint, and registered research records. See D-2026-10-26-145.

## CDSE Sentinel-1 IW SLC metadata protocol review (2026-10-26)

- [x] Run the frozen 12-AOI SLC metadata screen. It found 50 candidate datatake pairs across 10 disconnected active AOIs; no assets or pixels were accessed. This meets the threshold for protocol review only. See D-2026-10-26-146 and the [audit report](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/report.md).
- [x] Complete the asset-free review of usage rights, calibration/product lineage, cloud/time matching, geolocation/registration, and SLC target validity. MERLIN is not directly applicable to S1 IW/TOPS; deramping is required and I/Q decorrelation remains unresolved. See D-2026-10-26-147 and the [protocol review](../../experiments/data_audits/cdse_slc_iw_aoi_group_feasibility_20261026_v1/protocol_review.md).
- [ ] Await authorized CDSE credentials and freeze a product-bound deramp/decorrelation/calibration/geometry/target protocol before requesting candidate assets or sidecars.
- [ ] Do not create a split or run training/inference from this route until the target is validated and source-group-disjoint splitting is possible. Preserve B1 `not_promoted` / `stop_at_b1` and keep B2/B3 gated.

## Literature-review inventory status reconciliation (2026-10-26)

- [x] Replace the stale 46-row current-inventory statement in the first-pass literature review with the reconciled 98-row count and explicit unresolved historical-membership range. Keep the original 119-row recovery task open; no paper evidence or local results were added. See [D-2026-10-26-149](DECISION_LOG.md#d-2026-10-26-149--reconcile-literature-review-matrix-status-after-recovery).
- [x] Verify the root README's current related-work count against the parsed matrix: both show 99 data rows (the CSV has 12 columns), so no README correction was needed. The original 119-row inventory recovery remains open; see D-2026-10-26-155.

## H8 restoration-order literature boundary (2026-10-26)

- [x] Add the Muzeau full-text distinction to H8: denoising-state supervision is not a direct task-order or matched joint-versus-cascade experiment. Keep its one-scene/patch-split and pseudo-reference limits explicit; do not use its reported metrics as local evidence. See [D-2026-10-26-150](DECISION_LOG.md#d-2026-10-26-150--narrow-h8-task-order-evidence-using-the-muzeau-full-text).



## Future GPU-only model-forward runner (2026-10-27)

- [x] Create and run the versioned v3 direct-fusion diagnostic with explicit `--device cuda` after a successful GPU preflight. It completed on `cuda:0`; keep the v2 CPU run sealed and do not rerun it. See D-2026-10-27-014.
- [ ] Before the next model-forward task, repeat CUDA preflight in a GPU-enabled execution profile and use `--device cuda`; defer model compute if CUDA is unavailable. The ordinary restricted shell currently has zero visible CUDA devices. This does not change B1 `not_promoted` or authorize B2/B3.


## Native cache vs prospective source-group split (2026-10-27)

- [x] Compare only sibling SOTANet `train`/`val` filename metadata: all 30 candidate prefix groups occur in both native partitions, with no exact duplicate basenames. Do not treat those directory labels as source-group disjoint.
- [x] Confirm the separate frozen v2 cohort assigns 15/15 groups, maps all 1,800 selected names to the recorded group IDs, and is covered by the existing official-source pixel audit. This remains development evidence only; B1 is `not_promoted`, B2/B3 remain gated. See the [audit](../../experiments/data_audits/sen12_native_partition_group_overlap_20261027_v1/report.md).
- [ ] Resolve source and parent-group metadata for the separate numeric-name 2,000-PNG subset; require an authoritative manifest or crosswalk and do not infer lineage from numeric filenames.

## Reverse-task prior-art source review (2026-10-27)

- [x] Verify task direction and available source routes for SAG-CR and CGD-Net; add the documented alignment, multi-temporal input, split, and code-availability boundaries to the matrix and literature record. See D-2026-10-27-011 and the [audit](../../experiments/literature_audits/reverse_task_cloud_removal_alignment_review_20261027_v1/report.md).
- [ ] Keep the original matrix inventory unresolved until an exact pre-corruption source is available; do not infer historical membership from candidate identifiers.
- [ ] Keep B2/B3 gated. Any future model-forward diagnostic requires a new versioned CUDA-only runner and successful GPU preflight.


## Compute execution

- [x] Run the frozen v3 direct-fusion definition with `--device cuda` after an elevated preflight identified the RTX 3050. It completed on `cuda:0`; identical input/checkpoint bindings and near-identical per-sample CPU/GPU deltas were verified. The run remains interpretation-only; B1 is not promoted and B2/B3 remain gated. See [D-2026-10-27-014](DECISION_LOG.md#d-2026-10-27-014--record-the-cuda-repeat-as-interpretation-only-hardware-consistency-evidence).


## HSFMamba / OGSSR resource status

- [x] Recheck the author-linked resource catalog and repository license. OGSSR has a listed download route and repository-level research-only terms; an archive manifest, archive-specific rights, and an HSFMamba code link remain unverified. See the [screen](../../experiments/literature_audits/hsfmamba_resource_license_code_screen_20261027_v1/report.md).
- [ ] Continue resolving source and parent-group metadata for the local numeric-name 2,000-pair subset from an authoritative manifest or direct archive/source crosswalk. Do not identify it as OGSSR or SEN1-2 from counts, overlap ratios, or numeric filenames. On 2026-10-27, three focused metadata-only Google Drive searches failed because the connector backend returned `Unknown tool`; no files were returned. Retry this route only after the connector is available.

## Source-folder and ROI taxonomy follow-up (2026-10-27)

- [x] Reconcile the folder taxonomy against the primary paper and TUM supporting document: `1970` is the fall sampling seed; `s1_i` / `s2_i` are source-scene folders. The frozen assignment has 15/15 distinct source-scene folders, with 15 fall folders in validation. See D-2026-10-27-027 and the [scene taxonomy audit](../../experiments/data_audits/sen1_2_scene_folder_taxonomy_20261027_v1/report.md).
- [x] Update the README and retained interpretation-only reports to reflect the verified source-scene-folder semantics; preserve their original run metrics and add dated interpretation notes. See D-2026-10-27-029.
- [ ] Obtain a source-scene-to-coordinate/product crosswalk before making geographic-separation or acquisition-independence claims. Preserve the source-scene-folder-disjoint wording; keep B1 `not_promoted` and B2/B3 gated.

## Research-text disclosure through group 430 (2026-10-27)

- [x] Record clipped project-document excerpts; exclude surfaced historical development values. No test payload or test-metric file was opened, and task flags remain false/false. See D-2026-10-27-030 and the [exposure audit](../../experiments/data_audits/research_text_read_exposure_20261026_v1/report.md).


## MONet training-reference boundary — 2026-10-28

- [x] Review MONet's full author manuscript and the follow-up study's institutional abstract/author README. Record optical-derived synthetic supervision separately from inference-time optical guidance. The follow-up full protocol remains inaccessible; no local experiment or gate changes. See D-2026-10-28-049 and the [audit](../../experiments/literature_audits/monet_optical_training_reference_20261028_v1/report.md).

## 2026-10-30 更新

1. **待审阅：** 审阅[研究总纲候选 v1.5-proposed](RESEARCH_CHARTER_REVISION_PROPOSAL.md)；获批前不改写 `AGENTS.md` 与 `README.md` 的研究方向条款。
2. **继续文献核查：** 补齐 optical-guided SAR restoration、可靠性自适应生成与少步流模型的直接领域证据；区分正式发表与预印本。
3. **保持阶段约束：** B1 仍未晋级；先完成 C2 执行验收与数据/来源组独立性证据，再按既有决策判断是否恢复模型选择。不得因本轮 PDE 文献进入 B2/B3。

## 2026-10-30 更新：总纲采纳与 C2 代码验收

1. **已完成：** 用户批准并采纳 v1.6；正式总纲、`AGENTS.md`、README 研究摘要和状态入口已同步。该决定不改变历史阶段 gate。
2. **已完成首轮代码改进：** C2 输入生成器/读取器绑定冻结协议和 C1 来源清单；指标实现固定 epsilon、强制未定义/非有限指标整项失败并生成掩膜记录；四条件 parity 审计与训练/选模 epoch helper 已实现。C2 定向套件 29 项通过、3 项因无 CUDA 跳过。
3. **优先阻塞项：** 取得并校验 C1 v2 仅 train/development 的固定载荷及权威来源组—地理/采集交叉表；在此之前不执行精细影像扫描或正式 C2 固定输入生成。
4. **下一项实现：** 完成角色受限暂存、八次训练编排及 epoch-boundary checkpoint 保存/恢复、paired screen/结果 writer、CUDA/资源和唯一输出目录 preflight；已实现的 parity 与 mask record helper 必须接入正式 runner。
5. **阶段约束：** 正式 C2 尚未运行，B1 仍未晋级；不得进入 B2/B3，不读取确认集/测试集，不把合成验证写成方法效果。

## 2026-10-31 优先事项：恢复 C1 v2 可验证的数据入口

1. 查找或取得 `data/` 数字文件到 C1 v2 冻结官方身份的可核验交叉表；若不存在，则准备 C1 v2 训练与开发角色的匹配影像，并按官方文件校验值验证。无需获取或打开确认集影像。
2. 映射/载荷核验通过后，只暂存 1,600 对训练和 200 对开发数据，沿用冻结 C1 v2 角色，接入已完成 CUDA 预检的 C2 探索流程。
3. 在映射解决前，不随机重划本地 2k，不继续全量训练；旧 `test_100` 中已用于优化的 13 个样本不作评价或选模。
