# MMCANet/JDSR 仓库代理规范

本文件作用于仓库根目录及所有子目录。子目录若新增更具体的 `AGENTS.md`，其规则只补充或细化对应作用域；不得降低本文件关于数据安全、实验公平、可复现性和证据真实性的要求。

本项目已于 2026-11-06 正式采纳 [研究总纲 v1.7](docs/research/RESEARCH_CHARTER.md)。研究问题、假设、C1–C6 路线和学术主张边界以该总纲为准；当前阶段状态以 `docs/research/CURRENT_STATE.md` 为准。若总纲与本文件第 2–15 节及第 16.4–16.5 节的数据安全、实验公平、可复现性、测试隔离和历史门槛规则存在张力，执行时采用更严格的保护规则；总纲采纳不自动改变阶段资格。

## 项目文档与终端语言

项目文档标题、正文、README/报告、脚本面向用户的输出及项目工作产生的终端摘要均使用简体中文。论文原题、作者姓名、项目名、代码/API 标识、配置键、路径、可复制命令和标准术语保留原文。

## 1. 项目目标与完成标准

本仓库的核心任务是研究 SAR 联合去斑与超分辨率，当前主线为 `MMCANet/`，对照实现位于 `Early-FusionCascadeJDSR/`，论文位于 `paper/MMCANet/`。

所有代理都应把被授权的工作推进到以下状态：

- 功能已实现，而非只给建议或伪代码；
- 代码已在适用环境实际运行；
- 正确性、公平性和稳定性有直接验证；
- 配置、数据划分、随机性、checkpoint 和结果可复现；
- 产物、命令和证据便于第三方审查；
- 论文描述、公式、表图和结论与当前代码及最终实验一致。

“脚本不报错”“训练完成”“生成了图片”“已有 checkpoint”“某个指标更高”都不是算法正确或论文可投稿的充分条件。若正式结论所需实验未运行，必须如实标为未验证，不得用意图、旧结果或局部冒烟替代。

## 1A. 正式研究方向与总纲优先级（v1.7）

正式题目为：

> `光学先验引导与 SAR 物理一致性约束的生成式联合去斑和超分辨率重建方法研究`

题目和研究路线均属可检验框架。只有在测量表示与观测模型经核验后，才使用“物理一致性”；证据不足时改用“观测一致性”或“表示域”。完整研究问题 Q1–Q6、假设、C1–C6 阶段、晋级条件和论文主张边界见[正式研究总纲](docs/research/RESEARCH_CHARTER.md)。

执行顺序为 C1 数据/目标/表示/观测审计 → C2 光学效用 → C3 生成模型可行性 → C4 观测一致性 → C5 不可靠光学下的可靠性与回退 → C6 效率。阶段顺序不构成自动晋级；每阶段须满足总纲和本文件的执行前条件。

历史 B1/B1R/B1F 结果保持 `not_promoted / stop_at_b1`，不得改写为“Joint > Strong Cascade”。`fixed_val_dev2k`、旧 B1F inner-dev 及其他已审计污染划分不得作为独立泛化或模型选择证据。历史强级联对照、数据/表示限制和来源不确定性按本文件后续规则及总纲第 14 节保留。

## 2. 进入仓库后的必做检查

开始实质修改前，先读取本文件及作用域内更具体的指令，并记录：

```bash
pwd
git status --short --untracked-files=all
git log -1 --oneline
git diff --stat
```

随后确认：

- 当前分支、HEAD、已有修改与未跟踪文件；
- 数据目录、split、checkpoint、输出目录和历史实验；
- 当前 Python/PyTorch/CUDA 环境及可用设备；
- 训练、验证、推理、评价和论文构建入口；
- 当前任务的权威配置、结果和验收证据。

已有修改和实验产物默认属于用户。不要删除、覆盖、回退、批量格式化或借机重构与当前任务无关的内容。

## 3. 项目结构与权威入口

默认将以下路径视为重点审查范围，但不得因此忽略它们调用的依赖：

- `MMCANet/`：模型、数据、训练和推理主线；
- `Early-FusionCascadeJDSR/`：bicubic、SAR2SAR+SwinIR、SFT 等 baseline 与指标实现；
- `scripts/`：统一实验、评价、表格和图片生成入口；
- `data/`：本地数据镜像或工程子集，原始数据属性必须另有清单证明；
- `checkpoints/`、`weights/`：权重，不等同于可复现 run；
- `outputs/`：历史或正式产物，必须靠 manifest 区分；
- `paper/MMCANet/`：论文源文件与最终构建；
- `docs/`：协议、审计、复现和提交准备文档。

若同一算法存在多个历史入口，先通过代码、checkpoint 结构和运行证据确认当前主线。不要默认文件名中的 `final`、`v4` 或日期代表正确、最佳或可投稿版本。

## 4. Python、PyTorch 与实现规范

### 4.1 环境记录

每次正式 run 至少记录：

- Python、PyTorch、torchvision、CUDA、cuDNN 版本；
- GPU 型号、数量、驱动和显存；
- 操作系统、关键依赖版本和实际 Python 可执行文件；
- 代码 commit、dirty 状态和配置哈希。

仓库若提供锁定环境，优先使用并验证它；不得把“本机能 import”当成依赖完整。新增依赖须更新相应环境文件和复现说明。

### 4.2 张量契约

关键函数应明确并检查：

- shape，例如 SAR `[B,1,H,W]`、optical `[B,3,H,W]`；
- dtype，训练通常为浮点，保存前才显式量化；
- device，禁止意外 CPU/GPU 混用；
- 数据域与数值范围，例如线性幅度、强度、功率、dB、log-normalized；
- 输入、目标、输出的空间分辨率和配准关系。

禁止依赖隐式广播来掩盖尺寸或通道错误。需要 resize 时必须在代码和协议中明确位置、插值、`align_corners` 和物理含义；评价阶段默认禁止修复模型输出尺寸。

### 4.3 训练、验证与推理

- train/validation/test 必须使用独立数据集对象和固定清单；验证与测试禁止随机增强。
- 推理必须使用 `model.eval()` 和 `torch.inference_mode()` 或 `torch.no_grad()`。
- checkpoint 加载必须核对架构、通道数、upscale、normalization、degradation 和配置哈希；正式运行禁止用 `strict=False` 后忽略缺失参数继续。
- 恢复训练必须恢复 model、optimizer、scheduler、scaler、epoch、best metric 和 RNG 状态。
- 训练中检查 loss、梯度和输出的 NaN/Inf；正式 run 遇到非有限值应失败并保留诊断，不得静默跳过。
- 修改 Python 后至少执行 `python -m py_compile`；按风险运行定向 pytest、CPU smoke test 和适用完整回归。

### 4.4 随机性与可复现性

统一设置并记录 Python、NumPy、PyTorch CPU/CUDA seed。DataLoader 必须设置 generator 与 worker seed。正式对比中：

- 随机退化应预生成并以 manifest/hash 固定，测试时不得每个方法独立采样；
- 数据增强只用于训练并可由 seed 复现；
- cuDNN/算法确定性选择必须记录；若某操作无法确定性运行，应明确说明影响；
- 不得只保存“最好看”或最有利的 seed。

## 5. 数据安全、来源与划分

### 5.1 原始数据保护

- 不修改、覆盖、重编码或批量移动原始数据。
- 派生数据写入新的、带协议版本的目录。
- 大型数据、checkpoint、缓存和输出不得误加入 Git；提交前检查 `.gitignore` 和 staged files。
- 任何清理操作先确认路径、内容和用户授权；禁止破坏性地清理历史实验。

### 5.2 来源与语义

数据集名称、论文引用、传感器、产品级别、极化、波段、空间分辨率、地理范围、时间匹配、配准、裁剪和预处理必须来自可核验证据。不得根据文件名或笔记猜测后写成事实。

8-bit PNG、显示拉伸图或逐图归一化数据不得自动称为辐射定标的幅度/强度/功率。若缺少原始物理量，代码、文档和论文必须降低表述强度并说明限制。

### 5.3 固定划分与防泄漏

- train/val/test 清单必须显式保存、版本化并计算哈希。
- 三个集合文件名不得重叠；还应按原始场景、轨道、时间、地理 tile 或 patch 来源检查同源泄漏。
- 相邻或重叠 patch 应按 group 划分，不能随机逐 patch 分散到不同集合。
- 当前固定 train/validation 与 B1F fit/inner-dev 已发现精确 spatial/target overlap；不得将这些 partition 的结果解释为独立泛化或用于未来模型选择/晋级。任何新 development split 必须在实验前版本化，并以可核验的 source groups 和空间重叠审计证明其划分依据；不能把数值文件名或 overlap component 当作来源 ID。
- 所有正式方法使用相同测试清单；任何缺失样本都应使正式评价失败。
- test 仅用于最终评价，不得选择 epoch、权重、阈值、loss 权重、ROI 或展示样本。
- `mini` 子集只用于 smoke/debug，不能支撑论文结论。

## 6. 实验公平性

正式比较必须确保：

- 相同 SAR 输入、optical 输入、pseudo-target、测试样本和文件顺序；
- 相同固定 LR tensor 或无损保存的固定 LR 文件，而非“相同参数各自重新退化”；
- 相同输入/输出尺寸与评价域；
- 相同 ROI、mask、crop 和指标实现；
- 显示增强与指标输入完全分离；
- 不对某个方法单独使用 percentile stretch、min-max、clamp、resize 或额外后处理；
- baseline 使用与其任务匹配、结构完整、训练合理且来源可追溯的权重；
- 不用明显欠训练、checkpoint 不匹配、随机初始化残留或输入分布错误的 baseline 支撑优越性。

若方法本身需要不同训练超参数，可以合理调优，但必须只用训练/验证集并记录搜索空间、选择规则和计算预算。公平不等于强迫所有方法使用错误的同一优化器设置。

正式结果目录必须与历史/debug 目录分离。结果进入论文前，至少检查文件集合、尺寸、通道、dtype、范围、NaN/Inf、配置/权重/input manifest 哈希和代码版本。

## 7. SAR/JDSR 物理与数值约束

### 7.1 数据域

每个预处理、退化、loss 和 metric 都要注明在哪个域计算：

- 复数 SLC、幅度、强度/功率、dB/log 或归一化表示；
- 是否有辐射标定、clamp、percentile 或量化；
- 该域是否允许对结果作物理解释。

不得把在逐图 log/percentile 归一化域中实现的乘性噪声无条件称为真实 Gamma speckle。若只能使用显示域 PNG，应称为受控合成压力测试或 physics-inspired degradation，并说明 domain gap。

### 7.2 退化链路

频域截断、可选模糊、下采样、speckle、clamp 和 normalization 的顺序必须只有一个权威实现，并与配置、测试和论文一致。

- Gamma speckle 参数化应通过统计测试验证均值约为 1、方差约为 `1/L`；还要说明它作用于强度还是幅度，幅度模型不能直接照搬强度 Gamma。
- FFT 操作检查 `fftshift/ifftshift`、奇偶尺寸、中心窗口、共轭对称、能量变化和边界；不得用未说明的 `abs(IFFT)` 折叠负振铃。
- 模糊核应归一化，检查角度、range/azimuth 轴和边界处理。
- clamp 的饱和比例必须记录，尤其检查亮散射点。
- 已带 speckle 的 pseudo-target 上再次注入 speckle时，应明确这是复合噪声假设并分析局限。

### 7.3 SAR 指标

PSNR、SSIM、RMSE 对 noisy pseudo-target 只衡量一致性，不能等同于 clean-reference 恢复质量。论文结论必须联合结构、辐射、亮点、ratio 和鲁棒性证据。

- ENL：固定 homogeneous ROI/mask；同时报告边缘/结构损失，禁止以过度平滑单独提高 ENL。
- EPI：固定 edge operator、边界、归一化、mask 和方向；说明噪声边缘敏感性。
- TBR：固定 target/background 定义；同时考虑中位数或稳健统计，避免少数亮点操纵均值。
- Ratio image：固定比值方向和 epsilon，报告均值、方差、coefficient of variation、分位数及与结构边缘的相关性。
- 显示图可统一拉伸，但图注必须说明该操作不参与任何指标。

## 8. 模型与损失审查要求

对 MMCANet 的审查至少覆盖 SAR/optical encoder、bottleneck cross-attention、SAR skip、shallow gate、upsampling、输出头、初始化和不同尺寸行为。

- 记录总参数量、可训练参数量、关键特征尺寸、MACs/FLOPs（仅在可靠工具可用时）、运行时和峰值显存。
- 检查 attention 的空间复杂度，防止输入尺寸变化导致二次内存爆炸。
- 检查 attention/gate 的可训练缩放参数确实更新，并报告其最终值或分布。
- 必须测试 optical 全零、缺失、错位和错误纹理；光学分支不能只因能提高 SSIM 就被视为安全。
- 输出头若使用 sigmoid，需检查高亮散射点饱和和动态范围；修改输出参数化必须有 A/B 证据。

损失函数审查需记录每项未加权值、加权贡献和梯度规模。ratio loss 要避免暗区除法产生异常梯度；optical-guided TV 要检查 edge normalization 是否逐图稳定、是否完全关闭平滑；所有 epsilon、窗口和权重必须进入配置及 checkpoint 元数据。

## 9. 正式训练与实验管理

每个正式 run 建议采用以下最小目录：

```text
<run_dir>/
├── config.yaml                 # 或 config.json
├── command.txt
├── environment.txt
├── git_state.txt
├── manifest.json
├── splits/                     # 或 split manifest/hash
├── input_manifest.json
├── checkpoints/
├── train_log.csv
├── validation_metrics.csv
├── test_per_image_metrics.csv
├── test_summary.csv
├── runtime.json
└── figures/
```

`manifest.json` 至少记录 run id、method、seed、commit、dirty 状态、config hash、split hash、input manifest hash、checkpoint hash、输出数量、开始/结束时间、设备和 evaluator 版本。

训练保存 `last` 与按验证规则选出的 `best`，并写清选择指标和方向。多 seed 结果按预先声明规则汇总 mean/std；主要方法间优先报告逐图配对差值及 bootstrap 95% CI。单 seed 只能作为探索结果，除非论文明确限制并有额外稳定性检查。

## 10. 必要实验原则

修复顺序固定为：

```text
P0 正确性与泄漏
→ P1 公平性与复现性
→ 统一可信基线
→ 模型改进
→ 完整实验
→ 论文改写
→ 最终投稿审查
```

模型改动必须对应明确问题，先有 baseline，再做单因素 A/B，然后在完整固定测试集上决定保留、放弃或待研究。禁止一次堆叠多个模块后无法归因；新模块必须有消融。

核心实验通常包括：bicubic、合理的 SAR-only joint baseline、SAR2SAR+SR cascade、SFT/direct conditioning、完整 MMCANet；无 optical、无 cross-attention、无 shallow fusion、direct concatenation、逐项 loss 消融；退化/normalization/blur/speckle 敏感性；optical 平移/仿射、遮挡、错误 patch、全零和噪声压力测试。具体矩阵以当前论文主张为准，不为增加表格而无限扩展。

## 11. 推理与评价的失败策略

正式 evaluator 默认采取 fail-closed：

- prediction、GT、fixed LR、optical 和 mask 的文件名集合必须完全一致；
- 检查重复名、缺失/额外文件、无法解码文件、尺寸、通道、dtype、范围和 NaN/Inf；
- 禁止正式模式下使用静默交集、自动 RGB→灰度、预测 resize 或逐图自适应拉伸；
- 若允许 `--allow-missing`、`--resize-*` 等 debug 选项，输出必须醒目标记为非正式且不得被论文表生成器读取；
- canonical GT/input manifest 的哈希不一致时必须失败；
- 汇总前确认所有方法具有相同 `N` 和相同样本顺序。

指标已计算成功不代表定义正确。为每个指标提供已知输入的单元测试，并在协议文档中写出公式、域、data range、mask/ROI 和方向。

## 12. 论文、表图与引用

### 12.1 主张与证据

论文每项贡献、方法描述和定量结论必须映射到：代码位置、正式配置、checkpoint、实验 run、逐图 CSV、汇总统计和对应图表。未实现或未验证内容不得写成贡献。

摘要、引言贡献、结果讨论和结论只使用最终统一协议产生的数字。历史/debug 结果可以在内部审计文档出现，但不能进入投稿主表。

### 12.2 数据与术语

- 严格区分 SEN1-2、SEN12MS、SEN12MS-CR 及 Sentinel-1/2 传感器来源。
- `self-supervised`、`physics-guided`、`robust`、`generalized`、`radiometrically consistent` 等词必须有准确假设和实验支持。
- noisy HR SAR 作为 pseudo-target 时，应说明监督信号来源，避免暗示存在 clean SAR ground truth。
- synthetic degradation 的结论不得直接推广到真实低分辨率传感器数据。

### 12.3 引用真实性

只加入已核对原论文、出版社、DOI 或官方数据集页面的引用。不得虚构作者、题名、DOI、年份、数据集属性或实验结果。引用应直接支持相邻表述；不确定引用先留在审计记录中，不进入正式 bibliography。

### 12.4 自动生成与排版

论文表格和图片应由最终 CSV/JSON/manifest 自动生成，避免手工复制数字。定性样本按预先声明、可复现的规则选择，覆盖不同场景类型和失败案例，不得按 MMCANet 优势挑样本。

LaTeX 必须能从明确环境完整编译。检查 error、undefined citation/reference、重复 label、overfull/underfull box、双栏溢出、图片清晰度、字体、乱码、裁切和重叠。构建成功后仍需实际查看最终 PDF 页面。

作者、单位、邮箱、基金、致谢、数据/代码可用性和利益冲突等无法从仓库确认的信息不得自行编造，应集中列为待用户确认项。

## 13. 测试与验证最低要求

根据修改范围运行适用检查，至少考虑：

- split 文件互斥、覆盖和 group 泄漏；
- normalization 的已知样例与稳定性；
- physical degradation 的尺寸、确定性、Gamma 统计、FFT 中心和能量；
- model forward、输出尺寸、zero/missing optical fallback；
- checkpoint 配置/结构不匹配失败；
- metric known-case、mask/ROI、ratio 统计；
- evaluator 对缺失、额外、RGB、尺寸、dtype、NaN/Inf 的失败行为；
- 完整测试样本数和 manifest/hash；
- 论文表格数字与结果源一致；
- LaTeX 构建日志和 PDF 视觉检查。

CPU smoke test 只证明链路可运行；不能替代 GPU 完整训练、多 seed、全测试集或数值回归。任何未运行检查都应在交付中明确列出原因。

## 14. Git、日志与交付安全

- 禁止未经用户授权执行 `git reset --hard`、`git checkout --`、`git clean -fd`、改写历史、提交、推送或创建 PR。
- 不用删除历史输出来“让目录干净”；新 run 使用唯一目录，发现冲突时默认失败。
- 先定位根因，再做最小完整修复；失败后读取真实错误并改变诊断路径，不重复无信息的同一命令。
- 保留命令、stdout/stderr、环境、配置、hash 和失败证据；不得手工改指标 CSV 以匹配论文。
- 交付时区分：已验证事实、仍待验证、外部阻塞、无效历史结果和用户待确认信息。

如果 GPU、数据或外部元数据缺失，先保存具体证据和待运行命令，然后继续完成所有不依赖该条件的代码、CPU 测试、审计、脚本、论文删假和构建工作。不得因单一阻塞提前声称项目完成，也不得伪造未运行结果。

## 15. 最终投稿验收

在声称“可投稿、可复现”前，逐项提供直接证据：

- 训练/验证/推理/评价成功日志与可复制命令；
- 无泄漏 split 与固定测试输入 manifest；
- P0/P1 修复及覆盖它们的测试；
- 所有正式方法同 N、同输入、同输出尺寸、同 evaluator；
- 主比较、核心消融、退化敏感性、optical 错位和错误引导压力测试；
- 逐图指标、mean/std、配对不确定性统计和公平性审计；
- 视觉图、error map、ratio image、局部裁剪和可复现选样规则；
- claim–code–experiment–figure 对照；
- 无投稿前占位表述且数字由最终结果自动生成的论文；
- 无关键 LaTeX 警告并经过视觉检查的最终 PDF；
- 清晰的环境、数据准备、训练、推理、评价和论文重建指南；
- 所有外部阻塞和待用户确认项未被伪造填充。

缺少任一支撑核心结论的证据时，只能报告当前准备度与缺口，不能把“没有发现错误”解释为已经完成。

## 16. 长期科研执行与连续性

本项目正式研究目标、Q1–Q6 问题和 C1–C6 前瞻路线以 [研究总纲 v1.7](docs/research/RESEARCH_CHARTER.md) 为准；长期目标仍是研究 optical 信息对 SAR 联合恢复的增量价值及失效边界，并形成证据充分、可复现的论文。工作标题、MMCANet-v2、SAR-dominant residual fusion、`C_use/C_geo/C_sem` 和 diffusion 都是研究方向或候选假设，不能视为既定方法或已验证贡献。研究结果可以要求改换模型、任务拆分或论文主张。

### 16.1 科学问题与研究原则

正式总纲第 4 节的 Q1–Q6 是权威研究问题。核心判断涵盖匹配光学的增量价值、可迁移线索、生成模型的任务增益、SAR 观测相容性、不可靠 optical 的抑制/回退，以及联合与级联的质量和成本比较；不得把候选机制预写为既成贡献。

每项主张必须能追到原始文献、代码、配置、数据清单、运行、逐图结果和分析。研究记录及论文审查必须明确区分五类证据：**事实**（可由来源、代码或 manifest 直接核验）、**观察**（只对记录的 run、样本和 seed 范围成立）、**假设**（尚待检验的预测）、**开放问题**（当前缺少足够证据）和**无支持主张**（现有证据不能支撑，禁止写成事实或贡献）；同时记录反证。论文引用和新颖性判断必须先核验正式来源；不得只凭摘要写成全面文献结论。

### 16.2 研究流程

按需持续执行：

```text
literature and project evidence
→ question and falsifiable hypothesis
→ protocol and predeclared decision rule
→ minimal implementation or diagnostic
→ validation-only run
→ paired and failure analysis
→ decision log and next action
→ update manuscript only when evidence supports it
```

单次迭代只回答主要研究问题。小成本诊断优先于训练；短筛选优先于长训；多 seed 和预先声明的门槛用于主张稳定性。训练或大规模 GPU 消耗不得仅因“可能有用”而启动；先确认设备、数据和 protocol 均可用。2,000-pair development subset 足以继续探索，不得把约 20,000-pair 数据设为继续工作的先决条件。项目此前把约 20,000 对称为“完整 SEN1-2”，但该口径尚未与原始发布版本核对；SEN1-2 2018 年主论文报告 282,384 对 ([primary paper](https://doi.org/10.5194/isprs-annals-IV-1-141-2018))，而 QXS-SAROPT 原始论文报告 20,000 对、来自 Gaofen-3 与 Google Earth ([QXS-SAROPT paper](https://arxiv.org/abs/2103.08259))。两者源论文描述的 patch 重叠也不同：SEN1-2 为 stride 128/50%；QXS 论文称 20%，但“stride 52”措辞按常规步长定义存在歧义。本地 train/validation 审计观察到 204/205 像素平移、51/52 像素重叠，与 20% 几何相符，但不足以证明来源。这使数据集名称/规模混淆成为一种可能，但不能据此认定本地文件来自 QXS。当前工作区 `data/sar` 与 `data/opt` 只有 2,000 个同名 PNG（文件名 8000–9999）；2026-10-19 记录的 mini 副本各 100 个已于 2026-10-21 清理，原始 SAR/optical PNG 保留；这证明该 `data/` 树保存的是 2k 子集，不能证明其他位置不存在更大数据集。在恢复来源/版本清单前，将本地 2k 称为项目标注的 SAR–optical 子集、将约 20,000-pair 视为未核实来源的项目报告规模；不得将其称为已核实的完整官方 SEN1-2，也不得据此假定 TUM 场景目录适用于本地文件。

### 16.3 证据与阶段决策

- 先建立与目标任务匹配的 SAR-only joint baseline，再比较 optical 方法；Phase-A2 强级联审计保留为 reference，不据此宣称 `Joint > Strong Cascade`。
- B1/B1R/B1F 的失败状态和原 gate 不可回写。额外诊断可以解释模型行为，但不能当作原 gate 晋级结果。
- 一个 seed、开发验证集或 target-assisted oracle 只能支撑探索性结论。不能替代多 seed formal validation 或 inference-available 方法验证。
- Phase-B2 是 validation-only、zero-training 的错配诊断，只有 Phase-B1 晋级后才运行；Phase-B3 鲁棒融合训练只在 B2 暴露可复现失败模式后进行。当前 B1 未通过，因此停在 B1，不进入 B2/B3。
- B1 通过前，不运行 B2 错配晋级审计或 B3 鲁棒融合训练；stale/semantic rejection 继续延期。可做的只读诊断须独立记录并明确不推进原 gate。
- 扩大训练预算前，要有重复证据支持假设、可用算力、冻结配置和明确的成功/失败解释。连续失败后先做根因分析和定向文献复查，不堆模块或调 gate。
- 不得声称本项目首次联合 SAR 去斑与超分；DSRKD、ISSMSAR、HSFMamba 等已存在相关联合工作。FIRM 已研究算子感知条件流逆问题，Twist Flow 已讨论逆问题中的后验样本覆盖/欠离散。通用条件流、逆问题样本 spread 或常规门控不能直接作为本项目新贡献；没有参照后验时不得把样本 spread 称为经校准的不确定性。上述方法的高斯噪声、算子或后验假设不可未经核验迁移至当前 SAR 表示域。新颖性需结合准确任务、数据和机制重新审查。

### 16.4 数据、test 与实验治理

- 冻结 C1 v2 的 train/development/confirmation 划分直接沿用：样本数为 1,600/200/200，来源场景组为 92/19/20。confirmation 已于 2026-11-05 访问并用于一次访问后修订的描述性复核，现已永久封存；不得称为 untouched confirmation，也不得用于调参、选模或阶段晋级。C1 v2 没有独立 test 角色；必须将 confirmation 与项目历史 test 分开记录。development 已重复用于训练/选择，只支撑范围受限的探索性结论。
- C2 v2 gate 仍未判定；SAR-only C3-A 可按单独冻结依据开展，不会解锁光学条件 C3-B。C3-B 仍须先有匹配 optical 相对 SAR-only 和配对/置换控制的可重复效用证据。v1.7 采纳不追溯改变 C2、B1/B1R/B1F 或 B2/B3 门槛。

第 2–15 节的工程、安全、物理和复现规则继续生效。本节补充长期研究约束：

- 当前 `fixed_val_dev2k` 是反复使用的 development validation，不是 untouched holdout。按各阶段既定规则禁止打开 test 图像或读取 test 指标；每个新实验的 test flags 必须为 false。
- 只允许在 split-integrity audit 中查看 split membership 元数据，不读取 test 图像/测量结果；若任何工作违反这一界线，应在 state 和 manifest 如实披露，并从该样本集停止调参。
- 目标是 normalized 8-bit representation-domain PNG 时，使用“表示域受控退化”“noisy pseudo-target”等表述；没有原始幅度/强度/功率与定标信息时，不作辐射物理主张。
- 保存每次运行的配置、命令、环境、代码状态、数据/输入 manifest、checkpoint、逐图指标和决策；使用全新唯一输出目录。禁止重写历史 gate、手改指标、覆盖 checkpoint 或删除历史失败证据。
- 对改动执行与风险相称的验证。Python 修改执行 `py_compile`；算法/协议修改执行相应测试和 smoke。未执行的正式验证必须标明。

### 16.5 研究记录与会话连续性

`docs/research/CURRENT_STATE.md` 是当前阶段唯一状态入口；新状态须记录日期、当前阶段、已验证证据、未知项、风险、资源条件和下一步。每轮实质工作后同步：

- `MASTER_RESEARCH_PLAN.md`：阶段目标和晋级/停止条件；
- `EXPERIMENT_REGISTRY.md`：run id、protocol、manifest、结果和解释边界；
- `DECISION_LOG.md`：何时及为何继续、停止或切换路线；
- `NEXT_ACTIONS.md`：按优先级排序的可执行任务、依赖和阻塞；
- `RESEARCH_RISKS.md`：数据、退化、指标、计算、创新性与复现风险；
- `literature_review.md`、`related_work_matrix.csv`、`research_gaps.md`、`diffusion_feasibility.md`、`optical_information_analysis.md`、`research_hypotheses.md`：文献和方法证据。

每次会话开始时先读 `CURRENT_STATE.md`、最近的 decision 记录及其指向的主要 run；每次结束前更新状态和下一步。README 是项目入口，并链接到这些来源。旧的状态/投稿计划若与上述文件冲突，作为历史快照保留，不作为当前事实来源。

### 16.6 真实完成条件

完成一个 screen、代码路径、论文 PDF 或数据审计，都不代表研究课题完成或达到投稿准备。只有核心主张有足够的任务匹配基线、独立重复、正确的物理与指标边界、公平比较、完整误差和负结果分析、可复现产物以及论文一致性后，才可报告投稿准备状态。若研究不能支持 optical 增益或鲁棒性，应交付清晰、完整的负结果和经证据约束的替代问题，不编造贡献，也不承诺录用。

## 17. 用户长期工作交付规范

- 用户给出明确工作目标后，持续推进目标范围内所有已授权且必要的步骤；不得只完成调研、计划或局部代码就提前停止，也不得把单一路径的外部阻塞当作整个目标已经完成。
- 遇到阻塞时，记录可核验证据，继续完成不依赖该阻塞的代码、实验设计、分析、验证和研究记录；条件恢复后沿原目标继续。只有目标确已完成，或确实没有任何可独立推进的工作且必须由用户提供信息/改变外部条件时，才暂停，并清楚说明具体原因。
- 停止前对照用户原始目标核对交付项：所需实现、实际运行、适用验证、结果与限制记录、文档同步均已完成；未完成项不得写成完成。中间状态汇报不替代目标完成。
- 用户要求的研究/工程工作全部完成并通过适用验证后，检查改动和提交范围，再实际创建 Git commit 并推送到仓库已配置的目标远端/分支；不能只给命令或声称已推送。推送后报告 commit、分支、远端及验证结果，供用户通过 ChatGPT 审查并给出下一步工作计划与具体实施方案。在当前目标完成前，不得以等待审查或下一计划为由提前停止。
- 提交只包含当前授权目标产生且审查过的文件；保留已有用户修改，排除无关工作、数据、checkpoint、输出、密钥和凭证。若必须提交的内容受明确外部阻塞，继续推进可完成部分，不伪造完成状态，也不以不相关文件填充提交。
- 空间管理：长训练和评价前后检查项目所在分区余量；及时清理当前授权任务产生、确认无复用价值且可重建的缓存和临时中间文件，并记录清理路径与释放空间。不得删除原始/派生数据、历史实验结果或失败证据、checkpoint、其他项目文件及无法确认归属的内容；没有安全候选时保留文件并记录余量，不因清理而伪造或覆盖实验记录。
- 项目文档标题、正文及面向用户的终端摘要均使用简体中文，代码标识、路径、配置键、原论文题名和标准术语保留原文。
