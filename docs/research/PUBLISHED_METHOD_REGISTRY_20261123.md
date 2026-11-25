# SAR 联合去斑与超分方法注册表（更新于 2026-11-24）

本表用于区分**已发表原方法**、**任务适配实现**、**项目内部基线**和**新候选方法**。只有取得作者实现、原始配置/权重、数据划分和评价细节后，才可将本地代码称为论文复现。当前开发集已重复用于训练/选择，所有本地数值均限于探索观察。

本轮 SAR-only SwinIR 与两种 DDPM 条件已完成；结论见[Research Decision Review](SAR-only%20Published%20Baseline%20与%20Diffusion%20决策复核_20261124.md)。这更新的是本地任务适配和候选状态，不改变已发表方法来源证据，也不将本地数值冒称为原论文结果。

## 方法概览

| 方法 | 原论文任务 | 本项目中的身份 | 实现/权重/协议状态 | 当前处理 |
|---|---|---|---|---|
| DSRKD | SAR 联合去斑与超分 | `ORIGINAL_PUBLISHED_METHOD`，直接任务先例 | 作者主页概述编码器–上采样–解码器、特征蒸馏和目标蒸馏；已查来源没有可运行作者代码和权重，全文协议细节未完整核实；因无代码包，许可状态不可核验 | 优先联系到可验证代码/协议后再做同协议复现；当前不能伪造实现为 DSRKD |
| SwinIR | 通用图像 SR、去噪、压缩伪影恢复 | `TASK_ADAPTED_IMPLEMENTATION` | 官方论文、代码及通用图像权重公开；本地使用官方架构源码，但本轮单通道、Sigmoid、随机初始化、C1 SAR 数据训练，无 SAR 权重或论文原 recipe | 已完成 30 轮 SAR-only 适配；best epoch 30，development 组均衡 PSNR 21.6352 dB；与内部 C2 配对结果及限制见决策复核 |
| EDSR | 通用单图超分 | `TASK_ADAPTED_PUBLISHED` | CVPRW 论文及作者实现公开；本地保留 EDSR 主干，按 SAR 表示域改单通道和 Sigmoid 输出，随机初始化；未使用作者 SAR 权重 | 已完成 30 轮适配，best epoch 29；复用 development 组均衡 PSNR 21.7316 dB；相对本地 SwinIR 任务适配 ΔPSNR +0.0964 dB（19 组 bootstrap 95% CI [+0.0693,+0.1316]）。身份和限制见[基线比较](../../experiments/published_baselines_sar_only_20261123_v1/baseline_comparison.md)与[来源核验](PHASE_D1_PAIRED_BASELINE_PROVENANCE_AUDIT_20261124.md)；不是作者 SAR 权重或原论文完整训练方案复现 |
| SAR-BM3D + EDSR | SAR-BM3D 去斑 + EDSR 超分的组合级联 | `TASK_ADAPTED_COMPOSITE` | SAR-BM3D 是独立 SAR 去斑工作，EDSR 是独立通用 SR 工作；组合不是这两篇论文共同提出的直接 JDSR 方法。当前工作树无可核验 SAR-BM3D 可执行实现/参数，故无法核验其代码许可 | 先取得/核验实现、许可、噪声域及参数依据，再比较；否则不报告数值 |
| HSFMamba | optical 引导 SAR 联合超分与去噪 | `ORIGINAL_PUBLISHED_METHOD`，直接 optical 任务先例 | IEEE 论文摘要报告 3,200 optical–SAR 对及 8 种退化；作者相关目录可见 OGSSR 数据资源索引，未发现 HSFMamba 算法代码/权重链接，代码许可不可核验。原稿的 Gamma 数据域、配准及 split 细节需回到全文核对 | 当前 C2 v2 gate 未判定，且缺少新独立 development；不训练、不评估、不访问 optical。先解决 gate 和独立数据资格 |
| RSDM / LSIR | SAR-only 无监督 ×4 联合去斑与超分 | `ORIGINAL_PUBLISHED_METHOD`，直接生成式任务先例 | IEEE 正文称 RSDM 用 Stable Diffusion 2.1 LoRA，并用两幅代表 SAR 图适配；LSIR 在潜空间做区域模型整合。作者报告 RTX 4090、1,000 扩散步、30 步 DDIM、200 次 LoRA 更新；未发现公开作者代码，许可不可核验 | 与本项目 x2 监督 noisy pseudo-target、单卡 4 GiB 环境和当前数据域不匹配；不声称可以直接复现 |
| DiffusionSAR | SAR 去斑 | `ORIGINAL_PUBLISHED_METHOD`，生成式相邻先例 | TGRS 论文使用 log-Yeo–Johnson 变换与条件扩散，讨论 SAR 乘性 speckle；当前可访问来源未核实作者训练代码/权重，许可不可核验。原任务没有 SR | 作为去斑方法先例；迁移到 x2 JDSR 需要新的条件/观测定义，不把其结果当联合超分比较 |
| Frequency-enhanced diffusion prior / CD-SW | SAR ×4 联合去斑与超分 | `ORIGINAL_PUBLISHED_METHOD` 候选，直接生成式任务先例 | 找到题名对应的公开仓库（维护者与论文作者身份未核实）；README 自称包含实现，报告 FAIR-CSAR-V1.0、2,400/300/300 train/val/test、×4 结果与阶段训练设置。但 IOP 出版页被 robots 拒绝，README 仍写“paper submitted / 2025”、BibTeX journal 留空；出版元数据需再核实。代码树没有 README 所写 `train_swinir.py` / `train_cdformer_stage1.py` / `train_cdformer_stage2.py`，仓库根目录无 LICENSE，预训练权重链接标为待发布 | 高相关候选，不能当作当前已复现 baseline。现有 `main.py`/`trainer.py` 可供审查，但默认 DF2K/Set5、Windows 路径、训练中重新随机合成退化，与冻结 C1 v2 x2 固定输入不匹配；代码中 `model/cdformer.py` 把扩散步数硬编码为 4，与 README 的 1,000 steps/100-step DDIM 叙述冲突。需先核验出版/授权、还原配置并做 SAR 数据适配，才能开展明确标注的近似复现 |
| SAR-only Conditional Diffusion | 本项目表示域 noisy pseudo-target 条件生成 | `PROPOSED_METHOD` | 已完成单 seed 30 轮像素域 DDPM；四 draw，单次/pixel-mean PSNR 11.5083/11.7661 dB，亮区与背景均值明显压低 | STOP 当前配置；不追加同类 seed。结果不代表 clean SAR 或分布校准 |
| SwinIR + Residual Diffusion | 确定性预测上的条件残差采样 | `PROPOSED_METHOD` | 已完成单 seed 30 轮、train-only residual scale、20-step DDIM、4 draws；pixel-mean PSNR 21.0971 dB，较 SwinIR 低 0.5381 dB（95% CI [−0.6072,−0.4713]） | STOP 当前质量优先配置；残差条件相对直接 Conditional DDPM 改善，但采样成本高且仍低于确定性参照 |
| `C2MatchedJoint` | 本地 SAR-only 确定性联合恢复 | `INTERNAL_BASELINE` | 已有 96,209 参数 `best.pt`，SHA-256 `8255b522…ce27f7165`；seed 20261029，30 轮；详细谱系见[基线谱系审计](BASELINE_LINEAGE_AUDIT_20261123.md) | 作为同一开发清单上的项目内部确定性参考，不能称为发表强 baseline |
| C3-A SAR-only Flow | 项目内部 SAR-only 条件 Flow | `INTERNAL_PROPOSED_METHOD` | 冻结 C3-A v1/v3 已完成；相对 C2 `best.pt` 的 ΔPSNR −3.0724 dB、ΔSSIM −0.17818；没有达到追加 seed 条件 | 停止当前像素域 Flow 配置；该结果不否定扩散候选或其他 Flow 参数化 |

## 方法证据与匹配性

### DSRKD

DSRKD 是本表最直接的 SAR-only 联合去斑与超分论文先例。作者主页确认其知识蒸馏框架和编码器–上采样–解码器主线；IEEE DOI 记录可用于核对出版身份。当前检查没有取得作者代码、可下载权重或完整实验协议，因此状态为“高优先级待取得可复现材料”，不是“已完成复现”。[作者主页](https://circlelzy.github.io/)、[IEEE DOI](https://doi.org/10.1109/TGRS.2024.3432193)

### SwinIR 与 EDSR

SwinIR 原论文覆盖通用图像超分、去噪和压缩伪影恢复；官方仓库声明 Apache-2.0 许可，包含作者模型和通用图像权重。EDSR 原论文是单图超分方法；作者 PyTorch 仓库声明 MIT 许可。两者适合作为标准架构迁移到 SAR 单通道 noisy-target 设置的可执行参照，但若改数据域、单通道 head、输出约束或训练 recipe，结果应标注为**任务适配**，不可称论文原方法在 SAR 的公开表现。[SwinIR 论文](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)、[SwinIR 作者仓库与许可](https://github.com/JingyunLiang/SwinIR)、[EDSR 论文](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)、[EDSR 作者代码与许可](https://github.com/sanghyun-son/EDSR-PyTorch)

本轮 SwinIR 不采用作者发布的自然图像权重，因为输入/目标是项目自建的 SAR 退化表示域；采用随机初始化并在 SAR train 上训练。官方权重许可和把 RGB 权重转为单通道的映射并未纳入本轮，避免把未经核验的权重转换称作公平 SAR baseline。

### 光学先例与阶段资格

HSFMamba 是直接 optical 引导联合恢复先例，但论文发布不等于本地光学训练自动获准。当前 `C2 v2 gate` 为 `unjudged`；开发集已经重复选择，且没有新的来源独立 development。根据 v1.8 和仓库数据规则，不运行 optical 方法、不读取 optical payload，也不把现有 200 张 development 的结果当新独立证据。[HSFMamba IEEE 论文](https://ieeexplore.ieee.org/document/11045185/)、[作者数据与基准目录](https://github.com/mmic-lcl/Datasets-and-benchmark-code)

### 生成模型先例

RSDM/LSIR 是直接 SAR 联合去斑与超分的扩散先例，但其无监督 ×4 设置、Stable Diffusion/LoRA 方案和作者报告的 RTX 4090 预算与当前 C1 v2 的 x2 noisy pseudo-target 训练不匹配。[IEEE 正文](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?arnumber=11359000)、[DOI](https://doi.org/10.1109/JSTARS.2026.3655710)

DiffusionSAR 处理 SAR 去斑，使用 log-Yeo–Johnson 表示变换和条件扩散；它不能直接替代联合超分模型。[TGRS DOI](https://doi.org/10.1109/TGRS.2024.3419083)、[作者稿全文](https://ris.utwente.nl/ws/portalfiles/portal/447494127/Despeckling_SAR_Images_With_Log-YeoJohnson_Transformation_and_Conditional_Diffusion_Models.pdf)

直接 SAR 联合扩散候选 [Frequency-enhanced diffusion prior network](https://doi.org/10.1088/1361-6501/ae8686) 的[题名对应公开仓库](https://github.com/18291716943/CD-SW)自称提供模型/训练源码与 FAIR-CSAR 设置说明；当前未核实仓库维护者与论文作者的身份关系。此次浅克隆的仓库 HEAD 为 `9da9f56f93b48fc36fb1a4f85b859dec2ec543be`；README 报告 x4 FAIR-CSAR 上 19.26 dB/0.3502 SSIM，但该数值只标为仓库报告值。论文正文的出版社页面访问被 robots.txt 阻止；仓库 README 自己仍写 paper submitted、2025、journal 空白。当前代码树存在 `main.py` 与 `trainer.py`，但缺 README 里引用的三阶段脚本和权重，且步数/数据退化路径与文档不一致。仓库根没有许可文件；`SwinIR_main/LICENSE` 的 Apache-2.0 仅能证明该子目录文件的许可声明，不代表 CD-SW 其余代码授权。当前不直接复制或称为精确复现；若继续，先获取一手论文全文/许可/配置，或另行实现并标为近似任务适配。

FAIR-CSAR 官方数据页说明数据由 175 个 Gaofen-3 Level-1 SLC 场景构成、总量约 250 GB；下载需要注册/邮箱验证。数据使用条款允许科研和教学用途并要求引用，商业用途另需许可。该来源可作为后续 SAR-only 外部来源候选，但不是 optical 配对数据；全量下载超出当前分区余量，且需要先核查 `ImageInstruction`、场景到图像映射和原始数值表示，再决定可安全取得的 scene-disjoint 子集。当前未下载或访问任何数据载荷。[FAIR-CSAR 官方数据页](https://radars.ac.cn/web/data/getData?newsColumnId=59937a70-2a40-4027-a32d-df36753bf6ae&pageType=en)、[数据集论文 DOI](https://doi.org/10.1109/TGRS.2024.3519891)

SAR-BM3D 是 SAR 去斑滤波器，EDSR 是通用超分网络；二者顺序连接是一个候选级联 baseline，需要各自固定预处理/归一化、输入输出 domain、实现来源和参数，不能冒称 DSRKD 或另一篇联合方法。[SAR-BM3D 论文记录](https://ieeexplore.ieee.org/document/5651432)

## 后续研究顺序（2026-11-24）

1. 不重跑已停止的 Flow、Conditional DDPM 或 Residual DDPM 同配置实验；已归档的负结果作为候选筛选证据保留。
2. 先取得未使用且身份/来源组/配准可核验的新 development 数据，并冻结清单、空间重叠审计和模型选择规则。
3. 解决 C2 v2 gate 的未判定状态，重新预注册 matched Optical、SAR-only、组内置换、缺失与错位控制以及 seed/成本/停止门槛。
4. 只有匹配 Optical 增量在新 development 上通过预设门槛，才另行评估 C3-B；需用户另行授权，且不能使用本地复用 development 替代。
5. DSRKD 代码/权重和 CD-SW 作者身份/许可/协议的证据缺口仍待核实；没有充分来源时不冒称复现。
