# 算子感知条件流与后验样本覆盖文献审计

**审计日期：** 2026-11-05
**用途：** 为研究总纲 v1.7 候选稿界定 C3/C4 方法边界
**审阅范围：** 两篇 2026 年 arXiv 一手全文
**实验状态：** 未运行本地模型、未读取项目影像、未访问 confirmation/test 载荷

## 核心结论

本次新增两项与候选路线直接相关的方法证据：

1. 算子感知条件流已被用于通用图像逆问题；其理论和实验采用已知线性观测算子与加性高斯噪声。
2. 成对训练的条件生成模型可能欠用潜变量、低估欠约束部分的样本变化；Twist Flow 提出扩充传输状态并分别考察观测一致性和样本扩散。

两者都不是 SAR 去斑、光学引导或 SAR–光学联合 JDSR 证据。它们缩小通用方法新颖性空间，并提醒本项目将观测模型效应、光学条件效应和样本多样性分开验证。当前 8-bit SAR PNG/noisy pseudo-target 尚不支持直接移植其 Gaussian 似然或后验校准主张。

## 文献一：算子感知条件流

Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems，作者 Shirin Shoushtari、Edward P. Chandler、Xiao Shi、Ulugbek S. Kamilov，arXiv:2609.12953，2026-09-11。

论文从测量条件后验均值构造条件速度场，并以 operator-dependent 数据一致性更新参数化速度。作者将 forward operator 纳入训练目标，推理时不再附加外部 guidance；另分析采样步数与失真—感知权衡。方法推导假设已知线性算子和加性高斯测量噪声。

全文在 CelebA 与 AFHQ-Cat 上考察去噪、去模糊、超分辨率及 inpainting 等任务。论文中的超分条件包括 ×2 与 ×4。以上是作者论文所报的设置和结果，未在本项目复现。

**与本项目的关系：**它证明“把已知观测算子并入条件流速度场”已是通用逆问题中的先例，因此不能单独作为本项目贡献。其假设不能直接覆盖未核实字节映射的 SAR PNG、已带噪 pseudo-target、表示域复合退化或 Gamma 型乘性噪声。后续只可在本地表示、目标和算子经独立审计后，重新推导相应目标；不能把论文中的高斯一致性结论移植为本地 SAR 物理一致性。

**一手来源：**[arXiv 论文全文](https://arxiv.org/abs/2609.12953)

## 文献二：成对监督下的后验样本覆盖

Twist Flow for Inverse Problems，作者 Shiqin Zeng、Zijun Deng、Felix J. Herrmann，arXiv:2610.09281，2026-10-07。

论文指出，单观测—单目标的成对训练可能使条件生成模型主要依赖观测而弱化潜变量，造成对多峰或欠约束后验支持覆盖不足。提出的 joint twist-flow 在扩充状态中传输观测支路坐标与目标样本坐标，以增加观测相容性与目标变化之间的耦合。

证据分为两层：

- 二维 toy inverse problems 使用 MCMC 参考样本，可比较后验支持和边缘分布。
- 图像恢复使用 CelebA 及 Fourier 低通、模糊下采样和遮挡等人工算子；论文明确说明没有真实图像后验样本，主要报告观测一致性和欠约束分量的样本变化。论文还包含地震速度模型反演。

**与本项目的关系：**它提醒单个 noisy pseudo-target 监督可能无法识别真实目标分布，也不能因输出有随机差异就称为 calibrated posterior uncertainty。若本项目后续尝试多次生成，须分别报告固定观测相容性、重建均值、样本差异及可用的参考分布证据；没有后验参考或有效校准时，样本标准差只作为描述性生成诊断。该论文没有评估 SAR speckle、optical guidance 或本地观测似然。

**一手来源：**[arXiv 论文全文](https://arxiv.org/abs/2610.09281)

## 对现有研究路线的影响

- C1 数据和目标审计仍是 C4 似然推导的前置条件；现有论文不改变本地观测模型资格。
- C3 保留 SAR-only deterministic 与 SAR-only flow 对照。若进入光学条件模型，仍需 matched/permuted/missing optical 控制，不能把 flow 的一般收益归因于 optical。
- C4 可将 operator-aware conditional velocity 列为候选形式，但必须按本地表示和退化重新推导；Gaussian 线性模型仅是相关工作，不是可直接复用的局部假设。
- C3/C4 若研究多样本生成，需要区分样本 spread、观测一致性和后验校准；不得按目标从多次采样中挑最优输出。
- 新颖性不落在通用条件流、通用观测算子注入或生成多样性本身，而只能来自经验证的 SAR 目标/观测设定及可复现的任务级增量证据。

## 边界与后续

本次仅审阅公开的一手论文页面全文，没有获取作者代码、复现结果或 SAR 任务实验。两篇论文的同行评审/正式发表状态未由本审计核实。既有的 Flow Matching for Fast Posterior Sampling in Bayesian Inverse Problems（arXiv:2610.02377）已于 2026-10-30 收录，本次不重复增加矩阵行。

## 2026-11-06 补充：arXiv 版本题名核对

复核 arXiv:2609.12953 当前全文后确认其现行版本为 v2，题名为 *FIRM: Flow-based Imaging via Regularized Minimization*，版本日期为 2026-09-27；v1（2026-09-11）题名为 *Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems*。论文编号和作者未变。v2 明确将理论范围限定为已知线性观测算子与加性高斯噪声；其图像恢复实验涵盖 CelebA、AFHQ-Cat 上的去噪、去模糊、超分辨率及两种遮挡修复任务。候选总纲和相关工作矩阵采用 v2 当前题名，保留 v1 旧题名便于检索。该版本核对不改变既有方法边界或实验决策。

一手来源：[arXiv 当前全文与版本记录](https://arxiv.org/abs/2609.12953)。本次仍未核实作者代码、正式发表状态或本地 SAR 复现；未访问任何本地图像、test 数值或 confirmation 载荷。
