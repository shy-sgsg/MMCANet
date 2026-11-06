# 物理约束生成与后验一致性：文献核查报告

**核查日期：** 2026-10-30  
**记录编号：** LIT-2026-10-30-PHYSICS-01  
**状态：** 文献来源已核对；未改变项目实验阶段或晋级门槛。

## 核查范围

补充检查物理约束扩散、谱空间物理信息扩散，以及精确硬约束下后验测度选择问题。来源优先采用期刊出版社、会议论文集和 arXiv 官方页面；作者代码链接仅在正式来源明确提供时记录。

## 核实到的文献事实

| 工作 | 已核实信息 | 方法要点 | 适用边界 |
|---|---|---|---|
| [CoCoGen 期刊页面](https://epubs.siam.org/doi/abs/10.1137/24M1636071) | Christian Jacobsen、Yilin Zhuang、Karthik Duraisamy；发表于 *SIAM Journal on Scientific Computing*，47(2)，C399–C425（2025），DOI：10.1137/24M1636071。出版社标有代码与数据可复现标记，并链接作者仓库。 | 将离散 PDE 信息纳入 score-based 生成过程，研究 PDE 前向与反问题，包括稀疏观测下的场重建。 | 证据来自 PDE 科学计算问题；不能证明 SAR 表示域、散斑统计或光学引导有效。 |
| [PISD 会议论文集页面](https://proceedings.mlr.press/v306/gallon26a.html) | Davide Gallon、Philippe Von Wurstemberger、Patrick Cheridito、Arnulf Jentzen；ICML 2026，PMLR 306，32814–32831 页；页面提供作者代码。 | 在缩放谱表示的潜空间学习 PDE 参数与解的联合分布，并在扩散步骤中以 Adam 更新纳入物理约束和部分观测。 | 实验对象为 Poisson、Helmholtz 与不可压 Navier–Stokes 方程；不是 SAR/JDSR 或跨模态错配实验。 |
| [CoCoS arXiv 页面](https://arxiv.org/abs/2606.04804) | Jian Xu、Yanning Wu、Delu Zeng、John Paisley、Qibin Zhao；arXiv:2606.04804，页面当前列出 v3（2026-06-07 修订）。本次核查只确认 arXiv 版本，正式发表状态未确认。 | 论文针对精确 PDE 硬约束的零测度流形条件化，讨论 co-area/Fixman Jacobian 对后验测度的影响，并提出 CoCoS 采样器。 | 这是预印本提出的理论与实验主张，范围是特定硬约束反问题；不能直接外推至有噪 SAR 似然、软惩罚或本项目算子。 |

## 对当前研究的含义

1. 物理约束生成已有成熟科学计算先例；项目不得把“扩散结合物理项”本身写成新颖性主张。
2. PDE 残差下降只说明特定离散方程约束得到满足，不自动证明后验校准、辐射一致性、结构真实性或任务指标改善。
3. CoCoS 提醒精确硬约束下的测度定义可能影响后验解释。由于本次只核实到预印本，且其对象与本项目表示域不同，应作为审查问题而非普遍定理写入候选方案。
4. 这些来源均未提供 optical-guided SAR 联合去斑与超分辨率的直接证据；它们不改变 B1 停止状态，不授权运行 B2/B3，也不改变 test 边界。

## 本轮访问边界

本轮只检查论文官方页面、作者页面提供的代码链接及项目文献/研究文档；未打开数据图像，未读取测试指标，未运行项目代码或模型。项目累计记录仍为 `test_set_accessed=true`、`test_set_used=false`；本轮审计自身两项均为 false。
