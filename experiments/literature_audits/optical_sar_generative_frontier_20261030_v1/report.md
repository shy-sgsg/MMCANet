# 光学引导 SAR、联合恢复与生成式效率：前沿文献对照

**核查日期：** 2026-10-30  
**记录编号：** LIT-2026-10-30-RESTORE-01  
**性质：** 一手来源筛查与研究边界审查；不是系统综述，也不是本地复现实验。

## 范围与核验规则

本次按研究对象区分九类：Conditional Flow Matching、Rectified Flow、Flow Posterior Sampling、Diffusion Posterior Sampling、Physics-Constrained Generation、Optical-guided SAR Restoration、SAR Joint Despeckling and Super-Resolution、Reliability-Adaptive Generation、Few-step Generative Models。论文集、出版社、作者主页和作者代码库用于确认题名、发表状态或作者自述；只有核验到的论文正文/摘要才用于概述方法。仅有 DOI 或二手索引的条目继续标为待核验，不用于新颖性结论。

## 已确认的直接相关工作

| 方向 | 一手来源与已核实观察 | 与本项目的关系及边界 |
|---|---|---|
| SAR 联合去斑与超分 | DSRKD（IEEE TGRS，2024）作者主页列出论文及 IEEE Xplore 链接，概述其以知识蒸馏联合处理 SAR 去斑与超分；ISSMSAR（Remote Sensing，2024）由 MDPI 论文页面介绍为多时相 SAR 的集成散斑抑制与超分；Amieva 等（Acta Astronautica，2026）使用 Sentinel-1 IW 与 SM 模式开展非生成式联合恢复。 | “首次联合 SAR 去斑与超分”不成立。数据、参考目标、训练协议各不相同；不应把论文结果直接并表排名。 |
| 光学引导 SAR 联合恢复 | HSFMamba（IEEE JSTARS，2025）明确研究光学引导 SAR 联合超分与去噪，并报告跨模态特征选择和频率—空间聚合；OGSRN（Chinese Journal of Aeronautics，2022）以光学空间差异反馈训练 SAR 超分网络，测试阶段由 SAR 输入产生输出；OGSFlow 作者仓库 README 描述 UAV-SAR 光学引导两阶段流程和条件流。 | 一般性的光学引导、交叉注意力/选择或可靠性门控不是新颖性本身。OGSRN 的光学线索主要通过训练期反馈使用；OGSFlow 当前核验只到仓库 README，论文、固定源码版本和结果尚未确认。 |
| 可靠性自适应参考引导 | Ada-RefSR（ICLR，2026）研究参考图像超分，以隐式相关性门控自适应使用或抑制参考信息，并报告单步扩散方案。 | 这是自然图像参考超分的直接先例，不是 SAR—光学跨模态或 JDSR 证据。它要求本项目把候选贡献收窄到可验证的 SAR-specific 可靠性估计、错配损害降低和 SAR-only 回退。 |
| 少步流/扩散 | Blockwise Flow Matching（NeurIPS，2025）将轨迹划分为专用速度块并在 ImageNet 上评估效率；MeanFlow（NeurIPS，2025；arXiv 版本可查）研究单函数评估生成；LLE（NeurIPS，2025）为满足其规范形式的扩散逆问题算法学习少步线性外推。 | 这些结果分别属于通用图像生成或扩散逆问题求解器，不证明少步数会提升 SAR 恢复质量、观测相容性或光学效用。应将 NFE、重建质量、结构/亮回波、观测残差和运行成本一起比较。 |
| 流后验与物理约束 | 细节见[流后验采样审计](flow_posterior_sampling_frontier_20261030_v1/report.md)及[物理约束生成审计](physics_constrained_generation_20261030_v1/report.md)。 | 科学/贝叶斯反问题及 PDE 结果仅作算法先例和理论边界，不直接迁移为 SAR 证据。 |

### 一手来源

- DSRKD：[作者主页（含 IEEE Xplore 论文链接）](https://circlelzy.github.io/)、[IEEE Xplore 记录](https://ieeexplore.ieee.org/document/10606284/)。
- ISSMSAR：[MDPI 论文页面](https://www.mdpi.com/2072-4292/16/1/18)。
- HSFMamba：[IEEE Xplore 论文页面](https://ieeexplore.ieee.org/document/11045185/)。
- Sentinel-1 IW/SM 联合恢复：[Elsevier ScienceDirect 论文页面](https://www.sciencedirect.com/science/article/pii/S0094576525005004)。
- OGSRN：[Elsevier ScienceDirect 论文页面](https://www.sciencedirect.com/science/article/pii/S1000936121003459)。
- OGSFlow：[作者代码仓库](https://github.com/AG-WDS/OGSFlow)。
- Ada-RefSR：[OpenReview 论文页](https://openreview.net/forum?id=hYsqBf3Uad)。
- Blockwise Flow Matching：[NeurIPS 2025 论文集页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a67d9390e62237fb90986c48d12ff79-Abstract-Conference.html)。
- 少步扩散逆问题线性外推：[NeurIPS 2025 论文集页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2ca73f0633760e2e2740ccb2a0d3f8a1-Abstract-Conference.html)。
- MeanFlow：[arXiv 官方记录](https://arxiv.org/abs/2505.13447)。

## 尚未完成的一手来源核验

项目矩阵包含一条题名为“Unsupervised SAR Image Super-Resolution With Despeckling via Region-Specific Diffusion Models”的 2026 年 DOI 线索（10.1109/JSTARS.2026.3655710）。本轮未取得 IEEE 出版方页面或可核实的正式论文正文；现有二手索引不作为方法或效果证据。因此，该条目保留为高优先级核验任务，暂不据此宣称或否定“生成式 SAR JDSR 已存在”。在正式来源核对完成前，相关新颖性判断仍为开放问题。

DSRKD 的作者主页和 IEEE Xplore 链接可相互核对；IEEE 页面本轮触发浏览器验证，方法概述采用作者主页的自述。ISSMSAR、HSFMamba、Acta Astronautica、OGSRN、Ada-RefSR 与 NeurIPS 论文集均已查到其正式出版/会议来源页面或官方搜索索引；部分出版方页面限制自动访问，未将其指标数字用于本地比较。

## 对候选研究总纲的修订建议

1. 保留“光学效用、观测一致性和可靠性控制”为问题，不把 Flow、Diffusion、联合网络或光学条件预设成答案。
2. 明确既有光学引导 SAR JDSR、SAR-only JDSR、参考图自适应引导和少步生成先例；贡献须依赖任务匹配、协议匹配的增量证据。
3. 将可靠性贡献限定为推理时可用的条件信号、受控错配下的损害降低及缺失时回退；一般注意力/门控或自然图像 RefSR 结果不能代替该验证。
4. 将少步/加速放在质量和观测一致性证据之后；报告 NFE、时间、显存和结构/亮回波风险。
5. 先完成 C1 窄条/任意平移重叠审计与地理/采集来源交叉表；C2 当前协议虽然冻结，端到端运行器和运行前验收未完成。B1 仍为未晋级，因此不启动 B2/B3。

## 访问边界

本次仅读取项目文献/研究文档，并访问论文官方页面或作者页面；未打开项目图像，未读取测试集指标，未运行项目代码或模型。本轮测试集图像访问、指标访问、指标使用均为否；项目累计记录仍为测试图像曾被访问、测试指标未用于决策。
