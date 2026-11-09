# 多样本概率评分规则一手来源核验

## 审查问题

为 v1.8 生成分布评价设计核对三项候选评分：连续秩概率分数（CRPS）、能量分数和变差分数。重点确认经验集合形式、空间依赖敏感性及评分规则本身的限制。

## 一手来源

1. Gneiting 与 Raftery，*Strictly Proper Scoring Rules, Prediction, and Estimation*，2007，*Journal of the American Statistical Association* 102(477)，DOI：[10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)；作者/机构公开全文：[PDF](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。全文将 CRPS 描述为针对一维预测分布的评分，并将其纳入更一般的能量分数框架；能量分数适用于有限样本表示的预测分布。对欧氏距离幂指数 `0<β<2` 的能量分数，文中给出严格适当性结论及相应有限矩条件。
2. Scheuerer 与 Hamill，*Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities*，2015，*Monthly Weather Review* 143，1321 起，DOI：[10.1175/MWR-D-14-00269.1](https://doi.org/10.1175/MWR-D-14-00269.1)；作者机构存档全文：[NOAA PDF](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf)。论文给出以分量两两差值幂矩构造的变差分数，并用非负权重强调或降低特定配对。作者报告能量分数对分量相关性失配的敏感性可能有限；他们的模拟和风速案例显示，变差分数对相关结构的区分可更明显。

## 对评价设计的影响

- CRPS 可逐像素检查边缘分布，但它不包含跨像素联合依赖信息。
- 能量分数可以对整图向量的经验预测集合打分；其普适适当性不保证对每一种空间依赖错误都足够敏感。应与专门检查空间差异的量并列使用。
- 变差分数可将相邻像素对纳入局部空间检验。项目当前候选用 `p=1`，该阶数在来源中称为 madogram。变差分数是适当但非严格适当的规则，且把联合分布压缩为配对差矩会丢失信息；不能单独证明完整联合分布匹配。
- 上述论文来自概率天气预报/通用概率预测，不验证 SAR 图像域或 noisy pseudo-target。项目使用前仍须按已知案例检查实现、像素尺度、边界/掩膜和场景汇总方法。

## 与本项目合成诊断的对应

2026-11-09 图像域已知案例中，逐像素 CRPS 未区分边缘分布相同但空间依赖不同的候选；归一化整图能量与邻接变差分数均识别出该构造。该观察与来源对能量分数相关性敏感性的限制相容，也说明需要保留多种互补指标。玩具案例不替代真实 SAR 输出上的验证。

详见[图像域指标实验](../../synthetic_cpu/c3a_image_distribution_metrics_cpu_20261109_v1/report.md)、[补充评价设计](../../../docs/research/C3A_DISTRIBUTION_EVALUATION_ADDENDUM_DESIGN_20261109.md)和[正式评分规则文献综述](../../../docs/research/literature_review.md)。

## 证据边界

此审查只支持通用概率评分的定义、公式和理论性质。它不支持 SAR 物理主张、光学效用、模型质量、真实生成分布校准或任何阶段晋级，也不构成本地实验结果。
