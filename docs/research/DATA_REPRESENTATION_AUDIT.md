# 数据表示审计——C1 阶段

**日期：** 2026-10-28
**范围：** 仅限已认证的 `sen12_scene_dev1800_v2` 训练/开发集子集，共 1,600 组训练配对和 200 组开发配对。未访问测试目录、图像或测量值。
**状态：** 已核实该子集的来源身份；本地以数值命名的 2,000 组子集仍未完成来源映射。

## 有来源证据支持的表示信息

SEN1-2 官方补充文档说明：SAR 为 8 位单通道图像，表示以 dB 标度记录的后向散射系数 σ⁰（sigma-nought）；光学图像为由 4/3/2 波段构成的 8 位彩色图像。官方来源审计已将选取的 1,800 组 SAR/光学配对逐一匹配至来源路径，并在解码像素层面一致。该结果核实了所选文件及其编码说明，但没有提供字节值到 dB 的量化查找表、来源产品的定标元数据或原始线性功率值。

当前 B1 v2 加载器将字节值除以 255，并对 SAR 应用 `log1p(20x)/log(21)`。由于缺少从 PNG DN 到 σ⁰ dB、再到线性功率的有据可查逆变换，此操作是在已编码表示上额外施加的非线性变换。不得将其描述为把 SAR 转成定标强度，也不得称为物理正确的对数变换。光学 RGB 仅除以 255，没有逐图拉伸；通道与波段的对应关系依据官方数据集说明。

## 已核验子集的像素分布

本次只读分布检查仅重新打开了 SOTANet 同级缓存中的 1,800 张冻结 `train`/`val` 拼接 PNG。检查先核对拼接文件 SHA-256 和两个解码面板的哈希，再与保留的官方来源验证记录进行比较后更新计数；未读取任何 test 路径。生成的 [`summary.json`](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/summary.json) 报告 SAR 与各光学通道均包含 117,964,800 个像素。

| 通道 | DN 均值 | 总体标准差 | DN=0 | DN=255 | 第 1/50/99 百分位数 |
|---|---:|---:|---:|---:|---|
| SAR | 143.756 | 53.846 | 0.673% | 2.609% | 10 / 146 / 255 |
| 光学 R | 89.561 | 58.084 | 0.663% | 1.586% | 3 / 82 / 255 |
| 光学 G | 91.649 | 54.078 | 0.682% | 1.609% | 4 / 84 / 255 |
| 光学 B | 83.468 | 54.571 | 0.680% | 1.615% | 3 / 73 / 255 |

每个通道都出现了全部 256 个整数编码。端点计数仅描述 DN 频数；现有传递文档无法判定 DN=255 表示传感器饱和、编码上限，还是自然映射的高值。训练集与开发集的 SAR 汇总均值也不同（分别为 145.026 和 133.592 DN）；这是对子集的描述，不是独立分布检验。像素计数存在空间相关性，不得视为相互独立的样本数。

## 证据边界与下一步

- 以上统计属于经官方来源核验的 1,800 组子集，不属于独立的数值文件名 `data/sar` / `data/opt` 2,000 组子集。
- 模型输入表示是对 8 位源编码进行归一化变换的结果。不能由这些数值推断绝对辐射量、线性强度、定标功率或物理 ENL。
- 合成训练结论应限定在表示域。真实 SAR 评价需要来源产品元数据、明确的物理域转换，以及有依据的参考或独立测量。
- 不得使用这些描述性统计选择模型或调节阈值。本次未运行模型，也未改变任何阶段门槛。

## 来源

- SEN1-2 官方补充文档：[`SupportingDocument.txt`](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt)。
- 官方数据集论文：Schmitt、Hughes 与 Zhu，[SEN1-2 数据集论文](https://doi.org/10.5194/isprs-annals-IV-1-141-2018)。
- 来源与配对核验：[`sen12_scene_dev1800_v2_auth_20261021_v1`](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md)。
- 像素分布检查命令与代码：[`audit.py`](../../experiments/data_audits/mmcanet_phase_c1_representation_profile_20261028_v1/audit.py)。
- 当前预处理/退化代码：[`degradation.py`](../../MMCANet/src/degradation.py)、[`prepare_fixed_degradation.py`](../../scripts/prepare_fixed_degradation.py) 和 [`sen12_scene_dev1800_v2/b1_revalidation_v2.json`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.json)。

本次审计的 `test_set_accessed=false`、`test_set_used=false`、`training_or_inference_performed=false`、`model_selection_performed=false`。
