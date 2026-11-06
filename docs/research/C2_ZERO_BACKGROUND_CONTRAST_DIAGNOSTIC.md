# C2 零背景对比差诊断设计与合成验证

## 目的

C2 v2 的正式指标和 gate 保持冻结。其必需 median TBR 遇到零背景中位数时按原规则 fail-closed；本设计不改写该结果。新增函数 `compute_c2_contrast_diagnostic_metrics` 只提供独立的表示域描述统计，供另行预登记的诊断使用，不接入 C2 v2 评分器。

## 指标定义

所有统计都使用同一 SAR 输入生成、并在方法间共用的目标区掩膜 `T` 与背景区掩膜 `B`。对图像 `X` 定义：

```text
均值对比差：C_mean(X)   = mean(X[T])   - mean(X[B])
中位数对比差：C_median(X) = median(X[T]) - median(X[B])
对比差绝对误差：E = |C(prediction) - C(target)|
```

该差值不以背景值作分母，因此背景统计量为零时仍可计算。描述性 mean/median TBR 仅在对应背景均值/中位数严格大于零时返回；否则数值为 `null`，状态字段指出 target 或 prediction 的分母未定义。这里的 TBR 是归一化 PNG 表示域比值，不是辐射定标量，也不自动取得物理解释。

目标区和背景区掩膜必须非空、尺寸匹配且不重叠；影像必须是有限值并处于 `[0,1]`。函数不加 epsilon、不改变输入、不重新生成掩膜，也不参与模型或阈值选择。

## 实现与验证

- 实现：[`phase_c2_screen.py`](../../MMCANetV2/phase_c2_screen.py) 中的 `compute_c2_contrast_diagnostic_metrics`。
- 合成测试：[`test_c2_contrast_diagnostic.py`](../../tests/test_c2_contrast_diagnostic.py) 覆盖全零背景、正背景、预测端单侧零背景、重叠掩膜拒绝。
- 验证命令：`/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_c2_contrast_diagnostic.py tests/test_phase_c2_confirmation_metrics.py tests/test_evaluator.py --tb=short`。
- 结果：43 项通过；相关 Python 文件 `py_compile` 通过。
- 数据边界：本验证只构造合成数组，没有读取 train、development、confirmation 或 test 影像载荷，也没有重算 C2 模型输出。

## 后续使用边界

若要在真实 development 上比较该合同，必须先单独登记新的描述性诊断协议，固定使用同一输入身份和掩膜并报告来源组统计。该结果只能是已反复使用 development 上的探索观察；不能替代 C2 v2 gate，不能使用已封存 confirmation 作调参或筛选，也不能据此宣称独立泛化或物理一致性。
