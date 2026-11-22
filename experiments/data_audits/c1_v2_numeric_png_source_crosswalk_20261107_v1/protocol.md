# C1 v2 与本地数字 PNG 来源交叉核验方案

## 目标

核查冻结 C1 v2 的 train/development 来源 SHA-512，能否与仓库既有的 `source_hash_matches.csv` 精确对应，从而恢复本地 `data/sar`、`data/opt` 数字 PNG 名称。此次只复用已存在的来源摘要表，不读取或重新哈希任何影像。

## 输入与角色范围

- 冻结 C1 `sample_manifest.csv`：只提取 `role=train` 和 `role=development` 的样本键、来源路径及两个模态的 SHA-512。其他角色仅读取 `role` 字段后排除，不读取其路径、样本键或哈希字段。
- 既有 `experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/source_hash_matches.csv`：该表由 `scripts/audit_sen1_2_upstream_hash_crosswalk.py` 生成；脚本范围限于历史 `fixed_train_dev2k` 与 `fixed_val_dev2k`，并在源码中拒绝 test 路径。此次按 C1 train/development SHA-512 过滤，只有命中后才读取对应的本地路径与上游路径字段。
- 交叉核对不使用历史 split 的指标或模型结果，不读取本地 PNG、派生数组、checkpoint、confirmation/test 载荷，也不启动训练或 GPU。

## 判定

只有当模态及 SHA-512 唯一匹配、旧表本地路径形如 `data/sar/<数字>.png` 或 `data/opt/<数字>.png`，并且上游相对路径与冻结 C1 来源路径一致时，才把该样本计入映射。两模态还必须指向同一数字 basename，才计为完整配对。

零字节摘要命中仅表示这条精确字节哈希路线没有映射；不证明像素不同，也不排除经过重编码的同源图像。不得据此断言数据集身份、场景地理独立或 C1 gate 状态。

## 复现条件

复跑需要项目本地冻结 C1 样本清单和历史上游 `source_hash_matches.csv`。这些受控元数据不包含在本次提交中；提交仅提供脚本、协议和聚合结果，不能从干净检出独立重跑。脚本只应按本协议筛选 train/development 记录，不得读取 confirmation/test 影像载荷。
