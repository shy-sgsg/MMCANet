# C1 v2 train/development 与本地数字 PNG 映射未能建立

**审计日期：** 2026-11-07

**审计类型：** 只读来源元数据交叉核验
**训练/GPU：** 未运行

## 结论

冻结 C1 v2 的 1,600 对 train 与 200 对 development，共 3,600 个 SAR/optical 来源 SHA-512，在既有本地数字 PNG 上游哈希表中没有精确命中。因此，本次不能恢复这些 C1 来源路径与 `data/sar/<数字>.png`、`data/opt/<数字>.png` 文件名之间的映射。

## 方法与范围

从冻结 C1 `sample_manifest.csv` 中，只解析 `train` 和 `development` 的样本键、来源路径及 SHA-512。其他角色只读取 `role` 字段用于筛除，不读取其路径、样本键或哈希字段。

将这些来源摘要与既有上游哈希交叉表（项目本地路径：`experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/source_hash_matches.csv`）按模态和 SHA-512 精确连接。该历史表由本地既有审计脚本（`scripts/audit_sen1_2_upstream_hash_crosswalk.py`）生成，脚本输入范围为当时的 train/validation 文件清单，并拒绝 test 路径。本次只复用表内摘要和命中记录；不重新执行该旧审计，不读取任何 PNG，也不使用历史 validation 指标或模型结果。

## 结果

| 项目 | 结果 |
| --- | ---: |
| C1 train/development 样本 | 1,600 / 200 对 |
| 待连接来源模态 | 3,600 |
| 既有表精确 SHA-512 命中 | 0 |
| 可恢复本地数字 PNG 路径 | 0 |
| SAR/optical 同编号完整配对 | 0 / 1,800 对 |

## 解释边界

零精确字节哈希命中只说明这份历史表不能建立该映射；它不证明像素不同，也不排除重编码后的同源图像。此次不据此判定 C1 的数据集身份、传感器属性、地理/采集独立性或阶段 gate。历史 `fixed_val_dev2k` 可能受到既有污染；其记录在这里只作为文件来源索引，不作为泛化、模型选择或晋级证据。

此次仅读取 CSV/JSON 元数据，未访问影像、数组、checkpoint、confirmation/test 载荷或结果；未训练、未调用 GPU，也没有修改原始数据、协议或 gate。

## 复现边界

本次提交包含本审计的脚本、协议、聚合结果与运行摘要，不包含冻结 C1 样本清单或历史 `source_hash_matches.csv`。因此，干净检出不能单独重跑映射审计；复跑需要项目本地受控元数据，并须遵守本协议的 train/development 过滤规则。
