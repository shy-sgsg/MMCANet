# C1 v2 train/development 官方来源链核查

> **复现边界：** 本文总结项目此前保留的来源核验记录。本次提交包含本报告及聚合交叉核验产物，但不包含完整样本/来源清单、暂存文件清单、获取清单或旧上游哈希表。因此，不能仅凭本次提交重跑该来源审计；复核依赖项目本地受控元数据。本次提交不含 confirmation/test 影像载荷。

## 结论

冻结 C1 v2 的 train/development 暂存记录与 TUM MediaTUM 官方条目 1436631 及其发布的文件清单相连。现有证据支持这 1,600 组训练和 200 组开发载荷按冻结清单从该官方条目暂存，并逐文件核对字节数、SHA-512 和 PNG 结构。它不证明这些来源场景在地理位置、采集日期、轨道或精细空间范围上彼此独立。

项目旧 `fixed_train_dev2k` / `fixed_val_dev2k` 文件属于另一条历史输入链。旧交叉审计覆盖 3,600 个文件，在其检查范围内没有发现与上游文件索引的字节级精确匹配；这不能用于否定 C1 v2 train/development 暂存链，也不能排除旧文件经过重编码或其他变换。仓库 `./data/` 数字命名的 2,000 对 PNG 仍没有映射到 C1 v2 清单的可审计身份表。

## C1 v2 train/development 的本地证据

- C1 v2 来源组协议构造器（项目本地路径：`protocols/mmcanet_phase_c1/v2/build_source_group_protocol.py`）将公开 MediaTUM 导出节点固定为 `1436631`，使用 `m1436631` rsync 模块及官方 `checksums.sha512` 文件索引；来源根目录为 `ROIs1158_spring` 和 `ROIs2017_winter`。协议按季节根目录与 SAR/optical 来源场景文件夹构造 `source_group_id`，再按冻结规则抽取配对 patch。该协议生成阶段只读取元数据，不下载影像载荷。
- train/development 暂存运行清单（项目本地路径：`experiments/data_staging/c1_v2_train_development_20261031_parallel_v1/manifest.json`）记录 `status=complete`、冻结 C1 样本清单 SHA-256 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`、官方条目记录 SHA-256 `aaaaa19232e0f468adfadb06a2ce33bee0c5513ed873986c4e77bcf42120a7c4`、3,600 个请求文件、角色数 1,600/200、来源组数 92/19，以及完成后无 confirmation/test 载荷访问的标记。
- 暂存获取清单（项目本地路径：`data/derived/c1_v2_train_development_20261031_v1/acquisition_manifest.json`）记录暂存完成、3,600 个文件、786 个按 SHA-512 核验后复用的局部文件，以及同一官方条目记录摘要。暂存程序在合并后对冻结角色要求的全部文件重新核对文件集合、字节数和官方 SHA-512，再生成派生输入。
- 暂存结果证明的是**冻结 C1 v2 train/development 文件身份与官方发布文件清单的字节级对应关系**。它不自动验证论文描述的传感器产品属性，也不提供场景坐标、采集日期、轨道或 patch footprint。

## 旧历史文件哈希交叉审计的适用范围

旧上游哈希交叉审计摘要（项目本地路径：`experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/summary.json`）的 `scope` 明确为 `fixed_train_dev2k` 与 `fixed_val_dev2k`，合计检查 3,600 个旧文件，`exact_unique_file_matches=0`。该结果仅说明这些历史输入在该索引版本中没有字节级精确匹配；审计自身注明，这不排除像素等价重编码或其他转换，也没有从数字文件名推断来源 ID。

因此，旧审计不能套用到 2026-10-31 从 C1 v2 官方文件清单核验并暂存的 train/development 载荷，也不能证明或否定 `./data/` 当前所有文件的来源。`./data/` 数字命名子集与 C1 v2 身份的映射仍未知。

## 尚未解决的独立性问题

当前可报告 C1 v2 三个角色按来源场景文件夹分组互斥、train/development 文件身份互斥。现有来源清单没有将 `source_group_id` 对应到地理坐标、采集日期、轨道、产品 ID 或 patch footprint 的完整交叉表，故不能据此声称地理或采集独立，也不能排除邻近 patch 的部分空间重叠。TUM 在线条目及 OAI-PMH 在 2026-11-07 的直接访问曾返回 HTTP 429；这妨碍本轮在线刷新，但不抹去已保留的 2026-10-31 官方获取与逐文件校验记录。

## 阶段影响

不重划数据，不改变 C2 gate、C3-A 协议或 B1/B2/B3 阶段状态。用户要求暂停训练，因此不启动 GPU 训练、等待器或正式评价。此项为依据冻结协议、代码和既有获取 manifest 的元数据/记录核对；本轮未读取影像载荷。
