# C1 v2 训练/开发来源路径与派生输入交叉核验

## 结果摘要

只读冻结清单和派生 x2_light 的 train/development CSV/JSON 元数据，完成 1,800 对样本的来源交叉核验：C1 train 为 1,600 对、92 个来源组；development 为 200 对、19 个来源组。`source_group_id + patch_id` 在 C1 与派生输入 `files.csv` 间逐行唯一匹配，SAR 与 optical 的来源 SHA-512 对两角色全部相等。总计 48,739 条断言通过。

## 可追溯连接

1. C1 `sample_manifest.csv` 的 SHA-256 与 C1 主 manifest 中登记值一致；train/development staging manifest 和派生 x2_light 两个角色 manifest 都绑定同一来源清单摘要 `6a278e97c161164d9440b5388c5b3fcda87746a56e26adfb7cdf658b94924e57`。
2. train/development 的原始相对路径按来源组对应到 `season_root/s1_N/…` 与 `season_root/s2_N/…`。可见根目录标签为 `ROIs1158_spring`、`ROIs2017_winter`；这些是清单中的目录名，本审计不据此单独宣称数据集身份、地理坐标或独立采集。
3. 1,600/200 个 group/patch 键均能唯一连接到对应 x2_light `files.csv` 行；派生样本的 `sar_source_sha512`、`optical_source_sha512` 分别与 C1 记录的 SAR、optical 文件 SHA-512 完全相同。
4. C1 `payload_file_manifest.csv` 的摘要与其 payload audit manifest 登记值一致。train 的 3,200 个、development 的 400 个模态路径均有唯一元数据记录；记录中的官方/观测 SHA-512 与文件字节数相等。这里核对的是既有 哈希绑定元数据，本轮没有重读原始图像重算哈希。
5. train/development 之间来源组、原始相对路径、派生 `sample_id` 和模态来源 SHA-512 的交集均为 0。场景组隔离成立；它本身不证明地理独立、轨道/时间独立或像素无重叠。

## 对项目数据的含义与开放边界

这条元数据链把冻结 C1 train/development 的原始相对路径和哈希，连接到 C3-A 使用的派生 x2_light train/development 输入身份。它没有建立本地原始 `data/sar`、`data/opt` 中数字 PNG 文件名与该来源路径之间的逐文件映射；本地数字文件名仍不能据此被认定属于某一官方数据版本。该交叉核验也不重开任何模型训练、指标选择或阶段 gate。

本审计的逐样本连接只使用 train/development；其他角色 CSV 行只读取识别角色所需的前缀后即丢弃。为核验清单完整性，对整份 split metadata CSV 计算/核对 SHA-256，但不提取其他角色的路径、样本或逐行哈希，也不输出其内容。没有打开任何图像、数组、checkpoint 或历史结果，没有训练、推理或 GPU 调用。本轮未读取 confirmation/test 载荷；C1 v2 confirmation 按既有规则继续封存。

## 复现

历史审计脚本为 [audit_source_crosswalk.py](audit_source_crosswalk.py)，摘要输出见[stdout](stdout.txt)，输入和产物哈希见 `manifest.json`。固定历史命令见[运行说明](command.txt)。脚本固定写入本目录的 `results.json`；因当前目录已有历史结果，现会在读取输入前拒绝运行。禁止在原目录重跑。新运行须复制脚本和协议到同级的新唯一目录。

本审计的脚本和聚合结果随本次提交保存；复跑仍依赖项目本地的冻结 C1、暂存、payload audit 与 x2_light train/development 元数据，这些受控输入未随提交发布。因此，干净检出不能单独重现逐样本交叉核验。复跑时只可按协议读取 train/development 元数据，不得访问 confirmation/test 影像载荷。
