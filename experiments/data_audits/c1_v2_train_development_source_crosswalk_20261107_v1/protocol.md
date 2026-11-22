# C1 v2 训练/开发来源路径与派生输入交叉核验协议

## 目的

核验冻结 C1 v2 的 train/development 成员能否通过 `source_group_id + patch_id` 映射到清单中的 SAR/optical 原始相对路径，并进一步对应到派生 x2_light train/development `files.csv` 的来源 SHA-512。该项是 split-integrity 与来源元数据审计，不是图像质量或模型评价。

## 允许读取的元数据

- 冻结 C1 v2 的 `sample_manifest.csv`、`group_assignment.csv`、来源组摘要以及被这些清单绑定的 SHA-256。
- C1 train/development staging 的 `manifest.json` 和 `staging_files.csv`。
- 派生 x2_light 的 train 与 development `manifest.json`、`files.csv`。
- C1 payload audit 中绑定 `payload_file_manifest.csv` 的摘要，以及该文件中 train/development 对应原始相对路径的文件大小和 SHA-512 元数据。

所有角色行只用于提取 train/development；confirmation 与 test 的路径、样本、哈希和结果不输出、不用于连接。本轮不打开任何 PNG、Numpy 数组、SAR/optical 文件、checkpoint 或模型，也不遍历原始影像目录。

## 预定检查

1. 校验 C1 主清单绑定的 `sample_manifest.csv`、train/development staging manifest 记录的来源清单和 staging CSV 摘要，以及两份派生 x2_light 角色 manifest 的来源清单/`files.csv` 摘要。
2. 核对 train/development 的样本数分别为 1,600/200，来源组数为 92/19，来源组、原始相对路径与 SHA-512 均无跨角色交集。
3. 按 `(role, source_group_id, patch_id)` 将 C1 原始路径元数据与派生 `files.csv` 连接，逐样本核对 SAR 与 optical 的来源 SHA-512。
4. 按 `season_root/scene_id` 核对 SAR `s1_N` 与 optical `s2_N` 相对目录；并检查 train/development 选择路径在 hash-bound payload file manifest 中有唯一记录，且其记录的官方/观测文件大小及 SHA-512 一致。

## 解释限制

哈希与路径证据证明的是仓库清单之间的元数据连接，不是本次重读原始图像后重新计算的内容校验。结论只覆盖 train/development 的派生 x2_light 清单；它不自动建立用户原始 `data/sar`、`data/opt` 数字文件名与原始相对路径的逐文件映射，也不证明地理独立、像素级配准或传感器辐射定标属性。confirmation 永久封存；C1 v2 没有独立 test 角色。

## 复现条件

复跑脚本需要项目本地冻结 C1 样本/来源组元数据、train/development 暂存清单、C1 payload audit 元数据，以及派生 x2_light train/development 的 manifest 与 `files.csv`。这些受控输入不包含在本次 Git 提交中，因此干净检出后不能单独重跑本审计。若在项目数据环境复跑，必须遵守本协议的角色过滤规则；不得读取 confirmation/test 影像载荷或将其身份用于连接。

脚本固定写入本目录的 `results.json`；若该文件已存在，脚本会在读取输入前拒绝运行。当前历史目录已有结果，禁止在原目录重跑。新运行须复制脚本和协议到同级的新唯一目录，并依照本协议准备本地受控输入。
