# C1 v2 数据协议与当前使用资格审查

**审查日期：** 2026-11-05
**结论：** 近期探索直接沿用冻结 C1 v2 划分；不重新随机划分，不把 confirmation 当作未触碰确认集，也不把仓库数字命名 PNG 当作 C1 v2 身份。

## 冻结协议和分组依据

C1 v2 冻结协议位于 protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/。协议以官方 SEN1-2 根目录与匹配的 s1_i / s2_i 来源场景文件夹作为分组单位，先按整个来源文件夹分配角色，再在组内选 patch。协议记录 131 个来源场景组、1,600/200/200 个 train/development/confirmation 配对和 92/19/20 个来源组。

C1 v2 资格决定及 payload audit 记录：选定的 train/development 文件与官方 SHA-512 和文件大小相符，影像为 256×256 uint8；在已审计范围内没有跨角色精确文件或解码影像重复。受限的窄重叠扫描没有检出整数平移候选，但未排除小于扫描窗口、重采样、亚像素平移或被排除重复签名等情况。来源场景文件夹分离不证明地理位置、采集时间或影像 footprint 独立，细配准也未核实。

权威依据：
- [冻结 C1 v2 协议](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/protocol.json)
- [来源场景分组清单](../../protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/manifest.json)
- [C1 v2 development 使用资格决定](../../protocols/mmcanet_phase_c1/v2/c1_development_eligibility_20261028.md)
- [当前研究状态](CURRENT_STATE.md)

## 各角色的当前资格

| 角色 | 数量与来源组 | 当前用途边界 |
|---|---:|---|
| train | 1,600 对 / 92 组 | 可依各自已冻结探索协议训练；本轮未重新扫描或更改身份 |
| development | 200 对 / 19 组 | 已用于多轮评价与选模，只能作为重复使用的开发数据，不能称独立 holdout |
| confirmation | 200 对 / 20 组 | 2026-11-05 已访问并完成一次固定模型的描述性复核；复核经过载荷访问后的协议字段修订，结果为 inconclusive。现已封存，不再用于训练、调参、选模或独立确认主张 |
| test | 无独立角色 | C1 v2 方案没有单独 test 分区；仓库历史 test 污染另行记录，不与 confirmation 混同 |

本轮没有打开 C1 v2 confirmation 影像或测量文件；相关状态仅根据已有状态记录与复核报告确认。没有访问或使用 test 影像/测量结果。

## 研究使用决定

用户已明确要求沿用项目冻结的 C1 v2 划分。对当前有界探索，直接使用该协议绑定的 train/development 固定输入，不再请求目录路径、不新造随机 split、不以约 20,000 对数据作为开工前提。C3-A 的既有协议绑定 x2_light train/development 固定输入；其 GPU 训练资源门槛仍按该运行协议执行。

仓库 ./data/ 中数字命名的 2,000 对 PNG 尚未映射到 C1 v2 身份。该事实不阻止使用已冻结 C1 v2 train/development，也不允许把两个数据树假定为同一来源。若将来需要地理/采集独立的正式泛化或 untouched confirmation 证据，须另行获得可核实的来源 footprint/采集信息或新的独立来源组；不得重命名已访问的 confirmation。

## 当前结论边界

现有 PNG 与目标只能支持明确限定的表示域和 noisy pseudo-target 研究。它们不足以证明原始复数 SAR、定标幅度/功率、真实 clean-SAR 参考、物理 speckle 似然或细配准。C1 v2 的来源文件夹划分可支持限定范围的开发探索，不等于地理或采集独立性证明。
