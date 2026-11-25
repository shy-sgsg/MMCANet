# Phase D 更广 SAR 数据源审计

**日期：2026-11-25**

**状态：来源筛查与 BigEarthNet 元数据交叉审计完成；未下载额外数据载荷**
**目标：** 为 SARptical 单城区 pilot 之外的新 development 数据寻找可追溯的地理分组、SAR 表示和使用条件。

## 结论

下一优先候选仍为 **BigEarthNet v2/reBEN**，但**官方 train/validation 不能直接作为来源场景独立评价**。元数据交叉审计发现，`s1_name` 的 acquisition 时间代理在 train/validation 共享 241 组，覆盖 97.86% 的 train 记录和 99.95% 的 validation 记录；tile 级代理共享 267 组。完整 patch 标识没有重复，非测试 patch 格网代理也没有相同格点，但发现 23,453 对同 tile 八邻接格点，真实足迹和距离尚未核验。若继续，必须只从官方 train+validation 重新按来源组划分；官方 test 影像、来源字段和测量值不用于该实验。

BigEarthNet v2 是 10 m Sentinel-1 GRD 分类基准，不是干净 SAR 恢复集或高分辨率真值；它可支撑 SAR 表示域合成 ×2 pilot，不能直接证明真实去斑质量。SARptical 仍是当前用户已批准的有限 pilot 数据。

## 候选对照

| 数据源 | 可核验规模与来源 | 划分/许可/访问 | 研究适用性与限制 |
|---|---|---|---|
| **BigEarthNet v2/reBEN** | 549,488 个 Sentinel-1/2 配对 patch；S1 来自 312 个 IW GRD 场景；覆盖欧洲 10 国；SAR/Optical patch 为 10 m 网格，SAR 以校准后的 dB backscatter 保存且未做 speckle filter。官方 metadata 含 patch id、split、country、关联 S1 patch 名。 | 官方记录提供地理划分的 train/validation/test 和 metadata；许可为 CDLA-Permissive-1.0。S1 压缩档为 54.4 GB，metadata.parquet 为 3.6 MB。 | **有条件候选。** 不能直接沿用官方 validation；先按来源组重划 train+validation、核验 patch 邻近和完整 S1 产品身份，并解决本地 C1 地理/采集 crosswalk。只有通过这些审计后才适合用作更广 SAR-only development。数据仅 10 m、源自分类任务且无 clean SAR target；不应全量解包。 |
| **SEN12MS** | 180,662 个双极化 SAR、Sentinel-2 与 MODIS triplet；10 m；跨所有有人居住大陆和不同季节。官方存储为按区域/季节组织的 SAR、光学和标签压缩包。 | 作者工具箱提供按场景的 hold-out，约占 10%；余下 train 列表没有官方 validation 列表。官方页面说明下载需要较长时间；本次未读取文件清单或下载。 | 地理范围广、场景名可用于组划分，但要另从 train 场景冻结 development；需检查相邻/重叠 patches、scene leakage 及与本地 C1 来源的重合。用于 restoration 仍只有 noisy SAR pseudo-target。 |
| **QXS-SAROPT** | 作者论文报告 20,000 对 256×256、约 1 m SAR/Optical patch，来自 GaoFen-3 与 Google Earth，覆盖 San Diego、Shanghai、Qingdao 三个港口城市。 | 作者仓库通过问卷提供下载；README 要求研究引用论文。仓库未公开独立 license 文件，数据使用/再分发条件需先向作者核实。训练/测试地理分组和源 scene crosswalk 未在本次官方 README 确认。 | 与米级城市恢复更贴近，适合作为 SARptical 之后的高分辨率多城候选；访问门槛、Optical 权利和 split 元数据尚未解决。不能仅凭文件编号推断本地 2k PNG 就是该数据集。 |

## BigEarthNet v2 的准入审计项

1. 官方 Parquet 元数据已下载并核对 MD5；主文件 480,038 行、补充云/雪文件 69,450 行，合并后 549,488 个唯一 patch。官方 split 计数为 train 272,544、validation 139,577、test 137,367；test 行仅参与 split 计数和不展示 patch_id 的文件唯一性检查，未分析来源/空间字段或测量值。
2. train/validation 来源代理审计发现 241/301 个 acquisition 时间代理组、267 个 tile-acquisition 代理组跨角色；97.86% 的 train 与 99.95% 的 validation 记录落入共享时间代理组。完整 patch ID 无交集。空间格点代理无同格交集，但有 23,453 对同 tile 八邻接格点，尚未折算实际距离。首轮 12,050 为脚本漏计，已在专项报告更正。
3. 若采用 BigEarthNet，非测试 train+validation 有 306 个 acquisition 时间代理组；一个纯计数平衡示例可用 214/46/46 组构成约 70/15/15 的 patch 数，但它不是 split manifest，且未施加空间邻近约束。正式候选须仅从官方 train 与 validation 重新按可核验 source scene 建立 train/development/封存 confirmation；继续核验 `s1_name` 是否能映射到完整 S1 产品/足迹，并检查相邻 footprint 与本地 C1 来源重合。官方 test 载荷与来源组信息不用于该实验。
3. 本地 C1 v2 train/development staging 中，1,600/200 样本按 92/19 个来源组记在 `ROIs1158_spring`、`ROIs2017_winter` 下，但文件名没有坐标或 acquisition 时间。与 BigEarthNet 的跨项目重叠尚不能判断；需获取该 C1 ROI 的官方地理/采集 crosswalk，或选择能够核实为不重叠的新来源。不能只凭数据集名不同声称独立泛化。
4. 只读取获准角色的 SAR VV/VH GeoTIFF；Optical 通道不进入当前 SAR-only 流程。任何 optical utility 实验仍需遵守 C2 v2 gate 和独立开发协议。
5. 官方 S1 包约 54.4 GB，metadata 约 3.6 MB。当前实测项目根分区可用 6.1 GiB，挂载的 NTFS 数据卷可用约 90 GiB。优先先读 metadata，再决定是否将官方压缩包写入该卷并流式提取所选 SAR patches；不能先完整解包 54.4 GB 压缩档。

## 来源

- [BigEarthNet v2 官方页面](https://bigearth.net/)与[官方数据说明](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf)：patch 数、场景数、处理、metadata 字段、空间划分与许可。
- [BigEarthNet v2 Zenodo 记录](https://zenodo.org/records/10891137)：官方归档、metadata 文件大小与数据集许可。
- [SEN12MS 一手论文](https://isprs-annals.copernicus.org/articles/IV-2-W7/153/2019/isprs-annals-IV-2-W7-153-2019.html)及[作者工具箱](https://github.com/schmitt-muc/SEN12MS)：样本规模、地理范围和 scene hold-out 约束。
- [QXS-SAROPT 作者仓库](https://github.com/yaoxu008/QXS-SAROPT)及[原始论文](https://arxiv.org/abs/2103.08259)：来源城市、规模和请求式下载入口。

详细行数、校验值、代理键定义和边界见[BigEarthNet 元数据审计](PHASE_D_BIGEARTHNET_METADATA_AUDIT_20261125.md)。该审计只用公开论文、官方仓库、数据说明和文件元数据；没有下载或解码影像载荷，也没有改变 C2/C3 阶段权限。
