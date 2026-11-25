# BigEarthNet v2 官方元数据与划分交叉审计

**日期：2026-11-25**

**状态：元数据审计完成；官方 train/validation 暂不用于独立来源场景开发评价**
**访问边界：** 只读取官方 Parquet 元数据；未下载或读取任何 SAR/Optical 影像。`test` 行只参与 split 计数与全量 Parquet 文件唯一性检查；其 `patch_id` 未展示、解析或用于来源/空间分组，没有读取影像或测量值。

## 数据与校验

从 [BigEarthNet v2 Zenodo 记录](https://zenodo.org/records/10891137)取得两份公开元数据，并逐一核验 MD5：

| 文件 | 字节数 | MD5 | 行数 | 覆盖范围 |
|---|---:|---|---:|---|
| `metadata.parquet` | 3,616,349 | `55687065e77b6d0b0f1ff604a6e7b49c` | 480,038 | 不含季节性积雪、云或云影覆盖 patch |
| `metadata_for_patches_with_snow_cloud_or_shadow.parquet` | 710,162 | `fe31856f4986d446c9468b59d6387c91` | 69,450 | 上述被排除 patch 的补充元数据 |

合并后共 549,488 行、549,488 个唯一 `patch_id`，与官方总样本数相符。官方 split 标签计数为 train 272,544、validation 139,577、test 137,367。两份元数据均只暂存于 `/tmp`，不纳入仓库。

## train/validation 的来源与空间交叉

详细交叉仅对 `split=train` 与 `split=validation` 行计算。依据 `s1_name` 的命名字段构造两个可复现代理键：

- **acquisition 时间代理：** `s1_name` 前五段（卫星、模式、产品/极化标记与采集时间）。
- **acquisition×UTM tile 代理：** 前六段，再加入 UTM tile 标记。

这两个代理键不是已核验的完整 Sentinel-1 原始产品足迹 ID；它们用于发现重复采集关联，不用于宣称完成产品级来源追踪。

| 交叉层级 | train 唯一组 | validation 唯一组 | 共享组 | 被共享组覆盖的记录 |
|---|---:|---:|---:|---|
| acquisition 时间代理 | 301 | 246 | 241 | train 266,715/272,544（97.86%）；validation 139,501/139,577（99.95%） |
| acquisition×UTM tile 代理 | 346 | 272 | 267 | train 264,397/272,544（97.01%）；validation 139,501/139,577（99.95%） |
| 完整 `s1_name` patch 标识 | 272,544 | 139,577 | 0 | 无重复 patch 标识 |

因此，官方 train 与 validation 没有重复的完整 patch 标识，但绝大多数 patch 落在两种命名代理共同表示的 acquisition 组中。官方地理 split 的“无重复 patch”不能代替 acquisition/source-group 独立性证明。

另从非测试角色的 `patch_id` 解析 UTM tile 与水平/垂直 patch 序号作为空间格网代理：train 有 125,015 个唯一格点、validation 有 64,636 个；完全相同格点交集为 0，同 tile 的八邻接 train–validation 格点对为 23,453。修正后的计数按每个唯一 train 格点直接枚举八邻域中的 validation 格点；先前 12,050 的计算误用了字典序筛选条件，漏掉部分方向的配对，已废弃。邻接关系尚未用地理变换和真实 patch 足迹核验，故只能作为近邻风险信号，不能作米制距离结论。

## 新来源组划分的 metadata-only 可行性

只合并官方 train 与 validation 行，不使用 test 行，按 `s1_name` 前五段代理汇总得到 412,121 个 patch、306 个 acquisition 时间代理组。组大小中位数 832，P90 为 3,584，最大 7,618；301 组只含一个国家，5 组跨两个国家。

为估计分组数量是否足够，按排序后的代理组名、NumPy `PCG64` seed `33012` 产生一个 214/46/46 组的 70/15/15 计数平衡候选，对应 288,527/61,841/61,753 个 patch。该搜索只优化组数和 patch 数比例，**不是 split manifest**；没有验证候选角色之间的地理缓冲、tile 边界足迹、采集产品全覆盖、本地 C1 重合或数据授权。这个结果只说明候选 acquisition 组数量足以继续做独立协议设计。

## 与本地 C1 train/development 来源信息的交叉

只读本地 C1 v2 的既有 train/development staging CSV（1,600 train / 92 组；200 development / 19 组），没有读 confirmation/test staging 或影像。文件路径归于 `ROIs1158_spring` 与 `ROIs2017_winter`，样本/来源组名由 ROI 根目录、S1/S2 序号和 patch 序号组成；该 CSV 未提供地理坐标或 Sentinel acquisition 时间。这些目录命名与[SEN1-2 原论文](https://doi.org/10.5194/isprs-annals-IV-1-141-2018)公开的 season/seed 结构一致，但不足以映射本地具体 ROI。BigEarthNet 也由 Sentinel 数据构成且覆盖欧洲，因此现有 C1 staging 字段无法证明本地 C1 与 BigEarthNet 不重叠，也无法证明有重叠。跨项目独立性仍未解决，需先取得 C1 ROI 到地理位置/采集时间的官方 crosswalk，或选择能够由现有记录证实不重叠的全新来源。

## 决定与边界

1. 暂不把官方 validation 当成独立来源场景 development，也不直接沿用官方 split 做模型选择。
2. BigEarthNet 仍是较广 SAR-only 数据候选。非测试 train+validation 可形成 306 个 acquisition 时间代理组；元数据计数平衡候选为 214/46/46 组、288,527/61,841/61,753 个 patch，但该候选没有落盘且不具备空间独立性证明。官方 test 的影像、元数据来源组和测量值均保持不用。
3. 新分组须先按可核验 S1 source scene 设计，再在原始 tile/patch 足迹上审查跨角色同格、近邻和边界；确认组需预先冻结并封存载荷。现有两份元数据不足以确认产品级场景足迹；C1 staging 不含坐标/采集时间，故跨项目重合也尚未解决。
4. 即使完成重分组，该数据仍是 10 m、校准 dB backscatter 的 Sentinel-1 GRD 分类 patch，不含 clean SAR reference。它仅适合作为 SAR noisy pseudo-target 上的表示域受控退化/超分探索，不能单独支持真实去斑或 radiometric fidelity 主张。
5. 本次没有创建 split manifest、下载 54.4 GB S1 影像归档、解码 Optical、训练或推理；没有改变 C2/C3 阶段资格。

官方来源： [BigEarthNet v2 页面](https://bigearth.net/)、[数据说明 PDF](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf)、[Zenodo 记录](https://zenodo.org/records/10891137)、[SEN1-2 原论文](https://doi.org/10.5194/isprs-annals-IV-1-141-2018)。
