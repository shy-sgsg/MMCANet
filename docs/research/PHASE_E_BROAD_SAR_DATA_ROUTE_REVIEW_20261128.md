# 更广 SAR 与 Optical 数据路线复核（2026-11-28）

## 2026-11-28 追加：AllClear 样本可执行性与 Mendeley 来源身份纠正

### AllClear train/validation JSON 覆盖已核清

官方 train/validation 清单为 19,013/997 个互斥 ROI。对五份 train/validation JSON 交叉后发现，共享 `rois_metadata.csv` 中额外 34 个 train 角色 ROI 全部不在任何所查 train JSON；显式清单中另有 334 个 train ROI、13 个 validation ROI 未出现在任何相应 JSON。五份 JSON 均没有显式角色清单外 ID。当前可执行队列为显式清单与任务 JSON 的交集（18,679/984 个 ROI）；逐项缺失原因与 34 行产生原因仍未查明。结果见[筛查报告](../../experiments/data_audits/allclear_train_val_eligibility_crosscheck_20261128_v2/筛查报告.md)。

这一核对没有解决 AllClear 的任务错位：其输出是 S2 云去除产品，S1 为辅助模态，没有 clean SAR target。当前不把它作为 SAR restoration 训练数据，也不下载全部影像。

### 已用于 Phase E Task B 的 Mendeley Sentinel-1 来源

[Díaz-Paz 等 2026 年 Data in Brief 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13333279/)及[官方 Mendeley Data 记录](https://data.mendeley.com/datasets/6nzd6x25vw/1)描述全球 Sentinel-1 VV/VH 多时相融合数据，记录标注 CC BY 4.0。它不是本轮新发现：Phase E Task B 已使用其 GT25 VV 融合参考。冻结协议按官方元数据精确交集得到 467 个候选，排除 34 个缺元数据文件；train/development/confirmation split 为 327/70/70。仅读取 train/development 共 397 张影像，confirmation 仅保留成员元数据。既有 `fusion.rar` 归档位于外置持久卷，大小 1,827,618,397 字节。

同来源单景 SAR 包与 GT25 参考没有同地点、同轨道、同极化直接配对键，故单景包未进入训练。两项 Task B 模型试验将 GT25 参考先做固定表示域合成退化；结果不能解释为真实单景 SAR 输入与多时相参考的配对实验。GT25 是时间融合伪参考，不等于 clean SAR；时间变化、配准误差、融合平滑及 8-bit 表示限制其物理解释，也没有 HR SR 目标。

Mendeley 网页当前没有可见 Files 行，无认证 files API 返回 HTTP 401；这只限制匿名读取文件清单，不代表归档没有获取或既有实验不可用。本轮没有重新下载该数据。来源、输入/目标关系和协议证据详见[Task B 冻结协议](PHASE_E_TASK_B_GLOBAL_SENTINEL1_PILOT_PROTOCOL_20261126_V3.md)。

下一步不再把同一 Mendeley 来源计为独立数据候选；E5 转向筛查不同的、可授权且来源组可核验 SAR 数据。单一来源失败后继续其它候选，不启动当前 cohort 的第三个生成模型候选。

### DG-SARFM / SAR-NDIP：SAR-only 去噪预训练候选，许可与样本来源待核

出版社页面将 *Denoising-guided SAR foundation model* 列为 ISPRS Journal of Photogrammetry and Remote Sensing 论文，DOI `10.1016/j.isprsjprs.2026.10.019`；作者仓库称其于 2026-10-24 在线发布、卷期为 2027 年。出版社摘要描述 DG-SARFM 将 denoising-guided masked reconstruction 与 hierarchical contrastive learning 结合，用于 SAR 表征预训练，并在分类、检测、分割任务中评估。它不是联合去斑/超分图像输出方法，也没有 Optical 试验。

作者仓库 README 报告 SAR-NDIP 有 1,024,177 组 noisy/denoised SAR patch，来源为 Sandia MiniSAR、HRSID、MSAR、SARBuD、SAR-AIRcraft、OGSOD、ATRNet-STAR 和 FAIR-CSAR；使用 FANS 生成 denoised 伪目标，patch 尺寸为 128×128 或 256×256。README 提供百度网盘数据/权重包入口及公开提取信息；报告不复制提取码。仓库包含 MIT 代码许可证，但 README 明确第三方源数据仍遵循各自许可，因此不能把该代码许可证视为 SAR-NDIP 数据授权。

该资源可作为 SAR-only 表征/去噪预训练候选，不能因为有百万 patch 就当成百万独立场景或新 development cohort。FANS 伪目标可能带来平滑偏差；目前未取得逐样本来源清单、固定 split、包体量、源数据许可汇总及与本地历史训练数据的 overlap crosswalk。出版社全文本次返回 403；作者公开分享页的目录元数据接口返回 `errno=-12`，没有获得文件清单或体量，也没有继续尝试。所核证据限于出版社摘要和作者 README；未下载数据包或代码，也未运行实验。只有拿到源数据授权/许可映射、包清单与可按场景分组的样本 manifest 后，才评估有限 SAR-only 预训练用途；否则继续其它路线。

## 结论

目前没有已核验且独立于既有 Task B 的数据源可以直接代替正式 SAR 联合去斑/超分训练集。SAR-NDIP 提供表征预训练用的 FANS 去噪伪配对线索，但仍缺源数据许可、样本级 provenance、固定 split 和体量证据；它不提供 Optical 或 SR 目标。AllClear 的可执行队列已在任务 JSON 层明确，但其 S2 输出任务和无 clean SAR target 使其不适用。Mendeley 多时相 GT25 已用于表示域合成代理；缺直接对应键的单景 SAR 包被排除，因此它不是新的真实 SAR 配对 cohort。GUSO 覆盖广但下载与授权门槛很高。Umbra Open Data 有 SAR 时序线索但不含 Optical 配对；SARptical 仍只覆盖柏林城区。

本轮 AllClear 交叉审计只读取 train/validation 元数据和样本 JSON，没有打开 test 专属元数据文件或读取影像载荷；这与既有 Task B 已使用的 GT25 归档分开记录。confirmation/test/Optical 载荷均未访问。GUSO、OSDataset 和其它候选均未下载影像。

## 已完成来源和模型实验

### SARptical

用户批准的有限 pilot 已在 2026-11-26 完成：Berlin 单城区、12/8/8 个 250 m 空间组、916/306/306 张 train/development/封存 confirmation；confirmation 图像未读。当前 Flow Bridge 相对任务适配 SwinIR 的组等权 ΔPSNR 为 −2.6412 dB，高回波 ROI 门槛 0/8 组有效，按冻结规则停止。详见[SARptical 结果](PHASE_D_SARPTICAL_SPATIAL_PILOT_RESULTS_20261126.md)。

### Mendeley Data 全球 Sentinel-1 Task B 代理（既有来源）

Phase E 使用官方 Mendeley Data 的 GT25 VV 多时相高视数伪参考，构建固定表示域 ×2 合成退化；它不是已配对的真实单景 SAR。两项冻结生成候选都在 70 个 development 来源组上未通过 gate，详见[Phase E Research Decision Review](PHASE_E_TASKB_GLOBAL_PILOT_RESULTS_20261128.md)。不以该负结果否定其它数据或所有生成方法。

## Umbra Open Data：SAR 时序候选，尚非训练数据

本轮从 [Umbra Open Data 官方页面](https://umbra.space/open-data/)、[AWS Open Data Registry](https://registry.opendata.aws/umbra-open-data/)列出的公开 S3 前缀只读取 XML 对象清单和 STAC JSON 元数据；没有读取资产链接指向的 GeoTIFF、NITF 或 CPHD。

- `open-data/` 清单未截断，共 537 个对象，其中 134 份 STAC sidecar；对象列出的合计大小约 1.348 TB（没有下载这些资产）。134 份 sidecar 合计约 0.99 MB，压缩归档 SHA-256 为 `fe97d1aa8e37b67622d3ad855c3343049caeeb74ef6eae56238deb0700780860`。
- 按 item ID 去重后有 101 个唯一条目，采集日期为 2025-12-17 至 2026-01-18，共 26 天；涉及 Umbra-07/09/10、VV/HH、1/2/4 个 azimuth looks。sidecar 在公开前缀上的最后修改时间延至 2026-02-05；下载时未看到更晚的 acquisition date。Umbra 静态目录的 `Last-Modified` 为 2025-08-13，需与公开 S3 前缀分别看待。
- 按 bbox 经纬度平面相交占较小 bbox 面积至少 50% 建连通分量，得到 42 个**粗略候选位置组件**。此算法不是米制空间核验，也不代表真实 scene/source ID。
- 在 10 个组件内发现 11 对同平台、极化与 looks 匹配的重复采集；间隔 2.50–14.52 天，入射角差最大 11.12°。尚未核验绝对定标、共用像素网格、配准、地表变化或散斑独立性。该数量不足以直接形成规模充足的跨来源 train/development/confirmation 方案。

官方 Umbra 文档说明 Archive Catalog STAC API 需要认证，而开源 S3 前缀是另一个入口；本轮仅用公开前缀。元数据、对象清单、item 哈希、分组算法与详细限制见[Umbra 元数据筛查清单](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/元数据筛查清单.json)、[压缩 STAC 归档](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/umbra_stac_items_20261128.jsonl.gz)及[复算脚本](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/audit_umbra_metadata.py)。

## 更广配对源候选

### AllClear：角色表差异已交叉到任务 JSON，仍不适合 SAR 输出任务

[AllClear 官方网页](https://allclear.cs.cornell.edu/)报告 23,742 个全球 ROI；[作者 NeurIPS 论文](https://papers.nips.cc/paper_files/paper/2024/file/60095e1d7ebb292dbba93c4d8f7b2463-Paper-Datasets_and_Benchmarks_Track.pdf)最终统计为 23,708 个，train/validation/benchmark 为 19,013/997/3,698 个 ROI。[作者仓库](https://github.com/Zhou-Hangyu/allclear)列出显式角色清单和各任务 JSON。2026-11-28 的 train/validation-only 交叉审计显示：显式清单分别有 19,013/997 个互斥 ID；`rois_metadata.csv` 比 train 清单多 34 个角色行，34 个 ID 均未进入所查任一 train `tx12`/`tx3` JSON；清单内 334/13 个 train/validation ROI 也未出现在任何所查对应 JSON。五份 JSON 没有显式角色清单外 ID，因此当前可执行队列限定为显式清单与对应任务 JSON 交集，18,679/984 个 ROI。逐项缺失和额外 34 行的产生原因不推断；复算详见[交叉筛查报告](../../experiments/data_audits/allclear_train_val_eligibility_crosscheck_20261128_v2/筛查报告.md)。

此前 train/validation 中心距离、时间配对观察及限制仍见[原始筛查结果](../../experiments/data_audits/allclear_train_val_metadata_20261128_v1/筛查结果.json)：中心距离不证明 footprint 无重叠，时间接近也不证明像素配准或地表稳定。AllClear 仍是以 Sentinel-2 为主输出的 cloud removal，S1 为辅助模态；数据没有 clean SAR ground truth，且 S1/S2 表示不同。官方许可为 CC BY-NC 4.0，完整载荷规模约 435 万张；默认下载器会拉取所有角色影像。本轮不运行下载器、不下载影像，也不把该数据作为 SAR restoration 训练源。

### GUSO：地理覆盖大，但当前不适合作为直接路线

作者仓库报告 593,335 对 0.16–0.98 m SAR–Optical 图像，覆盖 319 城市、78 国；这在地理跨度上很有潜力。官方 README 列出的 train/validation/test 压缩包合计约 413 GB，下载链接通过 request form 提供；仅供学术用途，README 还要求对数据的二次开发或修改事先取得书面许可。受存储余量、授权和体量限制，本轮不请求、不下载、不读取它。

### OSDataset2.0、OSdataset、QXS-SAROPT 与 DFC2025

- [OSDataset2.0 官方论文/数据页](https://radars.ac.cn/en/article/2026/3)报告 6,476 对 512×512 配准块、覆盖 14 国，另有一对全幅场景；较早的官方数据页则报告 6,343 对，且列出 8.86 GB 下载包。该版本数量差异尚未核清。它面向 Optical–SAR 配准，patch-level 国家数不足以单独满足当前 20 个开发来源组的目标，而且没有去斑/超分 clean SAR 目标。
- [OSdataset 作者仓库](https://github.com/xym2009/OSdataset)报告 20 对 SAR/Optical 场景、增强显示域 uint8 SAR patch，并注明 patch 来源场景存在 50% 初始重叠、许可为 CC BY-NC。它适合配准检查参考，不适合作为独立 restoration benchmark。
- [QXS-SAROPT 作者仓库](https://github.com/yaoxu008/QXS-SAROPT)报告 20,000 个 GF-3/Google Earth patch pair，但来源集中在圣迭戈、上海、青岛，下载需填写表单，仓库未提供完整数据许可证。数量不等于 20,000 个独立来源组，不能据 patch 随机拆分。
- [2025 IEEE GRSS Data Fusion Contest 官方页](https://www.grss-ieee.org/community/technical-committees/2025-ieee-grss-data-fusion-contest/)的 Track 1 有约 4,300 对 Umbra SAR/航空 Optical、35 个区域、0.15–0.5 m；验证/测试部分公开的只有 SAR 图像。它有跨区域配对潜力，但 35 区域不足以按每个角色 20 组，原任务是 land-cover pseudo-label，不是恢复。

### OpenEarthMap-SAR 与 SAR2SAR

[OpenEarthMap-SAR 官方 Zenodo](https://zenodo.org/records/14622048)包含多区域 SAR 图像，但没有对应 clean restoration target；可作为 SAR-only 结构域补充，不是 optical 配对源。[SAR2SAR 原论文](https://arxiv.org/abs/2006.15037)是多时相 SAR 去斑方法先例，不是数据下载源；论文强调配准与时间变化补偿，提醒不能把 Umbra 重访影像直接平均成无噪 ground truth。

## 下一步有界计划

1. **筛查 SAR-NDIP 资格：** 先核作者包的大小、逐样本来源/场景映射、各上游数据许可、固定划分和本地交叉；未得到这些证据不下载或训练。该资源只可能支持 SAR-only 预训练，不能提供 Optical/SR 证据。若路径受限，继续审其它不同的一手 SAR 来源；不重复查询已用于 Task B 的 Mendeley 文件接口。
2. **建立独立来源组：** 只使用可授权的 train/development 元数据，与本地既有 train/development 建立 identity/footprint crosswalk；按场景、轨道、位置及采集时间分组，并确认缓冲距离和样本量。
3. **判定任务资格：** 检查目标 acquisition 是否与输入重叠、配准残差、时间稳定、多时相残余散斑与表示域。没有高低分辨率目标对时不称为 SR benchmark，多时相平均不称 clean truth。
4. **继续其它路线：** Umbra 的 11 对 SAR 重访须核验真实像素配准、地表稳定和产品域；GUSO 受下载体量及书面许可限制。某路线失败后记录缺口并继续筛选，不因此停止 E5。
5. **模型范围：** 本轮两项 Phase E 生成候选已用完，不在既有 development cohort 上训练第三项或事后调参。新训练需不同来源、新协议与明确授权。
6. C2 v2 继续 `UNJUDGED`、C3-B 关闭；confirmation/test/Optical 载荷保持未访问。
