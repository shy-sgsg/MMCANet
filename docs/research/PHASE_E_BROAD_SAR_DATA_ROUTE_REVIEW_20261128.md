# 更广 SAR 与 Optical 数据路线复核（2026-11-28）

## 结论

目前没有一个已核验的数据源可以直接代替正式的 SAR 联合去斑/超分训练集。AllClear 的 train/validation ROI 地理中心筛查显示组间距离较大，但官方清单与共享角色表的 train 计数相差 34 个 ROI，尚未解决版本一致性；它仍是有价值的广域来源候选，不能直接立为正式训练集。GUSO 覆盖广但下载与授权门槛很高。Umbra Open Data 提供 SAR 时序线索但不含 Optical 配对；SARptical 仍只覆盖柏林城区。

本轮仅下载并筛选 AllClear 官方元数据归档，未下载任何 SAR 或 Optical 影像。未打开 test 专属元数据文件；共享 ROI 表中的 test 行只按角色过滤，没有读取或保存其坐标。GUSO、OSDataset 和其它候选均未下载影像。本轮没有访问任何新来源的 Optical、confirmation 或 test 影像载荷。

## 已完成来源和模型实验

### SARptical

用户批准的有限 pilot 已在 2026-11-26 完成：Berlin 单城区、12/8/8 个 250 m 空间组、916/306/306 张 train/development/封存 confirmation；confirmation 图像未读。当前 Flow Bridge 相对任务适配 SwinIR 的组等权 ΔPSNR 为 −2.6412 dB，高回波 ROI 门槛 0/8 组有效，按冻结规则停止。详见[SARptical 结果](PHASE_D_SARPTICAL_SPATIAL_PILOT_RESULTS_20261126.md)。

### Mendeley Data 全球 Sentinel-1 Task B 代理

Phase E 使用官方 Mendeley Data 的 GT25 VV 多时相高视数伪参考，构建固定表示域 ×2 合成退化；它不是已配对的真实单景 SAR。两项冻结生成候选都在 70 个 development 来源组上未通过 gate，详见[Phase E Research Decision Review](PHASE_E_TASKB_GLOBAL_PILOT_RESULTS_20261128.md)。不以该负结果否定其它数据或所有生成方法。

## Umbra Open Data：SAR 时序候选，尚非训练数据

本轮从 [Umbra Open Data 官方页面](https://umbra.space/open-data/)、[AWS Open Data Registry](https://registry.opendata.aws/umbra-open-data/)列出的公开 S3 前缀只读取 XML 对象清单和 STAC JSON 元数据；没有读取资产链接指向的 GeoTIFF、NITF 或 CPHD。

- `open-data/` 清单未截断，共 537 个对象，其中 134 份 STAC sidecar；对象列出的合计大小约 1.348 TB（没有下载这些资产）。134 份 sidecar 合计约 0.99 MB，压缩归档 SHA-256 为 `fe97d1aa8e37b67622d3ad855c3343049caeeb74ef6eae56238deb0700780860`。
- 按 item ID 去重后有 101 个唯一条目，采集日期为 2025-12-17 至 2026-01-18，共 26 天；涉及 Umbra-07/09/10、VV/HH、1/2/4 个 azimuth looks。sidecar 在公开前缀上的最后修改时间延至 2026-02-05；下载时未看到更晚的 acquisition date。Umbra 静态目录的 `Last-Modified` 为 2025-08-13，需与公开 S3 前缀分别看待。
- 按 bbox 经纬度平面相交占较小 bbox 面积至少 50% 建连通分量，得到 42 个**粗略候选位置组件**。此算法不是米制空间核验，也不代表真实 scene/source ID。
- 在 10 个组件内发现 11 对同平台、极化与 looks 匹配的重复采集；间隔 2.50–14.52 天，入射角差最大 11.12°。尚未核验绝对定标、共用像素网格、配准、地表变化或散斑独立性。该数量不足以直接形成规模充足的跨来源 train/development/confirmation 方案。

官方 Umbra 文档说明 Archive Catalog STAC API 需要认证，而开源 S3 前缀是另一个入口；本轮仅用公开前缀。元数据、对象清单、item 哈希、分组算法与详细限制见[Umbra 元数据筛查清单](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/元数据筛查清单.json)、[压缩 STAC 归档](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/umbra_stac_items_20261128.jsonl.gz)及[复算脚本](../../experiments/data_audits/umbra_open_data_metadata_screen_20261128_v1/audit_umbra_metadata.py)。

## 更广配对源候选

### AllClear：完成 train/validation 元数据筛查，仍待任务资格核验

[AllClear 官方网页](https://allclear.cs.cornell.edu/)报告 23,742 个全球 ROI；[作者 NeurIPS 论文](https://papers.nips.cc/paper_files/paper/2024/file/60095e1d7ebb292dbba93c4d8f7b2463-Paper-Datasets_and_Benchmarks_Track.pdf)的最终统计为 23,708 个，train/validation/benchmark 为 19,013/997/3,698 个 ROI。[作者仓库](https://github.com/Zhou-Hangyu/allclear)列出 `train_rois_19k.txt`、`val_rois_1k.txt` 和 `test_rois_3k.txt`。2026-11-28 通过官方 `metadata.tar.gz` 仅筛查 train/validation：两个显式清单分别有 19,013/997 个 ROI，ID 无交集，全部在共享 `rois_metadata.csv` 中找到对应地理记录；validation ROI 到最近 train ROI 中心的距离最小 11.120 km、中位数 15.729 km，997 个中心均距 train 中心超过 5 km。该结果仅是经纬度中心距离筛查，不证明影像 footprint 不重叠。

同归档的 `rois_metadata.csv` 标记 train 的 ROI 有 19,047 行，比显式清单多 34 行；validation 两边都是 997 行。进一步流式读取 train/validation 的 `tx12_s2-s1_100pct.json` 后，数据 JSON 没有出现显式清单外 ROI，但只有 18,679 个 train ROI 与 984 个 validation ROI 实际包含 12 帧序列；显式清单中另有 334/13 个 ROI 没有 tx12 样本。共享表多出的 34 个 train 行未进入该 JSON 样本 cohort；其来源/版本仍待核，但显式清单与实际数据 JSON 的角色关系清楚，可仅以二者定义有限候选集合。

这些 train/validation JSON 共含 109,840/5,633 个序列实例，每个实例有 12 个 S2 输入时间点和一个 S2 target 时间点。对每个 S2 输入时间点查找最近 S1 时间戳，训练/验证分别有 45.4%/44.7% 在 48 小时内、61.1%/60.4% 在 72 小时内；对 target 时间点，训练/验证分别为 48.8%/48.3% 在 48 小时内、65.3%/65.4% 在 72 小时内。作者[数据加载器](https://github.com/Zhou-Hangyu/allclear/blob/main/allclear/dataset.py)参数写 `max_diff=2`，但实现通过 `timedelta.days > 2` 判拒，实际容许范围接近 72 小时；本审计报告最近日期差，不模拟加载器的多对一分配。该时间配对会限制按严格同日标准形成 SAR/Optical 配对，时间接近也不证明像素配准或地表稳定。

官方元数据归档 SHA-256 为 `7e941a8670b5cb9d7ac8d395521e1c7160838e5705b9751805b126ebcd701a63`；复算脚本与聚合记录见[AllClear train/validation 元数据筛查](../../experiments/data_audits/allclear_train_val_metadata_20261128_v1/筛查结果.json)和[审计脚本](../../experiments/data_audits/allclear_train_val_metadata_20261128_v1/audit_allclear_train_val_metadata.py)。

原任务是以 Sentinel-2 为主输出的 cloud removal，S1 是辅助模态；反向用于 SAR 恢复需另设任务。S1 是裁剪/归一化 GRD 表示、S2 为 L1C TOA，且数据没有 clean SAR ground truth。许可为 CC BY-NC 4.0，完整数据约 435 万张影像。官方 `download.py` 会先取元数据，再读取 train/validation/test ROI 清单并并行下载所有 ROI 影像；本轮没有运行该脚本。下一步应核对 34 个额外 train 角色行的版本/用途，再审计本地旧数据身份交叉、SAR 产品与时序目标、时间配对、footprint 和像素配准；之后才决定是否按显式清单与对应 JSON 冻结新的有限协议。该元数据审计不读取影像、不产生训练 split，也不解锁 C3-B。

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

1. **已完成：** 只下载官方元数据归档并解析 train/validation 清单、ROI 地理记录和 `tx12_s2-s1_100pct.json`。结果见上文；test 专属文件和全部影像载荷未访问。临时归档在分析后清理，保留归档 SHA-256 与不含坐标明细的聚合结果。
2. **待核：** 共享 ROI 表比显式 train 清单多 34 个角色行；这些 ID 没有出现在 `tx12` train JSON 中，但其角色来源/版本仍需核清。显式 train/validation 清单及对应 JSON 均没有跨角色 ROI ID。
3. **再评估数据资格：** 核查与本地已用数据的身份交叉、SAR 产品/时序目标、严格时间配对样本量、影像 footprint 与像素配准及最小子集存储；中心距离不能代替 footprint 审计，且没有 clean SAR target。
4. Umbra 另作为 SAR-only 多时相候选，先核验 11 对候选的真实配准与域合同；不与 Optical 效用证据混淆。C3-B 不启动。
5. 两项 Phase E Task B 生成候选已用完，不在相同 development cohort 上训练或调参；新模型实验需建立新数据协议和独立来源组。
