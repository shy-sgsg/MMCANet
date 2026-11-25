# Phase D 新 Optical 配对数据获取与独立划分进展

**日期：2026-11-24**  
**状态：来源候选确认；官方载荷清单与新 split 尚未建立；未下载影像**

## 当前判断

官方 TUM 数据目录将 SEN1-2 列为 282,384 对已配准 SAR–optical patch；原始 ISPRS 论文也报告 282,384 对 Sentinel-1/Sentinel-2 patch，并将其定位为跨传感器配对研究数据。它是继续调查**未使用来源组**的首选 Optical 数据候选，而不是已获准训练的新数据。官方页面：[TUM Open Datasets](https://www.asg.ed.tum.de/sipeo/our-open-dataset/)，[SEN1-2 原始论文](https://isprs-annals.copernicus.org/articles/IV-1/141/2018/index.html)。

本轮尝试读取官方 TUM 持久化记录 `https://mediatum.ub.tum.de/1436631` 的文件清单时，站点返回 HTTP 429。当前尚未核验当前数据包的文件数、压缩体积、目录结构、哈希、下载许可文本和可直接用于模型的 SAR 数值域。因此，本轮没有下载、解压或查看任何新数据载荷，也没有把本地 `data/` 或 C1 v2 的未核实来源映射成官方 SEN1-2。

## 新 split 的建立方案

1. 从官方持久化记录取得数据文件清单、版本、许可与逐文件大小；先估算磁盘使用，再只下载冻结计划所需的完整 source groups，不整包复制到项目 `data/`。
2. 依据官方目录、元数据和论文定义解析 scene/ROI/source group、SAR 与 optical 配对身份、采集时间和空间覆盖；任何名称推断必须由文件内容或官方说明确认。
3. 只使用 split-integrity audit 允许的旧 membership 元数据排除 C1 v2 train/development/confirmation 已用组；不打开对应确认影像/指标。若来源身份或与旧组重叠无法判定，整组排除。
4. 在新 payload 进入任何模型前，校验 SAR–optical 一一配对、像素网格/配准误差、传感器/极化/分辨率/时间信息、值域和量化；新文件清单按 source group 划分，检查同场景相邻/重叠 patch 不跨集合。
5. 冻结新的开发与确认组清单、SHA-256、选择规则和指标。建议每个角色先以至少 20 个可核验 source groups、约 200 个配对 patch 为目标；若来源量达不到，先报告可用组数并重新评审，不以逐 patch 随机拆分补数。确认组只在新协议晋级后作一次性最终评价，之前不读取载荷或指标。
6. 新 Optical screen 仍须满足总纲 C2 资格，或先取得用户对独立探索范围的批准；matched、within-group permuted、missing、misaligned Optical 和 SAR-only 必须使用相同组、目标、训练预算与评价规则。本轮 C2 v2 `UNJUDGED`，所以不训练 Optical 模型、不访问旧 confirmation/test。

## 证据与后续解锁

| 项目 | 当前状态 |
|---|---|
| 官方来源与配对规模 | 官方 TUM 目录及原始论文确认 SEN1-2 是 282,384 对 co-registered SAR–optical patch。 |
| 当前官方下载包清单/体积/哈希 | 未确认；本轮官方持久化记录返回 HTTP 429。 |
| C1 v2 本地文件—官方 patch 身份映射 | 未核实；不根据目录名或数字文件名推断。 |
| 新 source-group train/development/confirmation manifest | 尚无；需先取得官方文件清单与可核验的 source/ROI metadata。 |
| 本轮图像下载/载荷读取 | 否。 |
| C2/C3-B 状态 | C2 v2 仍 `UNJUDGED`；C3-B 不启动。 |

**最小下一步：** 通过官方持久化记录或其受支持的下载方式读取 SEN1-2 包级元数据和文件列表；确认体积/许可后，对照旧 split membership metadata 构造来源组排除清单。此动作只需元数据权限，不需要 confirmation/test 影像。


## 2026-11-24 访问路径复核

TUM 官方开放数据页面继续将 SEN1-2 列为 SAR–optical 配对数据，并链接到其论文；ISPRS 数据公告报告数据集包含 282,384 对且通过 TUM MediaTUM 记录发布。直接打开记录在本次查询返回 HTTP 429。另做一次匿名 `rsync --list-only` 顶层清单探测，服务端返回认证失败；没有传输文件。

公开口令只在非官方页面找到，自动审查拒绝使用该来源的口令访问服务。本项目没有绕过审查、没有下载数据，也没有读取任何 protected payload。后续最小解锁是用户提供官方页面给出的凭据/镜像访问方式，或明确授权经审查后使用的具体公开认证信息。

即使获得目录清单，也须先只读取元数据并依据可核验的 C1 v2 历史来源映射排除已分配来源组，再固定新的 train/development cohort；不得根据文件名、季节名或目录顺序推断历史成员关系。只有确认每个文件属于获准 train/development 角色后，才可下载对应 SAR/Optical 数据；confirmation/test 保持封存。

## 2026-11-24 官方发布与入口复核补记

ISPRS 官方发布公告明确指出 SEN1-2 的 282,384 对配对影像按 CC-BY 发布，并指向 TUM MediaTUM 持久记录；TUM 官方开放数据页仍列出同一规模。通过工具再次尝试官方 MediaTUM 记录返回 HTTP 429，dataserv 目录入口无法由当前浏览通道打开；没有向匿名 rsync 提供凭据，也没有传输文件。

- [TUM 官方开放数据页](https://www.asg.ed.tum.de/sipeo/our-open-dataset/)
- [ISPRS 官方发布公告](https://www.isprs.org/news/announcements/details.aspx?ID=175)
- [SEN1-2 原始论文](https://isprs-annals.copernicus.org/articles/IV-1/141/2018/index.html)

许可与发布规模已有官方证据，但当前包级文件清单、目录成员、体积、哈希和旧 split 来源映射仍未获取；因此不能建立新 split 或下载挑选影像。
