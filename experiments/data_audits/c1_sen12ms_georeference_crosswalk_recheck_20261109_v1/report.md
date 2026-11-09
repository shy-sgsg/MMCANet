# C1 v2 train/development 与 SEN12MS 地理元数据交叉路线复核

**复核日期：** 2026-11-09  
**性质：** 公开来源与本地 train/development 路径元数据审查；不读取影像载荷。

## 核查问题

检查 SEN12MS 官方 georeferenced GeoTIFF 是否能提供 C1 v2 train/development 来源场景的坐标或 patch footprint 交叉表，以补足现有按来源场景文件夹隔离证据。

## 新发现的来源事实

- SEN12MS 作者论文称其数据由 180,662 个 Sentinel-1、Sentinel-2 与 MODIS patch 组成，GeoTIFF 样本具有地理参考，GSD 统一为 10 m。该数字和多模态数据结构描述的是 SEN12MS 发布版，不等同于 SEN1-2 原始 PNG 清单。
- 论文将四个 ROI 随机种子与季节根目录关联，并描述单个 ROI/scene 的子目录及包含种子、季节、数据类型和 patch 编号的命名结构。C1 v2 train/development 的官方来源路径也出现相似目录词法。
- 作者维护仓库 README 说明其提供 `train_list`、`test_list` 和绘制 ROI 世界分布的 notebook。README 还说明这些列表属于其场景分类示例；它们不等同于 C1 v2 的冻结 train/development 划分。

## 本次访问和证据边界

- 只查看了 SEN12MS 作者论文、作者维护仓库 README 及公开 `train_list.txt` 页面元数据；该 4.91 MB 文件的内容未由网页显示。对若干已暂存 C1 train/development 文件名的精确公开检索未取得可核验的文件级交叉结果；这不是“无匹配”的证明。
- 本地只核对了 train/development 暂存 manifest 中的角色数和来源路径命名；没有读取图像像素或 GeoTIFF 内容。
- 没有打开 SEN12MS `test_list`、hold-out 列表内容或 overview notebook 输出；没有查看 confirmation/test 载荷、测量值、模型或 checkpoint。未训练、推理或使用 GPU。

## 结论与后续路线

目前只确认存在一个可能的官方元数据路线：对**仅 train/development** 的 C1 来源文件名，与 SEN12MS 的 georeferenced GeoTIFF 身份作逐文件核对。相似的 ROI/season/patch 命名本身不能证明两版数据中的同名文件对应同一切片；在得到精确、可追溯的文件级身份连接前，不能把 SEN12MS 地理参考传递给 C1 来源组，也不能宣称 C1 v2 有地理、采集或 footprint 独立性。

故 C1 的当前结论不变：冻结角色与来源场景文件夹互斥已核验；坐标、采集日期、轨道和空间 footprint 独立性未知。本报告不重试曾受限的 TUM 条目，不读取新影像，也不改变 C1 划分、C2/C3 gate、训练状态或数据权限。

## 来源

- Schmitt 等，SEN12MS 作者论文：[arXiv:1906.07789](https://arxiv.org/abs/1906.07789)。
- SEN12MS 作者维护仓库：[README 与工具说明](https://github.com/schmitt-muc/SEN12MS)；[公开训练清单页面](https://github.com/schmitt-muc/SEN12MS/blob/master/splits/train_list.txt)。
- SEN1-2 原始论文：[ISPRS Annals 页面](https://isprs-annals.copernicus.org/articles/IV-1/141/2018/)。
