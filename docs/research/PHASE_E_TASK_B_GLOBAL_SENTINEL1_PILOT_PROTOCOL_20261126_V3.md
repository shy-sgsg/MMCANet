# Phase E Task B 全球 Sentinel-1 表示域合成 Pilot 协议（2026-11-26 v3）

## 状态与修订范围

**状态：冻结；可准备 train/development 固定输入。** 本修订基于已授权 train-only 几何预检，在任何 development 载荷提取、输入生成或模型训练前冻结。此前 v1 因 SAR/GT25 归档无直接配对键被否决；v2 因 train 参考实际尺寸不符而停止。v2 的 train-only 审计读取了 327 张 train GT25 栅格以核对尺寸、dtype 和范围；未生成任何输入或模型结果。development、confirmation、历史 test 均未读取。

v3 仅将参考栅格固定为中心 512×512 窗口；不改变来源、样本成员、空间 split、噪声、退化、方法或评价规则。该任务仍是表示域合成代理，不是实测 SAR 输入或物理仿真。

## 数据来源与真实配对状态

- 来源：Mendeley Data 官方记录 [Labeled Dataset of SAR Sentinel-1 Imagery Despeckled with Multitemporal Fusion (5–25 Looks, VV/VH Polarizations)](https://data.mendeley.com/datasets/6nzd6x25vw/1)，DOI `10.17632/6nzd6x25vw.1`，CC BY 4.0；主论文：[Data in Brief 文章](https://pmc.ncbi.nlm.nih.gov/articles/PMC13333279/)。
- 当前公开单景 SAR 与 GT25 包在同地点、同轨道、同极化键上没有真实成对样本，故单景 SAR 包不进入训练。不得按最近邻、文件顺序或猜测对应关系配对。
- 仅使用 fusion 包中 GT25、VV 文件与官方 CSV 中同 ID/同轨道的元数据精确交集。467 个文件可匹配元数据；另有 34 个 GT25 VV 文件缺元数据而排除。
- GT25 是 25 acquisition temporal-average 高视数伪参考，不是真正 clean truth；时间变化、配准误差及融合平滑会限制参考质量。官方文章将这些参考描述为多时相 despeckled ground truth，但本项目结论保留“伪参考”表述。

## 栅格尺寸实测与冻结裁取

- 官方文章描述样本为 512×512，但本地 train-only 载荷实测为单通道 `uint8`、值域 `[0,255]`；原始高度 518–519，宽度 515–1465。该发布实现与论文所述固定尺寸不完全一致。
- 对每张允许角色的参考栅格，先检查为单通道、有限数值、值域 `[0,255]` 且两个维度都不小于 512，再按固定公式中心裁取：`row_start=(H−512)//2`，`col_start=(W−512)//2`，窗口为 `[row_start:row_start+512, col_start:col_start+512]`。不 resize、不逐图拉伸、不按指标选裁剪位置。记录原始 shape、dtype、min/max 与四个 crop offset。
- 若任一 development 影像不满足此合同，停止 v3 数据准备并报告；不得把 confirmation 用于补样本或调裁切。
- 8 km 分组按官方坐标中心、512 像素 crop 及论文所述约 10 m 像素间距预防邻近 crop 跨角色；GeoTIFF 地理变换未单独作为本轮模型输入，因此不作精确地面覆盖/配准主张。

## 固定 split

- 候选为 GT25 VV 与官方元数据精确匹配的 467 张参考。按大圆距离 ≤8,000 m 合并空间连通组；地点 ID 相同必同组。随机种子 `20261126`，train/development/confirmation 目标比例 70%/15%/15%，以轨道分层计数作贪心平衡并整组分配。
- 当前 metadata-only 结果为 467 个空间组：train 327 张/327 组，development 70 张/70 组，confirmation 70 张/70 组。最终输入构建时重算并保存 split manifest/hash；数量改变即停止，不改成逐图随机划分。
- confirmation 仅保留 membership 元数据；影像、派生输入、像素统计与指标均不可访问。无独立 test 角色。

## 表示域输入生成

1. 中心 crop 后按固定 `x = uint8_value / 255` 归一化，不作额外统计变换。保存目标为该 GT25 512×512 crop。
2. 每个 train/development 样本用稳定 sample ID 派生子 seed，预生成一张 `Gamma(k=1, θ=1)` 独立乘性因子 `n`，令 `x_noisy=x·n`；其理论均值 1、方差 1。这个噪声只定义在缩放栅格表示域，称为 controlled multiplicative speckle-like corruption，不称真实强度域 Gamma SAR speckle。
3. 低分辨率输入固定为 `y=AvgPool2d(kernel=2,stride=2)(x_noisy)`，大小 256×256。没有真实输入配对，不另加模糊、退化抽样或 synthetic noise 后处理。
4. `n` 与 `y` 对 train/development 预先固定并写入 manifest；同一 `y` 给所有方法。保存源 TIFF 与固定输入/目标 SHA-256、seed、crop window、shape、dtype、范围和生成配置。confirmation 不生成 `n` 或 `y`。

## Baseline 与候选方法

- Bicubic ×2：固定 LR 输入插值基线。
- EDSR 任务适配：vendored BasicSR EDSR，单通道、`num_feat=64`、`num_block=16`、`res_scale=1`、PixelShuffle ×2、随机初始化、Sigmoid 输出 `[0,1]`。
- SwinIR 任务适配：vendored SwinIR，单通道、`embed_dim=60`、`depths=(6,)`、`num_heads=(6,)`、`window_size=8`、`mlp_ratio=2`、PixelShuffle ×2、随机初始化、Sigmoid 输出 `[0,1]`。
- 唯一生成候选：冻结任务训练 SwinIR `x_b`；校正器 `μ(y,x_b)` 使用 bilinear ×2 LR 与 `x_b` 两通道输入，经 2→32 卷积、4 个双卷积残差块、零初始化 32→1 输出卷积，产生 `x_m=clip(x_b+μ,0,1)`；对中心残差 `r=x−x_m` 训练三层条件 epsilon U-Net，输入 noisy residual、`x_m`、LR 条件。U-Net 通道 32/64/128，每层两个残差块，128 维正弦 timestep embedding；1000 步线性 `β=1e-4…0.02`，固定 20 步 `η=0` DDIM，EMA 0.999。校正器 15 epoch、扩散 15 epoch，合计不超过 30 epoch。
- 生成结果分别保留一个预定 seed 的单次 draw、4 个固定 seed draw 的像素均值、确定性 `x_m` 消融及各 draw 样本。像素均值不代表单次生成质量；sample spread 不称校准不确定性。方法差异是设计事实，不预设学术新颖性。

## 训练和资源规则

- 单 seed=`20261126`。EDSR、SwinIR、生成候选分别最多 30 epoch、各不超过 4 小时，本轮 GPU 训练总预算不超过 12 小时；Bicubic 不训练。
- LR patch 64×64，对应 HR patch 128×128；每张 train 图每 epoch 一次随机裁剪，固定 seed 控制水平/垂直翻转与转置。batch=1；AdamW，`lr=1e-4`、`weight_decay=0`、无 scheduler；确定性重建与均值校正 L1，扩散 epsilon MSE。
- 仅用 development 来源组等权 PSNR 选 best checkpoint；所有方法接收同一固定输入/目标/样本集合。保存 last/best、训练日志、逐图指标、checkpoint、随机状态、计算资源和失败记录。
- 训练前记录 GPU、显存、环境、代码状态与外置卷余量。发生真实 CUDA OOM 时保留失败日志/last checkpoint，使用新 run 目录按预定 batch/crop 回退；不得覆盖目录或终止其它进程。

## 指标与决定

- 固定归一化表示域、全尺寸 development：PSNR（range=1）、11×11 Gaussian SSIM（range=1）、3×3 Sobel 梯度相关/强度比（忽略 1 像素边界）、每图 GT top 1% 的亮回波均值比/p95 比、每图 GT bottom 50% 的背景均值比、固定 `2×2` 面积平均重投影 MAE。
- 保存逐图 CSV；先按来源组聚合、组等权汇总。相对最强学习基线的配对来源组差用 10,000 次 bootstrap、seed=`20261127`，报告 95% CI；另报参数、推理时间、峰值显存、sample spread。
- 单 seed 只支撑探索性判断。继续门槛：候选相对最强学习基线 PSNR 至少 +0.05 dB 且来源组 bootstrap 95% CI 下界 >0；SSIM 与 Sobel 相关均值差不低于 0；亮回波/背景绝对均值比偏差和重投影 MAE 不高于基线。不满足则停止当前候选，不打开 confirmation。
- 最终表述限定为“GT25 多时相伪参考上的固定表示域合成任务”。不能外推为真实单景 SAR 输入、辐射物理一致、clean-SAR 恢复、跨来源泛化或 Optical 增益。
