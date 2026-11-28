# Phase E Task B 全球 Sentinel-1 表示域合成 Pilot 协议（2026-11-26 v2）

## 状态与问题

**状态：未执行；train-only 几何预检否决，已由 v3 替代。** v2 解包 train 后发现实际 GT25 栅格并非固定 512×512；准备脚本在生成任何固定输入前停止。随后仅检查 train 角色 327 张参考的几何、dtype 与数值范围。没有提取或读取 development/confirmation/test 影像，没有训练或模型选择。v3 冻结了基于实测尺寸的中心裁取规则。

公开归档审计发现，SAR 单景与 GT25 归档文件不满足论文描述的直接配对键，故本实验不使用 SAR 单景包。改用官方 GT25 参考的 train/development 子集，构造预先固定的**表示域合成任务**：从多时相高视数参考 `x_ref` 生成随机 speckle-like 噪声，再做 ×2 面积平均，预测原分辨率 `x_ref`。这是一项受控的去噪与超分辨率代理实验，不是实测低分辨率 SAR，不是真实 SAR 物理仿真，也不证明 clean-SAR 恢复。

## 来源和归档配对否决证据

- 数据来源为 Mendeley Data 官方记录 [Labeled Dataset of SAR Sentinel-1 Imagery Despeckled with Multitemporal Fusion (5–25 Looks, VV/VH Polarizations)](https://data.mendeley.com/datasets/6nzd6x25vw/1)，DOI `10.17632/6nzd6x25vw.1`，CC BY 4.0；主论文为 [Data in Brief 文章](https://pmc.ncbi.nlm.nih.gov/articles/PMC13333279/)。
- 归档成员清单（不读像素）显示：`dataset.rar` 有 87 个 `_0.tif` 地点 ID，82 个能与官方 CSV 的地点 ID 对上；`fusion.rar` 有 632 个 GT25 地点 ID，594 个能与 CSV 对上。两包地点 ID 交集只有 1 个，且没有相同地点/轨道/极化配对键。VV SAR 与同轨 VV GT25 的元数据最近邻最小距离约 17.7 km。
- 论文描述单景与多时相参考使用相同地点标识并可直接配对，但当前公开归档内容不符合该描述。v1 直接配对路线在零像素读取阶段否决；不得用空间最近邻、文件顺序或人工挑选样本补配对。
- 本 v2 仅读取 fusion 包内 GT25、VV、且地点 ID 与 CSV 轨道/坐标精确匹配的文件。metadata 清单审计共有 467 张候选参考，其中 34 个 GT25 VV 文件无 CSV 元数据而排除。这里只使用官方声称的 25 acquisition temporal-average reference；它是高视数伪参考，不是真正无噪声地面真值。时间变化、融合平滑及单景组成未知均是局限。
- 论文说明 GeoTIFF 数据被缩放到 `[0,255]`。本协议要求实际 train/development 文件逐一核验为单通道、有限值且在该范围；任何文件违反则停止并报告，不能改用逐图拉伸。

## 固定清单与空间角色

- 仅对 `fusion/SENTINEL1/{ASCENDING|DESCENDING}/VV/GT25/{id}.tif` 建记录；样本需与 `datos.csv` 中同一五字符 ID 和同一轨道精确匹配。每个样本有固定 `sample_id`、坐标、归档成员路径和来源组。
- 按球面大圆距离 8,000 m 形成连通空间组；同一地点 ID 必在同组。使用随机种子 `20261126`，按 70%/15%/15% 目标比例、轨道分层计数贪心平衡并整组分配。基于当前 metadata-only 清单，467 张均是各自独立的 8 km 空间组，预期为 train 327、development 70、confirmation 70；生成最终清单时重新核实上述数值并保存哈希。
- 划分完成后验证来源组互斥、样本覆盖完整、无重复。confirmation 只保留 ID/组/轨道/归档路径等 membership 元数据，不读取文件内容。
- 影像载荷访问标志必须为 `confirmation_payload_accessed=false`、`test_payload_accessed=false`、`optical_payload_loaded=false`。训练和数据构建入口只接收 train/development 清单，不可传入 confirmation 路径。

## 表示域合成退化

1. 对允许角色参考影像 `x_ref`，逐像素固定缩放 `x=x_ref/255`，不做逐图 min-max、百分位拉伸或隐式自动归一化。保留原始 dtype、范围、像素类型统计及文件哈希。
2. 以 `Gamma(k=1, θ=1)` 独立随机因子 `n` 乘上 `[0,1]` 参考得到 `x_speckled=x·n`；该变量均值为 1、方差为 1。这仅是对缩放栅格的 controlled multiplicative speckle-like corruption，**不称为真实强度域 Gamma SAR speckle**。
3. 固定 `2×2`、步长 2 面积平均 `D`，得到低分辨率模型输入 `y=D(x_speckled)`；预测目标为原分辨率 `x`。不另加模糊或其它噪声。
4. 对 train/development 每个样本各预生成一份 `n` 与 `y`，使用 `seed=20261126` 与稳定样本 ID 派生子 seed；同一固定 `y` 提供给所有方法。写 `input_manifest.json` 记录输入/目标哈希、shape、dtype、seed、噪声统计与转换配置。不得生成 confirmation 噪声或输入。
5. 这个合成观测在表示域定义，不对应真实传感器的点扩散函数、像素面积采样或 radiometric power。结果只能说明方法在此受控代理中的表现。

## 基线与唯一生成候选

### Baseline

- Bicubic ×2：输入 `y` 的确定性插值基线；输出统一裁剪到 `[0,1]`。
- EDSR：使用仓库 vendored BasicSR EDSR，`num_feat=64`、`num_block=16`、`res_scale=1`、PixelShuffle ×2，单通道随机初始化、Sigmoid 输出。命名为 EDSR 任务适配，不称 SAR 专用论文复现。
- SwinIR：使用仓库 vendored SwinIR，`embed_dim=60`、`depths=(6,)`、`num_heads=(6,)`、`window_size=8`、`mlp_ratio=2`、PixelShuffle ×2，单通道随机初始化、Sigmoid 输出。命名为 SwinIR 任务适配。
- 两个学习基线使用同一 train/development manifest、固定输入、目标、patch、增强、seed 和 30 epoch 上限；允许架构相关优化差异须预先写入 config。checkpoint 按 development 来源组等权 PSNR 选择。

### 候选：均值校正 + 中心残差条件扩散

1. 复用同一训练得到的 SwinIR checkpoint 作为冻结 `x_b`。
2. 确定性校正器 `μ(y,x_b)` 输入 bilinear ×2 的 `y` 与 `x_b`（2 通道），使用 2→32 的 3×3 卷积、4 个 32 通道双卷积残差块、32→1 零初始化输出卷积，预测 `x−x_b` 修正；`x_m=clip(x_b+μ,0,1)`。候选第 1–15 轮训练校正器。
3. 扩散阶段以 `r=x−x_m` 为目标，条件为 `y` 与 `x_m`。epsilon 网络为三层 U-Net、通道 32/64/128、每层两个残差块，stride-2 卷积下采样、转置卷积上采样并连接同尺度 skip；128 维正弦 timestep embedding 经两层 MLP 注入残差块。前向扩散 1,000 步线性 `β=1e-4…0.02`；DDIM 固定 20 步、`η=0`；EMA 衰减 0.999。候选第 16–30 轮训练扩散。
4. 推理统一裁剪预测到 `[0,1]`；分别报告预定 seed 的单次 draw、4 个固定 seed draw 的像素均值和 sample spread。像素均值不代表单次生成质量；spread 不称校准不确定性。
5. 与旧完整像素域 Flow、conditional DDPM、旧 residual DDPM 和 bicubic Flow Bridge 的区别是先独立学习确定性均值修正，再只对中心残差建模。此为设计差异，不预设新颖性。

## 训练、预算和停止规则

- 所有学习模型单 seed=`20261126`，最多 30 epoch、各不超过 4 h；EDSR、SwinIR 与候选合计 GPU 时间不超过 12 h。Bicubic 不训练。
- LR patch 为 64×64，HR patch 为 128×128；每个 train 样本每 epoch 一次随机裁剪，采用水平翻转、垂直翻转及转置。默认 batch size=1。AdamW，`lr=1e-4`、`weight_decay=0`、不使用 scheduler；确定性模型和均值校正用 L1，扩散 epsilon 用 MSE。随机增强由固定 seed 控制。
- 运行前记录 GPU 使用/可用显存、分区余量、Python/PyTorch/CUDA/cuDNN、代码 commit/dirty 状态、protocol/split/input manifest、配置和随机状态。外置实验卷保存缓存、checkpoint、预测及日志。发生真实 CUDA OOM 时保留日志/last checkpoint，并在新 run 目录按事先规则降 batch 或改 48×48 LR crop；不得覆盖失败运行或终止其它进程。
- 保存 `last`、development 来源组等权 PSNR 的 `best`、逐 epoch loss/开发指标、逐图指标、来源组统计、单次与均值预测、资源数据及 Research Decision Review。NaN/Inf、缺文件、哈希不符或结构不符均失败关闭。
- 若开发有效来源组少于 5、空间角色不互斥、文件不在 `[0,255]`、目标通道/尺寸错误、模型输出尺寸错误或资源预算不足，停止该 run 并保留证据；不读取 confirmation 解决问题。

## 评价和预先决定

- 统一在归一化表示域、同一 full-resolution development 样本集合计算：PSNR（range=1）、11×11 Gaussian SSIM（range=1）、3×3 Sobel 梯度幅值相关/强度比（忽略 1 像素边界）、每图 GT top 1% 的预测/GT 均值与 p95 比、每图 GT bottom 50% 的背景均值比、固定 `2×2` 面积平均重投影 MAE。显示增强不进入指标。
- 保存逐图 CSV；先在来源组内求均值，再对组等权汇总。相对最强学习基线的配对差使用 10,000 次来源组 bootstrap，seed=`20261127`，报告 95% 区间。另报参数量、推理秒数、峰值显存、抽样次数与 sample spread。
- 单 seed 仅作探索性决策。候选继续条件：development PSNR 较最强学习基线至少 `+0.05 dB` 且组配对 95% CI 下界 >0；SSIM 与 Sobel 相关差均值不低于 0；亮回波绝对均值比偏差、背景绝对均值比偏差和重投影 MAE 均不高于该基线。否则停止该候选配置，不用 confirmation 挽救结果。
- 无论结果如何，不读取 confirmation/test 影像，不生成其派生输入或指标；development 已用于 epoch 选择，只支撑有限探索结论。

## 来源限制与论文措辞

多时相平均依赖场景在采集期间稳定；地表变化与配准误差会模糊目标。当前单景与 GT25 包未形成可核验的真实成对样本。合成输入由 GT25 再退化而来，因此评价答案只对“25-acquisition temporal-average 表示域伪参考上的固定合成退化”成立；不能外推到原始 Sentinel-1 输入、真实低分辨率 SAR、clean SAR、其它来源或 optical-guided 重建。
