# Phase E Task B 全球 Sentinel-1 SAR 空间分组 Pilot 协议（2026-11-26 v1）

## 状态与问题

**状态：未执行；在像素访问前被配对审计否决。** 2026-11-26 对完整官方归档成员路径与 CSV 元数据核对后，SAR `_0.tif` 与 GT25 在相同地点、轨道、极化键上没有任何可配对样本。没有提取或解码任何影像。本文件保留最初计划；新候选见 [v2 合成代理协议](PHASE_E_TASK_B_GLOBAL_SENTINEL1_PILOT_PROTOCOL_20261126_V2.md)。

## v1 否决记录

- 官方公开归档大小与 SHA-256 已核验；SAR 包有 87 个单景地点标识，其中 82 个在 CSV 中；GT25 包有 632 个地点标识，其中 594 个在 CSV 中。两包地点标识仅 1 个相交，且该 ID 不构成相同轨道、极化的配对键；精确可配对键为 0。
- VV 输入地点与同轨 VV GT25 的元数据最近邻距离最小约 17.7 km；没有可据此建立同地点配对的候选。
- 官方数据论文描述文件使用相同地点标识并可直接配对，但当前公开归档与这一描述不一致。不得通过最近邻、逐图拉伸或猜测文件名制造真实输入/参考配对。
- 影像像素、confirmation 内容均未读取。v1 不产生训练或评价结果，也不改变任何历史阶段 gate。

主要问题：在有单景 SAR 输入和多时相高视数 SAR 伪参考的条件下，SAR-only 确定性强基线是否能完成受控的联合去斑与 ×2 超分辨率；一个由确定性残差均值校正引导的条件残差扩散候选，能否改善开发组上的结构恢复，同时保持亮回波和输入重投影表现。

本实验只支撑对该数据来源、该受控算子和该伪参考的探索性观察，不支撑 clean-SAR 恢复、真实成像物理一致性、跨来源泛化或光学增益主张。

## 数据来源与配对合同

- 来源：Mendeley Data 官方记录 [Labeled Dataset of SAR Sentinel-1 Imagery Despeckled with Multitemporal Fusion (5–25 Looks, VV/VH Polarizations)](https://data.mendeley.com/datasets/6nzd6x25vw/1)，DOI `10.17632/6nzd6x25vw.1`，记录标注 CC BY 4.0；配套数据论文：[Data in Brief article](https://www.sciencedirect.com/science/article/pii/S2352340926005494)。
- 配对候选只能由公开 `dataset.rar` 中 `SENTINEL1/{ASCENDING|DESCENDING}/VV/{location}_0.tif` 单景文件、`fusion.rar` 中相同地点/轨道/VV 的 `GT25` 参考文件及官方 `datos.csv` 元数据三者精确交集产生。仅核对归档成员路径不读取影像像素。缺任一项即排除并记数。
- Pilot 只用 VV；同一地点不论轨道或其它记录均归入同一来源组。输入是单景一视 SAR，目标是 25-look 多时相融合参考。该参考是 noisy/high-look pseudo-reference，不称 clean ground truth；单景是否被纳入 GT25 融合在现有来源描述中未确认，记录为目标依赖风险。
- 输入与 GT25 必须具有相同单通道栅格尺寸；不自动配准、裁剪、resize 或按图归一化。尺寸/波段不符即失败关闭。归档缺失项、坐标冲突及无法解码文件均保留审计记录。
- 文件角色、轨道、极化和来源组由 CSV 与归档成员路径建立。任何样本都不得因模型输出或开发指标被补入或移出数据清单。

## 表示域与固定退化

1. 仅对 train 角色 GT25 全像素计算一次全局 `p0.01` 与 `p99.99`，跨所有 train 影像合并统计；精确数值、输入/参考越界率写入输入 manifest。对所有 train/development 的输入和目标统一应用 `clip((v-p0.01)/(p99.99-p0.01), 0, 1)`。不做逐图拉伸；分位数相等或非有限时失败。
2. 先以同一固定变换归一化单景输入，再应用 `2×2` 非重叠面积平均、步长 2，得到 ×2 低分辨率模型输入。HR 目标是同位置 GT25。不得另加合成 speckle、模糊或可变退化。
3. 该算子定义的是归一化影像表示域受控退化；不声称是 SAR 仪器的真实空间响应或强度域 Gamma 噪声模型。保存单景与目标的配对 SHA-256、固定输入算子版本和 split manifest。

## 空间分组与数据角色

- 仅用官方经纬度元数据生成分组，不读取像素。相同五字符地点标识必在同组；任意两个元数据中心的大圆距离不超过 8,000 m 时合并连通分量。该半径覆盖 512×512、约 10 m 像素影像的对角线范围并留有定位余量；若实际地面采样距离证据不支持 10 m，则下调为保守的坐标分组描述并报告限制。
- 固定随机种子 `20261126`，按来源组分配 70%/15%/15% 的 train/development/confirmation 目标比例，以轨道分层计数作贪心平衡；整组不可拆分。保存每个样本的 role 与来源组，以及每角色图像数、组数和轨道计数。实际比例受组大小限制，不强行拆组。
- 划分生成后检查三个角色来源组两两不交、样本集合完全覆盖且没有重复。分组或比例不足以支撑三个非空角色时停止该数据集 pilot，不借用 confirmation。
- confirmation 的文件名和角色仅留在 metadata manifest；训练、预处理、解码、像素统计和评价脚本只接收 train/development 清单。运行 manifest 固定 `confirmation_payload_accessed=false`、`test_payload_accessed=false`、`optical_payload_loaded=false`。

## 方法与公平对照

### Baseline

- Bicubic ×2：低分辨率输入的确定性插值对照。
- EDSR：复用仓库 vendored BasicSR EDSR 主体，`num_feat=64`、`num_block=16`、`res_scale=1`、PixelShuffle ×2，适配单通道、随机初始化、Sigmoid 输出 `[0,1]`；报告为“EDSR 任务适配”，不称作者权重或完整论文复现。
- SwinIR：复用仓库 vendored SwinIR 主体，`embed_dim=60`、`depths=(6,)`、`num_heads=(6,)`、`window_size=8`、`mlp_ratio=2`、PixelShuffle ×2，适配单通道、随机初始化、Sigmoid 输出 `[0,1]`；报告为“SwinIR 任务适配”，不称 SAR 专用论文复现。
- 两个学习基线与主候选使用同一 train/development 文件清单、退化输入、目标、空间增强和 30 epoch 上限。所有选择仅基于 development 来源组等权 PSNR；保留每个 epoch 指标及 last/best checkpoint。架构必要的学习率等差异写进 config，不在 confirmation 上选择。

### 唯一生成式候选：冻结基线上的均值校正与中心残差扩散

1. 使用同一训练得到的 SwinIR checkpoint 作为冻结基线 `x_b`。
2. 训练轻量确定性校正器 `μ(y,x_b)` 预测 `GT25 - x_b` 的可预测均值修正：bilinear ×2 后的 LR 与 `x_b` 拼成 2 通道输入，经 2→32 的 3×3 卷积、4 个 32 通道双卷积残差块、32→1 的 3×3 零初始化输出卷积，得到 `x_m=clip(x_b+μ,0,1)`。候选前半预算用于该阶段。
3. 对中心残差 `r=GT25-x_m` 训练 SAR 条件 DDPM 噪声预测器。三层 U-Net 输入为 noisy residual、`x_m` 与 bilinear ×2 LR（3 通道）；每层两残差块、通道 32/64/128，以 stride-2 卷积下采样、转置卷积上采样并连接同尺度 skip；timestep 经 128 维正弦 embedding 和两层 MLP 注入残差块。候选后半预算用于扩散阶段。推理固定对 `x_m+r̂` 统一裁剪至 `[0,1]`，报告一组预定随机种子的单次 draw，并另报 4 次固定种子 draw 的像素均值；两者不混写。只报告样本 spread，不称校准不确定性。
4. 这与旧的完整像素域 Flow、旧 conditional DDPM、旧 residual DDPM 和旧 Bicubic Flow Bridge 配置不同；差别是明示的独立确定性均值校正阶段及其后的中心化条件残差分布。此区别是实验设计事实，不预设学术新颖性。
5. 候选拆为均值校正与扩散两个阶段，总计最多 30 个训练 epoch；冻结 SwinIR 基线不计入候选重复训练。候选单独计时不超过 4 h。
6. 扩散网络为 3 层 U-Net（基础通道 32、64、128，timestep embedding 128），预测加噪中心残差的 epsilon；前向 schedule 为 1,000 步线性 `β=1e-4…0.02`，采样用固定 20 步、`η=0` 的 DDIM。EMA 衰减 `0.999`。训练每个 crop 随机抽一个 timestep；随机状态随 seed 固定。不得根据开发结果改 schedule、网络宽度或采样步数。

## 训练、资源和停止条件

- 单 seed：`20261126`；EDSR、SwinIR、候选各最多 30 epoch、各不超过 4 h，本轮 GPU 新实验总计不超过 12 h。Bicubic 不训练。
- 固定 LR patch 为 64×64、对应 HR patch 为 128×128；水平翻转、垂直翻转、转置由固定 seed 控制。默认 batch size 1；只有记录到真实 CUDA OOM 后，才在新 run 目录按预案降为 48×48 patch 或降低 batch，保留旧失败日志/检查点。绝不终止其它进程。
- AdamW，初始学习率 `1e-4`、weight decay `0`，无学习率调度器；确定性网络用 L1，扩散 epsilon 用均方误差。均值校正阶段 15 epoch、扩散阶段 15 epoch；每轮完整遍历训练样本并每样本产生一次固定随机裁剪。首个 implementation smoke test 只检查张量、梯度和一次前向/反向，不训练或读取 development。
- 运行前记录 GPU 使用/显存、分区余量、Python/PyTorch/CUDA/cuDNN、代码 commit/dirty 状态、split/input manifest、配置和随机性。只在外置实验卷保存缓存、checkpoint、预测与日志。
- 保留 `last` 和按 development 来源组等权 PSNR 选出的 `best` checkpoint；NaN/Inf、样本缺失、配对不一致、哈希不匹配或真实 OOM 均留下失败证据，禁止静默跳过或覆盖。
- 实际可评价的开发来源组少于 5 组、任务输入/目标尺寸不匹配、训练/开发无有效配对、空间角色组不足或磁盘预算不够时，停止受影响 run 并继续其它可独立工作包。

## 指标与预先决策规则

- 全部指标只在固定归一化表示域、同一全尺寸 development 清单计算：PSNR（data range=1）、11×11 Gaussian-window SSIM（data range=1）、3×3 Sobel 梯度幅值相关/强度比（去除 1 像素边界）、每张 GT top 1% 像素的预测/GT 均值比与 p95 比、每张 GT bottom 50% 像素的背景均值比、固定 `2×2` 面积平均重投影 MAE。显示拉伸不进入指标。
- 逐样本 CSV 后按空间来源组等权汇总，并给出相对最强确定性基线的来源组配对差与 10,000 次来源组 bootstrap 95% 区间（bootstrap seed `20261127`）。另记录参数量、推理时长、峰值显存、生成次数和 sample spread。
- 单 seed pilot 只作探索性决策。生成候选继续门槛：相对最强学习基线 development PSNR 至少 `+0.05 dB` 且来源组配对 bootstrap 95% 区间下界大于 0；SSIM 与 Sobel 相关差均值不低于 0；亮回波绝对均值比偏差、背景绝对均值比偏差和重投影 MAE 均不高于该基线。未满足则停止该候选配置，不用 confirmation 挽救结果。
- 无论结果如何，本阶段不读取 confirmation 影像、不生成 confirmation 指标、不访问项目历史 test，也不宣称独立泛化或光学效用。任何来自 development 的结论须标为重复使用的开发证据。

## 运行边界

归档清单/哈希、metadata 配对审计、冻结 split、train-only 数值域审查、固定输入生成、定向测试、GPU smoke、训练/推理、逐样本评价、负结果和 Research Decision Review 是本协议执行链。归档下载中断或本候选失败只停止受影响工作包，不构成访问 confirmation、改协议门槛或重跑已停止旧模型的理由。
