# PCSDD 复数 SAR 扩散去斑论文全文审查

审查日期：2026-11-09  
论文：*A Novel PDE-Guided Complex Self-Supervised Diffusion Despeckling Net for SAR*，Xuying Xiong、Xinyu Zhang、Weidong Jiang、Li Liu、Yongxiang Liu、Tianpeng Liu，IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing，第 19 卷，8617–8632 页，2026。DOI：[10.1109/JSTARS.2026.3665611](https://doi.org/10.1109/JSTARS.2026.3665611)。

## 来源与审查范围

IEEE DOI 指向的官方页面可核对题名、期刊与 DOI，但本次自动访问在机器人验证页停止。审查了作者公开的 CC BY 4.0 全文副本；副本正文含 IEEE 期刊页眉、论文全文与相同 DOI：[作者公开全文](https://www.researchgate.net/publication/400891493_A_Novel_PDE-Guided_Complex_Self-Supervised_Diffusion_Despeckling_Net_for_SAR)。下述方法和实验描述来自论文正文；未运行作者代码或数据。

## 论文事实

- PCSDD 面向复数值 SAR SLC 去斑。作者把各次迭代写成各向异性 PDE 漂移与学习型 score 修正的组合；推理时移除随机扩散项。神经网络学习空间与时间变化的扩散传导系数，并将其限制在 `[0,1]`。
- 训练使用 blind-spot 自监督，不额外注入合成噪声。正文说明约 10% 像素作掩码、每个 minibatch 使用 4 个掩码子集，并描述从排除中心像素的 `5×5` 邻域替换被掩码位置、只在盲点位置计算监督。
- **正文内部歧义：** 掩码构造段将 `M=1` 解释为替换/监督位置；但输入式 `S_T=S_ori⊙M+P(S_ori;1−M)` 将 `M` 用于保留原观测，并将补全算子用于 `1−M`；后文又把 `M=0` 定义为盲点监督集合。符号约定互相不一致。作者声称不会读取中心像素，但仅凭当前可读版本无法独立确认实际实现满足该声明；不据此推断代码存在泄漏。
- 方法损失包括强度域 Gamma 负对数似然、ENL 约束、传导场平滑项与边缘加权项。正文把复杂值更新与强度统计损失放在不同表示域中说明。
- 实验使用 Gaofen-3 C 波段 SLC，文中报告 1 m 空间分辨率。输入预处理包括辐射定标、统一空间分辨率裁剪、幅度公共动态范围归一化及 ENL 设置。作者称各方法共享真实数据和 train/validation/test 划分，并为传统滤波器在不与 test 区域重叠的 validation 区域调参；可读正文未列出完整场景编号、各 split 数量或独立公开清单。
- 评价包括 ENL、speckle suppression index、edge-preservation index、mean preservation，以及 SAM 图像分割结果。论文报告 PCSDD 在所选实景对比中有较好的去斑和结构保持表现；这是作者报告，未在本地复现。

## 对 JDSR 项目的意义与边界

该工作是“复数 SLC 上以 PDE 先验和扩散/score 修正进行 SAR 去斑”的直接相邻先例。因此，不应把通用的物理引导扩散、PDE 去斑或复数 SAR 扩散去噪单独列为本项目的新颖性。论文没有超分辨率观测算子、光学引导、光学错配/回退实验，也没有与光学条件联合 SAR 去斑/超分辨率的匹配比较。

PCSDD 使用复数 SLC 与强度域统计损失；本项目当前实验数据是 normalized 8-bit PNG/noisy pseudo-target。其 PDE、Gamma 似然、视数设定或定标流程不能未经表示核验直接移植，也不构成本地 SAR 栅格观测一致性的证据。本次审查不改变 C2/C3/C4 gate、数据权限、C3-A v3 或训练暂停状态。

## 未核实项

- 未找到或运行作者代码；未下载论文数据或本地数据。
- 自动访问的 IEEE 官方全文受机器人验证阻挡；作者公开全文副本的上传者身份与版本差异未另行核验。
- 公开正文未提供可供本项目复用的完整场景级 split 清单；论文报告的对比结果不能替代本项目任务匹配基线。
- 盲点掩码符号的正文歧义尚未由作者代码或其他可核验实现消解。
