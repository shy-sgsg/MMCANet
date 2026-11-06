# 参考目标审计——C1 阶段

**日期：** 2026-10-28
**状态：** 已识别经核验 SEN1-2 子集的目标语义；当前协议中没有独立的 clean SAR 参考。

## 训练目标的性质

B1 v2 的高分辨率目标是同一来源 SAR 图像块。源文件为官方提供的 8 位 σ⁰ dB 表示；流程先按字节值缩放到 `[0,1]`，再执行配置中的对数归一化。因此它是归一化表示域中的**含噪 SAR 伪目标**，不是潜在反射率的无噪图像，也不是经过定标的线性强度真值。

对每组训练配对，低分辨率 SAR 输入由该目标确定性生成：使用冻结的 FFT/缩放/负值截断链，再施加表示域 Gamma 乘法。因此模型评价的是它与含噪目标之间的一致性；该目标与模型输入在统计上相关。PSNR、SSIM 及附加的结构/散射体指标描述的是声明的归一化和掩码下的伪目标一致性，不能直接衡量 clean SAR 恢复质量。

## 当前缺少的信息

所选 1,800 组数据不含原始 Sentinel-1 SAFE/SLC 数据、文档化的 DN↔σ⁰ 映射表、独立多视估计，或能建立 clean 参考的同场景重复采集。官方来源文件夹匹配可以证明该子集的文件来源，但不能证明目标无噪，也不能确认精确的采集时间配对。项目中以数值命名的另一组 2,000 对数据仍未映射来源，目标沿革也未经核验。

## 对研究的约束

- 当前结果应描述为：受控合成退化下，对归一化、含噪 8 位伪目标的重建结果。
- PSNR/SSIM 与由目标推导的 TBR、边缘、亮散射区域指标应作为表示域诊断。可与结构及 ratio 指标共同报告，但不得称为定标辐射保真度或真实去斑质量。
- 路线 A（受控合成退化）只能在经核验来源子集上开展，并且须先获得新的合格来源组划分。
- 路线 B（真实 SAR 评价）目前无法按既定定义执行。它需要真实低分辨率观测，以及可信的独立参考或测量协议；对现有伪目标再施加一次合成变换不构成路线 B。
- 未来若要声称使用 clean reference，必须记录目标生成流程，并明确论证目标残余噪声及其与 LR 输入的关系。

## 来源

- 运行绑定的语义与参数：[`b1_revalidation_v2.json`](../../protocols/sen12_scene_dev1800_v2/b1_revalidation_v2.json)。
- 固定退化与同目标生成链：[`degradation.py`](../../MMCANet/src/degradation.py)、[`prepare_fixed_degradation.py`](../../scripts/prepare_fixed_degradation.py) 和 [`MMCANetV2/data.py`](../../MMCANetV2/data.py)。
- SEN1-2 官方编码说明：[`SupportingDocument.txt`](../../experiments/data_audits/sen1_2_upstream_hash_crosswalk_20261019_v1/SupportingDocument.txt)。
- 来源与划分限制：[`sen12_scene_dev1800_v2_auth_20261021_v1`](../../experiments/data_audits/sen12_scene_dev1800_v2_auth_20261021_v1/report.md) 和 [`sen1_2_train_val_spatial_overlap_audit.md`](sen1_2_train_val_spatial_overlap_audit.md)。

本次未访问测试数据、未运行模型，也未改写任何历史结果或阶段门槛。
