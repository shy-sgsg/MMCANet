# 稿件沿革与证据来源审计

更新于 2026-10-19。本审计辨认稿件版本及其产物沿革。历史测试汇总仅用于核对来源与正文的对应关系，不用于当前模型选择。

## 稿件清单

| 角色 | 源文件与编译产物 | 角色证据 | 当前解释 |
|---|---|---|---|
| 正式 2k 投稿源文件、历史编译稿与修订预览 | `paper/MMCANet_formal/main.tex`、历史版 `main.pdf`、`main_caption_audited_20261018.pdf` | `README.md` 将其标为正式源文件；`main.pdf` 日期为 2026-08-21，且未被修改。修订源文件于 2026-10-18 另行编译为独立命名的 PDF。 | 这是基于旧双流模型的历史正式研究；不是 MMCANetV2 稿件，也未验证当前光学增益假设。 |
| 工程试点稿 | `paper/MMCANet/mmcanet_jdsr_draft.tex`、`paper/MMCANet/mmcanet_jdsr_draft.pdf` | 源文件自标为“Internal Pilot Draft, July 2026”，并报告 1,900/100 工程划分。`paper/pilot_archive/PILOT_ARCHIVE.md` 指向不可变的试点快照。 | 仅作归档。不得替换其 PDF，也不得将其复用为正式 1,600/200/200 研究。 |
| 当前 MMCANetV2 稿件 | 本次检查的 `paper/` 清单中没有 MMCANetV2 专属源文件。 | 仓库 README 和研究状态记录将 `MMCANetV2/model.py` 中的 B1 系列模型与以上两个版本区分开。 | 当前没有稿件能支撑关于活跃 V2 研究路线的主张。 |
| IEEEtran 模板目录中的独立草稿变体 | `IEEEtran/mmcanet_jdsr_draft.tex`、`IEEEtran/mmcanet_jdsr_refs.bib`；可重建 PDF 已于 2026-10-27 删除。 | 该 299 行源文件有独立题名和 8 条参考文献；`paper/MMCANet` 下的 7 月试点稿为 193 行、26 条参考文献。二者的源文件、PDF 和参考文献并非字节级相同。该源文件没有本地图片或生成表格输入；本地 IEEEtran 类文件与参考文献文件保留。 | 这是尚未分类的较早 MMCANet 草稿，不能作为 MMCANetV2 证据。其主张到运行记录的沿革未审计，不得复用其中结果或主张。为来源追踪保留源文件和配置。见 D-2026-10-27-007。 |

## 正式预览稿的主张与产物对应关系

| 主张类别 | 稿件位置 | 实现/运行沿革 | 审计状态 |
|---|---|---|---|
| SAR 主导的双流 U-Net、瓶颈交叉注意力与浅层光学残差 | `paper/MMCANet_formal/main.tex` 的引言与方法 | `MMCANet/src/train_attention_sr_common.py::TwoStreamFrequencyGuidedUNet`；seed-0 配置和运行 manifest 位于 `experiments/dev2k_formal/final_long_runs_nan_guard_v2/MMCANet_seed0/`；模型/checkpoint 对照见 `experiments/dev2k_formal/freeze_v1/model_freeze_manifest.json`。 | 架构/版本映射与旧实现一致，与 `MMCANetV2/model.py` 不同。 |
| ×2 表示域训练及选定的输出/损失配置 | 方法与实现细节 | seed-0 的 `config.json`、`manifest.json`，以及 freeze manifest 引用的训练和验证集选模 manifest。 | 配置沿革记录存在；尚未逐公式对照确切历史代码版本完成全面审计。 |
| 生成的主结果、配对比较、消融、压力测试、退化与效率表格 | `paper/MMCANet_formal/generated/` 下按规范生成的 `*.tex` 表格 | `generated/paper_tables_manifest.json` 将 15 个 TeX 产物绑定到直接汇总文件 `experiments/dev2k_formal/submission_results/submission_results.json`，并记录审稿补充项及直接拼接补充结果的输入。生成的 TeX 源码注释带有汇总文件路径/哈希。投稿汇总本身记录了基础正式汇总，以及冻结后外部逐图 CSV/manifest。 | 表格的直接来源是扩展后的投稿汇总，即使部分数值最初来自基础正式研究也是如此。因此，不应称这些生成文件直接取自 `formal_results.json`。扩展汇总声明 `protocol_role=adaptive_development_test`、`test_set_used=true`；这是历史开发证据，不是未触碰的确认集。 |
| 定量图 | `formal_main_multimetric.pdf`、`formal_optical_shift.pdf`、`formal_degradation_efficiency.pdf` | `formal_result_figures_manifest.json` 将图像哈希绑定至 `formal_results.json` 及其生成器哈希。 | 来源汇总已有记录；图表 manifest 声明使用了正式 test。不得把这些图重新解释为当前模型或未触碰留出集证据。 |
| 图 3 直接拼接方法的 seed 数 | `main.tex` 的图 3 图注、`generated/formal_main_multimetric.pdf`、表 I | 图表生成器读取基础 `formal_results.json`，其中直接拼接方法族只有冻结的 seed-0 运行。扩展后的三 seed 直接拼接结果位于 `generated/submission_main_results.tex` / 表 I。 | 旧图注曾称图表展示了完整三 seed 方法族。当前图注已说明图中柱状值为 seed 0，并指向表 I 中单独列出的冻结后、自适应开发集三 seed 结果。修订 PDF 第 7 页已目视检查；历史 `main.pdf` 保持不变。 |
| 方法/协议图与定性图 | `mmcanet_architecture.pdf`、`formal_protocol_flow.pdf`、`formal_qualitative_overview.pdf`、`formal_qualitative_diagnostics.pdf` 及补充材料 | 方法图和定性图各有独立 manifest。正式 README 记录了用于定性生成的固定输入/评价 manifest，以及预先声明的代表样本清单。 | 这些图与数值表的沿革分开。样本/显示来源不能证明方法更优；历史 test 身份集合已经作为自适应开发集使用。 |
| 可选的优化光学压力测试表 | `main.tex` 按条件导入 `generated/optimized_optical_stress.tex`；历史构建日志记录了该输入 | 更新后的汇总器为 `paper_tables_manifest.json` 增加表格/来源哈希和生成器指纹；标准表格生成器只有在两个文件仍匹配时才沿用此记录。审计器会在源文件包含该表时核对沿革。 | 当前历史表格和 manifest 未重新生成，因此旧 PDF 尚未纳入新的沿革核验。流程行为由临时夹具回归测试覆盖；不得把旧审计通过解释为已覆盖该表格。 |
| 当前光学效用与鲁棒性主张 | 正式预览稿的结果/讨论部分 | 当前研究状态记录：活跃 B1/B1R/B1F 以及后续 R/S/T 门槛均未晋级光学方法。 | 不得将正式预览稿结果或结论转移到 MMCANetV2；当前研究主张仍未建立。 |

## 测试集与时间边界

`paper/MMCANet_formal/README.md` 说明，投稿对比使用固定的 200 张测试输入，并标记 `protocol_role: adaptive_development_test`；该 README 明确说明这不是未触碰的独立留出集。因此，正式预览是历史开发产物，不是当前研究问题的独立确认。其生成表格和编译 PDF 均为历史产物。

正式预览 PDF 于 2026-08-21 编译，早于 `CURRENT_STATE.md` 汇总的后续 B1 系列筛查。自动投稿审计报告描述的是该历史构建，不代表 2026-10-18 MMCANetV2 活跃研究状态。

## 当前工作区的 checkpoint 与源快照审计（2026-10-18）

上述产物级对应关系已与当前 freeze、选模、训练和正式推理 manifest 核对；本次未打开 test 图像数组或逐图指标 CSV。freeze manifest 列出 10 个已选训练 checkpoint：3 个 SFT、3 个 MMCANet、3 个 SAR-only、1 个直接拼接 seed。这些 checkpoint 路径在当前工作区均不存在。对 `experiments/dev2k_formal/` 和 `outputs/` 下 115 个 `.pth`/`.pt` 文件进行有限扫描，按冻结文件大小筛选并再核对 SHA-256 后，没有找到与已选 checkpoint 字节匹配的文件。不得用当前 `best.pth` 替代这些权重；它们是不同产物。

当前 10 个选模 manifest 的哈希都与 freeze manifest 记录一致；相应的 10 个正式主套件推理 manifest 也都记录了冻结 checkpoint 哈希和 200 个输出。相反，当前 10 个训练 manifest 均与 freeze manifest 中记录的哈希不一致。SFT 主套件 manifest 未包含 `test_set_used`，与正式 README 记录的旧 manifest 限制一致；freeze 与周边协议记录仍是现存的边界证据。因此，历史输出与冻结 checkpoint 的身份关系有记录，但所选训练权重和确切训练 manifest 字节无法从当前检出版本独立恢复。这会限制重跑/复现主张，但本身不能证明历史预测或表格错误。

已检查的 MMCANet seed-0 运行元数据记录提交 `fc1d2e5` 且 `dirty=true`；其 `git_state.json` 只有提交号和 dirty 状态，没有源补丁或文件哈希快照。因此可以映射当前架构实现与运行配置，但无法独立重建历史训练时使用的确切 dirty 源代码状态。当前存在的 `best.pth` 并非 freeze manifest 中的 checkpoint 哈希。

原始与扩展表格/图像来源链现已映射到产物记录层级，图 3 的 seed 数/来源不匹配也已在稿件源文件中修正。本轮已完成生成来源与结果正文的核对（2026-10-19）；协议/评价器沿革仍有来源限制。剩余工作是恢复冻结训练权重和训练 manifest 的逐字节副本（若存在授权副本）。这些限制存在期间，应原样保留当前论文、freeze/选模/评价记录及历史 `main.pdf`。

## 结果正文与来源核对（2026-10-19）

范围：将摘要、结果和讨论中的字面数值主张与现存生成表格、绑定汇总及 OGSRN 配对比较汇总核对。未打开 test 图像数组或逐图 CSV；未重新计算指标；没有使用结果进行模型或 checkpoint 选择。投稿汇总标记 `protocol_role=adaptive_development_test` 和 `test_set_used=true`，因此下列所有基于 test 的数值都仍是历史自适应开发证据。

| 稿件主张 | 来源核对 | 处理结果 |
|---|---|---|
| 摘要中 MMCANet 三 seed 的 PSNR/SSIM/EPI 及其相对 SAR-only 的差值 | `generated/result_macros.tex` 绑定至 SHA-256 为 `26e7944c…5a06da2` 的 `submission_results.json`，与当前汇总哈希一致。展示数值（`17.275±0.053`、`0.1725±0.0009`、`0.1256±0.0009`；差值 `−0.035`、`+0.0012`、`+0.0001`）与相应生成的方法族汇总一致。 | 已核对；草稿已将其表述为自适应开发结果，而不是光学带来正向收益的主张。 |
| 主对比排序 | `generated/submission_main_results.tex` 报告 SwinIR-Joint 为 `17.341±0.004` dB、EPI 为 `0.1275`；直接拼接 EPI 为 `0.1287`，高于 SwinIR-Joint 和 MMCANet。 | 已修正文案：SwinIR-Joint 是 SAR-only 中 PSNR/EPI 最强的参考，同时指出直接拼接的 EPI 更高。 |
| 同图 seed 配对比较 | `generated/submission_paired.tex` 和 `generated/same_seed_paired.tex` 支持所有 seed 的 MMCANet−SwinIR PSNR 均值为负；直接拼接的 PSNR 区间在 seed 0/1 与零重叠，seed 2 为负，配对 EPI 更有利于直接拼接。SAR-only 对比中，seed 0 的 PSNR 接近零，seed 1/2 有利于 SAR-only；EPI 区间在 seed 0/1 与零重叠，seed 2 有利于 MMCANet。 | 已核对；无需修改方向性文案。 |
| OGSRN 与其 SAR-only 对照 | `experiments/dev2k_formal/ogsrn/test/ogsrn_paired_comparisons.csv` 按 seed 将 OGSRN 与 `SRUN-only` 对照配对。所有 seed 的 PSNR 均值差都为负，但 seed 0 区间与零重叠；三个 seed 的 SSIM 和 TBR-ratio utility 均为正。 | 已将含糊的“SAR-only 对应方法”改为配对的 SRUN-only 对照，说明 seed 0 的不确定性，并标注比较属于冻结后的自适应开发证据。 |
| 验证集选模与消融文案 | 生成的仅验证集输出头表中，B6 得分为 `13.765`，B0 为 `13.702`；融合表和选模敏感性表均保持仅验证集角色。 | 选模描述与来源表格一致；描述 B6 选择时未使用 test 结果。 |
| 原始与优化光学压力测试数值 | `generated/optical_stress.tex` 报告原始 seed-0 正常 PSNR `17.328`，正常到全零变化 `−0.0155`、错配 `−0.0104`、对角平移 8 像素 `−0.0151`。`generated/optimized_optical_stress.tex` 报告三 seed 对齐 PSNR `17.2697`，光学全零变化 `−0.0069`。 | 已核对；正文区分原始代表结果与冻结后追加的三 seed 结果。 |
| 退化敏感性与效率文案 | `generated/degradation_sensitivity.tex` 支持 `L={1,2,4,8}` 下保真/结构代理指标的变化。效率汇总记录 SAR-only seed-2 墙钟区间为 `922.289` 分钟，而 epoch 计算时间总和为 `801.45` 秒；草稿正确说明墙钟区间不是 GPU 计算时间。 | 已核对；无需修改措辞。 |
| 讨论中的边缘/HF/TBR ratio | `submission_results.json` 给出的 MMCANet 方法族均值为 `0.141669`、`0.007929` 和 `0.595765`。 | 已修正为三 seed 均值 `0.142`、`0.008`（展示舍入前为 `0.0079`）和 `0.596`。删除了与独立仅验证集损失筛查的直接“提升”比较，因为二者不是同一划分上的配对比较。 |

新稿件正文仅使用已有生成数值；生成 TeX 与汇总数据未更改。这种编辑性访问不改变实验 manifest 或门槛，也不验证历史模型。确切的 dirty 训练源快照、冻结训练权重和历史评价器实现仍未恢复，详见前文。

### 预览构建与视觉检查

修订源文件使用 `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=<temporary-directory> -jobname=main_numeric_audited_20261019 main.tex`，从 `paper/MMCANet_formal/` 编译到独立临时预览。命令成功退出，生成 16 页、2,217,587 字节的 PDF。提取文本定位到第 5、8、13 页的结果与讨论修订段落；三个页面的渲染稿均已目视检查，检查区域内无裁切、遮挡或表格位置问题。临时 PDF 和页面图已在审阅后删除；历史 PDF 均未替换。最终 LaTeX 日志没有未解析的引用或交叉引用警告；仍有 underfull box 警告、一页只含浮动体，以及该浮动页上 3.68 pt 的 overfull vbox，因此不能称为无警告构建。

## 剩余工作

- 恢复确切的冻结训练权重和训练 manifest 字节；若当前检出中无法恢复，应明确保留“无法从本工作区重跑”的结论。当前 freeze 与选模记录保留哈希，但 10 个训练权重路径均缺失；在实验/输出目录下的 115 个 `.pth`/`.pt` 文件中未发现哈希匹配项，当前训练 manifest 的冻结哈希匹配为 0/10。不得用当前 `best.pth` 替代。
- 在不使用历史自适应测试结果进行当前模型选择的前提下，继续核对正文手工填写的数值与生成表格/来源字段。（结果正文核对已于 2026-10-19 完成，见上文；协议/评价器沿革仍未解决。）
- 核实来源引用，以及当前论文源文件中没有明确证据支持的主张。
- 在当前科学门槛和证据缺口解决前，不得创建或标记声称有证据支撑的 MMCANetV2 稿件。

## 2026-11-06 当前内部 pilot 稿的有限审阅

本次只读了 `paper/MMCANet/mmcanet_jdsr_draft.tex` 的题名、摘要开头、引言、相关工作，以及方法部分至训练目标；在“Audited Pilot Protocol”和“Pilot Results”之前停止。稿件标记为 2026 年 7 月内部工程 pilot，自述进行了三轮、仅一个 seed、没有 validation，且数据物理域未经核实；因此它不是当前新主线的投稿稿，也不能支撑正式贡献。正文包含历史 test 身份数量及定性失败总结。本次据此披露 `test_set_accessed=true`（仅限历史文本）、`test_set_used=false`；未打开 test 影像、split 清单、测量值、逐图 CSV 或 checkpoint，也未利用这些文本作选择。旧稿沿革和 checkpoint 缺失的既有结论保持不变。
