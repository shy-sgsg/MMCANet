# D1 配对基线来源核验（训练后审计）

## 结论

EDSR 使用的 SwinIR 开发预测目录来自已完成的 SwinIR SAR 任务适配运行。EDSR 与 SwinIR 的 C1 v2 train/development 样本数、来源组数、数据 manifest 哈希及 `files.csv` 哈希一致。SwinIR 自身 best checkpoint 的 SHA-256 已在运行后记录，但原 SwinIR manifest 未登记该权重哈希；其 `baseline_checkpoint_*` 字段实际指向内部 C2 对照权重，后者的实际 SHA-256 与 manifest 一致，不能误当作 SwinIR 任务权重证据。200 个预测文件名与 SwinIR 运行输出、EDSR 逐图参考表及 D2 评价清单一致，当前逐文件 SHA-256 全部匹配 D2 清单。

这属于训练后的来源核验。EDSR 原运行没有在使用参考预测前写入逐文件哈希，也没有校验来源运行清单，因此本审计不能回溯证明 EDSR 每次读取时文件内容没有瞬时变更；其指标应解释为同数据、同来源运行、当前内容已交叉核验的探索性配对比较。

## 来源运行

- SwinIR run：`运行_swinir_20261123_v3`；状态 `complete`；最佳轮次 `30`；协议 SHA-256 `53b29c2be90b18e9ba9b111bb9f0ae8623a24703e2bf2e9e4f8a69b16f90046e`。
- SwinIR 运行清单 SHA-256：`45f0e43314d562218adc14afc040dc8dbcf0b3b987dbe650dc1cfd0d664ee6c9`。
- SwinIR 自身 `checkpoints/best.pt`：SHA-256 `b4d043d30ea25724cfca65fb7ed0b79ee2e6a7aa07e8e123ff710f2fccd63c61`（训练后计算；源 SwinIR manifest 未登记自身 checkpoint SHA-256）。
- SwinIR manifest 的 `baseline_checkpoint_*` 指向内部 C2 对照权重：`/home/shy/AIR/SAR+AI/project/JDSR/experiments/phase_c2_c1_v2_20261031_v1/matrix/runs/x2_light__sar_only/checkpoints/best.pt`；SHA-256 `8255b5229ca1e041cb59aed582d794bb67afdbe1601b6f4dc425c43ce27f7165`，该 C2 文件的实际 hash 与 manifest 匹配。
- EDSR run：`phase_d1_edsr_sar_20261124_v1`；其原始运行清单 SHA-256 `974139ba0f121682d3f64f1d7d2931f494207907c75f50b39d4bfb7732254db9`。
- D2 配对评价清单 SHA-256：`530f8df6c41bab3ec64b61388b634fa6e295fb3922a24e2029ca271ffaa461cb`；其中 200 个 SwinIR 预测哈希与当前文件逐一匹配。

## 数据角色与配对

| 角色 | 样本数 | 来源组数 | 数据 manifest SHA-256 | `files.csv` SHA-256 | SwinIR 匹配 | D2 匹配 |
|---|---:|---:|---|---|---|---|
| train | 1600 | 92 | `64d335f06812859b96b884a5583092df76dd4b6fa181c3a842b18ed5d3bf6eeb` | `97f35201125dcb3830e9d0583d3350159f0d6757056c7acf13c4c1a0910466ad` | 是 | 不适用 |
| development | 200 | 19 | `de3101a78105c05653b9779342cabe98d61a9cddac6153b25a7d6347d0ef0da9` | `70d762015b2b16a604d381758091ae3d282526a7137e9de51c6db827b60f2839` | 是 | 是 |

EDSR、SwinIR 和 D2 清单的 `test_set_accessed/used`、`confirmation_payload_accessed`、`optical_payload_loaded` 均为 `false`。本审计只哈希已生成的开发预测文件，不读取图像或受保护载荷。逐文件哈希见 [`baseline_reference_prediction_sha256_20261124.csv`](../../experiments/published_baselines_sar_only_20261123_v1/baseline_reference_prediction_sha256_20261124.csv)，机器可读交叉核验见 EDSR run 内的 `配对参考来源审计.json`。

## 运行计时口径

两个运行清单中的 `training_seconds` 从训练阶段计时并持续到最终 development 评价结束，包含逐轮 development checkpoint 选择评估、最终全量推理/评价与运行收尾；SwinIR 还包含最终内部 C2 对照推理。因此对比表将其标为“完整运行累计耗时”，不得解释为纯训练耗时。

## 解释边界

EDSR 与 SwinIR 都是发表通用架构的 SAR 单通道任务适配版，不是作者 SAR 权重或原论文完整方案复现。C1 v2 development 曾重复用于训练/选模；组均衡指标及 bootstrap 只支持这 19 个历史来源组上的探索观察。此审计不构成独立泛化证据，也不补足 SAR 专用已发表方法（如 DSRKD）缺少作者实现/权重的问题。
