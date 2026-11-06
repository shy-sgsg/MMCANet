# C3-A 按轮次边界续训设计

## 意图

让 C3-A 训练在进程中断后，从最近一个完整结束的 epoch 严格续跑。当前入口每轮写出 `last.pt`，但没有续训参数，也没有保存专用 DataLoader/Flow 随机数生成器状态；实验目录一旦创建便会拒绝再次启动。

## 目标

- 仅在显式传入 `--resume` 时恢复同一冻结运行目录；普通启动仍拒绝覆盖已有目录。
- checkpoint 恢复模型、Adam、已完成轮次、最佳轮次/分数、Python/NumPy/PyTorch CPU/CUDA RNG，以及 train DataLoader 和 Flow 生成器状态。
- 恢复前核对冻结协议、train/development 输入清单、代码哈希、兼容环境和运行状态；不匹配时 fail-closed。
- checkpoint 原子替换；日志以 checkpoint 中保存的轮次记录为准，续跑时重建，避免中断边界的重复或丢行。
- 仅从完整 epoch 边界续跑；中断 epoch 将从最近一个已保存轮次重新开始。

## 非目标

- 不改模型、损失、超参数、轮数、数据划分、评价指标、选轮规则或冻结协议。
- 不增加 batch 级续训、自动恢复、自动重启 GPU 等待器或自动开启训练。
- 不修改或覆盖任何历史运行；不把 CPU 合成恢复实验解释为模型效果证据。

## 约束

- 沿用冻结协议 `mmcanet_c3a_sar_only_flow_c1_v2_20261105_v1`，SHA-256 为 `147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`。
- 输入身份仍为 train/development；只读取 `lr` 和 `target`，confirmation/test/optical 标记必须为 false。
- 当前协议使用固定学习率且关闭混合精度，没有 scheduler 或 scaler 状态需要保存。
- 本轮实现和验证只能使用 CPU 合成状态；不得调用真实训练入口。

## 方案比较

1. **中断后另建全新运行。** 实现最少，但当前固定输出路径和协议运行身份无法安全创建第二个尝试，也会浪费已完成轮次。
2. **把 C3-A 改写为通用训练恢复框架。** 可复用更多模型，但会扩大改动范围并引入本协议没有的 scheduler、scaler 等选项。
3. **在 C3-A 入口实现轮次边界续训，并复用仓库现有原子 checkpoint 写入工具。** 改动集中在 C3-A runner 和定向测试，恢复字段直接对应当前冻结配置。

采用方案 3。它只补齐当前协议需要的续训状态，并保留普通启动的覆盖保护。

## 设计细节

- 新增布尔参数 `--resume`；不传时维持现有全新运行行为。传入时只接受预定输出目录中状态为 `running` 或 `failed` 且含有效 checkpoint 的同一运行。
- 每轮 checkpoint 记录协议哈希、轮次、当前/最佳分数与轮次、模型和优化器状态、所有全局 RNG 状态、两个专用 generator 状态，以及完整训练和开发逐图日志。
- 若 `best.pt` 与 `last.pt` 因写入中断处于不同轮次，验证二者后从较新且完整的 checkpoint 续跑；`best.pt` 仍依据已记录的最佳指标选定。
- 使用仓库已有 `save_checkpoint_atomic` 写 checkpoint。输入、代码和兼容环境记录在训练开始前落盘；续训时逐项比对，保留每次恢复时间、起始轮次和 checkpoint 哈希。
- 续训前从 checkpoint 重建 CSV；从 `checkpoint_epoch + 1` 继续。若 checkpoint 已到最后一轮，则只补齐最终汇总步骤，不再训练。

## 验收

- CPU 合成模型在“连续执行两轮”与“执行一轮、保存、重建对象并恢复后执行下一轮”后，参数和优化器状态一致。
- RNG 与两个专用 generator 的后续序列与连续执行一致。
- 协议、输入、代码、环境、运行状态或 checkpoint 不一致时拒绝恢复，且不静默修改训练选项。
- C3-A 定向测试和 `py_compile` 通过；不启动 GPU，不读取项目载荷。
