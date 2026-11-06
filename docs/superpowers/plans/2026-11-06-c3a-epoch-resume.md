# C3-A 按轮次边界续训实施计划

**目标：** 为冻结的 C3-A SAR-only Flow 训练入口增加显式、可复现、fail-closed 的 epoch 边界续训。

**结构：** 在 `scripts/run_phase_c3a_sar_flow.py` 中加入续训参数、checkpoint 状态收集/校验/恢复和 manifest 续写；复用 `scripts/training_resume.py` 的原子 checkpoint 写入。用独立 CPU 合成测试证明连续运行与恢复运行等价，不触及真实训练入口。

**技术栈：** Python、PyTorch、NumPy、pytest；不新增依赖。

**设计依据：** [设计说明](../specs/2026-11-06-c3a-epoch-resume-design.md)

## 全局约束

- 冻结协议 SHA-256：`147c361b3e91bc4cd8d76ce5c464ad9d2137f2d06b38f31a0bc7c7a7e93101a7`；不更改实验参数或晋级门槛。
- 只允许 train/development 的 `lr`/`target`；confirmation、test、optical 载荷不得读取。
- 续训必须显式请求；不得自动启动 GPU 等待器或训练。
- 普通启动遇到既有输出目录必须失败；只在相同冻结运行和兼容输入/代码/环境上续训。
- 项目文档和面向用户的输出使用简体中文。

## 审查重点

- 缺少专用 DataLoader/Flow generator 状态会改变后续样本顺序和训练噪声；CPU 等价测试需覆盖二者。
- 错误协议、输入清单或代码状态可能混入另一实验；恢复测试需确认不匹配时在更改模型/优化器前失败。
- 中断可能发生在 checkpoint 与 CSV 写入之间；checkpoint 必须包含足以重建日志的记录。
- `best.pt` 与 `last.pt` 原子写入间可能中断；恢复需从有效且较新的 checkpoint 判断下一轮。
- 已完成运行不得再次进入训练；恢复状态测试需拒绝 `complete`。

---

### 任务 1：先建立 CPU 续训行为测试

**文件：**
- 新建：`tests/test_c3a_training_resume.py`
- 测试现有入口与共享 helper：`scripts/run_phase_c3a_sar_flow.py`、`scripts/training_resume.py`

**接口：**
- 测试将调用 C3-A runner 暴露的 checkpoint 创建、验证和恢复 helper；若测试证明需要，接口固定为 `_make_checkpoint(...)`、`_validate_resume_checkpoint(...)`、`_restore_checkpoint(...)`。
- 复用共享 `save_checkpoint_atomic`，不得自行另写原子文件替换机制。

- [x] **步骤 1：** 编写连续两轮与“保存第一轮、重新构造模型/优化器/generator、恢复后执行第二轮”的 CPU 对照测试；断言模型参数、Adam 状态和两个 generator 的后续随机序列相同。
- [x] **步骤 2：** 运行新测试，确认因 C3-A checkpoint 缺少续训接口/状态而按预期失败。
- [x] **步骤 3：** 编写协议哈希不匹配、缺少 RNG/generator 状态、轮次越界以及完成态拒绝恢复的测试。
- [x] **步骤 4：** 运行这些测试，确认失败原因为对应的保护行为缺失。

### 任务 2：实现 C3-A 严格 epoch 恢复

**文件：**
- 修改：`scripts/run_phase_c3a_sar_flow.py`
- 测试：`tests/test_c3a_training_resume.py`

**实现要求：**
- 增加 `--resume`，只允许同一路径中未完成的冻结运行恢复；普通启动仍拒绝非空输出目录。
- 训练开始前记录并校验协议、输入 manifest/files.csv、相关代码哈希、关键环境版本和访问标记。
- checkpoint 保存当前/最佳 epoch 与分数、model/optimizer、Python/NumPy/PyTorch CPU/CUDA RNG、DataLoader/Flow generator 状态及训练/开发日志。
- 用 `save_checkpoint_atomic` 保存；保留现有 `best.pt` 与 `last.pt` 语义，并处理二者轮次不同的中断边界。
- 恢复时校验全部绑定信息，重建 CSV，恢复状态后从下一完整 epoch 开始；已到最终 epoch 时仅完成汇总。
- 每次恢复追加 manifest 历史，保存前次状态、checkpoint 哈希、恢复轮次和命令；错误继续 fail-closed。

- [x] **步骤 1：** 按任务 1 的失败测试增加最小 checkpoint 结构、完整状态恢复及绑定校验。
- [x] **步骤 2：** 运行任务 1 测试，确认所有行为通过。
- [x] **步骤 3：** 增加 CLI 与输出目录模式测试：默认启动拒绝重用，显式恢复接受兼容未完成目录并拒绝已完成/缺 checkpoint 目录。
- [x] **步骤 4：** 运行新续训测试与现有训练恢复定向测试。
- [x] **步骤 5：** 运行 C3-A Flow/evaluator/waiter 定向回归，随后对修改的 Python 文件执行 `py_compile`。
- [x] **步骤 6：** 只检查本次生成的轻量临时缓存，更新中文状态、实验登记、计划、决策及 README；记录 CPU 合成验证和资源余量。

## 执行边界

在用户恢复训练前，不运行真实 GPU 训练/评价，不读取 confirmation/test/optical 载荷。CPU 等价测试只证明续训状态管理正确，不证明模型效果。


## 执行结果 — 2026-11-07 00:26（中国标准时间）

- CPU 合成续训等价与 fail-closed 状态合同已实现并通过。
- C3-A 续训、Flow、评价器与等待器定向回归：67 项通过；相关 Python 文件 `py_compile` 通过。
- 训练、GPU 等待和正式评价保持暂停；没有读取 optical、confirmation 或 test 载荷。
- 按精确路径清理 13 个本任务临时日志及 14 个相关 Python 缓存后，项目分区可用 23,334,772,736 字节（约 21.73 GiB）；pytest 既有缓存与实验数据保留。
