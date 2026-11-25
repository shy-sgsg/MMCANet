# SARptical 空间分组 pilot 实施计划

> **执行方式：** 当前会话内由主代理逐项实现、验证并记录，不拆分给子代理。计划落实用户已接受的有限 SAR-only 空间 pilot 和[设计稿](../../research/PHASE_D_SARPTICAL_SPATIAL_PILOT_DESIGN_20261125.md)。

**目标：** 在 SARptical 的封存空间分组下，训练并公平比较 bicubic、EDSR、SwinIR 与 bicubic-anchored Flow Bridge，完成开发评价和 Research Decision Review。

**架构：** 新增一个实验专属数据模块负责只读 ZIP 索引、坐标 split、train/development SAR MAT 解析、统一缩放和固定输入清单；实验 runner 复用现有 EDSR、SwinIR 与 `SarConditionalFlow` 实现完成统一训练、checkpoint 选择、推理和评价。原始 ZIP 保持只读，confirmation 只进入 split 元数据，不进入任何载荷读取路径。

**技术栈：** Python、NumPy、SciPy MAT 读取、PyTorch、现有 MMCANetV2 Flow 模型、BasicSR SwinIR、项目 EDSR 适配器、pytest。

**规格：** `docs/research/PHASE_D_SARPTICAL_SPATIAL_PILOT_DESIGN_20261125.md`。

## 全局约束

- 使用独立 worktree `codex/phase-d-research-20261124`；保留主工作区既有未提交改动和历史实验产物。
- 数据角色固定为 train/development/confirmation = 12/8/8 个 250 m 网格组，75 m 边界内缩；候选样本数为 916/306/306，seed `187762`。
- 只读取 train/development SAR 载荷；confirmation/test 与 Optical 载荷不读。manifest 显式记录全部保护 flags。
- ×2 表示域输入由 SAR target 经 bicubic、antialias=True、`align_corners=False` 下采样；不称为物理 SAR 退化或 clean SAR ground truth。
- EDSR、SwinIR 使用单通道线性标准化输出；Flow 使用固定 bicubic 锚点、seed `20261125`、sigma `0.5`、8 步 Heun。
- 三种训练模型 batch size `4`、patch LR/HR `48×48`/`96×96`、最多 30 epochs、Adam `1e-4`；Flow 用速度场 MSE，其余用 L1。
- 统一按 development 来源组等权 PSNR 选 best；最终评价单次 draw 与 Flow 四 draw pixel-mean/集合统计分开报告。
- 2026-11-26 实测项目根分区 6.1 GiB、`/tmp` 7.5 GiB，独立 NTFS 结果盘剩 90 GiB。训练缓存、checkpoint、预测和日志写至唯一成功目录 `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/phase_d_sarptical_spatial_pilot_20261126_retry01`；预计峰值新增 3 GiB 后保留 5 GiB 的条件满足。实际 CUDA OOM、磁盘写满、NaN/Inf 或配对错误时停止并保留证据。未知 CUDA 进程和 3 GiB 显存门槛不作为预检条件，不终止外部进程。
- 项目文档、脚本用户输出、报告和终端摘要使用简体中文；路径、模型/API 标识和标准术语保留原文。
- 最终完成实验和复核后，只提交本次授权工作产生且审过的代码/文档；大数据、缓存、checkpoint、预测数组不进 Git。之后按用户长期规范提交并推送。

## 文件结构

- 新建 `experiments/phase_d_sarptical_spatial_pilot_20261125/pilot_data.py`：空间分组、ZIP 安全读、MAT 数值解析、训练统计、LR/HR 预处理和 train/development 清单。
- 新建 `experiments/phase_d_sarptical_spatial_pilot_20261125/run_pilot.py`：四种方法统一训练与验证、checkpoint 选择、最终推理、指标和资源日志、manifest/Research Decision Review 生成。
- 新建 `experiments/phase_d_sarptical_spatial_pilot_20261125/finalize_saved_run.py`：复用已保存 checkpoint 与缓存恢复最终推理/指标核对和 Review，不重训。
- 新建 `tests/test_phase_d_sarptical_pilot.py`：split/角色保护、MAT 解析、预处理张量契约、Flow 训练路径、评价缺失/额外文件失败。
- 最终运行产物写入唯一外部持久化目录 `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/phase_d_sarptical_spatial_pilot_20261126_retry01/`；之前损坏归档/入口异常产生的失败目录均另行保留，不覆盖，也不加入 Git。

## 五个重点故障输入

1. 场景边界、NaN 坐标或孤立点：空间组分配必须 fail-closed，结果不能静默丢点或跨角色。
2. 嵌套/损坏的 MAT 文件或多个候选 SAR 矩阵：只接受经核验的单一 112×112 数值矩阵；其他结构明确失败。
3. confirmation 成员请求载荷：读取函数必须在打开 ZIP member 前拒绝 confirmation/test/Optical。
4. 零方差或非有限训练统计：生成配置失败；development 不重新拟合统计量。
5. 预测文件缺失/额外、尺寸错、非有限输出或实际 OOM：正式汇总失败，保存日志和已写 checkpoint，不自动修复结果。

## 任务 1：空间清单与 train/development SAR 输入

**文件：**

- 新建 `experiments/phase_d_sarptical_spatial_pilot_20261125/pilot_data.py`
- 新建 `tests/test_phase_d_sarptical_pilot.py`

**接口：**

- `build_spatial_split(archive_path: Path, seed: int = 187762) -> dict`：从 ZIP 中只读坐标 MAT，返回排序后的网格组、角色、SAR sample ID 与完整性摘要；候选角色必须为 12/8/8 组和 916/306/306 样本。
- `load_sar_matrix(archive_path: Path, sample_id: str, role: str) -> np.ndarray`：只接受 `train` 或 `development`；确认角色权限后才读取 ZIP member；返回有限的单通道 `112×112 float32` SAR dB 矩阵。
- `prepare_fixed_inputs(archive_path: Path, split: dict, output_dir: Path) -> dict`：只缓存 train/development 输入；归一化参数只由 train HR target 得出；每个来源 member 记录 SHA-256。

- [x] 编写测试，验证 28 组的 PCG64 seed `187762` 分配可复现、角色数 12/8/8、样本数 916/306/306 且组互斥。
- [x] 编写测试，验证 payload reader 在打开 ZIP 前拒绝 `confirmation`、`test` 和 `optical`。
- [x] 编写测试，验证 MAT 解析对单一有限 `112×112` SAR 矩阵成功，对错误形状、歧义字段和 NaN 失败。
- [x] 运行定向测试确认预期失败，再实现 `pilot_data.py` 的最小接口。
- [x] 再运行定向测试；生成 split manifest 后人工核对其只含 confirmation 元数据，并审计 916/306 的 SAR input manifest 与分组哈希。

## 任务 2：统一方法模型与训练循环

**文件：**

- 新建 `experiments/phase_d_sarptical_spatial_pilot_20261125/run_pilot.py`
- 修改 `tests/test_phase_d_sarptical_pilot.py`
- 复用 `experiments/phase_d1_edsr_20261124_v1/edsr_sar.py`
- 复用 `experiments/published_baselines_sar_only_20261123_v1/train_swinir_sar.py`
- 复用 `MMCANetV2/phase_c3a_flow.py`

**接口：**

- `train_method(method: str, train_data, dev_data, config: dict, run_dir: Path) -> dict`：method 仅为 `edsr`、`swinir`、`flow_bridge`；保存 last/best、训练日志和验证记录。
- `flow_bridge_pair(target, anchor, time, noise, sigma: float) -> tuple[Tensor, Tensor]`：按规格的线性 bridge 返回状态与目标速度。

- [x] 编写测试验证 EDSR、SwinIR 与 Flow 在单通道输入上输出 `112×112` 有限张量，Flow 目标速度与冻结公式一致。
- [x] 编写训练循环的 CPU 单 batch 测试，确认有限 loss、参数更新、同步翻转/转置增强和 RNG 可复现。
- [x] 运行定向测试确认失败，再实现统一训练循环；训练期间保存模型/优化器/RNG、epoch、best metric、环境及显存/磁盘快照。
- [x] 运行定向测试确认通过，并对 Python 文件执行 `py_compile`。

## 任务 3：统一推理、指标和来源组决策统计

**文件：**

- 修改 `experiments/phase_d_sarptical_spatial_pilot_20261125/run_pilot.py`
- 修改 `tests/test_phase_d_sarptical_pilot.py`

- [x] 用合成已知样例编写 PSNR/RMSE、SSIM、EPI、边缘强度、高值区域误差、背景偏差与 LR 重投影误差的 known-case 测试。
- [x] 编写测试，缺失/额外预测、尺寸错、NaN/Inf 或不同样本顺序时 evaluator 必须失败。
- [x] 编写测试，来源组等权汇总/bootstrap 固定 seed 可重现，Flow 单 draw 与 4 draw pixel-mean 不合并。
- [x] 运行测试确认失败后实现评价器，生成逐图 CSV、来源组 CSV、方法配对差和 10,000 次组 bootstrap 95% 区间。
- [x] 运行定向测试并执行 Python 编译检查。

## 任务 4：实际 GPU 训练、推理和研究决策

- [x] 检查唯一输出目录不存在、空间余量、输入哈希和权限 flags；记录 `AISAR` Python/PyTorch/CUDA/cuDNN、GPU 与 Git 状态。
- [x] 训练 EDSR 与 SwinIR；若其结果完整，训练固定 Flow Bridge。每轮验证仅用 development；confirmation payload 始终不读。
- [x] 运行全 306 个 development 的固定方法推理；Flow 额外生成 4 draws，分别统计单 draw、pixel-mean 与样本稳定性。
- [x] 对照冻结门槛分析 PSNR/SSIM/RMSE/EPI、结构/高值/背景、重投影误差、推理时间、峰值显存和磁盘成本。
- [x] 形成中文 manifest、逐图与分组 CSV、资源记录、生成样例以及 Research Decision Review，明确 CONTINUE/PIVOT/STOP 与结论边界。

## 任务 5：文档同步与最终交付

- [x] 更新 `CURRENT_STATE.md`、`MASTER_RESEARCH_PLAN.md`、`EXPERIMENT_REGISTRY.md`、`DECISION_LOG.md`、`NEXT_ACTIONS.md`、`RESEARCH_RISKS.md`、`PROJECT_FACTS.md` 与 README。
- [x] 执行相关 pytest、项目 `tests/` 全集、`git diff --check`；核对 run 文件数量、权限 flags、manifest/CSV 和空间余量。根目录无筛选的 pytest 会收集裁剪版 vendored BasicSR 测试并因 `basicsr.data` 缺失而失败，见 Research Decision Review；项目测试目录 210 项均通过。
- [x] 复核仅本次授权的代码/文档进入提交；数据、缓存、checkpoint、预测和原有改动不得进入提交。
- [x] 按用户长期规范完成单次最终 commit 并推送到配置远端/分支，报告 SHA、目标与验证结果。

## 实际执行与验证结果（2026-11-26）

- **完成 run：** `/run/media/shy/C61C9A341C9A200B/Codex_JDSR_runs/phase_d_sarptical_spatial_pilot_20261126_retry01`；决策 `STOP_CURRENT_FLOW_BRIDGE_CONFIGURATION`。训练、推理和最终化均完成；confirmation/test/Optical payload 均未访问。
- **负结果与边界：** Flow 单 draw 相对 SwinIR 的 ΔPSNR 为 −2.6412 dB（8 组 bootstrap 95% CI [−2.7286,−2.5647]）；SSIM/EPI 下降，背景偏差 8/8 组恶化。高回波 ROI 0/8 组有效，按冻结 gate 失败。完整结论见[Research Decision Review](../../research/PHASE_D_SARPTICAL_SPATIAL_PILOT_RESULTS_20261126.md)。
- **异常保留：** 首个下载归档损坏、第二次入口 `NameError` 和汇总器全 NA 异常均各自保留证据；有效训练仅在 retry01 完成，finalizer 对 checkpoint/cache 的推理重放与原逐图指标一致。
- **验证：** 定向测试 22 项通过，项目 `tests/` 210 项通过；4 个 Python 文件编译通过，`git diff --check` 通过。无筛选的根目录 pytest 还收集 vendored BasicSR 测试并因其缺少 `basicsr.data` 报 12 项收集错误；未修改 vendored 代码。
- **产物核对：** 主 manifest 中 config/split/input SHA-256、6 个 best/last checkpoint hash 全部匹配；2,448 个预测文件与 manifest 数量一致。运行输出保留在外部卷，不纳入 Git。

## 决策日志

- 用户已接受 SAR-only 有限空间 pilot；按其具体书面方案执行，没有把候选统计当作模型结果。
- 用户要求并行 GPU 负载不作为启动拦截且仅真实 OOM 为 CUDA 停止边界；不终止外部任务。
- 用户要求工作全部完成后再提交推送；不在中途提交。
