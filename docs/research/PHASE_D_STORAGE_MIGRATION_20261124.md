# Phase D 磁盘空间解锁与实验资产迁移（2026-11-24）

## 迁移前后空间

| 位置 | 迁移前 | 迁移后 | 说明 |
|---|---:|---:|---|
| 根 ext4 分区（含仓库） | 可用约 49 GiB | 可用约 47 GiB | Phase D 持久 worktree 与 C3-A 归档写入此分区。 |
| `/tmp` tmpfs | 可用约 1.8 GiB | 可用约 3.6 GiB | 临时 worktree 已在逐文件 SHA-256 校验后移除；原路径改为指向持久副本的符号链接。 |

迁移前根分区 464 GiB、已用约 392 GiB、可用约 49 GiB；`/tmp` 总量 7.5 GiB、已用约 5.8 GiB。迁移后根分区已用约 394 GiB、可用约 47 GiB；`/tmp` 已用约 4.0 GiB、可用约 3.6 GiB。inode 使用率分别约 10% 与 2%，不是当前限制。

## 持久化位置与 Git 状态

- Phase D 独立 worktree：`/home/shy/AIR/SAR+AI/project/JDSR-phaseD-20261124`，分支 `codex/phase-d-research-20261124`，基于 `91ad31107e71f42e028cec984a3c02a62220a710`。
- C3-A 历史 worktree 归档：`/home/shy/AIR/SAR+AI/project/JDSR-phaseD-c3a-archive-20261124`，保留源 worktree 中的代码、未提交实验记录、checkpoint、样本与符号链接。
- `91ad311` 的父提交为 `5e6f2fd`；当前 `main` 为 `13a3c91`，且 `91ad311` 不是 `main` 的祖先。本阶段明确从 `91ad311` 开始，不假定其已合并。
- 主工作区 6,075 项既有状态未改动。论文目录未在此次存储迁移中操作。
- 为维持既有命令与清单中的路径，`/tmp/jdsr-c3a-final` 与 `/tmp/jdsr-c3a-20261123` 现在分别链接到上述持久化 worktree 和 C3-A 归档。链接和固定 C1 v2 train 输入路径已核验可解析。

## 文件级迁移核验

- Published Baseline/扩散 worktree：迁移 3,609 个文件、1,373,618,645 字节；目标缺失 0、多余 0、SHA-256 或符号链接目标不一致 0。
- C3-A worktree 归档：迁移 690 个常规文件、475,735,502 字节，另有 4 个符号链接；目标缺失 0、多余 0、文件哈希或链接目标不一致 0。
- 两组迁移分别于移除 `/tmp` 原 worktree 前对相对路径、文件大小和每个文件的 SHA-256，以及符号链接目标进行比较。worktree 根目录的 `.git` 指针是目标位置特有元数据，不参与文件内容比较。
- 另对持久化目录中的 2,823 个 `.pt`、`.npy`、`.npz` 资产计算 SHA-256，总计 1,791,925,388 字节；逐项校验表见[资产 SHA-256 清单](PHASE_D_STORAGE_SHA256_20261124.csv)。这包含模型 checkpoint、C3-A `generated_samples.npz` 和生成/基础预测 NPY 文件；数据图像没有加入校验清单或 Git。

## 数据与清理边界

C3-A 归档里的四条 `x2_light` 数据链接仍指向主仓库已有的固定 train/development 派生输入目录；该目录只按冻结清单提供 train 1,600/92 来源组与 development 200/19 来源组。未迁移、复制、改写或读取 confirmation/test 影像载荷。其他原始/派生数据、checkpoints、唯一实验结果与失败记录均予保留。

本次只移除了已完成完整持久化复制并验证的两个本任务临时 worktree；没有执行 `git clean`，没有删除原始数据、历史失败证据或无法确认归属的文件。大 checkpoint 与生成样本留在持久化本机目录，不进入 Git。

## 2026-11-24 收尾空间复测

- 根分区可用 **16.71 GiB**；inode 使用约 **9.5%**。`/tmp` 可用 **2.93 GiB**；inode 使用约 **1.2%**。
- Phase D worktree 占 **1.589 GiB**，其中 `experiments/` 占 **1.569 GiB**；主仓库 `data/` 占 **8.677 GiB**。EDSR run 占 **0.082 GiB**，D2 prior run 占 **0.029 GiB**，D2 evaluation run 占 **0.196 GiB**。
- 本轮可确认安全且可重建的缓存仅为专用临时目录中的 91 个 `.pyc` 文件，已清理并释放 1,991,907 字节。实验 checkpoint、预测数组、诊断 CSV 和其他任务的 `/tmp` 内容均保留；当前没有其他能确认归属且无复用价值的清理候选。
