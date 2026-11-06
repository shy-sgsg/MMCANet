# C3-A train/development CPU 数据完整性预检

## 目的

在 GPU 训练仍暂停时，先按冻结 C3-A `x2_light` 协议验证 train/development 的 SAR LR 与 noisy pseudo-target 固定数组，尽早发现哈希、尺寸、dtype、有限性或值域错误。

## 读取边界

- 仅打开 `train` 与 `development` 固定输入中的 `lr/*.npy`、`target/*.npy`。
- 不打开 `optical/*.npy`；来源清单和 `files.csv` 中相关身份/hash 仅作为元数据读取。
- 不打开 confirmation/test 影像或测量载荷；不初始化 CUDA，不创建训练/评价 checkpoint。

## 结果

train 的 1,600 个样本来自 92 个来源组，核验 3,200 个数组文件；development 的 200 个样本来自 19 个来源组，核验 400 个数组文件。两者均只包含 `lr` 与 `target`，哈希、shape、dtype、有限值和值域检查通过。CUDA 预检前后均未初始化；confirmation、test、optical 载荷访问标记均为否。逐角色字节数、固定输入 manifest 与 `files.csv` 哈希、环境和代码哈希见同目录 [`manifest.json`](manifest.json)。重放命令见 [`command.txt`](command.txt)。

## 解释边界

该预检只证明冻结 train/development SAR LR 与 pseudo-target 固定数组符合本地输入合同；不证明 Flow 收敛、重建效果、光学效用、独立泛化或物理一致性。GPU 训练仍按用户指示暂停。
