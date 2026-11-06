# C3-A Flow CPU 数值与确定性诊断

## 目的

用不含项目影像的合成张量核验 Heun 求解器的二阶误差收敛、函数调用数，以及逐样本源噪声的可重复性。该诊断只覆盖实现数值，不评价任何 SAR 重建模型。

## 协议

- 使用线性合成向量场 `dx/dt=x`，初值为 `0.1`，解析终值为 `0.1e`。
- Heun 积分步数为 `16、32、64`。预声明判据：步数翻倍时绝对误差比处于 `3.8–4.2`，且每步正好调用模型两次。
- 在 CPU 上检查同一 `run seed + sample_id` 得到完全相同的噪声，batch 顺序变化不改变对应样本噪声，不同样本噪声不同。
- 不读取 train/development、confirmation、test 或 optical 载荷，不初始化 CUDA。

## 结果

Heun 误差比与噪声确定性检查均通过；合成张量在 CPU 上执行 Adam 速度场单步后，loss/gradient 均有限且模型参数发生更新。另为多次采样评价器增加 CPU 合成运行清单验收：合法训练/input manifest 可通过；缺失清单、来源路径或 SHA-256 不一致、train/development 角色/数量矛盾、confirmation/test/optical 标记开启、声明读取 optical 数组均 fail-closed。实施前相关缺陷用例按预期失败；实施后 C3-A Flow、评价器、等待器四组定向测试共 51 项通过，相关 Python 文件编译通过。详细数值、代码哈希与环境见同目录 [`manifest.json`](manifest.json)。重放命令见 [`command.txt`](command.txt)。

## 解释边界

通过结果支持 Heun 数值实现及逐样本噪声映射符合上述软件性质；不支持训练收敛、图像质量、多次采样稳定性、后验校准、SAR 观测一致性或物理解释主张。正式 C3-A CUDA 训练与 development 评价仍未运行。
