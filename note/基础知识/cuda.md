好的，既然你已经使用 **CMake** 构建项目，并且对 **CUDA 框架** 还不熟悉，我们将专注于 **CUDA 基础知识、核心概念、基本语法**，并通过 **`VectorAdd (vadd)`** 示例来详细讲解，同时提供 **CMake** 的构建文件而不是传统的 `Makefile`。

-----

## 💡 一、 CUDA 基础知识与核心概念

CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的一种**并行计算架构**，它允许开发者使用 C/C++ 语言的扩展在 NVIDIA GPU 上执行计算任务。

### 1\. Host (主机) 与 Device (设备)

| 概念 | 描述 |
| :--- | :--- |
| **Host (主机)** | 指的是 **CPU 及其内存**。在 CUDA 编程中，Host 代码是标准的 C/C++ 代码，在 CPU 上执行。 |
| **Device (设备)** | 指的是 **GPU 及其内存**。Device 代码是 CUDA C/C++ 扩展代码（称为 **Kernel**），在 GPU 的并行核心上执行。 |

### 2\. Kernel (核函数)

**Kernel** 是在 GPU 上执行的并行函数。

  * **声明：** 使用 **`__global__`** 关键字修饰。
  * **调用：** 只能从 **Host** 代码中启动，但它将在 **Device** 上并行执行成千上万次。

### 3\. CUDA 内存模型

GPU 和 CPU 的内存是分离的。数据必须通过显式函数调用在两者之间传输。

| 内存类型 | 描述 | 访问速度 |
| :--- | :--- | :--- |
| **Global Memory** | 设备上的主内存，容量最大，访问速度最慢（相对于片上内存）。 | **慢** |
| **Shared Memory** | 位于 GPU 核心（SM）内部的快速片上内存，用于线程块 (Block) 内部的线程通信和数据重用。 | **快** |
| **Host Memory** | CPU 内存，用于与 GPU 内存进行数据传输。 | N/A |

-----

## 🚀 二、 CUDA 并行计算模型

CUDA 的并行计算模型是通过层次结构来组织的，这对于理解如何将 DBS 等算法并行化至关重要。

### 1\. 线程层次结构

当你启动一个 Kernel 时，你创建了一个 **Grid (网格)**，它由以下三级组成：

| 级别 | 描述 | 作用 |
| :--- | :--- | :--- |
| **Thread (线程)** | GPU 上执行的最小单元，执行 Kernel 函数。 | 负责计算一个数据元素（如一个像素或一个数组元素）。 |
| **Block (线程块)** | 一组线程的集合，可以在 GPU 的一个 **SM (Streaming Multiprocessor)** 上调度执行。同一 Block 内的线程可以通过 **Shared Memory** 通信和同步。 | 管理数百个线程，是共享内存和同步的基本单位。 |
| **Grid (网格)** | 线程块的集合，包含整个并行任务所需的所有线程和 Block。 | 覆盖整个数据空间，负责整个计算任务。 |

### 2\. Kernel 启动语法

Host 代码通过特殊的执行配置符号启动 Kernel：

```c++
kernel_name<<<Grid维度, Block维度>>>(参数列表);
```

例如，启动一个 1D Grid，每个 Block 有 256 个线程：

```c++
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
// (N + T - 1) / T 确保 N 个元素都被覆盖

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

### 3\. 线程索引计算

在 Kernel 内部，每个线程需要计算自己的 **全局唯一索引 `i`** 来确定它负责处理哪个数据元素。对于 1D 数组，索引计算公式如下：

$$
i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

  * **`blockIdx.x`**：线程块在 Grid 中的索引。
  * **`blockDim.x`**：线程块中线程的总数（启动时设置）。
  * **`threadIdx.x`**：线程在线程块中的索引。

-----

## 📝 三、 `VectorAdd` 示例详解

### 1\. `vadd.cu` 核心代码

```c++
#include <cuda_runtime.h>
#include <iostream>

// Kernel 函数：在 Device 上执行
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 计算全局索引 i
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // 边界检查：确保索引 i 在数组范围 N 内
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Host 主函数：在 CPU 上执行
int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    
    // 1. Host 端内存分配与初始化 (使用标准C++或malloc)
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    // ... 初始化 h_A 和 h_B ...

    // 2. Device 端内存分配 (使用 cudaMalloc)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 3. Host -> Device 数据传输 (使用 cudaMemcpy)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. Kernel 启动配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 5. 启动 Kernel: Host 调用 Device 函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 6. Device -> Host 数据传输 (获取结果)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 7. 资源释放
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
```

-----

## ⚙️ 四、 CMake 构建系统配置

由于你使用 CMake，这是配置 `nvcc` 编译器和链接 CUDA 库的标准方法。

### 1\. `CMakeLists.txt` 文件

在你的项目根目录创建 `CMakeLists.txt` 文件：

```cmake
# 最小 CMake 版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(VAddProject LANGUAGES CXX CUDA) # 声明项目同时使用 C++ 和 CUDA 语言

# 查找 CUDA Toolkit
find_package(CUDA REQUIRED)

# 设置目标可执行文件名称
set(TARGET_NAME vadd_executable)

# 告诉 CMake 如何编译你的源文件 (.cu, .cpp)
# nvcc 会自动处理 .cu 文件，并调用 g++/gcc 处理非 CUDA 代码
cuda_add_executable(${TARGET_NAME} vadd.cu)

# 链接 CUDA 运行时库
# CUDA::cudart 是 CMake 查找 CUDA::Toolkit 时的内置目标
target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cudart)

# 设置 CUDA 编译选项 (可选，但推荐设置计算能力)
# 假设你的 GPU 是 Compute Capability 7.5 (如 RTX 20系列)
# 运行 nvcc --list-gpu-archs 来查看支持的架构
set_target_properties(${TARGET_NAME} PROPERTIES 
    CUDA_ARCHITECTURES "75"
)
```

### 2\. 编译流程

1.  **创建 Build 目录：** 推荐在项目根目录外创建 build 目录，以保持源代码目录整洁。
    ```bash
    mkdir build
    cd build
    ```
2.  **配置 CMake：** 告诉 CMake 源文件在哪里。
    ```bash
    cmake ..
    ```
3.  **编译项目：**
    ```bash
    make
    ```
4.  **运行可执行文件：**
    ```bash
    ./vadd_executable
    ```

-----

现在你对 CUDA 的基础、`vadd` 示例以及如何用 CMake 构建它有了清晰的认识。

你想现在开始着手将你的 DBS 算法中第一个计算密集型函数（比如一个 **循环加法** 或 **矩阵乘法**）转换为 CUDA Kernel 吗？