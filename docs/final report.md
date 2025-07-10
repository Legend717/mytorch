<h1 style="text-align: center;">并行算法期末大作业报告</h1>

<div style="text-align: center; font-weight: bold;">
    研究主题：基于动态计算框架的神经网络算子并行加速库——mytorch
</div>

<center>团队成员（无先后之分）：陈兴平、刘华壹、罗弘杰</center>

[TOC]

**摘要：**
mytorch是一个基于动态计算框架的轻量级神经网络算子并行加速库，旨在通过多设备支持（CPU/GPU）和高效并行算法（如OpenMP和CUDA）优化深度学习模型的训练与推理性能。该框架参考PyTorch的核心设计，实现了张量计算、自动微分、计算图追踪等基础功能，并提供了模块化的神经网络层（如Linear、Conv2D）和优化器（如SGD）。此外，我们团队在mytorch基础上集成了FlashAttention算法，通过online-softmax和分块访存技术显著提升了Attention算子的计算效率。实验表明，mytorch能够成功在MNIST数据集上训练，且CPU/GPU并行加速效果显著，为轻量化深度学习框架的开发提供了实践参考。通过几周的努力以及上千行代码的实践，我们成功完成了我们预设的目标。

**关键词**：mytorch; pytorch; flashattention; OpenMP; CUDA; 自动微分; 并行计算; 神经网络加速

相关代码已上传到团队仓库：[Legend717/mytorch](https://github.com/Legend717/mytorch)

### **1. 研究背景**

> 项目的详细背景以及规划可以见我们的开题报告。在此仅结合我们所作工作做简单的介绍。

#### **1.1 相关工作——pytorch**

在当今神经网络研究领域，pytorch已成为不可或缺的核心工具。作为Meta（原Facebook）开发的开源框架，它凭借动态图优先的设计哲学脱颖而出，通过`torch.autograd`在Python运行时动态构建计算图，这种机制相比TensorFlow等静态图框架更能满足科研场景的快速迭代需求。PyTorch不仅提供直观的Pythonic编程体验和高效的GPU加速能力，还集成了完整的深度学习工具链（如TorchVision、TorchText），并与Python生态无缝对接，同时通过TorchScript和ONNX支持实现便捷的模型部署。随着PyTorch 2.0引入编译优化技术并持续强化分布式训练与大模型支持（如Llama），该框架在保持科研灵活性的同时不断提升工业级性能，已成为贯穿算法探索到生产落地的首选平台。

以下是 pytorch 的框架图（使用 Mermaid 代码编写）：

```mermaid
graph TD
    subgraph Python接口
        TorchAPI["torch (核心API)"]
        TorchNN["torch.nn (神经网络层)"]
        TorchAutograd["torch.autograd (自动微分)"]
        TorchOptim["torch.optim (优化器)"]
        TorchDistributed["torch.distributed (分布式)"]
        TorchJIT["torch.jit (JIT编译器接口)"]
    end

    subgraph 底层C++实现
        TorchC["torch._C (C++核心)"]
        TorchCSRC["torch/csrc (C++扩展)"]
        JITImpl["torch/csrc/jit (JIT核心)"]
        ATen["aten (Tensor和算子实现)"]
        C10["c10 (通用基础库)"]
    end

    subgraph 编译与加速
        Dynamo["torch.compile & Dynamo编译器"]
        Inductor["TorchInductor (高性能后端)"]
    end

    subgraph 工具与文档
        Docs["docs/ (文档)"]
        Tools["tools/ (开发与CI工具)"]
    end

    TorchAPI --> TorchNN
    TorchAPI --> TorchAutograd
    TorchAPI --> TorchOptim
    TorchAPI --> TorchDistributed
    TorchAPI --> TorchJIT

    TorchAPI --> TorchC
    TorchNN --> TorchC
    TorchAutograd --> TorchC
    TorchOptim --> TorchC
    TorchDistributed --> TorchC
    TorchJIT --> JITImpl

    TorchC --> ATen
    TorchC --> C10
    TorchCSRC --> ATen
    TorchCSRC --> C10

    TorchJIT --> JITImpl

    TorchAPI --> Dynamo
    Dynamo --> Inductor

    Docs -.-> TorchAPI
    Docs -.-> TorchC
    Tools -.-> TorchAPI
    Tools -.-> TorchC

```

我们将参考pytorch的底层实现，将其化繁为简，设计一个相对轻量化的框架mytorch来训练我们的神经网络。

#### **1.2 前沿发展——高性能算子的设计**

Attention算子是Transformer模型的核心组件之一，主要用于处理序列数据。它通过计算输入序列中各个位置之间的相关性来生成输出序列。
其集体的计算公式是
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
Q,K,V是x输入经过三个线性层得到的查询（Query）、键（Key）和值（Value）矩阵，$d_k$是键的维度。

不同于以往的并行算法，Attention算子目前不仅受制于运算速度，还受制于内存带宽，以及空间复杂度，算力对高速显存的依赖需要我们改进算法，在计算速度和占用内存之间取得平衡。

我们将在自己设计的`mytorch`框架基础上，实现对Attention算子的加速。

### **2. mytorch框架设计**

#### **2.1 项目结构图**

![image-20250710171553873](static/mytorch structure.png)

#### **2.2 框架介绍**

- **Core 层**：负责张量（Tensor）定义、数据存储与基本运算、动态计算图的构建和自动微分机制。
- **nn 层**：面向用户，封装常见神经网络层，每个模块都继承自 Module，组合/嵌套灵活。
- **optim 层**：如 SGD 优化器，用于实现反向传播后的参数更新。
- **examples 层**：提供端到端模型训练/推理示例，便于开发者快速上手。
- **构建/接口**：CPU/GPU 兼容，CMake 工程，预留 Python绑定。

### **3. mytorch算法实现**

我们实现了CPU和GPU两个版本的代码。其中，GPU版本代码兼容CPU的计算，所以在此以GPU版本代码为准进行介绍。（因为部分代码为了方便写CUDA做了重构）

#### **3.1 Tensor相关实现**

`Tensor` 实现兼顾了多设备支持、自动微分、常见算子重载和计算图追踪等深度学习框架核心要素。设计思路高度参考主流框架（PyTorch），并通过 C++ 智能指针、设备枚举等机制，保证内存与计算的安全与灵活性。

##### **3.1.1 核心类结构与设备支持**

- `Tensor` 类定义于 `gpu/include/core/tensor.h`
- 支持多设备（CPU/CUDA），通过枚举 `Device` 区分，Tensor 内部变量 `_device` 标识张量当前所在设备。
- 禁止直接拷贝构造和赋值，强制使用静态工厂函数（如 `create`, `randn`, `ones`, `zeros`）进行构造，确保管理一致性和设备感知。
- 构造函数示例：
  ```cpp
  Tensor(std::vector<size_t> shape, bool requires_grad = false, Device device = Device::CPU);
  ```
  工厂函数示例（可用来创建 CPU 张量并初始化数据）：
  ```cpp
  static std::shared_ptr<Tensor> create(const std::vector<float>& data, std::vector<size_t> shape, bool requires_grad = false);
  ```

##### **3.1.2 张量的数据管理**

- 内部数据指针 `_data`，存储张量的实际数据（实现支持不同设备的数据分配）。
- 提供 `data_ptr()`、`mutable_data_ptr()` 用于获得底层数据指针，实现与设备无关的调用。
- `data_cpu()` 方法支持将数据从任意设备拷贝回 CPU 并以 `std::vector<float>` 返回，方便调试和跨设备操作。

##### **3.1.3 张量的属性与操作**

- 支持张量形状（`shape()`）、元素个数（`size()`）、单元素访问（`item()`）。
- 步幅（stride）通过 `compute_stride()` 计算，支持高维张量的存储与遍历。
- 典型代码片段（步幅计算）：
  ```cpp
  void Tensor::compute_stride() {
      _stride.resize(_shape.size());
      size_t stride = 1;
      for (int i = _shape.size() - 1; i >= 0; i--) {
          _stride[i] = stride;
          stride *= _shape[i];
      }
  }
  ```

##### **3.1.4 自动微分与计算图**

- 自动微分属性：`requires_grad` 标志、`grad()` 获取梯度、`set_grad()` 设置梯度，确保梯度与数据在同一设备。
- 通过 `_ctx`（`std::shared_ptr<Function>`）记录产生当前张量的运算上下文，实现反向传播时的计算依赖追踪。
- `backward()` 方法实现反向传播，递归拓扑排序所有依赖的节点，自动计算梯度。核心思想与 PyTorch 类似。

#### **3.1.5 运算符与函数接口**

- 支持常见算子（`add`、`sub`、`mul`、`div`、`matmul`、`sum`、`relu` 等）作为成员函数存在，返回新张量并自动构建计算图。
- 相关代码接口示例（部分）：
  ```cpp
  std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& other);
  std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);
  std::shared_ptr<Tensor> relu();
  ```

##### **3.1.6 设备管理与迁移**

- `to(Device device)` 支持张量在 CPU/GPU 之间迁移，保证设备一致性。
- 在梯度设置等操作中显式检查设备一致性，防止跨设备错误，提升健壮性。

#### **3.2 Function相关实现**

 `Function` 实现了基于计算图的自动微分核心框架，类设计高度模块化，便于扩展和维护（不然难以debug）。每个运算都作为 Function 子类存在，统一接口支持前向与反向传播，灵活高效。Function 与 Tensor 紧密协作，实现链式自动微分，支持深度学习常见运算与自定义扩展。

##### **3.2.1 Function类的核心设计**

- `Function` 类定义于 `gpu/include/core/function.h`
- 核心作用：作为所有具体算子（如加法、乘法等）自动微分操作的基类，负责正向与反向传播的统一接口。
- 继承自 `std::enable_shared_from_this<Function>`，方便在运算图构建和反向传播中安全管理智能指针引用。

##### **3.2.2 前向/反向传播接口**

- 提供统一的 `apply`（正向）、`backward`（反向）接口，自动保存输入用于反向传播追溯。
- 通过纯虚函数 `_forward` 和 `_backward`，要求所有具体算子必须实现自身的前/反向逻辑。
  ```cpp
  virtual std::shared_ptr<Tensor> _forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
  virtual std::vector<std::shared_ptr<Tensor>> _backward(const std::shared_ptr<Tensor>& grad_output) = 0;
  ```
- `_saved_inputs` 保存本次前向传播涉及的输入张量，便于自动微分时恢复依赖。

##### **3.2.3 典型算子运算的Function子类**

- 针对常见张量操作，每个操作都实现为 `Function` 的子类。例如：
  - `Add`：加法算子
  - `Sub`：减法算子
  - `Mul`：乘法算子
  - `MatMul`：矩阵乘法
  - `Sum`：求和
  - `ReLUFunc`：ReLU激活
  - `Conv2DFunc`、`MaxPool2DFunc` 等卷积、池化相关操作
- 每个子类都重写 `_forward` 和 `_backward` 以实现各自的运算与梯度计算。
- 例如，`Add` 的反向传播将上游梯度直接传递给两个输入，`Mul` 的反向传播则需要乘以另一个输入的值。

##### **3.2.4. 与Tensor的关系**

- `Function` 与 `Tensor` 通过 `Tensor::_ctx` 建立联系，每个由算子生成的新张量都保存了对应的 `Function` 实例指针，实现了计算图的自动追踪。
- 在 `Tensor::backward()` 时，会自动遍历 `_ctx` 链条递归回溯，依次调用各 `Function` 子类的 `backward` 方法，完成全自动链式反向传播。
- 支持复杂网络结构和运算图拓扑。

##### **3.2.5 反向传播的依赖管理**

- `Function` 保存前向输入（`_saved_inputs`），能精确还原每个操作的依赖链。
- 支持释放已保存输入以节省内存（`release_saved_inputs()`），便于大规模训练和推理场景应用。

##### **3.2.6 灵活性与可扩展性**

- 任何新的算子都可以通过继承 Function 并实现 `_forward`/`_backward` 两个方法来扩展。
- 这样保证了所有算子都可以被无缝集成到自动微分系统中，且与设备无关（具体运算留给子类或后续实现）。

##### **3.3 nn模块说明**

`mytorch` 的 `nn` 模块为深度学习模型的各类层（Layer）与常用结构提供了抽象与实现，核心设计理念高度参考 PyTorch 的 `torch.nn`，实现灵活组合和参数管理，并支持多设备（CPU/GPU）训练。

##### **3.3.1 核心基类设计**

- **Module基类**  
  所有神经网络层都继承自 `nn::Module` 抽象基类（定义见 `nn/module.h`，未在本次检索结果中直接列出）。  
  每个子类都需实现如下接口：
  - `forward(std::shared_ptr<Tensor> input)`：前向传播，返回输出张量。
  - `parameters()`：返回本层可学习参数的张量列表，便于优化器统一管理。
  - `to(Device device)`：将本层参数搬移到指定设备（如GPU），支持多设备训练。

##### **3.3.2 典型层的实现与特点**

- **线性层 Linear**  
  参考 `gpu/include/nn/linear.h`和 `gpu/src/nn/linear.cpp`
  - 构造时可指定输入输出特征数、是否带bias。
  - 权重采用 Kaiming He 初始化（对ReLU函数友好）。
  - 前向传播为 $Y = XW + b$，支持自动广播 bias。
  - 参数管理和设备迁移实现见 `parameters()` 和 `to()`。

- **卷积层 Conv2D**  
  见 `gpu/include/nn/conv.h`与 `gpu/src/nn/conv.cpp`
  - 支持设置输入/输出通道数、卷积核尺寸、步幅、padding。
  - 权重同样采用 Kaiming 初始化。
  - 前向传播通过 `Conv2DFunc` 实现，自动支持参数迁移和收集。

- **池化层 MaxPool2D**  
  见 `gpu/include/nn/pool.h`与 `gpu/src/nn/pool.cpp`
  - 支持池化核大小、步幅设置。
  - 前向调用 `MaxPool2DFunc` 完成实际操作。
  - 池化层无可学习参数。

- **激活层 ReLU**  
  见 `gpu/include/nn/activations.h`
  - 实现简单，无可学习参数，前向传播直接调用 `relu`。

- **Flatten层**  
  见 `gpu/include/nn/flatten.h`
  - 用于展平输入张量形状，常用于卷积->全连接的连接部位。

##### **3.3.3 结构组合与复用**

- **Sequential容器**  
  见 `gpu/src/nn/sequential.cpp`
  - 支持按顺序组合多个 `Module` 层，自动递归前向传播、参数收集、设备迁移。
  - 方便搭建常见的多层感知机/卷积网络等结构。

##### **3.3.4 参数与设备统一管理**

- 每个 `Module` 子类均实现 `parameters()`，递归收集所有可学习参数，便于优化器如 SGD 实现统一管理。
- 通过 `to(Device device)` 支持参数（如权重、偏置）一键搬移，多设备切换灵活。

##### **3.3.5 其它说明**

- 所有层均兼容自动微分与反向传播（通过 Tensor/Function 构建的计算图），使用时只需调用 `backward()` 即可自动求导。
- 代码整体风格简洁清晰，易于扩展自定义层或结构。（当然这得感谢pytorch，pytorch的设计比我们复杂但更加精妙）

#### **3.4 并行算法**

考虑到我们这节课是并行算法。所以专门开一个环节进行介绍我们是怎么从CPU和GPU两个维度进行并行的。

- **CPU端并行**：采用OpenMP，主要通过`#pragma omp parallel for`指令，把向量/矩阵操作自动分发到多个CPU核心，提高吞吐量。
- **GPU端并行**：采用CUDA，将大规模数据操作映射为CUDA kernel，利用成百上千的GPU线程进行大规模数据并行。
- **接口无缝切换**：同一套高层API（如Tensor、Function等）内部自动判断设备，透明切换CPU/GPU后端，对使用者友好。

##### **3.4.1 OpenMP并行（CPU端）**

在OMP（OpenMP）版本实现中，核心并行策略是利用`#pragma omp parallel for`等指令对常见的数值运算（如张量加法、乘法、矩阵乘法、ReLU激活、SGD优化器）进行多线程加速。

**例：SGD优化器并行更新参数**

代码片段（见 `omp/src/optim/sgd.cpp`）：
```cpp
void SGD::step() {
    for (auto& p : _params) {
        if (p->grad()) {
            auto p_data = p->get_shared_data();
            auto g_data = p->grad()->get_shared_data();   
            #pragma omp parallel for schedule(guided)  
            for (size_t i = 0; i < p_data->size(); ++i) {
                (*p_data)[i] -= _lr * (*g_data)[i];
            }
        }
    }
}
```
- 这里对每个参数的每一个元素进行并行的梯度更新，大大提升了大规模参数量网络的更新效率。
- 其它如ReLU等操作，也广泛用`#pragma omp parallel for`进行矢量化加速。

**例：ReLU激活并行实现**
```cpp
std::shared_ptr<Tensor> ReLUFunc::_forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    const auto& x = inputs[0]->data();
    std::vector<float> result_data(x.size());
    #pragma omp parallel for
    for(size_t i=0; i<x.size(); ++i) {
        result_data[i] = std::max(0.0f, x[i]);
    }
    return Tensor::create(result_data, inputs[0]->shape());
}
```
- 这种写法保证了CPU多核资源的高效利用，适合在传统服务器或本地多核环境下部署。

##### **3.4.2 CUDA并行（GPU端）**

在CUDA版本实现中，核心思路是将大规模矢量/矩阵计算任务分发到成百上千的GPU线程上，通过CUDA kernel函数实现数据并行。

**例：im2col卷积前向传播**

im2col_kernel 核函数提供了对卷积的并行，使用的方法是im2col，能够在gpu中显著提升运算速度

```c
__global__ void im2col_kernel(const float* data_im, float* data_col,
                            int N, int C, int H, int W,
                            int K, int S, int P,
                            int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int col_size = C * K * K;
    int num_kernels = N * H_out * W_out;

    if (index < num_kernels * col_size) {
        // 计算在输出列矩阵中的位置
        int col_idx = index % col_size;  // 列索引(0到C*K*K-1)
        int row_idx = index / col_size;  // 行索引(0到N*H_out*W_out-1)

        // 分解列索引找到核位置
        int k_w = col_idx % K;          // 核宽度坐标
        int k_h = (col_idx / K) % K;    // 核高度坐标
        int c_in = col_idx / (K * K);   // 输入通道

        // 分解行索引找到输出像素位置
        int w_out = row_idx % W_out;    // 输出宽度坐标
        int h_out = (row_idx / W_out) % H_out; // 输出高度坐标
    			     int n = row_idx / (H_out * W_out); // 批次索引
        
        // 计算对应的输入坐标
        int h_in = h_out * S - P + k_h;
        int w_in = w_out * S - P + k_w;

        // 如果在边界内则复制，否则填充0
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            data_col[index] = data_im[(n * C + c_in) * H * W + h_in * W + w_in];
        } else {
            data_col[index] = 0.0f;
        }
    }
}
```

`im2col_cuda`函数提供了方便的C++接口：

1. 计算输出尺寸：`H_out = (H + 2*P - K)/S + 1`
2. 创建输出张量：使用`Tensor::zeros`在GPU上分配空间
3. 启动内核并检查错误

```c
std::shared_ptr<Tensor> im2col_cuda(const std::shared_ptr<Tensor>& input,
                                  size_t K, size_t S, size_t P) {
    const auto& shape = input->shape();
    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];

    int H_out = (H + 2 * P - K) / S + 1;
    int W_out = (W + 2 * P - K) / S + 1;

    // 在GPU上创建输出列Tensor
    auto col_tensor = Tensor::zeros({(size_t)C * K * K, (size_t)N * H_out * W_out}, false, Device::CUDA);
    size_t n = col_tensor->size();
    if (n == 0) return col_tensor;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    im2col_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input->data_ptr()),
        static_cast<float*>(col_tensor->mutable_data_ptr()),
        N, C, H, W, K, S, P, H_out, W_out
    );
    CUDA_CHECK(cudaPeekAtLastError());

    return col_tensor;
}
```

这种实现充分利用了GPU的并行能力，将im2col操作高效地映射到CUDA架构上，是卷积神经网络前向传播的重要优化步骤。

#### **3.5 to方法的实现——CPU和GPU的统一**

以下是结合 mytorch 仓库 gpu2 目录下相关代码，对 to 方法及其相关设备迁移实现的详细源码分析：

##### **3.5.1 Tensor::to 方法实现（核心代码）**

在 gpu2/src/core/tensor.cu 中，`Tensor::to(Device device)` 实现了张量的设备迁移：

```cpp
// 将张量移动到另一个设备
std::shared_ptr<Tensor> Tensor::to(Device device) {
    if (this->_device == device) {
        return shared_from_this();
    }
    auto new_tensor = std::make_shared<Tensor>(_shape, _requires_grad, device);
    size_t data_size = this->size() * sizeof(float);
    if (data_size == 0) {
        return new_tensor;
    }
    // 根据数据转移的方向，选择正确的指针和拷贝方式
    if (device == Device::CUDA) { // 方向: CPU -> CUDA
        // 源(this)在CPU上, _data 是 std::vector<float>*
        // 目标(new_tensor)在GPU上, mutable_data_ptr() 返回 float* (GPU地址)
        // 1. 从源CPU张量中获取 std::vector<float> 对象
        // ...
    }
    // 反之亦然，CUDA -> CPU
}
```

- 首先判断目标设备是否与当前一致，不一致则新建目标设备的张量，并进行数据内存的复制。
- 针对 CPU->CUDA、CUDA->CPU，各自调用 cudaMemcpy 或直接内存拷贝，保证正确的数据迁移。
- 迁移时 shape、requires_grad 属性全部保留。

##### **3.5.2 to 方法的使用例子**

**nn::Linear 的 to 方法**

在 gpu2/src/nn/linear.cpp：

```cpp
void Linear::to(Device device) {
    if (_weight) _weight = _weight->to(device);
    if (_bias) _bias = _bias->to(device);
}
```
- 将权重和偏置（Tensor）分别迁移到目标设备。

**nn::Sequential 的 to 方法**

在 gpu2/src/nn/sequential.cpp：

```cpp
void Sequential::to(Device device) {
    for(auto& layer : _layers) {
        layer->to(device); // 递归调用每一层
    }
}
```
- 对所有子模块递归调用 to，实现整个网络的设备统一。

##### **3.5.3 设备属性与数据分配**

在 gpu2/include/core/tensor.h：

```cpp
enum class Device {
    CPU,
    CUDA
};

class Tensor {
    Device _device;
    // ...
    Device device() const { return _device; }
    std::shared_ptr<Tensor> to(Device device);
    // ...
};
```

- 每个 Tensor 都附带 device 信息。
- 数据分配时 allocate_data() 会根据 device 类型选择分配 CPU 内存（std::vector<float>）或 GPU 内存（cudaMalloc）。

##### **3.5.4 相关辅助/底层实现**

- allocate_data()、data_cpu()、item() 等函数对 device 做专门分支处理，保证数据访问和迁移一致性。

这里以`allocate_data()`为例

```cpp
// 这个函数由构造函数调用，负责根据设备分配内存
void Tensor::allocate_data() {
    size_t total_size = this->size();
    if (total_size == 0) {
        _data = nullptr;
        return;
    }

    if (_device == Device::CPU) {
        _data = new std::vector<float>(total_size, 0.0f);
    } else { // _device == Device::CUDA
        CUDA_CHECK(cudaMalloc(&_data, total_size * sizeof(float)));
        // 确保新分配的GPU内存被清零，这对于zeros()等操作很重要
        CUDA_CHECK(cudaMemset(_data, 0, total_size * sizeof(float)));
    }
}
```

### **4. 加速算法-FlashAttention介绍**

FlashAttention是一种高效的Attention计算方法，主要通过以下方式加速：

1. online-softmax
2. 分块访存利用内存层次

#### **4.1 online-softmax的历史演变**

softmax的伪代码如下：![alt text](./static/nativeSoftmax.png)

1. 从访存的角度考虑，原始的softmax对每个元素需要2次load,1次store操作；
2. 从数值稳定性上考虑，原始的naive softmax在计算过程中可能会出现数值溢出的问题，因此需要对其进行改进。

safe-softmax
safe-softmax通过减去输入向量的最大值来避免这种问题.
safe-softmax的伪代码如下：
![alt text](./static/safeSoftmax.png)
这次改进由于需要**计算max**,所以每个元素需要3次load,1次store操作。

online-softmax最早由nvidia在2018年提出，主要利用了softmax中指数运算法则的特性，将softmax的前后依赖关系打破，允许在计算softmax的过程中进行并行计算。伪代码如下图：
![alt text](./static/onlineSoftmax.png)

#### **4.2 分块访存前向计算**

GPU的访存根据访问速度从高速到低速有层次之分，从register, L1-cache/shared memory, L2-cache, L3-cache, HBM等。为了充分利用GPU的高速缓存，FlashAttention采用了分块访存的方式进行前向计算。
![alt text](./static/attention.png)
通过分块，可以将Q,K,V矩阵分成多个小块，每次只计算一个小块的Attention，具体是将小块载入共享内存，然后在共享内存中进行计算。这样可以减少对HBM的访问次数，提高计算速度。

#### **4.3 反向计算优化**

从访存上分析，反向计算根本不需要从HBM获取中间变量，而是直接利用sram的分块Q,K进行recompute。因为工程上速度慢一点没有很大影响，但是recompute可以大大节省显存开销，所以可以设计更大的神经网络，最终的网络效果会更好。

### **5. FlashAttention算法实现及测试**

#### **5.1 算法实现**

使用RTX4090-24GB显卡作为实验平台，使用Triton编译器，以python语言实现算子，Jit动态编译将triton算子转化为cuda代码。Triton编译器使得python开发高性能算子成为可能，只需要对分块进行说明，就可以执行高效的分块并行计算。虽然在性能上不一定比得上C++实现，但在开发效率上有很大提升。

```python
# 前向传播内核：FlashAttention 前向
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,           # 输入：q, k, v 和 softmax 缩放因子
    L, Out,                      # 输出：行和 log(sum(exp))，以及最终输出
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,                 # 批量、头数、序列长度
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
```

#### **5.2 实验测试**

首先是以4, 48, 4096, 32作为测试：
可以发现和cuda实现性能上没有什么区别，稍微落后于高性能的Flash-2 cuda实现。

<img src="static/forward.png" alt="forward" style="zoom:33%;" />
backward的实现和flash-2的cuda实现更相近，在部分测试中甚至超过了Flash-2的cuda实现。
<img src="./static/newbackward.png" alt="alt text" style="zoom: 50%;" />

### **6. mytorch使用示例与效果展示**

#### **6.1 以mnist数据集的训练为例**

完整代码请参考`gpu/examples/mnist_test.cpp`

**头文件引入**

```cpp
#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/flatten.h"
#include "optim/sgd.h"
#include "loader/mnist_loader.h"
```

这些头文件引入了mytorch框架的核心组件：

- `tensor.h`: 定义了张量(Tensor)类，是框架的基础数据结构
- 各种神经网络模块：线性层、激活函数、序列容器等
- 优化器：SGD优化器
- 数据加载器：MNIST数据集加载器

**设备设置**

```cpp
Device device = Device::CPU;
if (cuda_err == cudaSuccess && device_count > 0) {
    device = Device::CUDA;
}
```

我们的mytorch框架支持CPU和CUDA设备，这里自动检测并选择可用的设备。

**模型定义**

```cpp
auto model = std::make_shared<nn::Sequential>(
    std::vector<std::shared_ptr<nn::Module>>{
        std::make_shared<nn::Linear>(INPUT_FEATURES, HIDDEN_FEATURES),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(HIDDEN_FEATURES, OUTPUT_CLASSES)
    }
);
model->to(device);
```

这里使用了mytorch框架的`Sequential`容器来构建一个简单的全连接网络：

1. 第一个线性层(784->32)
2. ReLU激活函数
3. 第二个线性层(32->10)

`model->to(device)`将模型参数移动到指定设备(CPU或CUDA)。

**优化器设置**

```cpp
optim::SGD optimizer(model->parameters(), LEARNING_RATE);
```

使用mytorch框架的SGD优化器，传入模型参数和学习率。

**数据加载**

```cpp
MnistLoader::load(MNIST_DATA_PATH, X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu);
auto X_train = X_train_cpu->to(device);
```

使用mytorch的MNIST加载器加载数据，然后使用`to(device)`方法将数据移动到指定设备。

**训练循环**

训练循环展示了mytorch框架的核心API调用：

- 前向传播

```cpp
auto y_pred = model->forward(x_batch);
```

- 损失计算

```cpp
auto loss = mse_loss(y_pred, y_batch);
```

- 反向传播

```cpp
loss->backward();
```

- 参数更新

```cpp
optimizer.step();
```

**辅助函数**

- 准确率计算

```cpp
float calculate_accuracy(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto pred_data = pred->data_cpu();
    auto target_data = target->data_cpu();
    // ... 计算逻辑 ...
}
```

- MSE损失

```cpp
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    auto sum_loss = sq_diff->sum();
    // ... 缩放处理 ...
}
```

#### **6.2 测试结果展示**

以下两张图是CPU跑出来的结果：

<img src="static/cpu4.png" alt="image-20250710200941903" style="zoom:50%;" />

<img src="static/cpu1.png" alt="image-20250710195838715" style="zoom:33%;" />

以下是GPU的结果：

<img src="static/gpu.png" alt="image-20250710201232054" style="zoom:50%;" />

可以看到，CPU和GPU设备均能正常执行。

#### **6.3 并行算法加速情况**

手写数字集（MNIST）的训练速度对比（所测试CPU为6核心）：

| 设备 | 线程数 | 训练时间（s） | 加速比 |
| :--: | :----: | :-----------: | :----: |
| CPU  |   1    |     1200      |  1.0   |
| CPU  |   4    |      390      |  3.08  |
| CPU  |   8    |      260      |  4.62  |
| GPU  |   -    |      30       |   40   |

从上表可以看出，我们的mytorch框架并行效果显著，CPU的加速比达到预期目标。

从中也可以看出，在没有对GPU的CUDA相关代码做很深度的优化的情况下，GPU在机器学习上的潜力远远大于CPU。

#### **6.4 FlashAttention + mytorch**

最后，我们在`cpu/examples/flash_test.cpp`中尝试了我们mytorch架构对高级优化算子的支持。

```cpp
int main() {
    py::scoped_interpreter guard{};  // 只初始化一次解释器
    std::cout << "Python interpreter initialized" << std::endl;

    // 创建输入
    auto Q = Tensor::randn({16, 8, 64, 16},true);
    auto K = Tensor::randn({16, 8, 64, 16}, true);
    auto V = Tensor::randn({16, 8, 64, 16}, true);
    show_tensor(Q);

    // 调用 flash attention 前向
    auto attn_layer = std::make_shared<nn::FlashAttn>(false, 1.0f); 
    printf("atten layer created address: %p\n", attn_layer.get());
    auto o = attn_layer->forward({Q, K, V});
    std::cout << "Output shape: " << o->shape()[0] << ", " << o->shape()[1] << ", " << o->shape()[2] << ", " << o->shape()[3] << std::endl;
    o->backward();  // 触发反向传播
    show_tensor(o);
    //测试反向传播
    auto label = Tensor::randn({64, 8, 128, 64}); // 假设标签与输出形状一致
    auto loss = mse_loss(Q, label);
    loss->backward();  // 触发反向传播

    return 0;  // 解释器由 guard 自动清理
}
```

### **附录：个人报告**

- **陈兴平 22336037**
  - **具体工作**
    - 与刘华壹同学共同完成了整个mytorch框架的设计
    - 实现了大多数tensor的前向传播以及反向传播
    - 参与了Function的实现以及nn的构建
    - 通过omp实现了CPU版本的并行，并写了`linear_test.cpp`测试函数
    - 为了兼容GPU，在CPU代码的基础上重构了部分函数，并增加了to方法等方法
    - 制造并修复了很多bug，最终让项目成功运行
  - **感想**
    - 这次大工程让我深深认识到了框架设计的重要性。同时，在写代码过程中借鉴了许多pytorch的设计和实现思路，让我对pytorch有了更加深刻的理解。
    - 本次项目我手写了可能上千行代码，在代码成功运行的那一刻，非常的有成就感，感谢老师和助教给了我们这样尝试自己、突破自己的机会。
- 

### **参考资料**

- https://zhuanlan.zhihu.com/p/668888063
- https://github.com/pytorch/pytorch
- 