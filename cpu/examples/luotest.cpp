#include <iostream>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "core/function.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/conv.h"
#include "nn/pool.h"
#include "nn/flatten.h"
#include "nn/flash-attn.h"
#include "optim/sgd.h"

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    auto loss = sq_diff->sum();
    // 实际应用中会除以元素数量，但为了梯度稳定，我们用一个小的系数
    return loss->mul(Tensor::create({0.001f}, {1}));
}

// print 10 ele of tensor
void show_tensor(const std::shared_ptr<Tensor>& t) {
    const auto& data = t->data();
    std::cout << "Tensor data: ";
    for (size_t i = 0; i < std::min(data.size(), size_t(10)); ++i) {
        std::cout << data[i] << " ";
    }
    if (data.size() > 10) {
        std::cout << "...";
    }
    std::cout << std::endl;
}

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
    // auto label = Tensor::randn({64, 8, 128, 64}); // 假设标签与输出形状一致
    // auto loss = mse_loss(Q, label);
    // loss->backward();  // 触发反向传播

    return 0;  // 解释器由 guard 自动清理
}


