#pragma once

#include "core/tensor.h"
#include <vector>
#include <memory>

namespace nn {

class Module {
public:
    virtual ~Module() = default;

    // 前向传播
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;

    // 获取所有可训练参数
    virtual std::vector<std::shared_ptr<Tensor>> parameters() {
        return {};
    }

    // 将所有参数梯度清零
    void zero_grad() {
        for (auto& p : parameters()) {
            if (p->grad()) {
                p->set_grad(nullptr); // 简单起见，直接设为nullptr
            }
        }
    }
    
    virtual void to(Device device) = 0;

    // 使模块可像函数一样调用
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) {
        return forward(input);
    }
};

} // namespace nn