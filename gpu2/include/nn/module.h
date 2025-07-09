#pragma once

#include "core/tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    // 前向传播
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
    virtual std::shared_ptr<Tensor> forward(std::vector<std::shared_ptr<Tensor>> inputs){
        throw std::runtime_error("该Module子类没有实现forward(std::vector<std::shared_ptr<Tensor>> inputs)方法");
    }
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

    // 使模块可像函数一样调用
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) {
        return forward(input);
    }

    virtual void to(Device device) = 0;
    // void to(Device device) {
    //     std::cerr << "to方法未实现" << std::endl;
    // }
};

} // namespace nn