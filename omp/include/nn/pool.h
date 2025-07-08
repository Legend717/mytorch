// in include/nn/pool.h
#pragma once
#include "nn/module.h"

namespace nn {
class MaxPool2D : public Module {
private:
    size_t _kernel_size;
    size_t _stride;
public:
    MaxPool2D(size_t kernel_size, size_t stride) : _kernel_size(kernel_size), _stride(stride) {}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
}