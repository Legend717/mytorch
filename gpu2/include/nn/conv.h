#pragma once

#include "nn/module.h"

namespace nn {

class Conv2D : public Module {
private:
    std::shared_ptr<Tensor> _weight;
    size_t _stride;
    size_t _padding;

public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(Device device) override {}
};

} // namespace nn