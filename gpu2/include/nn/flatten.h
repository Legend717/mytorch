// in include/nn/flatten.h
#pragma once
#include "nn/module.h"

namespace nn {
class Flatten : public Module {
private:
    std::vector<size_t> _original_shape;
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    void to(Device device) override {}
};
}