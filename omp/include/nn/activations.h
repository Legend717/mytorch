#pragma once

#include "nn/module.h"

namespace nn {

class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

} // namespace nn