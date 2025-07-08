#pragma once

#include "nn/module.h"
#include <vector>
#include <memory>

namespace nn {

class Sequential : public Module {
public:
    Sequential(std::vector<std::shared_ptr<Module>> layers);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;

private:
    std::vector<std::shared_ptr<Module>> _layers;
};

} // namespace nn