#pragma once

#include "nn/module.h"

namespace nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
    void to(Device device) override;

private:
    std::shared_ptr<Tensor> _weight;
    std::shared_ptr<Tensor> _bias; // 可能为nullptr
    bool _use_bias;
};

} // namespace nn