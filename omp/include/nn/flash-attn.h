#pragma once

#include "nn/module.h"

namespace nn {

class FlashAttn : public Module {
private:
    bool _causal;  //other parameters can be fetched from inputs(Tensor class)
    float _sm_scale;

public:
    FlashAttn(bool causal = false, float sm_scale = 1.0f);
    std::shared_ptr<Tensor> forward(std::vector<std::shared_ptr<Tensor>> input) override;
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
    std::vector<std::shared_ptr<Tensor>> parameters() override;
};

} // namespace nn