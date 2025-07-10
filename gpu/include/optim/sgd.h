#pragma once

#include "core/tensor.h"
#include <vector>
#include <memory>

namespace optim {

class SGD {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr = 0.01f);

    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> _params;
    float _lr;
};

} // namespace optim