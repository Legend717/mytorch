#include "nn/pool.h"
#include "core/function.h"

namespace nn {
std::shared_ptr<Tensor> MaxPool2D::forward(std::shared_ptr<Tensor> input) {
    auto pool_func = std::make_shared<MaxPool2DFunc>(_kernel_size, _stride);
    return pool_func->apply({input});
}
}