#include "nn/activations.h"
#include "core/function.h"

namespace nn {

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    return input->relu();
}


} // namespace nn