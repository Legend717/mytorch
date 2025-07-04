// Replace the entire content of this file with the corrected version below.

#include "nn/flatten.h"
#include "core/tensor.h"
#include "core/function.h"

namespace nn {

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    // Save the original shape for the backward pass of a potential subsequent layer if needed,
    // though for flatten, the autograd engine handles this implicitly via the ReshapeFunc.
    _original_shape = input->shape(); 

    // Calculate the new shape
    size_t batch_size = _original_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < _original_shape.size(); ++i) {
        flattened_size *= _original_shape[i];
    }
    
    // Use the now-correct Tensor::reshape method
    return input->reshape({batch_size, flattened_size});
}

} // namespace nn