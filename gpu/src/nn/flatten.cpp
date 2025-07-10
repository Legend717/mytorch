#include "nn/flatten.h"
#include "core/tensor.h"
#include "core/function.h"

namespace nn {

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    _original_shape = input->shape(); 

    // 计算输出的形状
    size_t batch_size = _original_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < _original_shape.size(); ++i) {
        flattened_size *= _original_shape[i];
    }
    
    // 调用Reshape函数，将输入张量reshape成输出张量的形状
    return input->reshape({batch_size, flattened_size});
}

} // namespace nn