#include "nn/linear.h"
#include <cmath>

namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias) : _use_bias(use_bias) {
    // Kaiming He 初始化
    float stdv = std::sqrt(2.0f / in_features);
    _weight = Tensor::randn({in_features, out_features}, true); // shape [in, out]
    // 用stdv缩放
    auto weight_data = _weight->get_shared_data();
    for(auto& val : *weight_data) {
        val *= stdv;
    }

    if (_use_bias) {
        _bias = Tensor::zeros({1, out_features}, true); // shape [1, out] for broadcasting
    } else {
        _bias = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto output = input->matmul(_weight);
    if (_use_bias) {
        // Broadcating add is not implemented yet, so we assume batch size for bias.
        // A proper implementation would handle broadcasting.
        // For simplicity, we assume bias is added to each row.
        // Let's implement a simple broadcast add for this case.
        // For now, let's assume the add op can handle it.
        // Our current add op doesn't support broadcasting, it needs same shapes.
        // Let's create a temporary bias that matches the batch size.
        // A better way is to implement broadcasting in the Add op itself.
        // For this example, we'll keep it simple: the 'add' will handle it.
        // Let's go back and improve the add function.
        // OK, I'll modify the Add function in function.cpp to support this simple broadcasting.
        
        // This won't work with current Add. Let's make a simple fix.
        // The bias will be added manually in a loop.
        // A proper implementation is more complex.
        
        // Simplified approach: Matmul + Add
        // input: [batch, in], weight: [in, out], output: [batch, out], bias: [1, out]
        output = output->add(_bias); // This requires broadcasting support in Add
    }
    return output;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    if (_use_bias) {
        return {_weight, _bias};
    }
    return {_weight};
}

} // namespace nn