#include "nn/linear.h"
#include <cmath>

namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias) : _use_bias(use_bias) {
    // Kaiming He 初始化
    float stdv = std::sqrt(2.0f / in_features);
    std::vector<float> weight_data_cpu(in_features * out_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for(auto& val : weight_data_cpu) {
        val = d(gen) * stdv; // 直接应用缩放
    }
    _weight = Tensor::create(weight_data_cpu, {in_features, out_features}, true);

    if (_use_bias) {
        _bias = Tensor::zeros({1, out_features}, true); // shape [1, out] for broadcasting
    } else {
        _bias = nullptr;
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto output = input->matmul(_weight);
    if (_use_bias) {
        output = output->add(_bias); // 这里用到了广播加法
    }
    return output;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    if (_use_bias) {
        return {_weight, _bias};
    }
    return {_weight};
}

void Linear::to(Device device) {
    if (_weight) _weight = _weight->to(device);
    if (_bias) _bias = _bias->to(device);
}

} // namespace nn