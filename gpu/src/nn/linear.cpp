#include "nn/linear.h"
#include <cmath>
#include <random> // 需要包含random头文件

namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias) : _use_bias(use_bias) {
    // Kaiming He initialization
    float stdv = std::sqrt(2.0f / in_features);

    // --- 修改: 更高效的初始化 ---
    // 1. 在CPU上创建随机数据
    std::vector<float> weight_data_cpu(in_features * out_features);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for(auto& val : weight_data_cpu) {
        val = d(gen) * stdv; // 直接应用缩放
    }

    // 2. 从CPU数据创建Tensor，它初始在CPU上
    // 注意：这里的 to(Device::CPU) 是多余的，但为了清晰可以保留
    _weight = Tensor::create(weight_data_cpu, {in_features, out_features}, true);


    if (_use_bias) {
        // 偏置初始化为0，可以直接在目标设备上创建
        _bias = Tensor::zeros({1, out_features}, true);
    } else {
        _bias = nullptr;
    }
}

// forward, parameters, to 方法保持不变
std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto output = input->matmul(_weight);
    if (_use_bias) {
        // 现在这个add操作将依赖于支持广播的CUDA核
        output = output->add(_bias);
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