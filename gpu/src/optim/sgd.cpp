#include "optim/sgd.h"
#include "core/tensor.h" // 包含Tensor头文件以使用Device
#include <vector>

void sgd_update_cuda(float* params, const float* grads, float lr, size_t n);

namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr)
    : _params(params), _lr(lr) {}

void SGD::step() {
    for (auto& p : _params) {
        if (p->grad()) {
            // 检查张量所在的设备
            if (p->device() == Device::CUDA) {
                // 在GPU上直接更新
                sgd_update_cuda(
                    static_cast<float*>(p->mutable_data_ptr()),
                    static_cast<const float*>(p->grad()->data_ptr()),
                    _lr,
                    p->size()
                );
            } else {
                // 保留原始的CPU更新逻辑
                // 注意：这里的 get_shared_data() 假设是返回一个CPU上的 std::vector<float>&
                auto& p_data = *static_cast<std::vector<float>*>(p->mutable_data_ptr());
                const auto& g_data = p->grad()->data_cpu(); // 梯度数据也需在CPU上
                for (size_t i = 0; i < p_data.size(); ++i) {
                    p_data[i] -= _lr * g_data[i];
                }
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : _params) {
        p->set_grad(nullptr);
    }
}

} // namespace optim