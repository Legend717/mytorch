#include "optim/sgd.h"
#include "core/tensor.h" // 包含Tensor头文件以使用Device
#include <vector>

// 前向声明CUDA更新函数，它的实现将在一个新的 .cu 文件或 function.cu 中
void sgd_update_cuda(float* params, const float* grads, float lr, size_t n);

namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr)
    : _params(params), _lr(lr) {}

void SGD::step() {
    for (auto& p : _params) {
        if (p->grad()) {
            // --- 关键修改 ---
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
                // 如果您的实现不同，请保留您原始的CPU逻辑
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
        // 这个逻辑是正确的，无需修改
        p->set_grad(nullptr);
    }
}

} // namespace optim