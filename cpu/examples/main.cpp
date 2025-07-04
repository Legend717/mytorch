#include <iostream>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/conv.h"
#include "nn/pool.h"
#include "nn/flatten.h"
#include "optim/sgd.h"

// 辅助函数：均方误差损失
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    auto loss = sq_diff->sum();
    // 实际应用中会除以元素数量，但为了梯度稳定，我们用一个小的系数
    return loss->mul(Tensor::create({0.001f}, {1}));
}


int main() {
    std::cout << "--- MiniTorchCPU CNN Training Example ---" << std::endl;

    // 1. 定义超参数
    const size_t BATCH_SIZE = 4;
    const size_t IN_CHANNELS = 1; // 灰度图
    const size_t IMG_HEIGHT = 28;
    const size_t IMG_WIDTH = 28;
    const size_t NUM_CLASSES = 10;
    const float LEARNING_RATE = 0.001f;
    const int EPOCHS = 50;

    // 2. 定义CNN模型 (类LeNet结构)
    auto model = std::make_shared<nn::Sequential>(
        std::vector<std::shared_ptr<nn::Module>>{
            // 输入: [B, 1, 28, 28]
            std::make_shared<nn::Conv2D>(IN_CHANNELS, 6, 5), // -> [B, 6, 24, 24]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),          // -> [B, 6, 12, 12]
            
            std::make_shared<nn::Conv2D>(6, 16, 5),         // -> [B, 16, 8, 8]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::MaxPool2D>(2, 2),          // -> [B, 16, 4, 4]

            std::make_shared<nn::Flatten>(),                // -> [B, 16*4*4] = [B, 256]

            std::make_shared<nn::Linear>(16 * 4 * 4, 120),  // -> [B, 120]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(120, 84),          // -> [B, 84]
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(84, NUM_CLASSES)   // -> [B, 10]
        }
    );

    // 3. 创建优化器
    optim::SGD optimizer(model->parameters(), LEARNING_RATE);

    // 4. 创建虚拟数据
    auto X_train = Tensor::randn({BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH});
    auto y_train = Tensor::randn({BATCH_SIZE, NUM_CLASSES});

    std::cout << "Training for " << EPOCHS << " epochs..." << std::endl;

    // 5. 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        optimizer.zero_grad();
        auto y_pred = model->forward(X_train);
        auto loss = mse_loss(y_pred, y_train);
        loss->backward();
        optimizer.step();

        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "], Loss: " << loss->item() << std::endl;
        }
    }

    std::cout << "--- Training Finished ---" << std::endl;

    return 0;
}