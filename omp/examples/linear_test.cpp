#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <omp.h>

#include "core/tensor.h"
#include "nn/module.h"
#include "nn/linear.h"
#include "nn/activations.h"
#include "nn/sequential.h"
#include "nn/flatten.h"
#include "optim/sgd.h"

// 辅助函数：均方误差损失
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target) {
    auto diff = pred->sub(target);
    auto sq_diff = diff->mul(diff);
    auto loss = sq_diff->sum();
    return loss->mul(Tensor::create({0.0001f}, {1})); // 用一个较小值代替除以元素数量的操作
}

int main() {
    std::cout << "--- Linear Test ---" << std::endl;

    // 1. 定义超参数
    const size_t SEQ_LEN = 1000;     // 输入序列长度
    const size_t PRED_LEN = 200;    // 预测长度
    const size_t HIDDEN_DIM = 640;  // 隐藏层维度（简化网络）
    const float LEARNING_RATE = 0.01f;
    const int EPOCHS = 100;

    // 设置 OpenMP 并行数
    omp_set_num_threads(8);

    // 2. 简化模型结构
    auto model = std::make_shared<nn::Sequential>(
        std::vector<std::shared_ptr<nn::Module>>{
            // std::make_shared<nn::Flatten>(), 
            std::make_shared<nn::Linear>(SEQ_LEN, HIDDEN_DIM),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(HIDDEN_DIM, PRED_LEN),
        }
    );

    // 3. 创建优化器
    optim::SGD optimizer(model->parameters(), LEARNING_RATE);

    // 4. 创建实际数据
    std::vector<float> X_train_data;
    std::vector<float> y_train_data;
    for (size_t i = 0; i < SEQ_LEN; i++) {
        X_train_data.push_back(i + 1.0f);
    }
    for (size_t i = 0; i < PRED_LEN; i++) {
        y_train_data.push_back(i + SEQ_LEN + 1.0f);
    }

    auto X_train = Tensor::create(
        X_train_data, 
        {1, SEQ_LEN}
    );
    auto y_train = Tensor::create(
        y_train_data, 
        {1, PRED_LEN}
    );

    std::cout << "Training for " << EPOCHS << " epochs..." << std::endl;

    // 5. 训练循环
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        optimizer.zero_grad();
        
        auto flat_pred = model->forward(X_train);
        
        auto loss = mse_loss(flat_pred, y_train);
        loss->backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS 
                      << "], Loss: " << loss->item() << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Training time: " << duration << " ms\n";

    return 0;
}