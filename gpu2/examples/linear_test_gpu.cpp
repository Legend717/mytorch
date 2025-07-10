#include <iostream>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
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
    
    // 创建标量张量，并确保它和pred在同一设备上
    auto scalar = Tensor::create({0.0001f}, {1})->to(pred->device());
    return loss->mul(scalar);
}

int main() {
    // 0. 设置设备
    Device device = Device::CUDA;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device = Device::CUDA;
        std::cout << "--- MiniTorchGPU: Using CUDA device ---" << std::endl;
    } else {
        device = Device::CPU;
        std::cout << "--- MiniTorchGPU: Using CPU device ---" <<std::endl;
    }
    // device = Device::CPU;

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

    model->to(device);
    std::cout << "模型成功加载到设备" << std::endl;

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

    auto X_train_cpu = Tensor::create(
        X_train_data, 
        {1, SEQ_LEN}
    );
    auto y_train_cpu = Tensor::create(
        y_train_data, 
        {1, PRED_LEN}
    );

    auto X_train = X_train_cpu->to(device);
    auto y_train = y_train_cpu->to(device);
    std::cout << "数据成功加载到设备" << std::endl;

    std::cout << "Training for " << EPOCHS << " epochs..." << std::endl;

    // 5. 训练循环
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        optimizer.zero_grad();
        
        auto flat_pred = model->forward(X_train);

       // std::cout << "forward finish!\n";
        
        auto loss = mse_loss(flat_pred, y_train);

        // std::cout << "loss finish\n";
        // std::cout << (int)loss->device() << '\n';
        loss->backward();
        // std::cout << "backward finish\n";
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