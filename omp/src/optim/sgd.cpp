#include "optim/sgd.h"
#include <iostream>

namespace optim {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float lr)
    : _params(params), _lr(lr) {}

void SGD::step() {
    for (auto& p : _params) {
        if (p->grad()) {
            auto p_data = p->get_shared_data();
            auto g_data = p->grad()->get_shared_data();   
            #pragma omp parallel for schedule(guided)  
            for (size_t i = 0; i < p_data->size(); ++i) {
                (*p_data)[i] -= _lr * (*g_data)[i];
            }
            //std::cout<<p_data->size()<<std::endl;
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : _params) {
        p->set_grad(nullptr);
    }
}

}