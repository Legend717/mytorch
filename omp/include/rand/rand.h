#pragma once

#include<random>

namespace MyRand{
    inline std::mt19937 global_rand_generater;
    
    inline void set_global_random_seed(unsigned int seed) {
        global_rand_generater.seed(seed);
    }   
}