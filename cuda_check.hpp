#pragma once
#include <cuda.h>
#include <sstream>


inline void cuda_check(const char* expr, cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::stringstream what;
        what << cudaGetErrorString(code) << "\n\t" << expr
             << " at " << file << "." << line << std::endl;
        throw std::runtime_error(what.str());
    }
}

#define CUDA_CHECK(ans) { cuda_check("" #ans, (ans), __FILE__, __LINE__); }
