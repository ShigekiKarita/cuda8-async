#include <future>
#include <thread>
#include <iostream>
#include "cuda_check.hpp"

const int N = 1 << 20;

__global__ void kernel(float *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

std::string launch_kernel() {
    float *data;
    CUDA_CHECK(cudaMalloc(&data, N * sizeof(float)));
    kernel<<<1, 64>>>(data, N);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(0));
    return "ok";
}


int main() {
    const int num_threads = 8;
    std::future<std::string> fs[num_threads];

    for (auto& f: fs) {
        try {
            std::packaged_task<std::string()> task(launch_kernel);
            f = task.get_future();
            std::thread(std::move(task)).detach();
        } catch(std::exception& e) {
            std::cerr << "Error creating thread: " << e.what() << std::endl;
        }
    }

    for (auto& f: fs) {
        try {
            std::cout << f.get() << std::endl;
        } catch(std::exception& e) {
            std::cerr << "Error joining thread: " << e.what() << std::endl;
        }
    }

    cudaDeviceReset();
}