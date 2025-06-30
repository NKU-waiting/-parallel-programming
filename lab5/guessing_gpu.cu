// --- START OF FILE guessing_gpu.cu ---

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "guessing_gpu.h"

// 定义最大密码长度，用于在GPU上分配固定大小的缓冲区
#define MAX_PASSWORD_LENGTH 64
// 每个Block的线程数
#define THREADS_PER_BLOCK 256

// 错误检查宏
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void init_gpu_device() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-enabled devices found.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(0); // Select GPU 0
    printf("GPU device initialized.\n");
}


__global__ void generation_kernel(
    char* d_output_guesses,
    const char* d_prefixes,
    const int* d_prefix_offsets,
    const int* d_prefix_lengths,
    const char* d_suffixes,
    const int* d_suffix_offsets,
    const int* d_suffix_str_offsets,
    const int* d_suffix_lengths,
    const int* d_num_suffixes_per_pt,
    const int* d_pt_start_indices,
    int num_pts,
    int total_guesses_to_generate) // Pass total guesses as an argument
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // FIX #1: Add a boundary check (guard clause).
    // This is the most critical fix. It stops any extra threads that are
    // launched beyond the actual number of work items.
    if (global_tid >= total_guesses_to_generate) {
        return;
    }

    // FIX #2: Use a simpler, safer loop to find which PT this thread belongs to.
    int pt_idx = 0;
    // The last PT's start index is not in d_pt_start_indices, so we loop up to num_pts - 1
    for (int i = 1; i < num_pts; ++i) {
        if (global_tid < d_pt_start_indices[i]) {
            break; // We found the right PT in the previous iteration
        }
        pt_idx = i;
    }

    // Now pt_idx correctly identifies the Pre-Terminal (PT) for this thread.

    // Calculate this thread's local suffix index within its PT
    int suffix_local_idx = global_tid - d_pt_start_indices[pt_idx];

    // Get prefix information
    int prefix_offset = d_prefix_offsets[pt_idx];
    int prefix_len = d_prefix_lengths[pt_idx];

    // Get suffix information
    // 1. Find the start of the suffix set for this PT
    int suffix_set_start_offset = d_suffix_offsets[pt_idx];
    // 2. Find the specific suffix string within that set
    int suffix_str_start_offset = d_suffix_str_offsets[suffix_set_start_offset + suffix_local_idx];
    int suffix_len = d_suffix_lengths[suffix_set_start_offset + suffix_local_idx];

    if (prefix_len + suffix_len >= MAX_PASSWORD_LENGTH) return; // Prevent buffer overflow

    // Get the pointer to where this thread should write its output
    char* output_ptr = d_output_guesses + (size_t)global_tid * MAX_PASSWORD_LENGTH;

    // Copy prefix
    for (int i = 0; i < prefix_len; ++i) {
        output_ptr[i] = d_prefixes[prefix_offset + i];
    }
    
    // Copy suffix
    for (int i = 0; i < suffix_len; ++i) {
        output_ptr[prefix_len + i] = d_suffixes[suffix_str_start_offset + i];
    }

    // Add string terminator
    output_ptr[prefix_len + suffix_len] = '\0';
}


void generate_passwords_on_gpu(GpuTaskData& task, std::vector<std::string>& out_guesses) {
    if (task.total_guesses_to_generate == 0) return;

    // 计算每个PT的起始索引，用于kernel内的查找
    std::vector<int> h_pt_start_indices(task.num_pts);
    h_pt_start_indices[0] = 0;
    for (int i = 1; i < task.num_pts; ++i) {
        h_pt_start_indices[i] = h_pt_start_indices[i-1] + task.h_num_suffixes_per_pt[i-1];
    }

    // --- 1. 分配设备内存 ---
    char* d_output_guesses;
    char* d_prefixes;
    int* d_prefix_offsets;
    int* d_prefix_lengths;
    char* d_suffixes;
    int* d_suffix_offsets;
    int* d_suffix_str_offsets;
    int* d_suffix_lengths;
    int* d_num_suffixes_per_pt;
    int* d_pt_start_indices;

    CUDA_CHECK(cudaMalloc(&d_output_guesses, (size_t)task.total_guesses_to_generate * MAX_PASSWORD_LENGTH));
    CUDA_CHECK(cudaMalloc(&d_prefixes, task.h_prefixes.size()));
    CUDA_CHECK(cudaMalloc(&d_prefix_offsets, task.h_prefix_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_lengths, task.h_prefix_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_suffixes, task.h_suffixes.size()));
    CUDA_CHECK(cudaMalloc(&d_suffix_offsets, task.h_suffix_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_suffix_str_offsets, task.h_suffix_str_offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_suffix_lengths, task.h_suffix_lengths.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_suffixes_per_pt, task.h_num_suffixes_per_pt.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pt_start_indices, h_pt_start_indices.size() * sizeof(int)));

    // --- 2. 拷贝数据到设备 ---
    CUDA_CHECK(cudaMemcpy(d_prefixes, task.h_prefixes.data(), task.h_prefixes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefix_offsets, task.h_prefix_offsets.data(), task.h_prefix_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prefix_lengths, task.h_prefix_lengths.data(), task.h_prefix_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_suffixes, task.h_suffixes.data(), task.h_suffixes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_suffix_offsets, task.h_suffix_offsets.data(), task.h_suffix_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_suffix_str_offsets, task.h_suffix_str_offsets.data(), task.h_suffix_str_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_suffix_lengths, task.h_suffix_lengths.data(), task.h_suffix_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_suffixes_per_pt, task.h_num_suffixes_per_pt.data(), task.h_num_suffixes_per_pt.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt_start_indices, h_pt_start_indices.data(), h_pt_start_indices.size() * sizeof(int), cudaMemcpyHostToDevice));


    // --- 3. 启动内核 ---
    int grid_size = (task.total_guesses_to_generate + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // THIS IS THE LINE THAT WAS FIXED.
    // We now pass all the required arguments to the kernel.
    generation_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_output_guesses, d_prefixes, d_prefix_offsets, d_prefix_lengths,
        d_suffixes, d_suffix_offsets, d_suffix_str_offsets, d_suffix_lengths,
        d_num_suffixes_per_pt, d_pt_start_indices, task.num_pts,
        task.total_guesses_to_generate); 

    CUDA_CHECK(cudaGetLastError());

    // --- 4. 拷贝结果回主机 ---
    std::vector<char> h_output_guesses( (size_t)task.total_guesses_to_generate * MAX_PASSWORD_LENGTH );
    CUDA_CHECK(cudaMemcpy(h_output_guesses.data(), d_output_guesses, (size_t)task.total_guesses_to_generate * MAX_PASSWORD_LENGTH, cudaMemcpyDeviceToHost));
    
    // --- 5. 释放设备内存 ---
    CUDA_CHECK(cudaFree(d_output_guesses));
    CUDA_CHECK(cudaFree(d_prefixes));
    CUDA_CHECK(cudaFree(d_prefix_offsets));
    CUDA_CHECK(cudaFree(d_prefix_lengths));
    CUDA_CHECK(cudaFree(d_suffixes));
    CUDA_CHECK(cudaFree(d_suffix_offsets));
    CUDA_CHECK(cudaFree(d_suffix_str_offsets));
    CUDA_CHECK(cudaFree(d_suffix_lengths));
    CUDA_CHECK(cudaFree(d_num_suffixes_per_pt));
    CUDA_CHECK(cudaFree(d_pt_start_indices));

    // --- 6. 将扁平化的char数组转换回std::vector<std::string> ---
    out_guesses.reserve(out_guesses.size() + task.total_guesses_to_generate);
    for (int i = 0; i < task.total_guesses_to_generate; ++i) {
        out_guesses.emplace_back(&h_output_guesses[ (size_t)i * MAX_PASSWORD_LENGTH ]);
    }
}