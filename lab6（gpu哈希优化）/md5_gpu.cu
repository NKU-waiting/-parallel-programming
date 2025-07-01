#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <iostream>

// 定义MD5相关的常量，这些将被拷贝到GPU的常量内存中
// __constant__ memory is cached and provides high-bandwidth access
// when all threads in a warp access the same location.
__constant__ unsigned int S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__constant__ unsigned int K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

// 辅助函数，在设备端（GPU）执行
// 使用 __device__ 关键字
__device__ inline unsigned int rotate_left(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

// MD5核心计算Kernel
// 每个线程负责计算一个输入字符串的MD5
__global__ void md5_kernel(
    const char* d_inputs,         // 所有输入字符串拼接成的大字符数组
    const int* d_offsets,         // 每个字符串在d_inputs中的起始偏移
    const int* d_lengths,         // 每个字符串的长度
    unsigned int* d_digests,      // 输出的摘要，每个摘要占4个int
    int num_inputs) 
{
    // 计算当前线程的全局ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，防止越界访问
    if (tid >= num_inputs) {
        return;
    }

    // 1. 初始化状态变量 (H0-H3)
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xefcdab89;
    unsigned int h2 = 0x98badcfe;
    unsigned int h3 = 0x10325476;

    // 2. 准备消息块并进行Padding
    int len = d_lengths[tid];
    const char* input_ptr = d_inputs + d_offsets[tid];
    
    // 计算padding后的总长度（以512位块为单位）
    int num_blocks = (len + 8) / 64 + 1;
    int padded_len = num_blocks * 64;

    // 使用栈上内存（或__shared__ memory）来处理消息块，速度快
    // 注意：栈大小有限，如果密码很长，这里需要更复杂的处理
    // 但对于典型密码长度，这是最高效的方式
    unsigned char M_padded[256]; // 假设密码最长不会导致padding后超过256字节
    if (padded_len > 256) {
        // 对于超长密码可以放弃处理或采用动态分配（不推荐在kernel中）
        // 在这里可以设置一个错误码
        return;
    }

    // 拷贝原始消息
    for(int i = 0; i < len; ++i) M_padded[i] = input_ptr[i];
    
    // 添加 0x80
    M_padded[len] = 0x80;

    // 清零剩余部分
    for(int i = len + 1; i < padded_len; ++i) M_padded[i] = 0;

    // 添加原始长度（64位小端）
    unsigned long long bit_len = (unsigned long long)len * 8;
    for(int i = 0; i < 8; ++i) {
        M_padded[padded_len - 8 + i] = (unsigned char)(bit_len >> (i * 8));
    }

    // 3. 逐块处理消息
    for (int i = 0; i < num_blocks; ++i) {
        unsigned int w[16];
        const unsigned char* block_ptr = M_padded + i * 64;

        // 将64字节块解码为16个32位字 (小端)
        for (int j = 0; j < 16; ++j) {
            w[j] = (unsigned int) block_ptr[j*4] |
                   ((unsigned int) block_ptr[j*4+1] << 8) |
                   ((unsigned int) block_ptr[j*4+2] << 16) |
                   ((unsigned int) block_ptr[j*4+3] << 24);
        }

        // 初始化此块的哈希值
        unsigned int a = h0;
        unsigned int b = h1;
        unsigned int c = h2;
        unsigned int d = h3;

        // 主循环 (64轮)
        for (int j = 0; j < 64; ++j) {
            unsigned int f, g;
            if (j < 16) {
                f = (b & c) | ((~b) & d);
                g = j;
            } else if (j < 32) {
                f = (d & b) | ((~d) & c);
                g = (1 * j + 1) % 16;
            } else if (j < 48) {
                f = b ^ c ^ d;
                g = (3 * j + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7 * j) % 16;
            }

            unsigned int temp = d;
            d = c;
            c = b;
            b = b + rotate_left(a + f + K[j] + w[g], S[j]);
            a = temp;
        }

        // 更新哈希值
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }

    // 4. 将最终结果写回全局内存
    // 每个digest占4个int
    d_digests[tid * 4 + 0] = h0;
    d_digests[tid * 4 + 1] = h1;
    d_digests[tid * 4 + 2] = h2;
    d_digests[tid * 4 + 3] = h3;
}

// C-style wrapper for C++ calls
// 这是提供给C++代码调用的主接口
extern "C" void md5_hash_on_gpu(
    const char* h_inputs,
    const int* h_offsets,
    const int* h_lengths,
    unsigned int* h_digests,
    int num_inputs) 
{
    // 分配GPU内存
    char* d_inputs;
    int* d_offsets;
    int* d_lengths;
    unsigned int* d_digests;

    size_t inputs_size = h_offsets[num_inputs - 1] + h_lengths[num_inputs - 1];
    
    cudaMalloc(&d_inputs, inputs_size);
    cudaMalloc(&d_offsets, num_inputs * sizeof(int));
    cudaMalloc(&d_lengths, num_inputs * sizeof(int));
    cudaMalloc(&d_digests, num_inputs * 4 * sizeof(unsigned int));

    // 从主机拷贝数据到设备
    cudaMemcpy(d_inputs, h_inputs, inputs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, num_inputs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, num_inputs * sizeof(int), cudaMemcpyHostToDevice);

    // 设置Kernel启动配置
    int threads_per_block = 256;
    int blocks_per_grid = (num_inputs + threads_per_block - 1) / threads_per_block;

    // 启动Kernel
    md5_kernel<<<blocks_per_grid, threads_per_block>>>(d_inputs, d_offsets, d_lengths, d_digests, num_inputs);
    
    // 检查Kernel启动是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // 将设备端的结果拷贝回主机
    cudaMemcpy(h_digests, d_digests, num_inputs * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // 释放GPU内存
    cudaFree(d_inputs);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_digests);
}