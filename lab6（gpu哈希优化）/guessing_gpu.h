
#ifndef GUESSING_GPU_H
#define GUESSING_GPU_H

#include <vector>
#include <string>
#include "PCFG.h"

// 定义一个结构体来打包发送到GPU的数据
struct GpuTaskData {
    // 主机端数据
    std::vector<char> h_prefixes;         // 所有前缀拼接成的字符数组
    std::vector<int> h_prefix_offsets;    // 每个前缀在h_prefixes中的起始偏移
    std::vector<int> h_prefix_lengths;    // 每个前缀的长度

    std::vector<char> h_suffixes;         // 所有后缀拼接成的字符数组
    std::vector<int> h_suffix_offsets;    // 每个后缀集的起始偏移
    std::vector<int> h_suffix_str_offsets; // 单个后缀字符串在h_suffixes中的相对偏移
    std::vector<int> h_suffix_lengths;     // 每个后缀的长度
    std::vector<int> h_num_suffixes_per_pt; // 每个PT有多少个后缀

    int total_guesses_to_generate;
    int num_pts;
};

// 初始化GPU设备
void init_gpu_device();

// GPU生成函数的主接口
void generate_passwords_on_gpu(GpuTaskData& task, std::vector<std::string>& out_guesses);

// 声明一个将由 `guessing_cpu.cpp` 实现的函数
// 这是为了让 `main.cpp` 能够调用 `q.DispatchToGpu`
void DispatchToGpu(PriorityQueue& q, const std::vector<PT>& pt_batch);


#endif // GUESSING_GPU_H