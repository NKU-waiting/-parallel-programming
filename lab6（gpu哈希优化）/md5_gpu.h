#ifndef MD5_GPU_H
#define MD5_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

// 声明提供给C++调用的GPU哈希函数
// 参数：
// h_inputs: 主机端的所有输入字符串拼接成的大字符数组
// h_offsets: 主机端的每个字符串在h_inputs中的起始偏移
// h_lengths: 主机端的每个字符串的长度
// h_digests: 主机端的输出缓冲区，用于存放计算好的摘要
// num_inputs: 输入字符串的总数
void md5_hash_on_gpu(
    const char* h_inputs,
    const int* h_offsets,
    const int* h_lengths,
    unsigned int* h_digests,
    int num_inputs
);

#ifdef __cplusplus
}
#endif

#endif // MD5_GPU_H