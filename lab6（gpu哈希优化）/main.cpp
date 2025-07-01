// 引入必要的头文件
#include "PCFG.h"
#include "guessing_gpu.h" // 引入GPU密码生成接口
#include "md5_gpu.h"      // 新增：引入GPU MD5哈希接口
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;

// 全局计时器 (Global Timers for Instrumentation)
// ==========================================================
double time_cpu_gen = 0.0;
double time_gpu_prepare = 0.0;
double time_gpu_compute = 0.0;
double time_queue_update = 0.0;
// ==========================================================


// --- 进阶要求相关参数 ---
// 目标：凑够约10万个猜测再送给GPU进行密码生成
const int GPU_BATCH_TARGET_GUESSES = 100000;
// 小于此阈值的PT由CPU处理，避免GPU开销
const int CPU_GENERATION_THRESHOLD = 4096;


// 编译指令建议:
// 1. nvcc -c -O3 -arch=sm_XX guessing_gpu.cu -o guessing_gpu.o
// 2. nvcc -c -O3 -arch=sm_XX md5_gpu.cu -o md5_gpu.o
// 3. g++ -c -O3 main.cpp -o main.o
// 4. g++ -c -O3 train.cpp -o train.o
// 5. g++ -c -O3 guessing_cpu.cpp -o guessing_cpu.o
// 6. g++ main.o train.o guessing_cpu.o guessing_gpu.o md5_gpu.o -o main -O3 -L/usr/local/cuda/lib64 -lcudart
// (注意: sm_XX 需替换为你的GPU架构, 如 sm_75)

int main()
{
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    
    // 初始化GPU设备
    init_gpu_device();

    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "Model training and queue initialization finished." << endl;

    auto start = system_clock::now();
    long long history = 0;
    
    // 用于暂存准备发往GPU进行密码生成的PT批次
    vector<PT> gpu_batch;
    int guesses_in_gpu_batch = 0;

    while (!q.priority.empty())
    {
        // 1. 从优先队列顶部取出一个PT
        PT current_pt = *q.priority.begin(); 
        
        // 2. 根据该PT生成新的PT并更新优先队列 (插桩)
        auto start_q_update = system_clock::now();
        q.UpdateQueueFromTop();
        auto end_q_update = system_clock::now();
        time_queue_update += duration_cast<microseconds>(end_q_update - start_q_update).count() / 1e6;

        // 3. 动态负载均衡决策 (密码生成阶段)
        int last_seg_idx = current_pt.content.size() - 1;
        int num_guesses_to_gen = (last_seg_idx >= 0 && !current_pt.max_indices.empty()) ? current_pt.max_indices[last_seg_idx] : 0;
        
        if (num_guesses_to_gen == 0) continue; // 跳过空任务

        // 如果任务太小，直接用CPU处理
        if (num_guesses_to_gen < CPU_GENERATION_THRESHOLD) {
            auto start_cpu = system_clock::now();
            q.Generate_CPU(current_pt);
            auto end_cpu = system_clock::now();
            time_cpu_gen += duration_cast<microseconds>(end_cpu - start_cpu).count() / 1e6;
        } else {
            // 否则，将其加入GPU批处理任务
            gpu_batch.push_back(current_pt);
            guesses_in_gpu_batch += num_guesses_to_gen;
        }

        // 4. 检查GPU批次是否达到目标大小，若达到则分发
        if (guesses_in_gpu_batch >= GPU_BATCH_TARGET_GUESSES || (q.priority.empty() && !gpu_batch.empty())) {
            q.DispatchToGpu(gpu_batch);
            gpu_batch.clear();
            guesses_in_gpu_batch = 0;
        }
        
        // 5. 内存管理和哈希验证阶段
        // 当累积的密码数量达到阈值时，进行一次批量的哈希验证
        if (q.guesses.size() >= 1000000)
        {
            cout << "Guesses generated so far: " << history + q.guesses.size() << ". Now hashing..." << endl;

            auto start_hash = system_clock::now();
            
            const int total_guesses_to_hash = q.guesses.size();
            if (total_guesses_to_hash > 0) 
            {
                // ==========================================================
                //  核心修改：调用GPU进行MD5哈希验证
                // ==========================================================

                // 1. 准备数据以适应GPU接口（扁平化）
                std::vector<char> h_inputs_flat;
                std::vector<int> h_offsets;
                std::vector<int> h_lengths;
                // 准备好接收结果的缓冲区, 每个MD5摘要是128位，即4个32位无符号整数
                std::vector<unsigned int> h_digests(total_guesses_to_hash * 4); 

                h_offsets.reserve(total_guesses_to_hash);
                h_lengths.reserve(total_guesses_to_hash);

                for (const auto& guess : q.guesses) {
                    h_offsets.push_back(h_inputs_flat.size());
                    h_lengths.push_back(guess.length());
                    h_inputs_flat.insert(h_inputs_flat.end(), guess.begin(), guess.end());
                }

                // 2. 调用GPU哈希函数
                md5_hash_on_gpu(
                    h_inputs_flat.data(),
                    h_offsets.data(),
                    h_lengths.data(),
                    h_digests.data(),
                    total_guesses_to_hash
                );

                // 3. (可选) 在此可以添加代码来处理或验证哈希结果
                // 例如: check_hashes(h_digests);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            
            // 清理已处理的密码，为下一批做准备
            history += q.guesses.size();
            q.guesses.clear();
            q.guesses.shrink_to_fit(); // 释放vector占用的多余内存
        }

        // 6. 终止条件
        int generate_n = 10000000;
        if (history + q.guesses.size() > generate_n) {
            break;
        }
    }

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    // ==========================================================
    // 修改后的最终报告 (Final Report with Instrumentation)
    // ==========================================================
    cout << "----------------- FINAL REPORT -----------------" << endl;
    cout << "Total Guesses generated: " << history + q.guesses.size() << endl;
    cout << "Model Train Time: " << fixed << setprecision(4) << time_train <<" seconds" << endl;
    cout << "----------------------------------------------" << endl;
    cout << "Total Time (Guess + Hash): " << fixed << setprecision(4) << time_guess << " seconds" << endl;
    cout << "  Guess Generation Time (Total): " << fixed << setprecision(4) << time_guess - time_hash << " seconds" << endl;
    cout << "  MD5 Hash Time (GPU): " << fixed << setprecision(4) << time_hash << " seconds" << endl;
    cout << "----------------- DETAILED GUESS TIME BREAKDOWN -----------------" << endl;
    cout << "  CPU Generation Time:         " << fixed << setprecision(4) << time_cpu_gen << " seconds" << endl;
    cout << "  GPU Data Preparation Time:   " << fixed << setprecision(4) << time_gpu_prepare << " seconds" << endl;
    cout << "  GPU Compute + Transfer Time: " << fixed << setprecision(4) << time_gpu_compute << " seconds" << endl;
    cout << "  Priority Queue Update Time:  " << fixed << setprecision(4) << time_queue_update << " seconds" << endl;
    double unaccounted_time = (time_guess - time_hash) - (time_cpu_gen + time_gpu_prepare + time_gpu_compute + time_queue_update);
    cout << "  Other (Loop logic, etc.):    " << fixed << setprecision(4) << unaccounted_time << " seconds" << endl;
    cout << "----------------------------------------------" << endl;
    
    return 0;
}