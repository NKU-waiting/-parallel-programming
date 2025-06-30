//# 编译CUDA代码
//nvcc -c -O3 -arch=sm_75 guessing_gpu.cu -o guessing_gpu.o


// 链接所有对象文件
//g++ main.cpp train.cpp guessing_cpu.cpp guessing_gpu.cu md5.cpp -o main -O2 -L/usr/local/cuda/lib64 -lcudart
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include "guessing_gpu.h" // 引入GPU接口

using namespace std;
using namespace chrono;

const int SIMD_BATCH = 4; // 使用4路并行
// 全局计时器 (Global Timers for Instrumentation)
// ==========================================================
double time_cpu_gen = 0.0;
double time_gpu_prepare = 0.0;
double time_gpu_compute = 0.0;
double time_queue_update = 0.0;

// --- 进阶要求相关参数 ---
// 1. PT层面的并行: 一次向GPU发送一批PT

// 目标：凑够约10万个猜测再送给GPU
const int GPU_BATCH_TARGET_GUESSES = 100000;
const int CPU_GENERATION_THRESHOLD = 524288;
// 2. CPU/GPU重叠: 在此未显式实现双缓冲流水线，因为主要瓶颈在GPU。
//    但动态负载均衡本身就是一种高级的CPU/GPU协同。

// 3. 动态负载均衡: 小于此阈值的PT由CPU处理，避免GPU开销


// 编译指令如下:
// nvcc -c -O3 -arch=sm_75 guessing_gpu.cu -o guessing_gpu.o  (sm_75请根据你的GPU架构修改)
// g++ -c -O3 main.cpp -o main.o
// g++ -c -O3 train.cpp -o train.o
// g++ -c -O3 guessing_cpu.cpp -o guessing_cpu.o
// g++ -c -O3 md5.cpp -o md5.o
// g++ main.o train.o guessing_cpu.o guessing_gpu.o md5.o -o main -O3 -L/usr/local/cuda/lib64 -lcudart

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
    long long history = 0; // Use long long for large numbers
    
    // 用于暂存准备发往GPU的PT批次
    vector<PT> gpu_batch;
    int guesses_in_gpu_batch = 0;

    while (!q.priority.empty())
    {
        // 1. 从优先队列顶部取出一个PT
        // For std::multiset, we use *q.priority.begin() instead of q.priority.front()
        PT current_pt = *q.priority.begin(); 
        
        // 2. 根据该PT生成新的PT并更新优先队列 (插桩)
        auto start_q_update = system_clock::now();
        q.UpdateQueueFromTop();
        auto end_q_update = system_clock::now();
        time_queue_update += duration_cast<microseconds>(end_q_update - start_q_update).count() / 1e6;

        // 3. 动态负载均衡决策
        int last_seg_idx = current_pt.content.size() - 1;
        int num_guesses_to_gen = (last_seg_idx >= 0 && !current_pt.max_indices.empty()) ? current_pt.max_indices[last_seg_idx] : 0;
        
        if (num_guesses_to_gen == 0) continue; // 空任务

        // 如果任务太小，直接用CPU处理 (插桩)
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

        // 4. 检查GPU批次是否达到目标大小，或者队列将空，需要清空批次
        if (guesses_in_gpu_batch >= GPU_BATCH_TARGET_GUESSES || (q.priority.empty() && !gpu_batch.empty())) {
            // cout << "Dispatching batch of " << gpu_batch.size() << " PTs to GPU, generating " << guesses_in_gpu_batch << " guesses." << endl;
            q.DispatchToGpu(gpu_batch);
            gpu_batch.clear();
            guesses_in_gpu_batch = 0;
        }
        
        // 5. 内存管理和进度报告
        if (q.guesses.size() >= 1000000)
        {
            cout << "Guesses generated so far: " << history + q.guesses.size() << endl;

            auto start_hash = system_clock::now();
            
            const int total = q.guesses.size();
            for (int i = 0; i < total; i += SIMD_BATCH) 
            {
                string batch_inputs[SIMD_BATCH];
                bit32 batch_states[SIMD_BATCH][4];
                for (int j = 0; j < SIMD_BATCH; ++j) {
                    int idx = i + j;
                    batch_inputs[j] = (idx < total) ? q.guesses[idx] : "";
                }
                MD5Hash(batch_inputs, batch_states);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            
            history += q.guesses.size();
            q.guesses.clear();
            q.guesses.shrink_to_fit(); // 释放vector内存
        }

        // 6. 终止条件
        int generate_n = 100000000;
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
    cout << "  MD5 Hash Time: " << fixed << setprecision(4) << time_hash << " seconds" << endl;
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