// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// g++ main.cpp train.cpp guessing_MP.cpp md5.cpp -fopenmp -o main
// g++ main.cpp train.cpp guessing_MP.cpp md5.cpp -fopenmp -o main -O2
// g++ main.cpp train.cpp guessing.cpp md5.cpp -pthread -o main -O2
// # 编译命令
// mpic++ main.cpp train.cpp guessing_MP.cpp md5.cpp -fopenmp -o main -O2

#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <algorithm> // for std::min
#include <cstring>   // for memcpy

// 在 guessing_MP.cpp 中声明的外部函数
void GenerateForWorker(model& m, PT& pt, vector<string>& generated_guesses);

using namespace std;
using namespace chrono;

const int SIMD_BATCH = 4;

// MPI 消息标签
const int WORK_TAG = 1;
const int STOP_TAG = 2;

// 辅助函数：序列化字符串向量
vector<char> serialize_strings(const vector<string>& strings) {
    vector<char> buffer;
    size_t num_strings = strings.size();
    buffer.insert(buffer.end(), (char*)&num_strings, (char*)&num_strings + sizeof(size_t));
    for (const auto& s : strings) {
        size_t len = s.length();
        buffer.insert(buffer.end(), (char*)&len, (char*)&len + sizeof(size_t));
        buffer.insert(buffer.end(), s.begin(), s.end());
    }
    return buffer;
}

// 辅助函数：反序列化字符串向量
vector<string> deserialize_strings(const vector<char>& data) {
    vector<string> strings;
    const char* buffer = data.data();
    size_t offset = 0;
    
    size_t num_strings;
    memcpy(&num_strings, buffer + offset, sizeof(size_t));
    offset += sizeof(size_t);

    strings.reserve(num_strings);
    for (size_t i = 0; i < num_strings; ++i) {
        size_t len;
        memcpy(&len, buffer + offset, sizeof(size_t));
        offset += sizeof(size_t);
        strings.emplace_back(buffer + offset, len);
        offset += len;
    }
    return strings;
}

// --- MASTER LOGIC ---
void master_main(int world_size) {
    double time_hash = 0, time_guess = 0, time_train = 0;
    // 删除了 total_sort_time 计时器
    double total_send_time = 0, total_recv_time = 0;
    PriorityQueue q;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        cout << "Master process (Rank 0) starting training..." << endl;
    }
    
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    time_train = duration_cast<microseconds>(end_train - start_train).count() / 1e6;

    q.init();
    
    if (world_rank == 0) {
        cout << "Master process started. Initial PTs in queue: " << q.priority.size() << endl;
    }

    int curr_num = 0;
    auto start = system_clock::now();
    long long history = 0;
    long long generate_n = 10000000;
    int print_interval = 1000000;
    int hash_batch_size = 1000000;
    int num_workers = world_size - 1;

    while (!q.priority.empty()) {
        // 排序逻辑已被彻底删除
        
        auto send_start = high_resolution_clock::now();
        int batch_size = min((int)q.priority.size(), num_workers);
        for (int i = 0; i < batch_size; ++i) {
            // 使用 top() 和 pop() 操作优先队列
            PT pt_to_send = q.priority.top(); 
            q.priority.pop();

            vector<char> serialized_pt = pt_to_send.serialize();
            MPI_Send(serialized_pt.data(), serialized_pt.size(), MPI_CHAR, i + 1, WORK_TAG, MPI_COMM_WORLD);
        }
        auto send_end = high_resolution_clock::now();
        total_send_time += duration_cast<microseconds>(send_end - send_start).count();

        auto recv_start = high_resolution_clock::now();
        for (int i = 0; i < batch_size; ++i) {
            MPI_Status status;
            int recv_size;

            MPI_Probe(i + 1, WORK_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_CHAR, &recv_size);
            vector<char> guess_data(recv_size);
            MPI_Recv(guess_data.data(), recv_size, MPI_CHAR, i + 1, WORK_TAG, MPI_COMM_WORLD, &status);
            vector<string> received_guesses = deserialize_strings(guess_data);
            q.guesses.insert(q.guesses.end(), received_guesses.begin(), received_guesses.end());

            MPI_Probe(i + 1, WORK_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_CHAR, &recv_size);
            vector<char> new_pt_data(recv_size);
            MPI_Recv(new_pt_data.data(), recv_size, MPI_CHAR, i + 1, WORK_TAG, MPI_COMM_WORLD, &status);
            
            size_t offset = 0;
            const char* buffer = new_pt_data.data();
            while(offset < new_pt_data.size()){
                size_t pt_size;
                memcpy(&pt_size, buffer + offset, sizeof(size_t));
                offset += sizeof(size_t);
                vector<char> single_pt_data(buffer + offset, buffer + offset + pt_size);
                offset += pt_size;
                PT new_pt;
                new_pt.deserialize(single_pt_data);
                q.CalProb(new_pt);
                q.priority.push(new_pt); // 使用 push() 添加
            }
        }
        auto recv_end = high_resolution_clock::now();
        total_recv_time += duration_cast<microseconds>(recv_end - recv_start).count();
        
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= print_interval) {
             cout << "Guesses generated: " << history + q.total_guesses << " / " << generate_n << endl;
             curr_num = q.total_guesses;
        }

        if (history + q.total_guesses >= generate_n) {
             cout << "Target number of guesses (" << generate_n << ") reached. Finishing..." << endl;
             break;
        }

        if (q.guesses.size() > hash_batch_size) {
            auto start_hash = high_resolution_clock::now();
            const int total = q.guesses.size();
            for (int i = 0; i < total; i += SIMD_BATCH) {
                string batch_inputs[SIMD_BATCH];
                bit32 batch_states[SIMD_BATCH][4];
                for (int j = 0; j < SIMD_BATCH; ++j) {
                    int idx = i + j;
                    batch_inputs[j] = (idx < total) ? q.guesses[idx] : "";
                }
                MD5Hash(batch_inputs, batch_states);
            }
            auto end_hash = high_resolution_clock::now();
            time_hash += duration_cast<microseconds>(end_hash - start_hash).count() / 1e6;

            history += q.guesses.size();
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    for (int i = 1; i < world_size; ++i) {
        MPI_Send(nullptr, 0, MPI_CHAR, i, STOP_TAG, MPI_COMM_WORLD);
    }
    
    auto end = system_clock::now();
    time_guess = duration_cast<microseconds>(end - start).count() / 1e6;

    if (!q.guesses.empty()) {
        auto start_hash = high_resolution_clock::now();
        const int total = q.guesses.size();
        for (int i = 0; i < total; i += SIMD_BATCH) {
            string batch_inputs[SIMD_BATCH];
            bit32 batch_states[SIMD_BATCH][4];
            for (int j = 0; j < SIMD_BATCH; ++j) {
                int idx = i + j;
                batch_inputs[j] = (idx < total) ? q.guesses[idx] : "";
            }
            MD5Hash(batch_inputs, batch_states);
        }
        auto end_hash = high_resolution_clock::now();
        time_hash += duration_cast<microseconds>(end_hash - start_hash).count() / 1e6;
        history += q.guesses.size();
        q.guesses.clear();
    }

    cout << "\n--- DETAILED TIME BREAKDOWN (Master) ---" << endl;
    cout << "Total Send Time (incl. serial):   " << total_send_time / 1e6 << " seconds" << endl;
    cout << "Total Recv Time (incl. deserial): " << total_recv_time / 1e6 << " seconds" << endl;
    double other_time = (time_guess * 1e6 - time_hash * 1e6 - total_send_time - total_recv_time) / 1e6;
    cout << "Other Logic & Wait Time:          " << other_time << " seconds" << endl;

    cout << "\n--- FINAL REPORT ---" << endl;
    cout << "Total guesses generated: " << history << endl;
    cout << "Guess time (excluding hash): " << time_guess - time_hash << " seconds" << endl;
    cout << "Total Hash time: " << time_hash << " seconds" << endl;
    cout << "Total Train time: " << time_train << " seconds" << endl;
    cout << "Total execution time (Train + Guess + Hash): " << time_train + time_guess << " seconds" << endl;
}

// --- WORKER LOGIC ---
void worker_main() {
    PriorityQueue q_worker;
    q_worker.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q_worker.m.order();
    q_worker.init();
    
    double total_compute_time = 0;
    double total_send_back_time = 0;
    int tasks_processed = 0;

    while (true) {
        MPI_Status status;
        int recv_size;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == STOP_TAG) {
            MPI_Recv(nullptr, 0, MPI_CHAR, 0, STOP_TAG, MPI_COMM_WORLD, &status);
            break;
        }
        
        MPI_Get_count(&status, MPI_CHAR, &recv_size);
        vector<char> pt_data(recv_size);
        MPI_Recv(pt_data.data(), recv_size, MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD, &status);
        
        PT received_pt;
        received_pt.deserialize(pt_data);
        
        auto compute_start = high_resolution_clock::now();
        vector<string> generated_guesses;
        GenerateForWorker(q_worker.m, received_pt, generated_guesses);
        vector<PT> new_pts = received_pt.NewPTs();
        auto compute_end = high_resolution_clock::now();
        total_compute_time += duration_cast<microseconds>(compute_end - compute_start).count();

        auto send_back_start = high_resolution_clock::now();
        vector<char> serialized_guesses = serialize_strings(generated_guesses);
        MPI_Send(serialized_guesses.data(), serialized_guesses.size(), MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD);
        
        vector<char> serialized_new_pts;
        for(const auto& pt : new_pts){
            vector<char> single_pt_data = pt.serialize();
            size_t pt_size = single_pt_data.size();
            serialized_new_pts.insert(serialized_new_pts.end(), (char*)&pt_size, (char*)&pt_size + sizeof(size_t));
            serialized_new_pts.insert(serialized_new_pts.end(), single_pt_data.begin(), single_pt_data.end());
        }
        MPI_Send(serialized_new_pts.data(), serialized_new_pts.size(), MPI_CHAR, 0, WORK_TAG, MPI_COMM_WORLD);
        auto send_back_end = high_resolution_clock::now();
        total_send_back_time += duration_cast<microseconds>(send_back_end - send_back_start).count();
        
        tasks_processed++;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "\n--- Worker " << rank << " Report ---" << endl;
    cout << "Tasks processed by this worker: " << tasks_processed << endl;
    cout << "Total Compute Time: " << total_compute_time / 1e6 << "s" << endl;
    cout << "Total Send-Back Time: " << total_send_back_time / 1e6 << "s" << endl;
    if (tasks_processed > 0) {
        cout << "Avg time per task (compute): " << total_compute_time / tasks_processed << " us" << endl;
        cout << "Avg time per task (send-back): " << total_send_back_time / tasks_processed << " us" << endl;
    }
}

// --- MAIN FUNCTION ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        if(world_rank == 0) {
            cerr << "This program requires at least 2 MPI processes (1 master, 1+ workers)." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        master_main(world_size);
    } else {
        worker_main();
    }

    MPI_Finalize();
    return 0;
}