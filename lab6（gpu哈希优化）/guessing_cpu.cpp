#include "PCFG.h"
#include "guessing_gpu.h"
#include <iostream>
#include <algorithm>
#include <chrono> // FIX: Added the missing <chrono> header for timing

using namespace std;
using namespace chrono; // FIX: Added namespace for chrono

// ==========================================================
// 引用 main.cpp 中定义的全局计时器
// Use 'extern' to declare global variables defined in another file.
// ==========================================================
extern double time_gpu_prepare;
extern double time_gpu_compute;
// ==========================================================


// --- Function definitions start here ---

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        float seg_prob = 0.0f;
        int total_freq = 1;

        if (pt.content[index].type == 1) {
            const auto& seg_data = m.letters[m.FindLetter(pt.content[index])];
            if (!seg_data.ordered_freqs.empty() && seg_data.total_freq > 0) {
                seg_prob = (float)seg_data.ordered_freqs[idx] / seg_data.total_freq;
            }
        } else if (pt.content[index].type == 2) {
            const auto& seg_data = m.digits[m.FindDigit(pt.content[index])];
            if (!seg_data.ordered_freqs.empty() && seg_data.total_freq > 0) {
                seg_prob = (float)seg_data.ordered_freqs[idx] / seg_data.total_freq;
            }
        } else if (pt.content[index].type == 3) {
            const auto& seg_data = m.symbols[m.FindSymbol(pt.content[index])];
            if (!seg_data.ordered_freqs.empty() && seg_data.total_freq > 0) {
                seg_prob = (float)seg_data.ordered_freqs[idx] / seg_data.total_freq;
            }
        }
        pt.prob *= seg_prob;
        index += 1;
    }
}

void PriorityQueue::init()
{
    // This function's logic becomes much simpler
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            } else if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            } else if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        if (m.total_preterm > 0) {
            pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        } else {
            pt.preterm_prob = 0.0f;
        }
        CalProb(pt);

        // Just insert it. The multiset handles sorting automatically.
        priority.insert(pt);
    }
}


void PriorityQueue::UpdateQueueFromTop()
{
    if (priority.empty()) return;

    // Get the first element (highest probability)
    PT top_pt = *priority.begin();
    
    // Erase it
    priority.erase(priority.begin());

    // Generate new PTs from the one we just removed
    vector<PT> new_pts = top_pt.NewPTs();
    for (PT& pt : new_pts)
    {
        CalProb(pt);
        // Simply insert the new PTs. The multiset will place them
        // in the correct sorted position in O(log N) time.
        priority.insert(pt);
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() <= 1) {
        return res;
    }

    int init_pivot = pivot;
    for (int i = pivot; i < curr_indices.size(); i++)
    {
        PT new_pt = *this;
        new_pt.curr_indices[i]++;
        if (new_pt.curr_indices[i] < new_pt.max_indices[i])
        {
            new_pt.pivot = i;
            res.emplace_back(new_pt);
        }
    }
    return res;
}

void PriorityQueue::Generate_CPU(PT pt)
{
    string base_guess;
    int seg_idx = 0;
    for (int idx : pt.curr_indices)
    {
        if (pt.content[seg_idx].type == 1) {
            base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 2) {
            base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 3) {
            base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        }
        seg_idx += 1;
    }

    const segment* last_seg_ptr;
    int last_seg_type = pt.content.back().type;

    if (last_seg_type == 1) {
        last_seg_ptr = &m.letters[m.FindLetter(pt.content.back())];
    } else if (last_seg_type == 2) {
        last_seg_ptr = &m.digits[m.FindDigit(pt.content.back())];
    } else {
        last_seg_ptr = &m.symbols[m.FindSymbol(pt.content.back())];
    }
    
    for (const auto& suffix : last_seg_ptr->ordered_values)
    {
        guesses.emplace_back(base_guess + suffix);
    }
}


// FIX: This is the single, correct, instrumented version of the function.
void PriorityQueue::DispatchToGpu(const vector<PT>& pt_batch) {
    if (pt_batch.empty()) return;
    
    // --- Instrumentation Start: GPU Data Preparation ---
    auto start_prepare = system_clock::now();

    GpuTaskData task;
    task.num_pts = pt_batch.size();
    task.total_guesses_to_generate = 0;

    for (const auto& pt : pt_batch) {
        string prefix_str;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                prefix_str += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                prefix_str += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else {
                prefix_str += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
        }
        task.h_prefix_offsets.push_back(task.h_prefixes.size());
        task.h_prefix_lengths.push_back(prefix_str.length());
        task.h_prefixes.insert(task.h_prefixes.end(), prefix_str.begin(), prefix_str.end());
        
        const segment* last_seg_ptr;
        if (pt.content.back().type == 1) last_seg_ptr = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2) last_seg_ptr = &m.digits[m.FindDigit(pt.content.back())];
        else last_seg_ptr = &m.symbols[m.FindSymbol(pt.content.back())];

        task.h_suffix_offsets.push_back(task.h_suffix_str_offsets.size());
        int num_suffixes = last_seg_ptr->ordered_values.size();
        task.h_num_suffixes_per_pt.push_back(num_suffixes);
        task.total_guesses_to_generate += num_suffixes;

        for (const auto& suffix : last_seg_ptr->ordered_values) {
            task.h_suffix_str_offsets.push_back(task.h_suffixes.size());
            task.h_suffix_lengths.push_back(suffix.length());
            task.h_suffixes.insert(task.h_suffixes.end(), suffix.begin(), suffix.end());
        }
    }

    auto end_prepare = system_clock::now();
    time_gpu_prepare += duration_cast<microseconds>(end_prepare - start_prepare).count() / 1e6;
    // --- Instrumentation End: GPU Data Preparation ---

    if (task.total_guesses_to_generate > 0) {
        // --- Instrumentation Start: GPU Compute + Transfer ---
        auto start_compute = system_clock::now();
        generate_passwords_on_gpu(task, this->guesses);
        auto end_compute = system_clock::now();
        time_gpu_compute += duration_cast<microseconds>(end_compute - start_compute).count() / 1e6;
        // --- Instrumentation End: GPU Compute + Transfer ---
    }
}