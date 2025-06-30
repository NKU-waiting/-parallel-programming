#include "PCFG.h"
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring> // For memcpy

using namespace std;

// 序列化 PT 对象到字节流 (vector<char>)
vector<char> PT::serialize() const {
    vector<char> buffer;
    auto append = [&](const void* data, size_t size) {
        const char* bytes = static_cast<const char*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    };

    append(&pivot, sizeof(pivot));
    append(&preterm_prob, sizeof(preterm_prob));
    append(&prob, sizeof(prob));

    size_t content_size = content.size();
    append(&content_size, sizeof(content_size));
    for (const auto& seg : content) {
        append(&seg.type, sizeof(seg.type));
        append(&seg.length, sizeof(seg.length));
    }

    size_t curr_indices_size = curr_indices.size();
    append(&curr_indices_size, sizeof(curr_indices_size));
    buffer.insert(buffer.end(), (char*)curr_indices.data(), (char*)(curr_indices.data() + curr_indices_size));

    size_t max_indices_size = max_indices.size();
    append(&max_indices_size, sizeof(max_indices_size));
    buffer.insert(buffer.end(), (char*)max_indices.data(), (char*)(max_indices.data() + max_indices_size));

    return buffer;
}

// 从字节流反序列化，恢复 PT 对象
void PT::deserialize(const vector<char>& data) {
    const char* buffer = data.data();
    size_t offset = 0;
    auto read = [&](void* dest, size_t size) {
        memcpy(dest, buffer + offset, size);
        offset += size;
    };

    read(&pivot, sizeof(pivot));
    read(&preterm_prob, sizeof(preterm_prob));
    read(&prob, sizeof(prob));

    size_t content_size;
    read(&content_size, sizeof(content_size));
    content.resize(content_size, segment(0,0));
    for (size_t i = 0; i < content_size; ++i) {
        read(&content[i].type, sizeof(content[i].type));
        read(&content[i].length, sizeof(content[i].length));
    }

    size_t curr_indices_size;
    read(&curr_indices_size, sizeof(curr_indices_size));
    curr_indices.resize(curr_indices_size);
    read(curr_indices.data(), curr_indices_size * sizeof(int));
    
    size_t max_indices_size;
    read(&max_indices_size, sizeof(max_indices_size));
    max_indices.resize(max_indices_size);
    read(max_indices.data(), max_indices_size * sizeof(int));
}


unordered_map<string, vector<float>> precomputed_probs;

void PriorityQueue::PrecomputeFreqs()
{
// ... (此函数内容保持不变) ...
    for (const auto& seg : m.letters)
    {
        vector<float> probs;
        for (float f : seg.ordered_freqs)
            probs.push_back(f / seg.total_freq);
        precomputed_probs["L" + to_string(seg.length)] = probs;
    }

    for (const auto& seg : m.digits)
    {
        vector<float> probs;
        for (float f : seg.ordered_freqs)
            probs.push_back(f / seg.total_freq);
        precomputed_probs["D" + to_string(seg.length)] = probs;
    }

    for (const auto& seg : m.symbols)
    {
        vector<float> probs;
        for (float f : seg.ordered_freqs)
            probs.push_back(f / seg.total_freq);
        precomputed_probs["S" + to_string(seg.length)] = probs;
    }
}

vector<PT> PT::NewPTs()
{
// ... (此函数内容保持不变) ...
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

void PriorityQueue::CalProb(PT &pt)
{
// ... (此函数内容保持不变) ...
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        const segment& seg = pt.content[index];
        string key;
        if (seg.type == 1) key = "L" + to_string(seg.length);
        if (seg.type == 2) key = "D" + to_string(seg.length);
        if (seg.type == 3) key = "S" + to_string(seg.length);
        pt.prob *= precomputed_probs[key][idx];
        index += 1;
    }
}

void PriorityQueue::init()
{
// ... (此函数内容保持不变) ...
    PrecomputeFreqs();
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.push(pt);
    }
}

// 这个函数现在被修改为接收 model 和输出 vector，以便于在工作进程中调用
void GenerateForWorker(model& m, PT& pt, vector<string>& generated_guesses)
{
    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int n = pt.max_indices[0];
        vector<string> local_guesses(n);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            local_guesses[i] = a->ordered_values[i];
        
        generated_guesses.insert(generated_guesses.end(), local_guesses.begin(), local_guesses.end());
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            const segment& seg = pt.content[seg_idx];
            if (seg.type == 1)
                prefix += m.letters[m.FindLetter(seg)].ordered_values[idx];
            else if (seg.type == 2)
                prefix += m.digits[m.FindDigit(seg)].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx];

            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        const segment& last_seg = pt.content.back();
        segment *a;
        if (last_seg.type == 1)
            a = &m.letters[m.FindLetter(last_seg)];
        else if (last_seg.type == 2)
            a = &m.digits[m.FindDigit(last_seg)];
        else
            a = &m.symbols[m.FindSymbol(last_seg)];

        int n = pt.max_indices.back();
        vector<string> local_guesses(n);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            local_guesses[i] = prefix + a->ordered_values[i];
        
        generated_guesses.insert(generated_guesses.end(), local_guesses.begin(), local_guesses.end());
    }
}

// 原有的 Generate 和 PopNext 函数仍然保留，供单进程或主进程内部使用
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int n = pt.max_indices[0];
        vector<string> local_guesses(n);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            local_guesses[i] = a->ordered_values[i];

        #pragma omp critical
        {
            guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
            total_guesses += n;
        }
    }
    else
    {
        string prefix;
        // ... (原有逻辑不变) ...
        // ... (为简洁省略，与 GenerateForWorker 内部逻辑相同，但写入 this->guesses)
        #pragma omp critical
        {
            // ... (写入 this->guesses)
        }
    }
}

void PriorityQueue::PopNext()
{
    PT top_pt = priority.top();
    
    Generate(top_pt);
    vector<PT> new_pts = top_pt.NewPTs();

    priority.pop();

    for (PT& pt : new_pts)
    {
        CalProb(pt);
        priority.push(pt); // 使用 .push()
    }
}

void PriorityQueue::BatchGenerate(int batch_size)
{
    int count = 0;
    while (!priority.empty() && count < batch_size)
    {
        PopNext();
        count++;
    }
}