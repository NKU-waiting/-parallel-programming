#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// 初始化矩阵和向量
void initializeMatrixAndVector(int n, vector<vector<int>>& matrix, vector<int>& vec) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = i + j; // 矩阵元素按 A[i][j] = i + j 初始化
        }
        vec[i] = i; // 向量元素简单地设为 i
    }
}

// 平凡算法：逐列计算内积
vector<int> naiveAlgorithm(const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    vector<int> result(n, 0); // 声明并初始化结果向量
    for (int j = 0; j < n; ++j) { // 遍历每一列
        for (int i = 0; i < n; ++i) { // 遍历每一行
            result[j] += matrix[i][j] * vec[i]; // 计算内积
        }
    }
    return result;
}

// 缓存优化算法：按行访问矩阵元素
vector<int> cacheOptimizedAlgorithm(const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    vector<int> result(n, 0); // 声明并初始化结果向量
    for (int i = 0; i < n; ++i) { // 遍历每一行
        for (int j = 0; j < n; ++j) { // 遍历每一列
            result[j] += matrix[i][j] * vec[i]; // 计算内积
        }
    }
    return result;
}

// 测试算法性能
void testAlgorithmPerformance(const string& algorithmName, const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    auto start = high_resolution_clock::now(); // 记录开始时间
    vector<int> result;

    if (algorithmName == "naive") {
        result = naiveAlgorithm(matrix, vec, n); // 调用平凡算法
    }
    else if (algorithmName == "cache_optimized") {
        result = cacheOptimizedAlgorithm(matrix, vec, n); // 调用缓存优化算法
    }

    auto stop = high_resolution_clock::now(); // 记录结束时间
    auto duration = duration_cast<microseconds>(stop - start); // 计算执行时间

    cout << "算法名称: " << algorithmName << endl;
    cout << "执行时间: " << duration.count() << " 微秒" << endl;
    // 输出部分结果（前5个元素）以验证正确性
    cout << "前5个结果: ";
    for (int i = 0; i < min(5, n); ++i) {
        cout << result[i] << " ";
    }
    cout << endl << endl;
}

int main() {
    int n;
    cout << "请输入矩阵的大小 (n): ";
    cin >> n;

    vector<vector<int>> matrix(n, vector<int>(n)); // 定义矩阵
    vector<int> vec(n); // 定义向量

    initializeMatrixAndVector(n, matrix, vec); // 初始化矩阵和向量

    cout << "正在测试矩阵大小为 " << n << "x" << n << " 的算法性能" << endl;

    testAlgorithmPerformance("naive", matrix, vec, n); // 测试平凡算法
    testAlgorithmPerformance("cache_optimized", matrix, vec, n); // 测试缓存优化算法

    return 0;
}