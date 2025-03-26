#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

// 平凡算法：逐个累加
int sum_plain(vector<int>& arr) {
    int sum = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// 优化算法：递归分治（二重循环实现）
int sum_optimized(vector<int>& arr) {
    size_t n = 65536;
    for (size_t m = n; m > 1; m /= 2) {
        for (size_t i = 0; i < m / 2; i++) {
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
        }
    }
    return arr[0];
}

// 循环展开优化算法
int sum_unrolled(vector<int>& arr) {
    size_t n = 65536;
    for (size_t m = n; m > 1; m /= 2) {
        size_t half = m /2;
        for (size_t i = 0; i < half; i += 4) { // 循环展开4倍
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
            arr[i + 1] = arr[(i + 1) * 2] + arr[(i + 1) * 2 + 1];
            arr[i + 2] = arr[(i + 2) * 2] + arr[(i + 2) * 2 + 1];
            arr[i + 3] = arr[(i + 3) * 2] + arr[(i + 3) * 2 + 1];
        }
    }
    return arr[0];
}
// 循环展开平凡算法
int sum_unrolled1(vector<int>& arr) {
    int sum = 0;
    for (size_t i = 0; i < arr.size(); i+=4) {
        sum += arr[i]; sum += arr[i + 1];
        sum += arr[i+2]; sum += arr[i + 3];
    }
    return sum;
}
int main() {
    size_t n = 65536; // 数组大小
    vector<int> arr(n);

    // 初始化数组
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1; // 填充数组
    }

    // 测试平凡算法
    auto start = chrono::high_resolution_clock::now();
    int sum1 = sum_plain(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_plain = end - start;
    cout << "平凡算法结果：" << sum1 << endl;
    cout << "平凡算法耗时：" << time_plain.count() << "秒" << endl;

    // 测试优化算法
    start = chrono::high_resolution_clock::now();
    int sum2 = sum_optimized(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_optimized = end - start;
    cout << "优化算法结果：" << sum2 << endl;
    cout << "优化算法耗时：" << time_optimized.count() << "秒" << endl;
    // 初始化数组
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1; // 填充数组
    }
    // 测试循环展开平凡算法
    start = chrono::high_resolution_clock::now();
    int sum3 = sum_unrolled1(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_unrolled1 = end - start;
    cout << "循环展开平凡算法结果：" << sum3 << endl;
    cout << "循环展开平凡算法耗时：" << time_unrolled1.count() << "秒" << endl;

    // 测试循环展开优化算法
    start = chrono::high_resolution_clock::now();
    int sum4 = sum_unrolled(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_unrolled = end - start;
    cout << "循环展开优化算法结果：" << sum4 << endl;
    cout << "循环展开优化算法耗时：" << time_unrolled.count() << "秒" << endl;

    return 0;
}