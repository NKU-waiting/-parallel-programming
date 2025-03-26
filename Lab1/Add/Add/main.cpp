#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

// ƽ���㷨������ۼ�
int sum_plain(vector<int>& arr) {
    int sum = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }
    return sum;
}

// �Ż��㷨���ݹ���Σ�����ѭ��ʵ�֣�
int sum_optimized(vector<int>& arr) {
    size_t n = 65536;
    for (size_t m = n; m > 1; m /= 2) {
        for (size_t i = 0; i < m / 2; i++) {
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
        }
    }
    return arr[0];
}

// ѭ��չ���Ż��㷨
int sum_unrolled(vector<int>& arr) {
    size_t n = 65536;
    for (size_t m = n; m > 1; m /= 2) {
        size_t half = m /2;
        for (size_t i = 0; i < half; i += 4) { // ѭ��չ��4��
            arr[i] = arr[i * 2] + arr[i * 2 + 1];
            arr[i + 1] = arr[(i + 1) * 2] + arr[(i + 1) * 2 + 1];
            arr[i + 2] = arr[(i + 2) * 2] + arr[(i + 2) * 2 + 1];
            arr[i + 3] = arr[(i + 3) * 2] + arr[(i + 3) * 2 + 1];
        }
    }
    return arr[0];
}
// ѭ��չ��ƽ���㷨
int sum_unrolled1(vector<int>& arr) {
    int sum = 0;
    for (size_t i = 0; i < arr.size(); i+=4) {
        sum += arr[i]; sum += arr[i + 1];
        sum += arr[i+2]; sum += arr[i + 3];
    }
    return sum;
}
int main() {
    size_t n = 65536; // �����С
    vector<int> arr(n);

    // ��ʼ������
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1; // �������
    }

    // ����ƽ���㷨
    auto start = chrono::high_resolution_clock::now();
    int sum1 = sum_plain(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_plain = end - start;
    cout << "ƽ���㷨�����" << sum1 << endl;
    cout << "ƽ���㷨��ʱ��" << time_plain.count() << "��" << endl;

    // �����Ż��㷨
    start = chrono::high_resolution_clock::now();
    int sum2 = sum_optimized(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_optimized = end - start;
    cout << "�Ż��㷨�����" << sum2 << endl;
    cout << "�Ż��㷨��ʱ��" << time_optimized.count() << "��" << endl;
    // ��ʼ������
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1; // �������
    }
    // ����ѭ��չ��ƽ���㷨
    start = chrono::high_resolution_clock::now();
    int sum3 = sum_unrolled1(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_unrolled1 = end - start;
    cout << "ѭ��չ��ƽ���㷨�����" << sum3 << endl;
    cout << "ѭ��չ��ƽ���㷨��ʱ��" << time_unrolled1.count() << "��" << endl;

    // ����ѭ��չ���Ż��㷨
    start = chrono::high_resolution_clock::now();
    int sum4 = sum_unrolled(arr);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_unrolled = end - start;
    cout << "ѭ��չ���Ż��㷨�����" << sum4 << endl;
    cout << "ѭ��չ���Ż��㷨��ʱ��" << time_unrolled.count() << "��" << endl;

    return 0;
}