#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// ��ʼ�����������
void initializeMatrixAndVector(int n, vector<vector<int>>& matrix, vector<int>& vec) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = i + j; // ����Ԫ�ذ� A[i][j] = i + j ��ʼ��
        }
        vec[i] = i; // ����Ԫ�ؼ򵥵���Ϊ i
    }
}

// ƽ���㷨�����м����ڻ�
vector<int> naiveAlgorithm(const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    vector<int> result(n, 0); // ��������ʼ���������
    for (int j = 0; j < n; ++j) { // ����ÿһ��
        for (int i = 0; i < n; ++i) { // ����ÿһ��
            result[j] += matrix[i][j] * vec[i]; // �����ڻ�
        }
    }
    return result;
}

// �����Ż��㷨�����з��ʾ���Ԫ��
vector<int> cacheOptimizedAlgorithm(const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    vector<int> result(n, 0); // ��������ʼ���������
    for (int i = 0; i < n; ++i) { // ����ÿһ��
        for (int j = 0; j < n; ++j) { // ����ÿһ��
            result[j] += matrix[i][j] * vec[i]; // �����ڻ�
        }
    }
    return result;
}

// �����㷨����
void testAlgorithmPerformance(const string& algorithmName, const vector<vector<int>>& matrix, const vector<int>& vec, int n) {
    auto start = high_resolution_clock::now(); // ��¼��ʼʱ��
    vector<int> result;

    if (algorithmName == "naive") {
        result = naiveAlgorithm(matrix, vec, n); // ����ƽ���㷨
    }
    else if (algorithmName == "cache_optimized") {
        result = cacheOptimizedAlgorithm(matrix, vec, n); // ���û����Ż��㷨
    }

    auto stop = high_resolution_clock::now(); // ��¼����ʱ��
    auto duration = duration_cast<microseconds>(stop - start); // ����ִ��ʱ��

    cout << "�㷨����: " << algorithmName << endl;
    cout << "ִ��ʱ��: " << duration.count() << " ΢��" << endl;
    // ������ֽ����ǰ5��Ԫ�أ�����֤��ȷ��
    cout << "ǰ5�����: ";
    for (int i = 0; i < min(5, n); ++i) {
        cout << result[i] << " ";
    }
    cout << endl << endl;
}

int main() {
    int n;
    cout << "���������Ĵ�С (n): ";
    cin >> n;

    vector<vector<int>> matrix(n, vector<int>(n)); // �������
    vector<int> vec(n); // ��������

    initializeMatrixAndVector(n, matrix, vec); // ��ʼ�����������

    cout << "���ڲ��Ծ����СΪ " << n << "x" << n << " ���㷨����" << endl;

    testAlgorithmPerformance("naive", matrix, vec, n); // ����ƽ���㷨
    testAlgorithmPerformance("cache_optimized", matrix, vec, n); // ���Ի����Ż��㷨

    return 0;
}