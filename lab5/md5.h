// --- START OF FILE md5.h ---

#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5哈希函数声明
// 这个函数会处理一个包含4个字符串的批次
void MD5Hash(const string inputs[4], bit32 states[4][4]);

#endif // MD5_H