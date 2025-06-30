// --- START OF FILE md5.cpp ---

#include "md5.h" // 必须首先包含自己的头文件
#include <iomanip>
#include <cstring> // for memcpy
#include <vector>  // for std::vector
#include <assert.h>

using namespace std;

// MD5 constants
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

// Basic MD5 functions
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

// Rotates x left n bits.
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// MD5 transformations for rounds 1, 2, 3, and 4.
#define FF(a, b, c, d, x, s, ac)          \
    {                                     \
        (a) += F((b), (c), (d)) + (x) + (ac); \
        (a) = ROTATE_LEFT((a), (s));      \
        (a) += (b);                       \
    }
#define GG(a, b, c, d, x, s, ac)          \
    {                                     \
        (a) += G((b), (c), (d)) + (x) + (ac); \
        (a) = ROTATE_LEFT((a), (s));      \
        (a) += (b);                       \
    }
#define HH(a, b, c, d, x, s, ac)          \
    {                                     \
        (a) += H((b), (c), (d)) + (x) + (ac); \
        (a) = ROTATE_LEFT((a), (s));      \
        (a) += (b);                       \
    }
#define II(a, b, c, d, x, s, ac)          \
    {                                     \
        (a) += I((b), (c), (d)) + (x) + (ac); \
        (a) = ROTATE_LEFT((a), (s));      \
        (a) += (b);                       \
    }


void MD5_core_transform(bit32 state[4], const Byte block[64]) {
    bit32 a = state[0], b = state[1], c = state[2], d = state[3];
    bit32 x[16];

    // Decode block into 16 32-bit words
    for (int i = 0, j = 0; j < 64; i++, j += 4) {
        x[i] = ((bit32)block[j]) | (((bit32)block[j + 1]) << 8) |
               (((bit32)block[j + 2]) << 16) | (((bit32)block[j + 3]) << 24);
    }

    /* Round 1 */
    FF(a, b, c, d, x[0], S11, 0xd76aa478);
    FF(d, a, b, c, x[1], S12, 0xe8c7b756);
    FF(c, d, a, b, x[2], S13, 0x242070db);
    FF(b, c, d, a, x[3], S14, 0xc1bdceee);
    FF(a, b, c, d, x[4], S11, 0xf57c0faf);
    FF(d, a, b, c, x[5], S12, 0x4787c62a);
    FF(c, d, a, b, x[6], S13, 0xa8304613);
    FF(b, c, d, a, x[7], S14, 0xfd469501);
    FF(a, b, c, d, x[8], S11, 0x698098d8);
    FF(d, a, b, c, x[9], S12, 0x8b44f7af);
    FF(c, d, a, b, x[10], S13, 0xffff5bb1);
    FF(b, c, d, a, x[11], S14, 0x895cd7be);
    FF(a, b, c, d, x[12], S11, 0x6b901122);
    FF(d, a, b, c, x[13], S12, 0xfd987193);
    FF(c, d, a, b, x[14], S13, 0xa679438e);
    FF(b, c, d, a, x[15], S14, 0x49b40821);

    /* Round 2 */
    GG(a, b, c, d, x[1], S21, 0xf61e2562);
    GG(d, a, b, c, x[6], S22, 0xc040b340);
    GG(c, d, a, b, x[11], S23, 0x265e5a51);
    GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);
    GG(a, b, c, d, x[5], S21, 0xd62f105d);
    GG(d, a, b, c, x[10], S22, 0x02441453);
    GG(c, d, a, b, x[15], S23, 0xd8a1e681);
    GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);
    GG(a, b, c, d, x[9], S21, 0x21e1cde6);
    GG(d, a, b, c, x[14], S22, 0xc33707d6);
    GG(c, d, a, b, x[3], S23, 0xf4d50d87);
    GG(b, c, d, a, x[8], S24, 0x455a14ed);
    GG(a, b, c, d, x[13], S21, 0xa9e3e905);
    GG(d, a, b, c, x[2], S22, 0xfcefa3f8);
    GG(c, d, a, b, x[7], S23, 0x676f02d9);
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);

    /* Round 3 */
    HH(a, b, c, d, x[5], S31, 0xfffa3942);
    HH(d, a, b, c, x[8], S32, 0x8771f681);
    HH(c, d, a, b, x[11], S33, 0x6d9d6122);
    HH(b, c, d, a, x[14], S34, 0xfde5380c);
    HH(a, b, c, d, x[1], S31, 0xa4beea44);
    HH(d, a, b, c, x[4], S32, 0x4bdecfa9);
    HH(c, d, a, b, x[7], S33, 0xf6bb4b60);
    HH(b, c, d, a, x[10], S34, 0xbebfbc70);
    HH(a, b, c, d, x[13], S31, 0x289b7ec6);
    HH(d, a, b, c, x[0], S32, 0xeaa127fa);
    HH(c, d, a, b, x[3], S33, 0xd4ef3085);
    HH(b, c, d, a, x[6], S34, 0x04881d05);
    HH(a, b, c, d, x[9], S31, 0xd9d4d039);
    HH(d, a, b, c, x[12], S32, 0xe6db99e5);
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8);
    HH(b, c, d, a, x[2], S34, 0xc4ac5665);

    /* Round 4 */
    II(a, b, c, d, x[0], S41, 0xf4292244);
    II(d, a, b, c, x[7], S42, 0x432aff97);
    II(c, d, a, b, x[14], S43, 0xab9423a7);
    II(b, c, d, a, x[5], S44, 0xfc93a039);
    II(a, b, c, d, x[12], S41, 0x655b59c3);
    II(d, a, b, c, x[3], S42, 0x8f0ccc92);
    II(c, d, a, b, x[10], S43, 0xffeff47d);
    II(b, c, d, a, x[1], S44, 0x85845dd1);
    II(a, b, c, d, x[8], S41, 0x6fa87e4f);
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0);
    II(c, d, a, b, x[6], S43, 0xa3014314);
    II(b, c, d, a, x[13], S44, 0x4e0811a1);
    II(a, b, c, d, x[4], S41, 0xf7537e82);
    II(d, a, b, c, x[11], S42, 0xbd3af235);
    II(c, d, a, b, x[2], S43, 0x2ad7d2bb);
    II(b, c, d, a, x[9], S44, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

void process_single_input(const string& input, bit32 state[4]) {
    // 1. Initialize state
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // 2. Padding
    uint64_t bitLength = (uint64_t)input.length() * 8;
    // Calculate the new length of the message in bytes, which is a multiple of 64
    uint64_t newLength = (input.length() / 64 + 1) * 64;
    if (input.length() % 64 > 55) {
        newLength += 64;
    }
    
    vector<Byte> paddedMsg(newLength, 0);
    memcpy(paddedMsg.data(), input.c_str(), input.length());
    paddedMsg[input.length()] = 0x80;

    // Append original length as 64-bit little-endian integer
    memcpy(&paddedMsg[newLength - 8], &bitLength, 8);

    // 3. Process message in 64-byte chunks
    for (size_t i = 0; i < newLength; i += 64) {
        MD5_core_transform(state, &paddedMsg[i]);
    }
}

// Main function to hash a batch of 4 strings
void MD5Hash(const string inputs[4], bit32 states[4][4]) {
    for (int i = 0; i < 4; ++i) {
        if (!inputs[i].empty()) {
            process_single_input(inputs[i], states[i]);
        }
    }
}