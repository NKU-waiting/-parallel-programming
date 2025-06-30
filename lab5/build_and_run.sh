#!/bin/bash

# ==============================================================================
# PCFG Password Guesser - Build, Run, and Clean Script
# ==============================================================================
#
# This script performs three main actions:
# 1. Compiles the entire project from source.
# 2. Runs the compiled executable, redirecting output to log files.
# 3. Cleans up all intermediate build files, leaving only the final logs.
#
# ==============================================================================

# --- 设定变量 ---
# 根据您的GPU型号修改计算能力。sm_75 适用于 RTX 20系列/T4。
# 例如: RTX 30系列 -> sm_86, GTX 10系列 -> sm_61
CUDA_ARCH=sm_75
EXECUTABLE="./main"
STDOUT_FILE="test.o"
STDERR_FILE="test.e"

# !! 重要 !!: 确保训练数据文件和此脚本在同一目录
# 或者在 main.cpp 中使用绝对路径
TRAINING_FILE="Rockyou-singleLined-full.txt"

# ==============================================================================
# Part 1: Compilation
# ==============================================================================
echo "==> [Phase 1/3] Starting Compilation..."

# 清理旧的编译产物和日志文件
rm -f *.o $EXECUTABLE $STDOUT_FILE $STDERR_FILE

# --- 编译 CUDA 代码 ---
echo "    -> Compiling CUDA code (guessing_gpu.cu)..."
nvcc -c -O3 -arch=${CUDA_ARCH} guessing_gpu.cu -o guessing_gpu.o
if [ $? -ne 0 ]; then
    echo "Error: nvcc compilation failed. Aborting."
    exit 1
fi

# --- 编译 C++ 代码 ---
echo "    -> Compiling C++ code..."
g++ -c -O3 main.cpp -o main.o
g++ -c -O3 train.cpp -o train.o
g++ -c -O3 guessing_cpu.cpp -o guessing_cpu.o
g++ -c -O3 md5.cpp -o md5.o
if [ $? -ne 0 ]; then
    echo "Error: g++ compilation failed. Aborting."
    exit 1
fi

# --- 链接所有对象文件 ---
echo "    -> Linking all object files..."
g++ main.o train.o guessing_cpu.o guessing_gpu.o md5.o -o $EXECUTABLE -O3 -L/usr/local/cuda/lib64 -lcudart
if [ $? -ne 0 ]; then
    echo "Error: Linking failed. Aborting."
    exit 1
fi

echo "==> Build successful! Executable is '$EXECUTABLE'"
echo ""

# ==============================================================================
# Part 2: Execution
# ==============================================================================
echo "==> [Phase 2/3] Running the program..."
echo "    -> Standard output will be saved to: $STDOUT_FILE"
echo "    -> Standard error will be saved to:  $STDERR_FILE"
echo "    -> This may take a while..."

# 预检查训练文件
if [ ! -f "$TRAINING_FILE" ]; then
    echo "Warning: Training file not found at '$TRAINING_FILE'. The program will likely fail."
    echo "Make sure '$TRAINING_FILE' is in the same directory as this script,"
    echo "and main.cpp is set to read from a relative path."
fi

# --- 执行程序并重定向输出 ---
$EXECUTABLE 1> "$STDOUT_FILE" 2> "$STDERR_FILE"

# 捕获程序退出码
EXIT_CODE=$?

# ==============================================================================
# Part 3: Cleanup & Final Report
# ==============================================================================
echo "==> [Phase 3/3] Cleaning up build files and reporting status..."

# 清理中间文件 - 明确指定要删除的文件，避免误删 test.o
echo "    -> Removing intermediate files: main.o, train.o, guessing_cpu.o, guessing_gpu.o, md5.o, and main"
rm -f main.o train.o guessing_cpu.o guessing_gpu.o md5.o $EXECUTABLE

# 检查程序是否成功运行
if [ $EXIT_CODE -eq 0 ]; then
    echo "==> Program finished successfully."
    if [ -s "$STDERR_FILE" ]; then
        echo "==> Note: Some messages were written to the error log ($STDERR_FILE)."
    else
        rm "$STDERR_FILE" # 如果错误文件为空，就删除它
    fi
    echo "==> Final output has been saved to '$STDOUT_FILE'."
else
    echo "==> Program exited with an error (Exit Code: $EXIT_CODE)."
    echo "==> Please check '$STDERR_FILE' for error details."
fi

echo "==> Script finished. Only log files '$STDOUT_FILE' and potentially '$STDERR_FILE' remain."