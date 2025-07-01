#!/bin/bash

# ==============================================================================
# PCFG Password Guesser with GPU Hashing - Build, Run, and Clean Script
# ==============================================================================
#
# This script performs three main actions:
# 1. Compiles the entire project, including CPU code, GPU password generation,
#    and the new GPU MD5 hashing module.
# 2. Runs the compiled executable, redirecting output to log files.
# 3. Cleans up all intermediate build files and the executable.
#
# ==============================================================================

# --- Configuration Variables ---
set -e # Exit immediately if a command exits with a non-zero status.

# Set the CUDA Compute Capability for your GPU.
# sm_75: Tesla T4, V100, RTX 20-series
# sm_86: RTX 30-series, A100
# sm_61: GTX 10-series
CUDA_ARCH=sm_75

# File and executable names
EXECUTABLE="./main"
STDOUT_FILE="run_output.log" # Changed to a more descriptive name
STDERR_FILE="run_error.log"

# !! IMPORTANT !!: Ensure the training data file is in the same directory
# or use an absolute path in main.cpp
TRAINING_FILE="Rockyou-singleLined-full.txt"

# List of all C++ source files
CPP_SOURCES="main.cpp train.cpp guessing_cpu.cpp"

# List of all CUDA source files
CUDA_SOURCES="guessing_gpu.cu md5_gpu.cu"

# List of all object files that will be generated
OBJECT_FILES="main.o train.o guessing_cpu.o guessing_gpu.o md5_gpu.o"


# ==============================================================================
# Part 1: Compilation
# ==============================================================================
echo "==> [Phase 1/3] Starting Compilation..."

# Clean up old build artifacts and logs
echo "    -> Cleaning up old files..."
rm -f *.o $EXECUTABLE $STDOUT_FILE $STDERR_FILE

# --- Compile CUDA Source Files ---
for source_file in $CUDA_SOURCES; do
    object_file="${source_file%.cu}.o"
    echo "    -> Compiling CUDA code: $source_file -> $object_file"
    nvcc -c -O3 -arch=${CUDA_ARCH} "$source_file" -o "$object_file"
done

# --- Compile C++ Source Files ---
for source_file in $CPP_SOURCES; do
    object_file="${source_file%.cpp}.o"
    echo "    -> Compiling C++ code: $source_file -> $object_file"
    g++ -c -O3 "$source_file" -o "$object_file"
done

# --- Link All Object Files ---
echo "    -> Linking all object files to create executable: $EXECUTABLE"
g++ $OBJECT_FILES -o $EXECUTABLE -O3 -L/usr/local/cuda/lib64 -lcudart

echo "==> Build successful! Executable is '$EXECUTABLE'"
echo ""


# ==============================================================================
# Part 2: Execution
# ==============================================================================
echo "==> [Phase 2/3] Running the program..."
echo "    -> Standard output will be saved to: $STDOUT_FILE"
echo "    -> Standard error will be saved to:  $STDERR_FILE"
echo "    -> This may take a while..."

# Pre-flight check for the training file
if [ ! -f "$TRAINING_FILE" ]; then
    echo "Warning: Training file not found at '$TRAINING_FILE'. The program will likely fail."
fi

# --- Run the executable and redirect output ---
# Use 'set +e' to temporarily disable exit-on-error, so we can capture the exit code
set +e
$EXECUTABLE > "$STDOUT_FILE" 2> "$STDERR_FILE"
EXIT_CODE=$?
set -e # Re-enable exit-on-error

# ==============================================================================
# Part 3: Cleanup & Final Report
# ==============================================================================
echo "==> [Phase 3/3] Finalizing..."

# --- Report Status ---
if [ $EXIT_CODE -eq 0 ]; then
    echo "==> Program finished successfully."
    if [ -s "$STDERR_FILE" ]; then
        echo "==> Note: Some messages were written to the error log ($STDERR_FILE). Please review."
    else
        # If the error log is empty, remove it for a cleaner workspace
        rm -f "$STDERR_FILE"
    fi
    echo "==> Final output has been saved to '$STDOUT_FILE'."
else
    echo "==> ERROR: Program exited with a non-zero status (Exit Code: $EXIT_CODE)."
    echo "==> Please check '$STDERR_FILE' for detailed error messages."
fi

# --- Clean Up Build Artifacts ---
echo "    -> Cleaning up build artifacts (*.o files and the executable)..."
rm -f $OBJECT_FILES $EXECUTABLE

echo "==> Script finished."

# Return the program's exit code to the calling shell
exit $EXIT_CODE