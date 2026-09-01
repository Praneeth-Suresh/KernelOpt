cmake_minimum_required(VERSION 3.18)
project(vector_add_kernel CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Add executable
add_executable(vector_add_benchmark kernel.cu)

# Set CUDA architectures (adjust for your GPU)
# Common options:
# - sm_70: V100, P100
# - sm_80: A100, RTX 30-series
# - sm_89: RTX 40-series
set_target_properties(vector_add_benchmark PROPERTIES CUDA_ARCHITECTURES "80;89")

# Optimization flags
target_compile_options(vector_add_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -O3 -Xcompiler -march=native>
)

# Output directory
set_target_properties(vector_add_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
