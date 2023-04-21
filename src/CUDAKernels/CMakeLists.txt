cmake_minimum_required (VERSION 3.19 FATAL_ERROR)
project (${PROJECT_NAME} LANGUAGES CXX CUDA)

add_library (CUDAKernels STATIC
	Fundamental.cu
	Fundamental.cuh
	Kernels.cu
	Kernels.cuh
	Preinitialize.cu
	Preinitialize.cuh)

target_include_directories (CUDAKernels PUBLIC ./)

set_target_properties(CUDAKernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(CUDAKernels PUBLIC cxx_std_17)
target_link_libraries (CUDAKernels PUBLIC imgui)