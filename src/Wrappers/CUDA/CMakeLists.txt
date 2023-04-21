cmake_minimum_required (VERSION 3.19 FATAL_ERROR)
project (${PROJECT_NAME} LANGUAGES CXX)

add_library (WrappersCUDA
	ErrorException.hpp
	ErrorException.cpp
	MyLittleCUDAWrapper.cuh
	MyLittleCUDAWrapper.cu)

target_include_directories (WrappersCUDA PUBLIC ./)

set_target_properties(WrappersCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(WrappersCUDA PUBLIC cxx_std_17)

target_link_libraries (WrappersCUDA PUBLIC imgui)