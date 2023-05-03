cmake_minimum_required (VERSION 3.19)
project (${PROJECT_NAME}-src LANGUAGES CXX CUDA)

#include CUDA
find_package (CUDAToolkit)

add_library (${PROJECT_NAME}
	PUBLIC
	Params.hpp
	Perlin.cuh
	Perlin.cu
	Fundamental.cuh
	Kernels.cuh
)

set_target_properties (${PROJECT_NAME}
	PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories (${PROJECT_NAME} PUBLIC /)

target_link_libraries (${PROJECT_NAME}
	INTERFACE CUDA::cudart
	INTERFACE CUDA::cuda_driver
)

add_library ("${pnk_namespace}::${PROJECT_NAME}" ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
	PROPERTIES
	EXPORT_NAME ${pnk_namespace}
	OUTPUT_NAME ${pnk_namespace}
)