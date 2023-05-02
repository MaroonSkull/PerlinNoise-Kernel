cmake_minimum_required (VERSION 3.19)
project (${PROJECT_NAME}-src LANGUAGES CXX CUDA)

#include CUDA
find_package (CUDAToolkit)

add_library (
	${PROJECT_NAME}
	Params.hpp
	Perlin.cuh
	Perlin.cu
	Fundamental.cuh
	Kernels.cuh
)

set_target_properties (${PROJECT_NAME} PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_SEPARABLE_COMPILATION ON
	PUBLIC_HEADER Perlin.cuh
)
target_include_directories (${PROJECT_NAME} PUBLIC /)

#mb it must be public linkage, not an interface
#[[add_library (Common INTERFACE
	src/Fundamental.cuh
	src/Kernels.cuh
)
target_include_directories (Common INTERFACE src)

target_link_libraries(${PROJECT_NAME} INTERFACE Common)]]

target_link_libraries(${PROJECT_NAME}
	PRIVATE CUDA::cudart)