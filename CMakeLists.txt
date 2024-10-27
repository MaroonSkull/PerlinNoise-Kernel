cmake_minimum_required (VERSION 3.23...3.31 FATAL_ERROR)

project (PNK
    VERSION 1.5.2
    DESCRIPTION "Perlin noise generation library with NVidia CUDA"
    HOMEPAGE_URL https://github.com/MaroonSkull/PerlinNoise-Kernel
    LANGUAGES CXX CUDA
)

add_library (${PROJECT_NAME} INTERFACE)

target_sources (${PROJECT_NAME}
	INTERFACE
	FILE_SET
		the_${PROJECT_NAME}_headers
	TYPE
		HEADERS
	BASE_DIRS
		include
	FILES
		include/pnk/Fundamental.cuh
		include/pnk/Kernels.cuh
		include/pnk/Perlin.cuh
		include/pnk/Params.hpp
)

target_compile_features (${PROJECT_NAME}
	INTERFACE
		cxx_std_17
)