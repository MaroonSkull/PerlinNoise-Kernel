cmake_minimum_required (VERSION 3.19)
project ("PerlinNoise-Kernel" LANGUAGES CXX)

#include CUDA
find_package (CUDAToolkit)

add_library (${PROJECT_NAME})

#include code
add_subdirectory ("src")

#include 3rd party libraries
#add_subdirectory ("dependencies")

#link libraries
#target_link_libraries (${PROJECT_NAME} PRIVATE ${library})]]



# as this is PUBLIC, anything that links to ${PROJECT_NAME} will also build with cxx_std_17
target_compile_features (${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties (${PROJECT_NAME} PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_SEPARABLE_COMPILATION ON
	PUBLIC_HEADER Perlin.cuh
)

#[[add_library (Common INTERFACE
	src/Fundamental.cuh
	src/Kernels.cuh
)
target_include_directories (Common INTERFACE src)
]]
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}-src)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)