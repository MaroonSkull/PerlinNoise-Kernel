cmake_minimum_required (VERSION 3.19)
project ("CMakeCUDA" LANGUAGES CXX CUDA)

add_executable (${PROJECT_NAME} "src/Main.cpp")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set (CUDAKERNELS_DIR "CUDAKernels")
set (ENTITIES_DIR "Entities")
set (SERVICES_DIR "Services")
set (SHADERS_DIR "Shaders")
set (WRAPPERSCUDA_DIR "Wrappers/CUDA")

#include OpenGL
find_package (OpenGL REQUIRED)

#include CUDA
find_package (CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#include 3rd party libraries
add_subdirectory ("dependencies")

#link libraries
foreach (library OpenGL::GL imgui)
    target_link_libraries (${PROJECT_NAME} PRIVATE ${library})
endforeach ()

# include code
add_subdirectory ("src")

foreach (library ${CUDAKERNELS_DIR} ${ENTITIES_DIR} ${SERVICES_DIR} ${SHADERS_DIR} WrappersCUDA)
    target_link_libraries (${PROJECT_NAME} PRIVATE ${library})
endforeach ()