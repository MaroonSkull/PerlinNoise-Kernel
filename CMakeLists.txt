cmake_minimum_required(VERSION 3.23)

project(PNK
  VERSION 1.1
  DESCRIPTION "Library for compute Multidimensional Perlin Noise via NVidia CUDA."
  HOMEPAGE_URL https://github.com/MaroonSkull/PerlinNoise-Kernel
  LANGUAGES CUDA CXX
)

include(GNUInstallDirs)

# setup CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set (CMAKE_CUDA_STANDARD 17)
else()
  message(FATAL_ERROR "CUDA not found")
endif(CMAKE_CUDA_COMPILER)

add_subdirectory (dependencies/cuda-api-wrappers)

add_library(${PROJECT_NAME}
  INTERFACE
  src/Perlin.hpp
)
# as this is PUBLIC, anything that links to ${PROJECT_NAME} will also build with cxx_std_17
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  PUBLIC_HEADER Perlin.hpp
)
target_link_libraries(${PROJECT_NAME}
  INTERFACE
    cuda-api-wrappers::runtime-and-driver
)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
