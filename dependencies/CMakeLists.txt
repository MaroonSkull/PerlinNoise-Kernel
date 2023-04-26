#[[cmake_minimum_required (VERSION 3.19)
project (${PROJECT_NAME} LANGUAGE CXX CUDA)
# here we add all dependencies

# cuda-api-wrappers
add_subdirectory (cuda-api-wrappers)

# public linkage
target_link_libraries (${PROJECT_NAME} PUBLIC cuda-api-wrappers::runtime-and-driver)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)]]