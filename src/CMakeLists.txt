cmake_minimum_required (VERSION 3.19)
project (${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

#include our libraries
foreach (library ${CUDAKERNELS_DIR} ${ENTITIES_DIR} ${SERVICES_DIR} ${SHADERS_DIR} ${WRAPPERSCUDA_DIR})
    add_subdirectory (${library})
endforeach ()

#link libraries