cmake_minimum_required (VERSION 3.19)

if (WIN32 AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	# Avoid a build failure when the BUILD_SHARED_LIBS variable is set to true.
	set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")
endif ()

project (PerlinNoise-Kernel
    VERSION 1.5.1
    DESCRIPTION "Perlin noise generation algorithm with NVidia CUDA"
    HOMEPAGE_URL https://github.com/MaroonSkull/PerlinNoise-Kernel
    LANGUAGES CXX
)

include (GNUInstallDirs)
set (pnk_namespace "pnk")
set (pnk_lib "pnk_${PROJECT_NAME}")

#include CUDA
find_package (CUDAToolkit 9.0 REQUIRED)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")


add_library (${pnk_lib} INTERFACE)
target_compile_features (${pnk_lib} INTERFACE cxx_std_17)

target_include_directories (${pnk_lib}
    INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries (${pnk_lib} INTERFACE
    CUDA::cudart
	CUDA::cuda_driver
)

add_library ("${pnk_namespace}::${PROJECT_NAME}" ALIAS ${pnk_lib})

set_target_properties(${PROJECT_NAME}
	PROPERTIES
	EXPORT_NAME ${pnk_namespace}
	OUTPUT_NAME ${pnk_namespace}
)

#install lib
install(
	TARGETS ${pnk_lib}
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

#[[install(
	DIRECTORY src/
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	FILES_MATCHING REGEX "\\.(hpp|cuh)$"
)]]

#[[
#include code
add_subdirectory ("src")

#include 3rd party libraries
add_subdirectory ("dependencies")

#link libraries
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}-src)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)]]