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
		include/${PROJECT_NAME}/Fundamental.cuh
		include/${PROJECT_NAME}/Kernels.cuh
		include/${PROJECT_NAME}/Perlin.cuh
		include/${PROJECT_NAME}/Params.hpp
)

target_compile_features (${PROJECT_NAME}
	INTERFACE
		cxx_std_17
)

# Установка файлов
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}Targets
	FILE_SET the_${PROJECT_NAME}_headers
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE PNK::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PNK
)

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate pkg-config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/PNKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PNK
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PNK
)

# Add possibility to find_package without install
export(EXPORT ${PROJECT_NAME}Targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        NAMESPACE ${PROJECT_NAME}::)