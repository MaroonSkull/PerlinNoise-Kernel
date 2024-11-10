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

# Установка файлов
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}Targets
	FILE_SET the_${PROJECT_NAME}_headers
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Генерация конфигурационного файла
include(CMakePackageConfigHelpers)

configure_package_config_file(
    cmake/PNKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/PNK
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PNKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/PNK
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE PNK::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/PNK
)