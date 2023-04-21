cmake_minimum_required (VERSION 3.19 FATAL_ERROR)
project (${PROJECT_NAME} LANGUAGES CXX)

add_library (Shaders STATIC
	1D/1D.frag
	1D/1D.vert
	1D/1D_noise.vert
	1D/1D_white.vert
	Shader.hpp
	Shader.cpp)

target_include_directories (Shaders PUBLIC 1D)
target_include_directories (Shaders PUBLIC ./)

target_compile_features(Shaders PUBLIC cxx_std_17)
target_link_libraries (Shaders PRIVATE glad)