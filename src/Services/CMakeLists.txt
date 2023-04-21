cmake_minimum_required (VERSION 3.19 FATAL_ERROR)
project (${PROJECT_NAME} LANGUAGES CXX)

add_library (Services STATIC
	"Definitions.hpp"
	Params.cpp
	Params.hpp
	Window.cpp
	Window.hpp)

target_include_directories (Services PUBLIC ./)

target_compile_features(Services PUBLIC cxx_std_17)
target_link_libraries (Services PUBLIC imgui)