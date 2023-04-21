cmake_minimum_required (VERSION 3.19 FATAL_ERROR)
project (${PROJECT_NAME} LANGUAGES CXX)

add_library (Entities STATIC
	Interfaces/IDrawable.hpp
	Interfaces/IDrawable.cpp
	Interfaces/ICalculable.hpp
	Background.hpp
	Background.cpp
	GUI.hpp
	GUI.cpp
	Perlin.hpp
	Perlin.cpp
	Plot.hpp
	Plot.cpp)

target_include_directories (Entities PUBLIC Interfaces)
target_include_directories (Entities PUBLIC ./)

target_compile_features(Entities PUBLIC cxx_std_17)
target_link_libraries (Entities PUBLIC imgui)