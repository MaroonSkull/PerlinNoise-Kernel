cmake_minimum_required (VERSION 3.19)
project (${PROJECT_NAME})
# here we add all dependencies

# glad
add_library (glad
	glad/include/glad/glad.h
	glad/src/glad.c)

target_include_directories(glad PUBLIC glad/src)
target_include_directories(glad PUBLIC glad/include)

# GLFW
set (GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set (GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory (glfw)

# Dear ImGui
add_library (imgui
		imgui/imgui.cpp
		imgui/imgui.h
		imgui/imgui_demo.cpp
		imgui/imgui_draw.cpp
		imgui/imgui_tables.cpp
		imgui/imgui_internal.h
		imgui/imgui_widgets.cpp
		imgui/imstb_rectpack.h
		imgui/imstb_textedit.h
		imgui/imstb_truetype.h
		imgui/backends/imgui_impl_glfw.cpp
		imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories (imgui PUBLIC imgui/backends)
target_include_directories (imgui PUBLIC imgui/)
target_compile_definitions (imgui PUBLIC -DIMGUI_IMPL_OPENGL_LOADER_GLAD)

# public linkage
target_link_libraries (imgui PUBLIC glad glfw)
