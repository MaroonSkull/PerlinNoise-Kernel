#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>
#include <stdint.h>
#include <string>
#include <vector>
#include "../Wrappers/CUDA/MyLittleCUDAWrapper.cuh"
#include "../Wrappers/CUDA/ErrorException.hpp"
#include "../Services/Window.hpp"
#include "../Shaders/Shader.hpp"
#include "../Services/Params.hpp"
#include "../Entities/Interfaces/IDrawable.hpp"
#include "../Entities/Interfaces/ICalculable.hpp"
#include "../Entities/Background.hpp"
#include "../Entities/Plot.hpp"
#include "../Entities/Perlin.hpp"
#include "../Entities/GUI.hpp"

int PerlinNoise(Window &MainWindow, std::string_view vertexShaderPathNoise, std::string_view vertexShaderPathLinear, std::string_view fragmentShaderPath);

// Functions for OpenGL
void GLFWErrorCallback(int error, const char *description);
void framebufferSizeCallback(GLFWwindow *window, int32_t width, int32_t height);
void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
