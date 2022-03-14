#ifndef __definitions_h__
#define __definitions_h__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>		// getline
#include <iostream>		// cout
#include <fstream>		// ifstream
#include <stdint.h>		// объявление int
#include <typeinfo>		// typeid
#include "Shader.h"		// shader class

// Объявляем функции
extern "C" cudaError_t
Perlin1DWithCuda_f(float *noise, const float *k, float step, uint32_t numSteps, uint32_t controlPoints, uint32_t resultDotsCols, uint32_t octaveNum);
extern "C" cudaError_t
Perlin1DWithCuda_d(double *noise, const double *k, double step, uint32_t numSteps, uint32_t controlPoints, uint32_t resultDotsCols, uint32_t octaveNum);

// Pathes to source of OpenGL vertex shader & fragment 
template <typename T>
int Perlin1D(const char *vertexShaderPathNoise, const char *fragmentShaderPath);
template <typename T>
int Perlin2D(const char *vertexShaderPath, const char *fragmentShaderPath);

// Functions for OpenGL
void framebuffer_size_callback(GLFWwindow *window, int32_t width, int32_t height);
void framebuffer_size_callback_texture_edition(GLFWwindow *window, int32_t width, int32_t height);
void processInput(GLFWwindow *window);

#endif