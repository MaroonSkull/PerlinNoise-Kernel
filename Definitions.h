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

// Объявляем функции
template<typename T>
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, uint32_t numSteps, uint32_t controlPoints, uint32_t resultDotsCols, uint32_t octaveNum);

void framebuffer_size_callback(GLFWwindow *window, int32_t width, int32_t height);

void processInput(GLFWwindow *window);

#endif