#pragma once

#include "Perlin.cuh"
#include "cuda/runtime_api.hpp"
#include "Fundamental.cuh"
#include "Kernels.cuh"

/**
* Вспомогательная функция для вычисления шума Перлина на оси с использованием GPU.
*
* \param noise – массив с результатом вычисления шума перлина на оси.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* \param controlPoints – количество узлов.
* \param resultDotsCols - количество точек для просчёта.
* \param octaveNum - количество накладывающихся октав на шум.
*
* \return noise – функция изменяет переданный массив.
* \return cudaError_t
*/
template<typename T> void Perlin1D(T* vertices, const Params1D& p) {
	T* dev_vertices = nullptr; // pointer to noise array in VRAM
	T* dev_noise = nullptr; // pointer to noise array in VRAM
	T* dev_octave = nullptr; // pointer to temp array in VRAM
	T* dev_k = nullptr; // pointer to array with tilt angle (tg slope angle) in VRAM

	// Choose which GPU to run on.
	cudaSetDevice(0);

	// Allocate GPU buffers for arrays.
	cudaMalloc((void**)&dev_vertices, 2 * (p.resultDotsCols + 1) * sizeof(T));

	// последняя точка (+1) замыкает шум, она всегда = 0
	cudaMalloc((void**)&dev_noise, (p.resultDotsCols + 1) * sizeof(T));

	// Массив для октав займёт в 2 раза меньше памяти.
	if (p.octaveNum > 0)
		// округление вниз при целочисленном делении - не ошибка
		cudaMalloc((void**)&dev_octave, p.resultDotsCols * sizeof(T) / 2);

	cudaMalloc((void**)&dev_k, p.controlPoints * sizeof(T));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_k, p.k_.data(), p.controlPoints * sizeof(T), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(p.resultDotsCols, 1, 1);
	dim3 blocksPerGrid(1, 1, 1);

	// 256 взято с потолка из каких-то общих соображений, забейте.
	if (p.resultDotsCols > 256) {
		threadsPerBlock.x = 256;
		blocksPerGrid.x = (p.resultDotsCols + 255) / 256;
	}

	// Launch a kernel on the GPU with one thread for each element.
	Perlin1D_kernel<T> << <blocksPerGrid, threadsPerBlock >> >
		(dev_noise, dev_octave, dev_k, p.resultDotsCols, p.step, p.numSteps, static_cast<bool>(p.octaveNum));

	std::function<void(void)> sharedKernel;
	// Выполняем наложение октав на получившийся шум.
	if (p.octaveNum) {
		if (p.resultDotsCols <= 2 * 32 * 1024 / sizeof(T)) // если вся октава помещается в разделяемую память, вызываем простое ядро
			Perlin1Doctave_shared_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (dev_noise, dev_octave, p.resultDotsCols, p.octaveNum);
		else // если октава не помещается целиком, вызываем сложное ядро.
			Perlin1Doctave_shared_unlimited_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (dev_noise, dev_octave, p.resultDotsCols, p.octaveNum);
	}

	Perlin1Dvertices_kernel<T> << <blocksPerGrid, threadsPerBlock >> >
		(dev_vertices, dev_noise, p.resultDotsCols);

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(vertices, dev_vertices, 2 * (p.resultDotsCols + 1) * sizeof(T), cudaMemcpyDeviceToHost);

	// clean-up
	cudaFree(dev_vertices);
	cudaFree(dev_noise);
	cudaFree(dev_octave);
	cudaFree(dev_k);
}