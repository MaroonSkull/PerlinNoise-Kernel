#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

template<typename T>
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, int numSteps, int controlPoints, int resultDotsCols);

/**
* линейная интерполяция точки t на промежутке [0, 1] между двумя прямыми с наклонами k0 и k1 соответственно.
* 
* \param k0 – значение наклона прямой в точке 0.
* \param k1 – значение наклона прямой в точке 1.
* \param t – точка, значение в которой интерполируется.
* 
* \return Результат интерполяции.
*/
template <typename T>
__device__ inline
T lerp_kernel(T k0, T k1, T t) {
	return fma(t, k1 - k0, k0); // (1-t)*k0 + t*k1 = k0 - t*k0 + t*k1 = t*(k1 - k0) + k0
}


/**
* 
*/
template <typename T>
__device__ inline
T smootherstep_kernel(T x) {
	return fma(static_cast<T>(6), x * x, fma(static_cast<T>(-15), x, static_cast<T>(10))) * x * x * x; // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
}

/**
* Одна октава шума Перлина на промежутке [n, n+1] в точке t
* 
* \param res – массив с результатом вычисления шума перлина на оси.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* 
* \return res – функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1D_kernel(T *res, const T *k, T step, int numSteps) {
	int id = threadIdx.x;		// [0..] – всего точек для просчёта
	int n = static_cast<T>(id) * step;			// 0 0 / 1 1 / 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	int dotNum = id % numSteps;	// 0 1 / 0 1 / 0 1 / .. – какую позицию занимает точка между левой и правой функцией
	T t = dotNum * step;		// 0.33 0.66 / 0.33 0.66 / .. – численное значение точки для интерполяции
	res[id] = lerp_kernel<T>(k[n], k[n+1], smootherstep_kernel<T>(t));
}

int main() {
	constexpr int controlPoints = 5;
	constexpr int numSteps = 10;
	constexpr int resultDotsCols = controlPoints * numSteps;
	constexpr float step = 1.0 / numSteps;
	const float k[controlPoints] = {-1.0, 0.2, 0.8, -0.3, -1.0}; // значения наклонов на углах отрезков (последний наклон равен первому)
	float noise[resultDotsCols] = {0};

	// Calculate Perlin in parallel.
	cudaError_t cudaStatus = Perlin1DWithCuda<float>(noise, k, step, numSteps, controlPoints, resultDotsCols);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "Perlin1DWithCuda failed!\r\n";
		return 1;
	}

	for (int i = 0; i < resultDotsCols; i++)
		std::cout << "noise[" << i << "] = " << noise[i] << "\r\n";

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();
	return 0;
}

/**
* Вспомогательная функция для вычисления шума Перлина на оси с использованием GPU.
*
* \param res – массив с результатом вычисления шума перлина на оси.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* \param controlPoints – количество узлов.
* \param resultDotsCols - количество точек для просчёта.
* 
* \return res – функция изменяет переданный массив.
* \return cudaError_t
*/
template<typename T>
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, int numSteps, int controlPoints, int resultDotsCols) {
	T *dev_res = 0;
	T *dev_k = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\r\n";
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void **)&dev_res, resultDotsCols * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaMalloc failed!\r\n";
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_k, controlPoints * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaMalloc failed!\r\n";
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_k, k, controlPoints * sizeof(T), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaMemcpy failed!\r\n";
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	Perlin1D_kernel<T> <<<1, resultDotsCols>>> (dev_res, dev_k, step, resultDotsCols/controlPoints);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "addKernel launch failed: %s\n" << cudaGetErrorString(cudaStatus) << "\r\n";
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaDeviceSynchronize returned error code %d after launching addKernel!\n" << cudaStatus << "\r\n";
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(res, dev_res, resultDotsCols * sizeof(T), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaMemcpy failed!\r\n";
		goto Error;
	}

Error:
	cudaFree(dev_res);
	cudaFree(dev_k);

	return cudaStatus;
}
