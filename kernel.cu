#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

template<typename T>
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, int numSteps, int controlPoints, int resultDotsCols);

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

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
* Сигмоидальная функция из семейства smoothstep, используется для создания более интенсивного градиента шума. 
* Подробнее см. https://en.wikipedia.org/wiki/Smoothstep#Variations
* 
* \param x – значение градиента (он же t)
* 
* \return возвращает классический smootherstep(x). Используется оригинальный второй полином Кена Перлина.
*/
template <typename T>
__device__ inline
T smootherstep_kernel(T x) {
	return fma(static_cast<T>(6), x * x, fma(static_cast<T>(-15), x, static_cast<T>(10))) * x * x * x; // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
}

/**
* Одна октава одномерного шума Перлина на промежутке [n, n+1] в точке t
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
	int id = threadIdx.x;						// [0..] – всего точек для просчёта
	int n = static_cast<T>(id) * step;			// 0 0 / 1 1 / 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	int dotNum = id % numSteps;					// 0 1 / 0 1 / 0 1 / .. – какую позицию занимает точка между левой и правой функцией
	T t = dotNum * step;						// 0.33 0.66 / 0.33 0.66 / .. – численное значение точки для интерполяции
	res[id] = lerp_kernel<T>(k[n], k[n+1], smootherstep_kernel<T>(t));
}

int main() {
	constexpr int controlPoints = 5;
	constexpr int numSteps = 10;
	constexpr int resultDotsCols = (controlPoints - 1) * numSteps;
	constexpr float step = 1.0f / numSteps;
	const float k[controlPoints] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // значения наклонов на углах отрезков (последний наклон равен первому)
	float noise[resultDotsCols] = {0};

	// Create OpenGL 3.3 context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	GLFWwindow *window = glfwCreateWindow(400, 200, "Perlin Noise Generator", nullptr, nullptr);
	if(window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Setting up viewport
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Устанавливаем callback на изменение размеров окна

	// Initialize GLAD
	if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// Calculate Perlin in parallel.
	cudaError_t cudaStatus = Perlin1DWithCuda<float>(noise, k, step, numSteps, controlPoints, resultDotsCols);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "Perlin1DWithCuda failed!\r\n";
		return 1;
	}

	// Print dots to console
	for (int i = 0; i < resultDotsCols; i++)
		std::cout << "noise[" << i << "] = " << noise[i] << "\r\n";/**/

	// Create render cycle
	while(!glfwWindowShouldClose(window)) {
		// Input processing
		processInput(window);

		// Rendering
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();
	// glfwTerminate must be called before exiting in order for clean up
	glfwTerminate();
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
	Perlin1D_kernel<T> <<<1, resultDotsCols>>> (dev_res, dev_k, step, resultDotsCols/(controlPoints-1));

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

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

// Обработка всех событий ввода: запрос GLFW о нажатии/отпускании клавиш на клавиатуре в данном кадре и соответствующая обработка данных событий
void processInput(GLFWwindow *window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}