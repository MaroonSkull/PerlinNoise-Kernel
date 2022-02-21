#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdint.h>

// Объявляем функции
template<typename T>
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, int numSteps, int controlPoints, int resultDotsCols, int octaveNum);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// Source of OpenGL vertex shader
const char *vertexShaderSource =	"#version 330 core\n"
									"layout (location = 0) in vec3 aPos;\n"
									"void main() {\n"
									"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
									"}\0";

// Source of fragment shader
const char *fragmentShaderSource =	"#version 330 core\n"
									"out vec4 FragColor;\n"
									"void main() {\n"
									"	FragColor = vec4(1.0f, 0.25f, 0.25f, 1.0f)\n;"
									"}\0";

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
void Perlin1D_kernel(T *res, T *octave, const T *k, int size, T step, int numSteps, int octaveNum) {
	uint32_t id = blockIdx.x*blockDim.x+threadIdx.x;// [0..] – всего точек для просчёта
	uint32_t n = static_cast<T>(id) * step;			// 0 0 / 1 1 / 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	uint32_t dotNum = id % numSteps;				// 0 1 / 0 1 / 0 1 / .. – какую позицию занимает точка между левой и правой функцией
	T t = dotNum * step;							// 0.33 0.66 / 0.33 0.66 / .. – численное значение точки для интерполяции
	t = smootherstep_kernel<T>(t);		// Применяем сигмоидальную функцию для сглаживания входного параметра и, следовательно, итогового шума
	T y0 = k[n] * t;					// kx+b (b = 0)
	T y1 = k[n+1] * (t - 1);			// kx+b (b = -k) = k(x-1)
	res[id] = lerp_kernel<T>(y0, y1, t);// Интерполяцией находим шум, пишем сразу в выходной массив
	
	// Если нужно вычислять октавы, делаем это
	if(octaveNum != 0) {
		// Сохраняем в глобальной памяти первую октаву шума
		if(id % 2 == 0)
			octave[id >> 1] = res[id] * 0.5;

		// синхронизируем выполнение, убеждаясь, что все данные октавы заполнены
		__syncthreads();

		// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
		for(int j = 1; j <= octaveNum; j++) {
			int octavePov = 1 << j;
			for(int i = 0; i < octavePov; i++) {
				if((id < (i + 1) * size / octavePov) && (id >= i * size / octavePov))
					res[id] += octave[(id - i * size / octavePov) * (octavePov >> 1)] / (octavePov >> 1);
			}
		}
	}
}

int main() {
	constexpr int32_t controlPoints = 6;
	constexpr int32_t numSteps = 204; // пока что код хоста, запускающий cuda, не позволяет запускать более 1024 потоков
	constexpr int32_t octaveNum = 1;
	constexpr int32_t resultDotsCols = (controlPoints - 1) * numSteps; // 204 * (6-1) = 1020 потоков
	constexpr float step = 1.0f / numSteps;
	constexpr float k[controlPoints] = {.6f, -.3f, 1.0f, -.6f, -0.1, .6f}; // значения наклонов на углах отрезков (последний наклон равен первому)
	// Perlin noise coords
	float noise[resultDotsCols] = {};
	float vertices[3 * resultDotsCols] = {}; //x, y, z to 1 dot -> length = 3*cols

	// Create OpenGL 3.3 context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	GLFWwindow *window = glfwCreateWindow(1800, 600, "Perlin Noise Generator", nullptr, nullptr);
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
	cudaError_t cudaStatus = Perlin1DWithCuda<float>(noise, k, step, numSteps, controlPoints, resultDotsCols, octaveNum);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "Perlin1DWithCuda failed!\r\n";
		return -5;
	}

	// Save dots into 3d coords
	for(int i = 0; i < resultDotsCols; i++) {
		vertices[3 * i] = 2 * static_cast<float>(i) / static_cast<float>(resultDotsCols) - 1; // x = 2x(norm)-1, нормализуем и смещаем влево
		vertices[3 * i + 1] = noise[i]; // y
		/*std::cout << "x[" << i << "] = " << vertices[3 * i] << "\t"
					<< "y[" << i << "] = " << vertices[3 * i + 1]	<< "\t"
					<< "z[" << i << "] = " << vertices[3 * i + 2]	<< "\r\n";/**/
	}

	// Create vertex array object.
	uint32_t VAO;
	glGenVertexArrays(1, &VAO);
	std::cout << "Vertex array object have been created with ID = " << VAO << "\r\n";

	// Связываем объект вершинного массива.
	glBindVertexArray(VAO);
	
	// Create vertex buffer object.
	uint32_t VBO;
	glGenBuffers(1, &VBO);
	std::cout << "Vertex buffer object have been created with ID = " << VBO << "\r\n";
	
	// Связываем буфер. Теперь все вызовы буфера с параметром GL_ARRAY_BUFFER
	// будут использоваться для конфигурирования созданного буфера VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	// Копируем данные вершин в память связанного буфера
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Сообщаем, как OpenGL должен интерпретировать данные вершин,
	// которые мы храним в vertices[]
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	// Create vertex shader
	uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);

	// Compile vertex shader source code
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);

	// Check vertex shader compile errors
	int32_t success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		return -2;
	} else std::cout << "Vertex shader have been compiled!\r\n";

	// Create and compile fragment shader
	uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);

	// Check fragment shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		return -3;
	}
	else std::cout << "Fragment shader have been compiled!\r\n";

	// Создаём объект шейдерной программы
	uint32_t shaderProgram = glCreateProgram();

	// Прикрепляем наши шейдеры к шейдерной программе
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// Check shader program linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if(!success) {
		glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		return -4;
	}
	else std::cout << "Shader program have been linked!\r\n";

	// Delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// Create render cycle
	while(!glfwWindowShouldClose(window)) {
		// Input processing
		processInput(window);

		// Rendering
		// Активируем созданный объект
		glUseProgram(shaderProgram);

		// Отменяем связывание???
		glBindVertexArray(VAO);

		// Рисуем ось OX


		// Рисуем шум Перлина
		glDrawArrays(GL_LINE_STRIP, 0, resultDotsCols);

		// Swap buffers
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
cudaError_t Perlin1DWithCuda(T *res, const T *k, T step, int numSteps, int controlPoints, int resultDotsCols, int octaveNum) {
	T *dev_res = 0; // pointer to result array in VRAM
	T *dev_octave = 0; // pointer to temp array in VRAM
	T *dev_k = 0; // pointer to array with tilt angle (tg slope angle) in VRAM
	cudaError_t cudaStatus;

	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\r\n";
		goto Error;
	}

	// Allocate GPU buffers for arrays.
	cudaStatus = cudaMalloc((void **)&dev_res, resultDotsCols * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << "cudaMalloc failed!\r\n";
		goto Error;
	}

	// Массив для октав займёт максимально в 2 раза меньше памяти.
	cudaStatus = cudaMalloc((void **)&dev_octave, resultDotsCols * sizeof(T) / 2);
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
	Perlin1D_kernel<T> <<<1, resultDotsCols>>> (dev_res, dev_octave, dev_k, resultDotsCols, step, resultDotsCols/(controlPoints-1), octaveNum);

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

// Обработка ресайза окна
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

// Обработка всех событий ввода: запрос GLFW о нажатии/отпускании клавиш на клавиатуре в данном кадре и соответствующая обработка данных событий
void processInput(GLFWwindow *window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}