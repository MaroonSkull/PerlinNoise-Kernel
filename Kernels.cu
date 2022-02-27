#include "Definitions.h"

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
* Вычисление одномерного шума Перлина.
* Вычисляет массив любой длины, допустимой видеокартой (для CC3.0+ это (2^31 − 1)*2^10 ≈ 2.1990233e+12 значений)
* 
* \param noise – массив с результатом вычисления шума перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param size – длина массива noise.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* \param isOctaveCalkNeed – будут ли в дальнейшем вычисляться октавы.
* 
* \return noise – (см. описание параметра) функция изменяет переданный массив (хранится в памяти GPU).
* \return octave – (см. описание параметра) функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1D_kernel(T *noise, T *octave, const T *k, uint32_t size, T step, uint32_t numSteps, bool isOctaveCalkNeed) {
	uint32_t id = blockIdx.x*blockDim.x+threadIdx.x;// [0..] – всего точек для просчёта
	if(id >= size) return;
	uint32_t n = static_cast<T>(id) * step;			// 0 0 / 1 1 / 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	uint32_t dotNum = id % numSteps;				// 0 1 / 0 1 / 0 1 / .. – какую позицию занимает точка между левой и правой функцией
	T t = dotNum * step;							// 0.33 0.66 / 0.33 0.66 / .. – численное значение точки для интерполяции
	t = smootherstep_kernel<T>(t);					// Применяем сигмоидальную(на промежутке [0, 1]) функцию, реализуя градиент
	T y0 = k[n] * t;								// kx+b (b = 0)
	T y1 = k[n+1] * (t - 1);						// kx+b (b = -k) = k(x-1)
	noise[id] = lerp_kernel<T>(y0, y1, t);			// Интерполяцией находим шум, пишем сразу в выходной массив

	// Если пользователю нужно вычислять октавы, сохраняем в памяти первую окатву шума
	if(isOctaveCalkNeed)
		// Первая октава занимает в два раза меньше памяти, чем исходный шум
		if(id % 2 == 0)
			octave[id >> 1] = noise[id] * 0.5;
}

/**
* Накладывает на готовый одномерный шум Перлина указанное количество октав.
* Данная версия алгоритма предполагает, что в разделяемую память полностью помещается первая октава.
* Это позволяет вычислять октавы для шума fp64 длиной вплоть до 8192, либо fp32 до 16384 значений.
*
* \param noise – массив с результатом наложения октав на шум Перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param size – количество изменяемых значений шума, длина массива noise.
* \param octaveNum – количество октав.
*
* \return noise – функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1Doctave_shared_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64.
	*/

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= size) return;

	if(size > 2*sharedOctaveLength) {
		if (id == 0) printf("size = %d, 2*sharedOctaveLength = %d. exit.\r\n\r\n", size, 2*sharedOctaveLength);
		return;
	}

	// Сохраняем в разделяемой памяти первую октаву шума
	// 32 - warp size, именно столько потоков на всех картах nvidia параллельно выполняют этот код.
	// Выполняем целочисленное деление с округлением вверх, чтобы скопировать всю октаву.
	uint32_t maxI = (size % 32 == 0) ? (size / 32) : (size / 32 + 1);
	for(uint32_t i = 0; i < maxI; i++) {
		uint32_t sharedId = 32 * i + threadIdx.x;
		if(size > 2 * sharedId) // контроллируем выход за пределы массива
			sharedOctave[sharedId] = octave[sharedId] * 0.5;
	}

	// Синхронизируем выполнение на уровне блока.
	__syncthreads();
	// На этом моменте вся первая октава записана в разделяемую память данного блока
		
	// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
	for(int j = 1; j <= octaveNum; j++) {
		int octavePov = 1 << j;
		for(int i = 0; i < octavePov; i++) {
			if((id < (i + 1) * size / octavePov) && (id >= i * size / octavePov))
				noise[id] += sharedOctave[(id - i * size / octavePov) * (octavePov >> 1)] / (octavePov >> 1);
		}
	}
}

/**
* Накладывает на готовый одномерный шум Перлина указанное количество октав.
* Данная версия алгоритма позволяет накладывать на шум октавы произвольной длины.
*
* \param noise – массив с результатом наложения октав на шум Перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param size – количество изменяемых значений шума, длина массива noise.
* \param octaveNum – количество октав.
*
* \return noise – функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1Doctave_shared_unlimited_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64.
	*/

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= size) return;

	noise[id] = -1.f;

	// Сохраняем в разделяемой памяти первую октаву шума
	// 32 - warp size, именно столько потоков на всех картах nvidia параллельно выполняют этот код.
	// Выполняем целочисленное деление с округлением вверх, чтобы скопировать всю октаву.
	uint32_t maxI = (sharedOctaveLength % blockDim.x == 0) ? (sharedOctaveLength / blockDim.x) : (sharedOctaveLength / blockDim.x + 1);
	for(uint32_t i = 0; i < maxI; i++) {
		uint32_t sharedId = blockDim.x * i + threadIdx.x;
		if(size > 2 * sharedId) // контроллируем выход за пределы массива
			sharedOctave[sharedId] = octave[sharedId] * 0.5;
	}

	// Синхронизируем выполнение на уровне блока.
	__syncthreads();
	// На этом моменте вся первая октава записана в разделяемую память данного блока

	// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
	for(int j = 1; j <= octaveNum; j++) {
		int octavePov = 1 << j;
		for(int i = 0; i < octavePov; i++) {
			if((id < (i + 1) * size / octavePov) && (id >= i * size / octavePov))
				noise[id] = sharedOctave[(id - i * size / octavePov) * (octavePov >> 1)] / (octavePov >> 1);
		}
	}
}

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
template<typename T>
cudaError_t Perlin1DWithCuda(T *noise, const T *k, T step, uint32_t numSteps, uint32_t controlPoints, uint32_t resultDotsCols, uint32_t octaveNum) {
	T *dev_noise = nullptr; // pointer to noiseult array in VRAM
	T *dev_octave = nullptr; // pointer to temp array in VRAM
	T *dev_k = nullptr; // pointer to array with tilt angle (tg slope angle) in VRAM
	cudaError_t cudaStatus;

	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\r\n";
		goto Error;
	}

	// Allocate GPU buffers for arrays.
	cudaStatus = cudaMalloc((void **)&dev_noise, resultDotsCols * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMalloc failed!\r\n";
		goto Error;
	}

	// Массив для октав займёт в 2 раза меньше памяти.
	if(octaveNum > 0) {
		cudaStatus = cudaMalloc((void **)&dev_octave, resultDotsCols * sizeof(T) / 2);
		if(cudaStatus != cudaSuccess) {
			std::cout << stderr << ": cudaMalloc failed!\r\n";
			goto Error;
		}
	}

	cudaStatus = cudaMalloc((void **)&dev_k, controlPoints * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMalloc failed!\r\n";
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_k, k, controlPoints * sizeof(T), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMemcpy failed!\r\n";
		goto Error;
	}

	dim3 threadsPerBlock(resultDotsCols, 1, 1);
	dim3 blocksPerGrid(1, 1, 1);

	// 256 взято с потолка из каких-то общих соображений, забейте.
	if(resultDotsCols > 256) {
		threadsPerBlock.x = 256;
		blocksPerGrid.x = (resultDotsCols % 256 == 0) ? resultDotsCols / 256 : resultDotsCols / 256 + 1;
	}

	// Launch a kernel on the GPU with one thread for each element.
	Perlin1D_kernel<T> <<<blocksPerGrid, threadsPerBlock>>>
		(dev_noise, dev_octave, dev_k, resultDotsCols, step, numSteps, static_cast<bool>(octaveNum));

	/*bool isOctaveCalkNeed = octaveNum > 0 ? true : false;
	void *args[] = {&dev_noise, &dev_octave, &dev_k, &step, &numSteps, &isOctaveCalkNeed};
	cudaLaunchCooperativeKernel((void *)Perlin1D_kernel<T>, blocksPerGrid, threadsPerBlock, args, 0, 0);/**/
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1D_kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\r\n";
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaDeviceSynchronize returned " << cudaGetErrorString(cudaStatus) << " after launching Perlin1D_kernel!\r\n";
		goto Error;
	}

	// Выполняем наложение октав на получившийся шум.
	if(octaveNum)
		if(resultDotsCols <= 32 * 2048 / sizeof(T)) // если вся октава помещается в разделяемую память, вызываем простое ядро
			Perlin1Doctave_shared_kernel<T> <<<blocksPerGrid, threadsPerBlock>>> (dev_noise, dev_octave, resultDotsCols, octaveNum);
		else // если октава не помещается целиком, вызываем сложное ядро.
			Perlin1Doctave_shared_unlimited_kernel<T> <<<blocksPerGrid, threadsPerBlock>>> (dev_noise, dev_octave, resultDotsCols, octaveNum);
		
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1Doctave_shared(_unlimited)_kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\r\n";
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaDeviceSynchronize returned " << cudaGetErrorString(cudaStatus) << " after launching Perlin1Doctave_shared(_unlimited)_kernel!\r\n";
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(noise, dev_noise, resultDotsCols * sizeof(T), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMemcpy failed!\r\n";
		goto Error;
	}

Error:
	cudaFree(dev_noise);
	cudaFree(dev_octave);
	cudaFree(dev_k);

	return cudaStatus;
}